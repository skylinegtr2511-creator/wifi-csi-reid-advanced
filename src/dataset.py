import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms


def extract_pid(filename: str) -> str:
    """
    Extract Person ID (PID) from Wi-PER81 heatmap filename.
    Handles both folder-organized PIDs and flat filenames (e.g., '000012mB_...' -> '000012mB').
    """
    stem = Path(filename).stem
    return stem.split("_")[0]


def get_wifi_transforms(img_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    RF-safe transformations for Wi-Fi CSI 2D magnitude heatmaps.
    Strictly avoids horizontal/vertical flips and rotations that corrupt physical
    time-step and OFDM subcarrier ordering.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        # Slight resize + RandomCrop simulates minor timing/positional shifts
        transforms.Resize((img_size + 16, img_size + 16)),
        transforms.RandomCrop((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        # Random rectangular erasing simulates Wi-Fi packet loss / multipath fading
        transforms.RandomErasing(
            p=0.5,
            scale=(0.02, 0.15),
            ratio=(0.3, 3.3),
            value="random",
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return train_transform, eval_transform


class WiPER81Dataset(Dataset):
    """
    Dataset loader for Wi-PER81 ImageData splits (D1, D2v, D2t).
    Supports loading either a single subset ('Gallery' or 'Probe') or merging both
    for Session A (D1) training.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "D1",
        subsets: Tuple[str, ...] = ("Gallery", "Probe"),
        transform: Optional[Callable] = None,
        pid_to_label: Optional[Dict[str, int]] = None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.subsets = subsets
        self.transform = transform

        self.samples: List[Tuple[Path, str]] = []
        valid_exts = {".png", ".jpg", ".jpeg"}

        for subset in subsets:
            subset_dir = self.root_dir / split / subset
            if not subset_dir.exists():
                raise FileNotFoundError(f"Expected dataset directory not found: {subset_dir}")

            # Recursively find all .png heatmaps (works whether nested by PID folder or flat)
            for path in sorted(subset_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in valid_exts:
                    # Use parent folder name if nested inside PID folder, else parse filename
                    pid = path.parent.name if path.parent != subset_dir else extract_pid(path.name)
                    self.samples.append((path, pid))

        if not self.samples:
            raise RuntimeError(f"No heatmap images found under {self.root_dir / split} for subsets {subsets}")

        # Build deterministic contiguous integer label mapping (0 .. num_classes-1)
        if pid_to_label is None:
            unique_pids = sorted({pid for _, pid in self.samples})
            self.pid_to_label = {pid: idx for idx, pid in enumerate(unique_pids)}
        else:
            self.pid_to_label = pid_to_label

        self.label_to_pid = {idx: pid for pid, idx in self.pid_to_label.items()}
        self.targets = [self.pid_to_label[pid] for _, pid in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        img_path, pid = self.samples[index]
        label = self.pid_to_label[pid]

        with Image.open(img_path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label, pid


class PKSampler(Sampler[List[int]]):
    """
    Balanced Identity Sampler for Batch-Hard Triplet Loss.
    Every batch contains P distinct identities, each with K heatmap instances
    (Batch Size = P * K, default 8 * 4 = 32).
    """

    def __init__(self, targets: List[int], p: int = 8, k: int = 4, seed: int = 42):
        self.targets = targets
        self.p = p
        self.k = k
        self.batch_size = p * k
        self.seed = seed
        self.epoch = 0

        self.index_dic: Dict[int, List[int]] = defaultdict(list)
        for idx, label in enumerate(targets):
            self.index_dic[label].append(idx)

        self.pids = sorted(self.index_dic.keys())
        if len(self.pids) < self.p:
            raise ValueError(f"Dataset has only {len(self.pids)} identities, fewer than P={self.p}")

        # Estimate number of batches per epoch so each identity is visited evenly
        self.num_batches = len(self.targets) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)

        # Pre-sample K instances per identity into shuffled pools
        avai_pids = list(self.pids)
        pid_pools: Dict[int, List[int]] = {}
        for pid in avai_pids:
            idxs = list(self.index_dic[pid])
            rng.shuffle(idxs)
            # Pad with replacement only if an identity has fewer than K samples
            while len(idxs) < self.k:
                idxs.append(rng.choice(self.index_dic[pid]))
            pid_pools[pid] = idxs

        batches = []
        while len(avai_pids) >= self.p:
            selected_pids = rng.sample(avai_pids, self.p)
            batch = []
            for pid in selected_pids:
                batch.extend(pid_pools[pid][: self.k])
                pid_pools[pid] = pid_pools[pid][self.k :]
                if len(pid_pools[pid]) < self.k:
                    avai_pids.remove(pid)
            batches.append(batch)

        for batch in batches:
            yield batch

    def __len__(self) -> int:
        # Exact number of full P x K batches yielded per epoch
        total_chunks = sum(max(1, len(idxs) // self.k) for idxs in self.index_dic.values())
        return total_chunks // self.p


def build_dataloaders(
    data_root: str = "data/Wi-PER81Dataset/ImageData",
    img_size: int = 224,
    p: int = 8,
    k: int = 4,
    eval_batch_size: int = 64,
    num_workers: int = 2,
) -> Dict[str, DataLoader]:
    """
    Build all DataLoaders for training (D1 with PKSampler) and retrieval evaluation
    (D2v & D2t Probe and Gallery).
    """
    train_tf, eval_tf = get_wifi_transforms(img_size=img_size)

    # 1. Training split: Session A (D1 Gallery + Probe merged = 800 heatmaps)
    train_dataset = WiPER81Dataset(
        root_dir=data_root,
        split="D1",
        subsets=("Gallery", "Probe"),
        transform=train_tf,
    )
    pk_sampler = PKSampler(targets=train_dataset.targets, p=p, k=k)

    # Note: pin_memory=True speeds up host-to-GPU transfers on Windows RTX 4060
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=pk_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 2. Validation split: Session B (D2v Gallery & Probe)
    val_gallery_ds = WiPER81Dataset(
        root_dir=data_root,
        split="D2v",
        subsets=("Gallery",),
        transform=eval_tf,
    )
    val_probe_ds = WiPER81Dataset(
        root_dir=data_root,
        split="D2v",
        subsets=("Probe",),
        transform=eval_tf,
        pid_to_label=val_gallery_ds.pid_to_label,
    )

    # 3. Test split: Session B (D2t Gallery & Probe)
    test_gallery_ds = WiPER81Dataset(
        root_dir=data_root,
        split="D2t",
        subsets=("Gallery",),
        transform=eval_tf,
    )
    test_probe_ds = WiPER81Dataset(
        root_dir=data_root,
        split="D2t",
        subsets=("Probe",),
        transform=eval_tf,
        pid_to_label=test_gallery_ds.pid_to_label,
    )

    eval_kwargs = dict(batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return {
        "train": train_loader,
        "val_gallery": DataLoader(val_gallery_ds, **eval_kwargs),
        "val_probe": DataLoader(val_probe_ds, **eval_kwargs),
        "test_gallery": DataLoader(test_gallery_ds, **eval_kwargs),
        "test_probe": DataLoader(test_probe_ds, **eval_kwargs),
    }