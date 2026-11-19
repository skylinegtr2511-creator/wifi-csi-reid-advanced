import random
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def set_seed(seed: int = 42) -> None:
    """Fix random seeds across Python, NumPy, and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Extract L2-normalized embeddings, integer labels, and string PIDs for a split.
    """
    model.eval()
    all_embs = []
    all_labels = []
    all_pids = []

    for images, labels, pids in dataloader:
        images = images.to(device, non_blocking=True)
        embs = model(images)
        all_embs.append(embs.cpu())
        all_labels.append(labels)
        all_pids.extend(pids)

    return torch.cat(all_embs, dim=0), torch.cat(all_labels, dim=0), all_pids


def evaluate_reid_metrics(
    probe_embs: torch.Tensor,
    probe_labels: torch.Tensor,
    gallery_embs: torch.Tensor,
    gallery_labels: torch.Tensor,
    max_rank: int = 10,
) -> Dict[str, object]:
    """
    Compute open-world Re-ID retrieval metrics (mAP, Rank-1, Rank-5, Rank-10, and CMC curve)
    by ranking Gallery embeddings against each Probe embedding via Euclidean distance.
    """
    # Compute pairwise Euclidean distance matrix: shape (num_probe, num_gallery)
    dist_mat = torch.cdist(probe_embs, gallery_embs, p=2).numpy()
    q_pids = probe_labels.numpy()
    g_pids = gallery_labels.numpy()

    num_q, num_g = dist_mat.shape
    if num_g < max_rank:
        max_rank = num_g

    # Sort gallery indices by ascending Euclidean distance for each probe
    indices = np.argsort(dist_mat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    all_cmc = []
    all_ap = []
    num_valid_q = 0

    for q_idx in range(num_q):
        raw_cmc = matches[q_idx]
        if not np.any(raw_cmc):
            # Skip probe if its identity does not exist in gallery
            continue

        # Cumulative Matching Characteristic (CMC) for this query
        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1

        # Average Precision (AP) for this query
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        precision_at_k = tmp_cmc / (np.arange(len(tmp_cmc)) + 1.0)
        ap = (precision_at_k * raw_cmc).sum() / num_rel
        all_ap.append(ap)

    if num_valid_q == 0:
        raise RuntimeError("No valid probe identities matched any gallery identities.")

    cmc_curve = np.asarray(all_cmc, dtype=np.float32).sum(axis=0) / num_valid_q
    mAP = float(np.mean(all_ap))

    return {
        "mAP": round(mAP, 4),
        "Rank-1": round(float(cmc_curve[0]), 4),
        "Rank-5": round(float(cmc_curve[min(4, max_rank - 1)]), 4),
        "Rank-10": round(float(cmc_curve[min(9, max_rank - 1)]), 4),
        "cmc_curve": cmc_curve.tolist(),
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, object],
    save_path: Path,
) -> None:
    """Save best model weights and evaluation metadata."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        save_path,
    )


def save_history(history: Dict[str, list], save_path: Path) -> None:
    """Save training & validation metrics history via joblib for notebook plotting."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(history, save_path)