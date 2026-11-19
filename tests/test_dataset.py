import pytest
import torch
from pathlib import Path
from src.dataset import build_dataloaders


DATA_ROOT = Path("data/Wi-PER81Dataset/ImageData")


@pytest.mark.skipif(not DATA_ROOT.exists(), reason="Wi-PER81Dataset link not found locally")
def test_wipers81_dataloaders_and_pk_sampler():
    loaders = build_dataloaders(
        data_root=str(DATA_ROOT),
        img_size=224,
        p=8,
        k=4,
        num_workers=0,  # 0 workers for fast Windows unit testing
    )

    train_ds = loaders["train"].dataset
    assert len(train_ds) == 810, f"Expected 810 D1 training heatmaps (81 IDs x 10), got {len(train_ds)}"
    assert len(train_ds.pid_to_label) == 81, f"Expected 81 unique identities, got {len(train_ds.pid_to_label)}"
    # Inspect first P x K batch
    images, labels, pids = next(iter(loaders["train"]))
    assert images.shape == (32, 3, 224, 224), f"Unexpected tensor shape: {images.shape}"
    assert labels.shape == (32,)

    unique_labels, counts = torch.unique(labels, return_counts=True)
    assert len(unique_labels) == 8, f"Expected P=8 identities per batch, got {len(unique_labels)}"
    assert torch.all(counts == 4), f"Expected K=4 samples per identity, got {counts.tolist()}"

    # Ensure validation and test splits are non-empty and aligned
    assert len(loaders["val_gallery"].dataset) > 0
    assert len(loaders["val_probe"].dataset) > 0
    assert len(loaders["test_gallery"].dataset) > 0
    assert len(loaders["test_probe"].dataset) > 0