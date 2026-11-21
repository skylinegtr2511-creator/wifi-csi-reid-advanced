import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when running `python scripts/train.py`
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import torch
from tqdm import tqdm
import yaml

from src.dataset import build_dataloaders
from src.losses import BatchHardTripletLoss
from src.models import build_model
from src.utils import (
    evaluate_reid_metrics,
    extract_embeddings,
    save_checkpoint,
    save_history,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Wi-Fi CSI Person Re-ID Model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_vit.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="Optional override for model.backbone (e.g., faster_vit_0, eva02_tiny, convnext_tiny).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.backbone is not None:
        cfg["model"]["backbone"] = args.backbone

    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    arch = cfg["model"]["arch"]
    backbone = cfg["model"]["backbone"]
    run_name = f"{arch}_{backbone}"
    print(f"=== Starting Run: {run_name} on {device} ===")

    # 1. DataLoaders (D1 P x K train, D2v val, D2t test)
    loaders = build_dataloaders(
        data_root=cfg["data"]["root_dir"],
        img_size=cfg["data"]["img_size"],
        p=cfg["data"]["p_identities"],
        k=cfg["data"]["k_samples"],
        eval_batch_size=cfg["data"]["eval_batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )

    # 2. Model & Loss
    model = build_model(
        arch=arch,
        backbone=backbone,
        pretrained=cfg["model"]["pretrained"],
        embedding_dim=cfg["model"]["embedding_dim"],
        dropout=cfg["model"]["dropout"],
        num_stripes=cfg["model"].get("num_stripes", 4),
        stripe_dim=cfg["model"].get("stripe_dim", 128),
        img_size=cfg["data"]["img_size"],
    ).to(device)

    criterion = BatchHardTripletLoss(
        margin=cfg["loss"]["margin"],
        soft_margin=cfg["loss"].get("soft_margin", False),
    )

    # 3. Differential Learning Rates + Optional Linear Warmup & Cosine Decay
    head_lr = cfg["training"]["lr"]
    backbone_lr = cfg["training"].get("backbone_lr", head_lr)

    if hasattr(model, "backbone") and hasattr(model, "head") and backbone_lr != head_lr:
        param_groups = [
            {"params": model.backbone.parameters(), "lr": backbone_lr},
            {"params": model.head.parameters(), "lr": head_lr},
        ]
        print(f"Using differential learning rates -> backbone_lr: {backbone_lr}, head_lr: {head_lr}")
    else:
        param_groups = [{"params": model.parameters(), "lr": head_lr}]

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=cfg["training"]["weight_decay"],
    )

    epochs = cfg["training"]["epochs"]
    warmup_epochs = cfg["training"].get("warmup_epochs", 0)
    if warmup_epochs > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-6
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs]
        )
    else:
        scheduler = None

    use_amp = cfg["training"].get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # 4. Training & Validation Loop
    patience = cfg["training"]["patience"]
    best_val_map = -1.0
    epochs_no_improve = 0

    ckpt_path = Path(cfg["outputs"]["checkpoint_dir"]) / f"{run_name}_best.pth"
    history_path = Path(cfg["outputs"]["log_dir"]) / f"{run_name}_history.joblib"
    results_csv = Path(cfg["outputs"]["results_dir"]) / f"final_test_metrics_{arch}.csv"

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_mAP": [],
        "val_rank1": [],
        "val_rank5": [],
        "val_rank10": [],
    }

    for epoch in range(1, epochs + 1):
        if hasattr(loaders["train"].batch_sampler, "set_epoch"):
            loaders["train"].batch_sampler.set_epoch(epoch)

        model.train()
        running_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            loaders["train"],
            desc=f"Epoch [{epoch:02d}/{epochs:02d}]",
            leave=False,
            unit="batch",
        )
        for images, labels, _ in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                embeddings = model(images)
                loss = criterion(embeddings, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        if scheduler is not None:
            scheduler.step()

        avg_train_loss = running_loss / max(1, num_batches)

        # Validation retrieval evaluation & validation loss on D2v
        val_p_embs, val_p_labels, _ = extract_embeddings(model, loaders["val_probe"], device)
        val_g_embs, val_g_labels, _ = extract_embeddings(model, loaders["val_gallery"], device)
        val_metrics = evaluate_reid_metrics(val_p_embs, val_p_labels, val_g_embs, val_g_labels)

        val_all_embs = torch.cat([val_p_embs, val_g_embs], dim=0)
        val_all_labels = torch.cat([val_p_labels, val_g_labels], dim=0)
        val_loss = criterion(val_all_embs, val_all_labels).item()

        history["epoch"].append(epoch)
        history["train_loss"].append(round(avg_train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_mAP"].append(val_metrics["mAP"])
        history["val_rank1"].append(val_metrics["Rank-1"])
        history["val_rank5"].append(val_metrics["Rank-5"])
        history["val_rank10"].append(val_metrics["Rank-10"])

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"D2v mAP: {val_metrics['mAP']:.4f} | "
            f"Rank-1: {val_metrics['Rank-1']:.4f} | "
            f"Rank-5: {val_metrics['Rank-5']:.4f}"
        )

        # Checkpoint & Early Stopping check
        if val_metrics["mAP"] > best_val_map:
            best_val_map = val_metrics["mAP"]
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, epoch, val_metrics, ckpt_path)
            print(f"  --> Saved new best checkpoint (D2v mAP: {best_val_map:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at Epoch {epoch} (no D2v mAP improvement for {patience} epochs).")
                break

    save_history(history, history_path)

    # 5. Load Best Checkpoint and Evaluate on Unseen Test Split (D2t)
    print(f"\n=== Evaluating Best Checkpoint on Unseen Test Split (D2t) ===")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_p_embs, test_p_labels, _ = extract_embeddings(model, loaders["test_probe"], device)
    test_g_embs, test_g_labels, _ = extract_embeddings(model, loaders["test_gallery"], device)
    test_metrics = evaluate_reid_metrics(test_p_embs, test_p_labels, test_g_embs, test_g_labels)

    print(
        f"[{run_name}] Best Epoch: {checkpoint['epoch']} | "
        f"D2t mAP: {test_metrics['mAP']:.4f} | "
        f"Rank-1: {test_metrics['Rank-1']:.4f} | "
        f"Rank-5: {test_metrics['Rank-5']:.4f} | "
        f"Rank-10: {test_metrics['Rank-10']:.4f}"
    )

    # Append or update row in outputs/results/final_test_metrics_<arch>.csv
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "Architecture": arch.upper(),
        "Backbone": backbone,
        "Best Epoch": checkpoint["epoch"],
        "Test mAP": test_metrics["mAP"],
        "Test Rank-1": test_metrics["Rank-1"],
        "Test Rank-5": test_metrics["Rank-5"],
        "Test Rank-10": test_metrics["Rank-10"],
    }
    if results_csv.exists():
        df = pd.read_csv(results_csv)
        df = df[df["Backbone"] != backbone]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(results_csv, index=False)
    print(f"Saved test results table to {results_csv}")


if __name__ == "__main__":
    main()