import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import yaml

from src.dataset import build_dataloaders
from src.models import build_model
from src.utils import evaluate_reid_metrics, extract_embeddings


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone Retrieval Evaluation on Wi-PER81")
    parser.add_argument("--config", type=str, default="configs/train_pcb.yaml")
    parser.add_argument("--backbone", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = build_dataloaders(
        data_root=cfg["data"]["root_dir"],
        img_size=cfg["data"]["img_size"],
        p=cfg["data"]["p_identities"],
        k=cfg["data"]["k_samples"],
        eval_batch_size=cfg["data"]["eval_batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )

    model = build_model(
        arch=cfg["model"]["arch"],
        backbone=args.backbone,
        pretrained=False,
        embedding_dim=cfg["model"]["embedding_dim"],
        dropout=cfg["model"]["dropout"],
        num_stripes=cfg["model"].get("num_stripes", 4),
        stripe_dim=cfg["model"].get("stripe_dim", 128),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    probe_loader = loaders[f"{args.split}_probe"]
    gallery_loader = loaders[f"{args.split}_gallery"]

    p_embs, p_labels, _ = extract_embeddings(model, probe_loader, device)
    g_embs, g_labels, _ = extract_embeddings(model, gallery_loader, device)
    metrics = evaluate_reid_metrics(p_embs, p_labels, g_embs, g_labels)

    print(f"\n=== Evaluation Results ({args.split.upper()} | {cfg['model']['arch']} - {args.backbone}) ===")
    print(f"Checkpoint Epoch : {ckpt.get('epoch', 'N/A')}")
    print(f"mAP              : {metrics['mAP']:.4f}")
    print(f"Rank-1           : {metrics['Rank-1']:.4f}")
    print(f"Rank-5           : {metrics['Rank-5']:.4f}")
    print(f"Rank-10          : {metrics['Rank-10']:.4f}")


if __name__ == "__main__":
    main()