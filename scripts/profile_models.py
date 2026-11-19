import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import torch
from thop import profile

from src.models import build_model


def main():
    backbones = ["convnext_tiny", "efficientnet_v2_s", "mobilenetv4"]
    dummy_input = torch.randn(1, 3, 224, 224)
    rows = []

    print("=== Computational Profiling (PCB Architecture | 224x224 Input) ===")
    for backbone in backbones:
        model = build_model(
            arch="pcb",
            backbone=backbone,
            pretrained=False,
            embedding_dim=128,
            num_stripes=4,
        )
        model.eval()
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        gflops = (macs * 2) / 1e9  # 1 MAC = 2 FLOPs
        params_m = params / 1e6

        rows.append({
            "Model Architecture": f"PCB-{backbone}",
            "GFLOPS": round(gflops, 3),
            "Total Parameters (M)": round(params_m, 2),
        })

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))

    out_path = Path("outputs/results/profile_pcb.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved profiling summary to {out_path}")


if __name__ == "__main__":
    main()