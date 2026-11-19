from typing import Dict, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


BACKBONE_REGISTRY: Dict[str, str] = {
    "convnext_tiny": "convnext_tiny.fb_in22k_ft_in1k",
    "efficientnet_v2_s": "tf_efficientnetv2_s.in21k_ft_in1k",
    "mobilenetv4": "mobilenetv4_conv_medium.e500_r256_in1k",
}


class PCBHead(nn.Module):
    """
    Part-Based Convolutional Baseline (PCB) Head for Wi-Fi CSI Heatmaps.
    Slices the 2D spatial feature map (B, C, H, W) horizontally into `num_stripes`
    OFDM subcarrier bands via AdaptiveAvgPool2d((num_stripes, 1)), refines each
    frequency stripe independently, and projects the concatenated local descriptors
    onto an L2-normalized 128D unit hypersphere.
    """

    def __init__(
        self,
        in_channels: int,
        num_stripes: int = 4,
        stripe_dim: int = 128,
        embedding_dim: int = 128,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.num_stripes = num_stripes
        self.pool = nn.AdaptiveAvgPool2d((num_stripes, 1))

        # Independent 1x1 conv bottleneck per horizontal subcarrier stripe
        self.local_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, stripe_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(stripe_dim),
                nn.ReLU(inplace=True),
            )
            for _ in range(num_stripes)
        ])

        self.dropout = nn.Dropout(p=dropout)
        self.projector = nn.Linear(num_stripes * stripe_dim, embedding_dim, bias=False)
        self.bn_out = nn.BatchNorm1d(embedding_dim)

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_map: 4D spatial feature tensor of shape (B, C, H, W).
        Returns:
            L2-normalized embedding tensor of shape (B, embedding_dim).
        """
        # Pool along time-step width W -> (B, C, num_stripes, 1)
        striped_feat = self.pool(feat_map)

        part_embeddings = []
        for i in range(self.num_stripes):
            # Extract i-th horizontal subcarrier stripe: (B, C, 1, 1)
            stripe = striped_feat[:, :, i : i + 1, :]
            refined = self.local_convs[i](stripe).flatten(1)  # (B, stripe_dim)
            part_embeddings.append(refined)

        # Concatenate all subcarrier band descriptors -> (B, num_stripes * stripe_dim)
        concat_feat = torch.cat(part_embeddings, dim=1)
        concat_feat = self.dropout(concat_feat)

        emb = self.bn_out(self.projector(concat_feat))
        return F.normalize(emb, p=2, dim=1)


class PCBReIDModel(nn.Module):
    """
    Modular Wi-Fi CSI Re-ID Network pairing a CNN backbone with a PCB head.
    """

    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        num_stripes: int = 4,
        stripe_dim: int = 128,
        embedding_dim: int = 128,
        dropout: float = 0.5,
    ):
        super().__init__()
        if backbone_name not in BACKBONE_REGISTRY:
            raise ValueError(
                f"Unsupported PCB backbone '{backbone_name}'. "
                f"Available options: {list(BACKBONE_REGISTRY.keys())}"
            )

        timm_model_name = BACKBONE_REGISTRY[backbone_name]
        # features_only=True returns multi-stage 4D feature maps; [-1] is the final stage
        self.backbone = timm.create_model(
            timm_model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(-1,),
        )
        in_channels = self.backbone.feature_info.channels()[-1]

        self.head = PCBHead(
            in_channels=in_channels,
            num_stripes=num_stripes,
            stripe_dim=stripe_dim,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_maps = self.backbone(x)
        final_feat_map = feat_maps[-1]  # (B, C, H, W)
        return self.head(final_feat_map)


def build_model(
    arch: str = "pcb",
    backbone: str = "convnext_tiny",
    pretrained: bool = True,
    embedding_dim: int = 128,
    dropout: float = 0.5,
    num_stripes: int = 4,
    stripe_dim: int = 128,
) -> nn.Module:
    """
    Unified model factory for PCB, SMGN, and Vision Transformers.
    """
    arch = arch.lower()
    if arch == "pcb":
        return PCBReIDModel(
            backbone_name=backbone,
            pretrained=pretrained,
            num_stripes=num_stripes,
            stripe_dim=stripe_dim,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
    elif arch == "smgn":
        raise NotImplementedError("SMGN architecture will be added in the next milestone.")
    elif arch in {"vit", "swin"}:
        raise NotImplementedError("Vision Transformer architecture will be added in the final milestone.")
    else:
        raise ValueError(f"Unknown architecture '{arch}'. Expected one of: ['pcb', 'smgn', 'vit', 'swin'].")