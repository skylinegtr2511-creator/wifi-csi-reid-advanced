from typing import Dict, List

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


CNN_BACKBONE_REGISTRY: Dict[str, str] = {
    "convnext_tiny": "convnext_tiny.fb_in22k_ft_in1k",
    "efficientnet_v2_s": "tf_efficientnetv2_s.in21k_ft_in1k",
    "mobilenetv4": "mobilenetv4_conv_medium.e500_r256_in1k",
}

# Candidate lists allow automatic fallback if an optional timm model name isn't registered locally
HYBRID_VIT_REGISTRY: Dict[str, List[str]] = {
    "faster_vit_0": ["faster_vit_0_224", "caformer_s18.sail_in22k_ft_in1k"],
    "caformer_s18": ["caformer_s18.sail_in22k_ft_in1k", "caformer_s18.sail_in1k"],
}

PURE_VIT_REGISTRY: Dict[str, str] = {
    "eva02_tiny": "eva02_tiny_patch14_224.mim_in22k",
}


class PCBHead(nn.Module):
    """
    Part-Based Convolutional Baseline (PCB) Head for Wi-Fi CSI Heatmaps.
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
        striped_feat = self.pool(feat_map)

        part_embeddings = []
        for i in range(self.num_stripes):
            stripe = striped_feat[:, :, i : i + 1, :]
            refined = self.local_convs[i](stripe).flatten(1)
            part_embeddings.append(refined)

        concat_feat = torch.cat(part_embeddings, dim=1)
        concat_feat = self.dropout(concat_feat)

        emb = self.bn_out(self.projector(concat_feat))
        return F.normalize(emb, p=2, dim=1)


class SMGNHead(nn.Module):
    """
    Multiple Granularity Network (SMGN) Head for Wi-Fi CSI Heatmaps.
    Forks a 4D spatial feature map (B, C, H, W) into Global (1x1), Coarse (2x1),
    and Fine (4x1) OFDM subcarrier branches.
    """

    def __init__(
        self,
        in_channels: int,
        branch_dim: int = 128,
        embedding_dim: int = 128,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.coarse_pool = nn.AdaptiveAvgPool2d((2, 1))
        self.fine_pool = nn.AdaptiveAvgPool2d((4, 1))

        def make_conv_block():
            return nn.Sequential(
                nn.Conv2d(in_channels, branch_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(branch_dim),
                nn.ReLU(inplace=True),
            )

        self.global_conv = make_conv_block()
        self.coarse_convs = nn.ModuleList([make_conv_block() for _ in range(2)])
        self.fine_convs = nn.ModuleList([make_conv_block() for _ in range(4)])

        total_parts = 1 + 2 + 4
        self.dropout = nn.Dropout(p=dropout)
        self.projector = nn.Linear(total_parts * branch_dim, embedding_dim, bias=False)
        self.bn_out = nn.BatchNorm1d(embedding_dim)

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        g_feat = self.global_conv(self.global_pool(feat_map)).flatten(1)

        c_pooled = self.coarse_pool(feat_map)
        c_feats = [
            self.coarse_convs[i](c_pooled[:, :, i : i + 1, :]).flatten(1)
            for i in range(2)
        ]

        f_pooled = self.fine_pool(feat_map)
        f_feats = [
            self.fine_convs[i](f_pooled[:, :, i : i + 1, :]).flatten(1)
            for i in range(4)
        ]

        concat_feat = torch.cat([g_feat, *c_feats, *f_feats], dim=1)
        concat_feat = self.dropout(concat_feat)

        emb = self.bn_out(self.projector(concat_feat))
        return F.normalize(emb, p=2, dim=1)


class PCBReIDModel(nn.Module):
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
        if backbone_name not in CNN_BACKBONE_REGISTRY:
            raise ValueError(f"Unsupported PCB backbone '{backbone_name}'.")

        self.backbone = timm.create_model(
            CNN_BACKBONE_REGISTRY[backbone_name],
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
        return self.head(self.backbone(x)[-1])


class SMGNReIDModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        branch_dim: int = 128,
        embedding_dim: int = 128,
        dropout: float = 0.5,
    ):
        super().__init__()
        if backbone_name not in CNN_BACKBONE_REGISTRY:
            raise ValueError(f"Unsupported SMGN backbone '{backbone_name}'.")

        self.backbone = timm.create_model(
            CNN_BACKBONE_REGISTRY[backbone_name],
            pretrained=pretrained,
            features_only=True,
            out_indices=(-1,),
        )
        in_channels = self.backbone.feature_info.channels()[-1]

        self.head = SMGNHead(
            in_channels=in_channels,
            branch_dim=branch_dim,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)[-1])


class HybridTransformerReIDModel(nn.Module):
    """
    Post-2022 Hierarchical Hybrid Vision Transformer (FasterViT-0 / CAFormer-S18)
    paired with Multi-Granularity Subcarrier Pooling (SMGNHead) to preserve
    both local OFDM frequency bands and global self-attention context.
    """

    def __init__(
        self,
        backbone_name: str = "faster_vit_0",
        pretrained: bool = True,
        branch_dim: int = 128,
        embedding_dim: int = 128,
        dropout: float = 0.5,
    ):
        super().__init__()
        if backbone_name not in HYBRID_VIT_REGISTRY:
            raise ValueError(f"Unsupported Hybrid ViT '{backbone_name}'.")

        candidates = HYBRID_VIT_REGISTRY[backbone_name]
        self.backbone = None
        self.resolved_timm_name = None

        for candidate in candidates:
            if timm.is_model(candidate):
                self.backbone = timm.create_model(
                    candidate,
                    pretrained=pretrained,
                    features_only=True,
                    out_indices=(-1,),
                )
                self.resolved_timm_name = candidate
                break

        if self.backbone is None:
            raise RuntimeError(
                f"None of the candidate timm models {candidates} are available in your timm version."
            )

        print(f"[HybridTransformerReIDModel] Loaded backbone: {self.resolved_timm_name}")
        in_channels = self.backbone.feature_info.channels()[-1]

        self.head = SMGNHead(
            in_channels=in_channels,
            branch_dim=branch_dim,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.backbone(x)[-1]
        # Ensure NCHW layout in case a transformer stage returns NHWC
        if feat_map.ndim == 4 and feat_map.shape[-1] == self.backbone.feature_info.channels()[-1]:
            feat_map = feat_map.permute(0, 3, 1, 2).contiguous()
        return self.head(feat_map)


class PureViTReIDModel(nn.Module):
    """
    Post-2022 Pure Vision Transformer (EVA-02 Tiny with 2D RoPE + MIM pretraining).
    """

    def __init__(
        self,
        backbone_name: str = "eva02_tiny",
        pretrained: bool = True,
        img_size: int = 224,
        embedding_dim: int = 128,
        dropout: float = 0.5,
    ):
        super().__init__()
        if backbone_name not in PURE_VIT_REGISTRY:
            raise ValueError(f"Unsupported Pure ViT '{backbone_name}'.")

        self.backbone = timm.create_model(
            PURE_VIT_REGISTRY[backbone_name],
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size,
        )
        in_features = self.backbone.num_features

        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.head(self.backbone(x))
        return F.normalize(emb, p=2, dim=1)


def build_model(
    arch: str = "pcb",
    backbone: str = "convnext_tiny",
    pretrained: bool = True,
    embedding_dim: int = 128,
    dropout: float = 0.5,
    num_stripes: int = 4,
    stripe_dim: int = 128,
    img_size: int = 224,
) -> nn.Module:
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
        return SMGNReIDModel(
            backbone_name=backbone,
            pretrained=pretrained,
            branch_dim=stripe_dim,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )
    elif arch == "vit":
        if backbone in HYBRID_VIT_REGISTRY:
            return HybridTransformerReIDModel(
                backbone_name=backbone,
                pretrained=pretrained,
                branch_dim=stripe_dim,
                embedding_dim=embedding_dim,
                dropout=dropout,
            )
        elif backbone in PURE_VIT_REGISTRY:
            return PureViTReIDModel(
                backbone_name=backbone,
                pretrained=pretrained,
                img_size=img_size,
                embedding_dim=embedding_dim,
                dropout=dropout,
            )
        else:
            raise ValueError(
                f"Unsupported ViT backbone '{backbone}'. "
                f"Available: {list(HYBRID_VIT_REGISTRY.keys()) + list(PURE_VIT_REGISTRY.keys())}"
            )
    else:
        raise ValueError(f"Unknown architecture '{arch}'. Expected one of: ['pcb', 'smgn', 'vit'].")