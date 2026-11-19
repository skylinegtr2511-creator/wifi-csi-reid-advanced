import pytest
import torch
from src.models import build_model


@pytest.mark.parametrize("backbone", ["convnext_tiny", "efficientnet_v2_s", "mobilenetv4"])
def test_pcb_backbones_forward_and_unit_norm(backbone):
    model = build_model(
        arch="pcb",
        backbone=backbone,
        pretrained=False,
        embedding_dim=128,
        num_stripes=4,
    )
    model.eval()

    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        emb = model(x)

    assert emb.shape == (4, 128), f"{backbone} output shape mismatch: {emb.shape}"
    norms = torch.norm(emb, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), f"{backbone} embeddings are not unit-normalized"