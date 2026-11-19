import torch
import torch.nn.functional as F
from src.losses import BatchHardTripletLoss


def test_batch_hard_triplet_loss_shape_and_grad():
    torch.manual_seed(42)
    # Keep raw_embeddings as the leaf tensor so PyTorch populates .grad
    raw_embeddings = torch.randn(32, 128, requires_grad=True)
    embeddings = F.normalize(raw_embeddings, p=2, dim=1)
    labels = torch.repeat_interleave(torch.arange(8), repeats=4)

    criterion = BatchHardTripletLoss(margin=0.3)
    loss = criterion(embeddings, labels)

    assert loss.ndim == 0, "Loss must be a scalar tensor"
    assert loss.item() > 0.0, "Random unit vectors should produce positive loss"

    loss.backward()
    assert raw_embeddings.grad is not None, "Gradients must flow back to embeddings"
    assert not torch.isnan(raw_embeddings.grad).any(), "Gradients must not contain NaNs"


def test_batch_hard_exact_value():
    emb = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.0, -1.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1])

    criterion = BatchHardTripletLoss(margin=0.3)
    loss = criterion(emb, labels)

    assert torch.isclose(loss, torch.tensor(0.3), atol=1e-5)