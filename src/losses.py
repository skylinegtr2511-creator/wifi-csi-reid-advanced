import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwise_euclidean_distance(embeddings: torch.Tensor, squared: bool = False) -> torch.Tensor:
    """
    Compute the 2D matrix of pairwise Euclidean distances between L2-normalized embeddings.

    Args:
        embeddings: Tensor of shape (B, D) — e.g., (32, 128) unit-hypersphere vectors.
        squared: If True, return squared Euclidean distances.

    Returns:
        dist_mat: Tensor of shape (B, B) where dist_mat[i, j] = ||emb[i] - emb[j]||_2.
    """
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a, b>
    dot_product = torch.matmul(embeddings, embeddings.t())
    sq_norm = torch.diag(dot_product)
    distances = sq_norm.unsqueeze(0) - 2.0 * dot_product + sq_norm.unsqueeze(1)

    # Clamp negative numerical noise to 0.0
    distances = torch.clamp(distances, min=0.0)

    if not squared:
        # Add small epsilon mask where distance is 0 to prevent NaN gradients during sqrt backward
        mask = (distances == 0.0).float()
        distances = distances + mask * 1e-16
        distances = torch.sqrt(distances)
        distances = distances * (1.0 - mask)

    return distances


class BatchHardTripletLoss(nn.Module):
    """
    Batch-Hard Triplet Margin Loss (Hermans et al.).
    For each anchor i in a P x K batch, mines:
      - Hardest positive: max_{j: y_j == y_i} d(emb_i, emb_j)
      - Hardest negative: min_{k: y_k != y_i} d(emb_i, emb_k)
    and optimizes L = mean(max(d(a, p_hard) - d(a, n_hard) + margin, 0)).
    """

    def __init__(self, margin: float = 0.3, soft_margin: bool = False):
        super().__init__()
        self.margin = margin
        self.soft_margin = soft_margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: L2-normalized embeddings of shape (B, D).
            labels: Integer identity labels of shape (B,).

        Returns:
            Scalar loss tensor.
        """
        dist_mat = pairwise_euclidean_distance(embeddings, squared=False)
        batch_size = embeddings.size(0)

        # Boolean masks for valid positive and negative pairs
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)
        dist_ap_mask = labels_equal.float()
        dist_an_mask = (~labels_equal).float()

        # 1. Hardest Positive per anchor: max over same-identity samples (includes self=0.0 as lower bound)
        hardest_positive_dist, _ = (dist_mat * dist_ap_mask).max(dim=1)

        # 2. Hardest Negative per anchor: min over different-identity samples
        # Add large value to same-identity entries so they are never selected as minimum
        max_anchor_dist, _ = dist_mat.max(dim=1, keepdim=True)
        dist_an = dist_mat + max_anchor_dist * (1.0 - dist_an_mask)
        hardest_negative_dist, _ = dist_an.min(dim=1)

        # 3. Compute Triplet Margin Loss
        if self.soft_margin:
            losses = F.softplus(hardest_positive_dist - hardest_negative_dist)
        else:
            losses = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)

        return losses.mean()


class StandardTripletLoss(nn.Module):
    """
    Baseline Triplet Margin Loss (margin=0.3) used in Assignments 2 & 3
    for explicit anchor, positive, negative triplet tuples.
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        return self.loss_fn(anchor, positive, negative)