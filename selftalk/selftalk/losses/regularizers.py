"""Stages 5-6: geometric and temporal regularizers on the Gaussian deltas.

Per the plan these are added to the objective after the photometric and lip
terms, always behind a weight flag so ablations can disable them:

- displacement magnitude   (weights: dynamic regions weak, static strong)
- neighborhood consistency (KNN mean of Δmu)
- local rigidity / Laplacian on Δmu
- inter-frame velocity and acceleration

All take per-Gaussian region weights w_i so that mouth/eyes move freely while
hair/ears/background stay near-static.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def knn_neighbors(means: torch.Tensor, k: int = 8) -> torch.Tensor:
    """Neighbor indices in canonical space.

    means: [N, 3]. Returns [N, k] indices (a point can be its own neighbor
    only if N <= k, which we reject).
    """
    if means.ndim != 2:
        raise ValueError(f"means must be [N, 3], got {tuple(means.shape)}")
    n = means.shape[0]
    if n <= k:
        raise ValueError(f"need N > k for KNN, got N={n}, k={k}")
    dist = torch.cdist(means, means)  # [N, N]
    dist.fill_diagonal_(float("inf"))
    return dist.topk(k, dim=-1, largest=False).indices


def displacement_regularizer(delta_means: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """lambda-reg: sum_i w_i ||Δmu_i||^2. delta_means [B, N, 3], weights [B, N]."""
    if weights.shape != delta_means.shape[:2]:
        raise ValueError("weights must be [B, N]")
    return (weights * delta_means.norm(dim=-1) ** 2).mean()


def neighborhood_consistency(delta_means: torch.Tensor, neighbor_idx: torch.Tensor,
                             weights: torch.Tensor) -> torch.Tensor:
    """||Δmu_i − mean_{j∈N(i)} Δmu_j||^2 averaged over Gaussians.

    delta_means: [B, N, 3]; neighbor_idx: [N, k]; weights: [B, N].
    """
    if neighbor_idx.shape[0] != delta_means.shape[1]:
        raise ValueError("neighbor_idx does not match N")
    neigh = delta_means[:, neighbor_idx]  # [B, N, k, 3]
    diff = delta_means - neigh.mean(dim=2)
    return (weights * diff.norm(dim=-1) ** 2).mean()


def laplacian_rigidity(delta_means: torch.Tensor, neighbor_idx: torch.Tensor,
                       weights: torch.Tensor) -> torch.Tensor:
    """Local rigidity: Laplacian of Δmu over the canonical KNN graph."""
    return neighborhood_consistency(delta_means, neighbor_idx, weights)


def velocity_regularizer(delta_means: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """||Δmu_t − Δmu_{t−1}||_1 along the batch (time) axis; weights [B, N]."""
    if delta_means.shape[0] < 2:
        raise ValueError("need >= 2 consecutive frames")
    vel = (delta_means[1:] - delta_means[:-1]).abs().mean(dim=-1)
    return (weights[1:] * vel).mean()


def acceleration_regularizer(delta_means: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """||Δmu_{t+1} − 2Δmu_t + Δmu_{t−1}||_1 along time; weights [B, N]."""
    if delta_means.shape[0] < 3:
        raise ValueError("need >= 3 consecutive frames")
    acc = (delta_means[2:] - 2 * delta_means[1:-1] + delta_means[:-2]).abs().mean(dim=-1)
    return (weights[1:-1] * acc).mean()


def region_weight_map(region_weights_per_gaussian: torch.Tensor) -> torch.Tensor:
    """Convenience: normalize a per-Gaussian weight map to sum to N."""
    return region_weights_per_gaussian / region_weights_per_gaussian.mean().clamp_min(1e-8)


def temporal_warping_error(images: torch.Tensor) -> torch.Tensor:
    """Mean |I_t − I_{t−1}| over consecutive frames; images [B, 3, H, W]."""
    if images.ndim != 4:
        raise ValueError(f"images must be [B, 3, H, W], got {tuple(images.shape)}")
    if images.shape[0] < 2:
        raise ValueError("need >= 2 frames")
    return (images[1:] - images[:-1]).abs().mean()
