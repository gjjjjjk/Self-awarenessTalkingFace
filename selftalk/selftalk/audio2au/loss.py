"""Stage 2: Audio-to-AU loss.

    L_AU = Huber(Û_t, U_t)
           + lambda_v * ||ΔÛ_t − ΔU_t||_1
           + lambda_a * ||Δ²Û_t − Δ²U_t||_1

with first/second temporal differences Δ taken along the frame axis. Each
term is optionally weighted by the OpenFace detection confidence so that
poorly tracked frames contribute less.
"""
from __future__ import annotations

import torch


def _first_difference(x: torch.Tensor) -> torch.Tensor:
    """Δx_t = x_t − x_{t−1}; [B, T, A] -> [B, T−1, A]."""
    return x[:, 1:] - x[:, :-1]


def _second_difference(x: torch.Tensor) -> torch.Tensor:
    """Δ²x_t = x_{t+1} − 2 x_t + x_{t−1}; [B, T, A] -> [B, T−2, A]."""
    return x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]


def au_regression_loss(
    pred_au: torch.Tensor,
    gt_au: torch.Tensor,
    confidence: torch.Tensor | None = None,
    beta: float = 1.0,
) -> torch.Tensor:
    """Confidence-weighted Huber loss on AU intensities.

    Args:
        pred_au: [B, T, A] predicted intensities.
        gt_au: [B, T, A] OpenFace intensities.
        confidence: [B, T, A] optional weights in [0, 1].
        beta: Huber transition point.
    """
    if pred_au.shape != gt_au.shape:
        raise ValueError(f"shape mismatch {tuple(pred_au.shape)} vs {tuple(gt_au.shape)}")
    if pred_au.ndim != 3:
        raise ValueError(f"expected [B, T, A], got {tuple(pred_au.shape)}")
    loss = torch.nn.functional.huber_loss(pred_au, gt_au, delta=beta, reduction="none")
    if confidence is not None:
        if confidence.shape != pred_au.shape:
            raise ValueError(f"confidence shape {tuple(confidence.shape)} mismatch")
        loss = loss * confidence
        denom = confidence.sum().clamp_min(1e-6)
    else:
        denom = pred_au.numel()
    return loss.sum() / denom


def audio2au_loss(
    pred_au: torch.Tensor,
    gt_au: torch.Tensor,
    confidence: torch.Tensor | None = None,
    lambda_vel: float = 0.2,
    lambda_acc: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Full stage-2 objective L_AU with velocity/acceleration terms.

    Returns a dict of scalar tensors: `total`, `huber`, `vel`, `acc`.
    """
    if pred_au.shape != gt_au.shape:
        raise ValueError(f"shape mismatch {tuple(pred_au.shape)} vs {tuple(gt_au.shape)}")
    if pred_au.shape[1] < 3:
        raise ValueError("need at least 3 frames for the second-difference term")

    huber = au_regression_loss(pred_au, gt_au, confidence)

    vel = (_first_difference(pred_au) - _first_difference(gt_au)).abs().mean()
    acc = (_second_difference(pred_au) - _second_difference(gt_au)).abs().mean()

    total = huber + lambda_vel * vel + lambda_acc * acc
    return {"total": total, "huber": huber, "vel": vel, "acc": acc}
