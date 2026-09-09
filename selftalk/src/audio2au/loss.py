"""Stage 2: confidence- and AU-balanced Audio-to-AU losses.

    L_AU = Huber(Û_t, U_t)
           + lambda_v * ||delta Û_t - delta U_t||_1
           + lambda_a * ||delta2 Û_t - delta2 U_t||_1
           + lambda_c * BCE(confidence_hat_t, confidence_t)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _first_difference(x: torch.Tensor) -> torch.Tensor:
    """First temporal difference: [B, T, A] -> [B, T-1, A]."""
    return x[:, 1:] - x[:, :-1]


def _second_difference(x: torch.Tensor) -> torch.Tensor:
    """Second temporal difference: [B, T, A] -> [B, T-2, A]."""
    return x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]


def _weighted_mean(
    values: torch.Tensor,
    confidence: torch.Tensor | None,
    au_weights: torch.Tensor | None,
) -> torch.Tensor:
    weights = torch.ones_like(values)
    if confidence is not None:
        if confidence.shape != values.shape:
            raise ValueError(
                f"confidence shape {tuple(confidence.shape)} != {tuple(values.shape)}"
            )
        weights = weights * confidence
    if au_weights is not None:
        if au_weights.ndim != 1 or au_weights.shape[0] != values.shape[-1]:
            raise ValueError(f"au_weights must be [A], got {tuple(au_weights.shape)}")
        weights = weights * au_weights.to(values).view(1, 1, -1)
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def au_regression_loss(
    pred_au: torch.Tensor,
    gt_au: torch.Tensor,
    confidence: torch.Tensor | None = None,
    beta: float = 1.0,
    au_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Confidence-weighted, per-AU-balanced Huber regression loss."""
    if pred_au.shape != gt_au.shape:
        raise ValueError(f"shape mismatch {tuple(pred_au.shape)} vs {tuple(gt_au.shape)}")
    if pred_au.ndim != 3:
        raise ValueError(f"expected [B, T, A], got {tuple(pred_au.shape)}")
    values = F.huber_loss(pred_au, gt_au, delta=beta, reduction="none")
    return _weighted_mean(values, confidence, au_weights)


def audio2au_loss(
    pred_au: torch.Tensor,
    gt_au: torch.Tensor,
    confidence: torch.Tensor | None = None,
    lambda_vel: float = 0.2,
    lambda_acc: float = 0.1,
    au_weights: torch.Tensor | None = None,
    pred_confidence: torch.Tensor | None = None,
    lambda_confidence: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Compute intensity, temporal, and optional confidence objectives.

    Args:
        pred_au: Predicted AU intensities [B, T, A].
        gt_au: OpenFace AU intensities [B, T, A].
        confidence: OpenFace frame confidence broadcast to [B, T, A].
        lambda_vel: First-difference loss weight.
        lambda_acc: Second-difference loss weight.
        au_weights: Optional per-AU weights [A], normally inverse standard
            deviation normalized to mean one.
        pred_confidence: Optional predicted confidence [B, T, A].
        lambda_confidence: BCE weight for predicted confidence.
    """
    if pred_au.shape != gt_au.shape:
        raise ValueError(f"shape mismatch {tuple(pred_au.shape)} vs {tuple(gt_au.shape)}")
    if pred_au.ndim != 3 or pred_au.shape[1] < 3:
        raise ValueError("pred_au/gt_au must be [B, T>=3, A]")

    huber = au_regression_loss(pred_au, gt_au, confidence, au_weights=au_weights)

    vel_values = (_first_difference(pred_au) - _first_difference(gt_au)).abs()
    vel_confidence = None
    if confidence is not None:
        vel_confidence = torch.minimum(confidence[:, 1:], confidence[:, :-1])
    vel = _weighted_mean(vel_values, vel_confidence, au_weights)

    acc_values = (_second_difference(pred_au) - _second_difference(gt_au)).abs()
    acc_confidence = None
    if confidence is not None:
        acc_confidence = torch.minimum(
            torch.minimum(confidence[:, 2:], confidence[:, 1:-1]),
            confidence[:, :-2],
        )
    acc = _weighted_mean(acc_values, acc_confidence, au_weights)

    confidence_loss = torch.zeros((), device=pred_au.device, dtype=pred_au.dtype)
    if pred_confidence is not None:
        if confidence is None:
            raise ValueError("confidence target required with pred_confidence")
        if pred_confidence.shape != confidence.shape:
            raise ValueError("predicted and target confidence shapes must match")
        confidence_loss = F.binary_cross_entropy(
            pred_confidence.clamp(1e-6, 1.0 - 1e-6), confidence
        )

    total = (
        huber
        + lambda_vel * vel
        + lambda_acc * acc
        + lambda_confidence * confidence_loss
    )
    return {
        "total": total,
        "huber": huber,
        "vel": vel,
        "acc": acc,
        "confidence": confidence_loss,
    }
