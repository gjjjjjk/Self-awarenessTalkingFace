"""Stage 5: constrained Gaussian parameter update.

    mu_t     = mu_c + Δmu_t
    r_t      = normalize(Δq_t ⊗ r_c)
    s_t      = exp(log s_c + Δlog s_t)
    alpha_t  = sigmoid(logit(alpha_c) + Δalpha_t)

Deformation is NEVER applied by overwriting canonical parameters — only via
these legal delta updates, which keep rotations unit-norm, scales positive
and opacities in (0, 1) by construction.
"""
from __future__ import annotations

import torch

from ..canonical import CanonicalGaussians
from .types import DeformedGaussians, DeformationDeltas

_LOGIT_EPS = 1e-4


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product a ⊗ b; both [.., 4] in (w, x, y, z) order."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def safe_logit(alpha: torch.Tensor, eps: float = _LOGIT_EPS) -> torch.Tensor:
    """logit of alpha with clamping to avoid ±inf at 0/1."""
    return torch.logit(alpha.clamp(eps, 1.0 - eps), eps=eps)


def constrained_gaussian_update(
    canonical: CanonicalGaussians, deltas: DeformationDeltas
) -> DeformedGaussians:
    """Apply the stage-5 constrained update.

    Args:
        canonical: canonical parameters (all [B, N, ...]).
        deltas: predicted deltas of matching shapes.

    Returns:
        DeformedGaussians with means [B, N, 3], rotations [B, N, 4] (unit),
        scales [B, N, 3] (> 0), sh [B, N, K, 3], opacities [B, N, 1] ∈ (0, 1).
    """
    if canonical.means.shape != deltas.means.shape:
        raise ValueError(
            f"means shape mismatch {tuple(canonical.means.shape)} vs {tuple(deltas.means.shape)}"
        )
    if canonical.scales.shape != deltas.log_scales.shape:
        raise ValueError("scales/log_scales shape mismatch")

    means = canonical.means + deltas.means
    rotations = torch.nn.functional.normalize(
        quaternion_multiply(deltas.rotations, canonical.rotations), dim=-1
    )
    log_scales = torch.log(canonical.scales.clamp_min(1e-8)) + deltas.log_scales
    scales = torch.exp(log_scales)
    sh = canonical.sh + deltas.sh
    opacities = torch.sigmoid(safe_logit(canonical.opacities) + deltas.logit_opacities)
    return DeformedGaussians(
        means=means, rotations=rotations, scales=scales, sh=sh, opacities=opacities
    )
