"""Stages 3-5: typed containers shared by the deformation networks."""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch


@dataclass
class ConditionTokens:
    """Condition tokens C_t = [A_t, U_t, E_t, V_t, S_t] (stage 3).

    audio:  [B, T_a, d] per-frame audio tokens.
    au:     [B, T_u, d] one token per AU intensity (editable, stage 7).
    blink:  [B, 1, d] eye-closure token E_t.
    view:   [B, 1, d] head-pose/viewpoint token V_t.
    style:  [B, 1, d] learned identity/style token S_t.
    """

    audio: torch.Tensor
    au: torch.Tensor
    blink: torch.Tensor
    view: torch.Tensor
    style: torch.Tensor

    def as_sequence(self) -> torch.Tensor:
        """Concatenated condition sequence C_t: [B, T_a + T_u + 3, d]."""
        return torch.cat([self.audio, self.au, self.blink, self.view, self.style], dim=1)

    def without(self, names: tuple[str, ...]) -> "ConditionTokens":
        """Zero out the named token groups (for stage-3 ablations)."""
        out = {}
        for f in fields(self):
            value = getattr(self, f.name)
            out[f.name] = torch.zeros_like(value) if f.name in names else value
        return ConditionTokens(**out)


@dataclass
class DeformationDeltas:
    """Per-Gaussian parameter deltas (stage 3 heads, stage 5 constrained use).

    means: Δμ [B, N, 3]; rotations: Δq [B, N, 4] (applied as Δq ⊗ r_c);
    log_scales: Δlog s [B, N, 3]; sh: ΔSH [B, N, K, 3];
    logit_opacities: Δα [B, N, 1] (added on the canonical logit).
    """

    means: torch.Tensor
    rotations: torch.Tensor
    log_scales: torch.Tensor
    sh: torch.Tensor
    logit_opacities: torch.Tensor


@dataclass
class DeformedGaussians:
    """Result of the constrained update (stage 5); ready for rendering."""

    means: torch.Tensor
    rotations: torch.Tensor
    scales: torch.Tensor
    sh: torch.Tensor
    opacities: torch.Tensor
