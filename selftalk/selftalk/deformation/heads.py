"""Stage 3: Gaussian query MLP and deformation delta heads.

    q_i = MLP[f(mu_i), PE(mu_i), g_i^can, r_i^region]

with f the triplane feature, PE a sinusoidal position encoding, g_i^can a
summary of the canonical attributes (r_i, s_i, alpha_i) and r_i^region a
learned region embedding. The heads map the fused feature z_{i,t} to the
parameter deltas (Δmu, Δq, Δlog s, ΔSH, Δalpha); final layers are
zero-initialized so training starts exactly at the canonical Gaussians.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import rearrange

from .region_weights import NUM_REGIONS
from .types import DeformationDeltas


def position_encoding(x: torch.Tensor, mult: int) -> torch.Tensor:
    """Sinusoidal PE: x [B, N, C] -> [B, N, C * mult * 2] (sin|cos per band)."""
    if mult <= 0:
        raise ValueError("mult must be positive")
    freqs = torch.exp(torch.arange(mult, dtype=x.dtype, device=x.device)
                      * (-math.log(10000.0) / max(mult - 1, 1)))
    phases = x.unsqueeze(-1) * freqs
    return torch.cat([phases.sin(), phases.cos()], dim=-1).flatten(-2)


class GaussianQueryMLP(nn.Module):
    """Per-Gaussian query q_i from canonical feature + PE + attrs + region."""

    def __init__(
        self,
        triplane_dim: int,
        d_model: int,
        hidden: int = 128,
        pe_mult: int = 4,
        region_feat_dim: int = 16,
    ) -> None:
        super().__init__()
        self.pe_mult = pe_mult
        self.region_embed = nn.Embedding(NUM_REGIONS, region_feat_dim)
        in_dim = triplane_dim + 3 * pe_mult * 2 + 8 + region_feat_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, d_model), nn.GELU(),
        )

    def forward(
        self,
        features: torch.Tensor,
        means: torch.Tensor,
        canonical_attrs: torch.Tensor,
        region_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
            features: f(mu_i) [B, N, F].
            means: mu_i [B, N, 3].
            canonical_attrs: g_i^can [B, N, 8] = (r 4, s 3, alpha 1).
            region_ids: r_i^region [N] or [B, N] integer region ids.

        Returns q_i: [B, N, d].
        """
        if canonical_attrs.shape[-1] != 8:
            raise ValueError("canonical_attrs must be [B, N, 8]")
        pe = position_encoding(means, self.pe_mult)
        region = self.region_embed(region_ids)
        if region.ndim == 2:
            region = region.unsqueeze(0).expand(features.shape[0], -1, -1)
        return self.mlp(torch.cat([features, pe, canonical_attrs, region], dim=-1))


def canonical_attribute_summary(
    rotations: torch.Tensor, scales: torch.Tensor, opacities: torch.Tensor
) -> torch.Tensor:
    """g_i^can = concat(r_i, s_i, alpha_i): [B, N, 4/3/1] -> [B, N, 8]."""
    return torch.cat([rotations, scales, opacities], dim=-1)


class DeformationHeads(nn.Module):
    """Independent heads z_{i,t} -> (Δmu, Δq, Δlog s, ΔSH, Δalpha)."""

    def __init__(self, d_model: int, hidden: int, num_sh: int) -> None:
        super().__init__()
        self.num_sh = num_sh

        def head(out_dim: int, bias_init: str = "zeros") -> nn.Sequential:
            final = nn.Linear(hidden, out_dim)
            nn.init.zeros_(final.weight)
            if bias_init == "identity_quat":
                nn.init.zeros_(final.bias)
                with torch.no_grad():
                    final.bias[0] = 1.0
            else:
                nn.init.zeros_(final.bias)
            return nn.Sequential(nn.GELU(), nn.Linear(d_model, hidden), nn.GELU(), final)

        self.head_means = head(3)
        self.head_rotations = head(4, bias_init="identity_quat")
        self.head_log_scales = head(3)
        self.head_sh = head(3 * num_sh)
        self.head_logit_opacities = head(1)

    def forward(self, z: torch.Tensor) -> DeformationDeltas:
        """z: [B, N, d] -> DeformationDeltas (all [B, N, ...])."""
        if z.ndim != 3:
            raise ValueError(f"z must be [B, N, d], got {tuple(z.shape)}")
        sh = self.head_sh(z)
        return DeformationDeltas(
            means=self.head_means(z),
            rotations=self.head_rotations(z),
            log_scales=self.head_log_scales(z),
            sh=rearrange(sh, "b n (k c) -> b n k c", k=self.num_sh),
            logit_opacities=self.head_logit_opacities(z),
        )
