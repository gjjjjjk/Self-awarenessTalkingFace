"""Stage 3: minimal viable deformation baseline (direct cross-attention).

    z_{i,t} = q_i + CrossAttn(q_i, C_t)

Every Gaussian query attends over the concatenated condition tokens C_t =
[A_t, U_t, E_t, V_t, S_t]; independent heads then predict the deltas. No
global motion tokens here — that comparison model lives in
`deformation/motion_tokens.py` (stage 4).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..canonical import CanonicalGaussians
from ..configs import DeformationConfig
from .heads import DeformationHeads, GaussianQueryMLP, canonical_attribute_summary
from .types import ConditionTokens, DeformationDeltas


class CrossAttentionBlock(nn.Module):
    """Post-norm cross-attention residual block: x + Attn(x, ctx)."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model),
        )
        self.norm_ff = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or context.ndim != 3:
            raise ValueError("x and context must be [B, M, d]")
        attended, _ = self.attn(x, context, context, need_weights=False)
        x = self.norm(x + attended)
        return self.norm_ff(x + self.ff(x))


class DirectConditionDeformation(nn.Module):
    """Stage-3 baseline: Gaussian queries attend the condition tokens.

    Args:
        config: DeformationConfig.
        triplane_dim: dimension F of f(mu_i).
        num_sh: number of SH coefficients per channel K = (deg+1)^2.
        drop_tokens: condition groups to zero out (stage-3 ablations),
            e.g. ("au",) for audio-only, ("au", "blink", "view") etc.
    """

    def __init__(
        self,
        config: DeformationConfig,
        triplane_dim: int,
        num_sh: int = 16,
        drop_tokens: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        self.config = config
        self.drop_tokens = tuple(drop_tokens)
        self.query_mlp = GaussianQueryMLP(
            triplane_dim=triplane_dim,
            d_model=config.d_model,
            region_feat_dim=config.region_feat_dim,
        )
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(config.d_model, config.n_heads)
            for _ in range(config.cross_attn_layers)
        ])
        self.heads = DeformationHeads(config.d_model, hidden=config.d_model, num_sh=num_sh)

    def forward(
        self,
        canonical: CanonicalGaussians,
        condition: ConditionTokens,
        region_ids: torch.Tensor,
    ) -> DeformationDeltas:
        """Compute parameter deltas for all Gaussians.

        Args:
            canonical: stage-1 output (features included).
            condition: stage-3 condition tokens C_t.
            region_ids: [N] integer region ids.

        Returns:
            DeformationDeltas with all fields [B, N, ...].
        """
        cond = condition.without(self.drop_tokens) if self.drop_tokens else condition
        context = cond.as_sequence()
        if context.shape[-1] != self.config.d_model:
            raise ValueError(
                f"condition dim {context.shape[-1]} != d_model {self.config.d_model}"
            )
        attrs = canonical_attribute_summary(
            canonical.rotations, canonical.scales, canonical.opacities
        )
        q = self.query_mlp(canonical.features, canonical.means, attrs, region_ids)
        z = q
        for block in self.blocks:
            z = block(z, context)
        return self.heads(z)
