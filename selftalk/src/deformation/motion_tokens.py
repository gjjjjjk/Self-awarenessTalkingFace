"""Stage 4: global motion tokens.

    M_t = TransformerDecoder(Q_motion, [A_t, U_t, E_t, V_t, S_t])
    z_{i,t} = q_i + CrossAttn(q_i, M_t)

with learned motion queries Q_motion ∈ R^{K×d} (K small, ablated). The N
Gaussian queries ATTEND TO the K motion tokens (never the reverse: a few
global tokens must not query Gaussians and emit all offsets directly).
Audio and AU streams are fused by a gated residual so AU conditioning cannot
overwrite audio detail: A_fused = A + σ(g) ⊙ U_res, g = MLP([pool(A), pool(U)]).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..canonical import CanonicalGaussians
from ..configs import DeformationConfig
from .baseline import CrossAttentionBlock
from .heads import DeformationHeads, GaussianQueryMLP, canonical_attribute_summary
from .types import ConditionTokens, DeformationDeltas


class GatedAudioAUFusion(nn.Module):
    """Parallel residual gated fusion of audio and AU token streams."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.au_residual = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model), nn.Sigmoid()
        )

    def forward(self, audio: torch.Tensor, au: torch.Tensor) -> torch.Tensor:
        """audio [B, Ta, d], au [B, Tu, d] -> fused audio stream [B, Ta, d]."""
        pooled = torch.cat([audio.mean(dim=1), au.mean(dim=1)], dim=-1)
        gate = self.gate(pooled).unsqueeze(1)  # [B, 1, d]
        return audio + gate * self.au_residual(au).mean(dim=1, keepdim=True)


class MotionTokenizer(nn.Module):
    """Learned motion queries decoded from the condition tokens (stage 4)."""

    def __init__(self, config: DeformationConfig) -> None:
        super().__init__()
        if config.num_motion_tokens <= 0:
            raise ValueError("num_motion_tokens (K) must be positive")
        self.motion_queries = nn.Parameter(
            0.02 * torch.randn(config.num_motion_tokens, config.d_model)
        )
        self.fusion = GatedAudioAUFusion(config.d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=4 * config.d_model,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.decoder_layers)

    def forward(self, condition: ConditionTokens) -> torch.Tensor:
        """Condition tokens -> motion tokens M_t: [B, K, d]."""
        audio = self.fusion(condition.audio, condition.au)
        memory = torch.cat([
            audio, condition.au, condition.blink, condition.view, condition.style
        ], dim=1)
        b = memory.shape[0]
        queries = self.motion_queries.unsqueeze(0).expand(b, -1, -1)
        return self.decoder(queries, memory)


class MotionTokenDeformation(nn.Module):
    """Stage-4 model: Gaussian queries attend to K motion tokens."""

    def __init__(
        self,
        config: DeformationConfig,
        triplane_dim: int,
        num_sh: int = 16,
    ) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = MotionTokenizer(config)
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
        """Compute deltas; canonical/condition as in stage 3, region_ids [N]."""
        motion = self.tokenizer(condition)  # [B, K, d]
        attrs = canonical_attribute_summary(
            canonical.rotations, canonical.scales, canonical.opacities
        )
        q = self.query_mlp(canonical.features, canonical.means, attrs, region_ids)
        z = q
        for block in self.blocks:
            z = block(z, motion)
        return self.heads(z)
