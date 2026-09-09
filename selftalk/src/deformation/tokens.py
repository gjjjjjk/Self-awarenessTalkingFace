"""Stage 3: condition token encoders C_t = [A_t, U_t, E_t, V_t, S_t].

Per-AU tokens (one token per AU intensity) make stage-7 editing trivial:
scaling AU26's intensity scales exactly that token. The style token is a
learned global embedding. Stage-3 ablations drop token groups via
`ConditionTokens.without(("au", "blink", ...))`.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .types import ConditionTokens


def sinusoidal_pe(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal position encoding of a [B, M] scalar field -> [B, M*dim]."""
    if dim % 2 != 0:
        raise ValueError("dim must be even")
    half = dim // 2
    freqs = torch.exp(torch.arange(half, dtype=torch.float32, device=x.device)
                      * (-math.log(10000.0) / max(half, 1)))
    phases = x.unsqueeze(-1) * freqs  # [B, M, half]
    return torch.cat([phases.sin(), phases.cos()], dim=-1).flatten(1)


class ConditionEncoder(nn.Module):
    """Encode raw frame conditions into d-dim tokens.

    Input (all per frame t, batched):
        audio_window A_{t-w:t+w}: [B, T, D_audio]
        au: [B, A] intensities (all AUs; lower-face masked if requested)
        blink E_t: [B, 1]
        pose V_t: [B, 6]
    """

    def __init__(self, d_model: int, audio_feature_dim: int = 768, num_au: int = 17,
                 blink_pe_dim: int = 16) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        self.d_model = d_model
        self.audio_proj = nn.Linear(audio_feature_dim, d_model)
        self.au_embed = nn.Sequential(
            nn.Linear(1 + 16, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.au_intensity_scale = nn.Linear(1, 16)
        self.blink_embed = nn.Sequential(nn.Linear(blink_pe_dim, d_model), nn.GELU(),
                                         nn.Linear(d_model, d_model))
        self.blink_pe_dim = blink_pe_dim
        self.view_embed = nn.Sequential(nn.Linear(6, d_model), nn.GELU(),
                                        nn.Linear(d_model, d_model))
        self.style = nn.Parameter(0.02 * torch.randn(1, 1, d_model))

    def forward(
        self,
        audio_window: torch.Tensor,
        au: torch.Tensor,
        blink: torch.Tensor,
        pose: torch.Tensor,
    ) -> ConditionTokens:
        if audio_window.ndim != 3:
            raise ValueError(f"audio_window must be [B, T, D], got {tuple(audio_window.shape)}")
        if au.ndim != 2:
            raise ValueError(f"au must be [B, A], got {tuple(au.shape)}")
        if blink.ndim != 2 or blink.shape[-1] != 1:
            raise ValueError(f"blink must be [B, 1], got {tuple(blink.shape)}")
        if pose.ndim != 2 or pose.shape[-1] != 6:
            raise ValueError(f"pose must be [B, 6], got {tuple(pose.shape)}")

        b = au.shape[0]
        audio_tokens = self.audio_proj(audio_window)

        au_feat = torch.cat([au.unsqueeze(-1), self.au_intensity_scale(au.unsqueeze(-1))], dim=-1)
        au_tokens = self.au_embed(au_feat)  # [B, A, d]

        blink_pe = sinusoidal_pe(blink, self.blink_pe_dim)
        blink_tokens = self.blink_embed(blink_pe).unsqueeze(1)  # [B, 1, d]
        view_tokens = self.view_embed(pose).unsqueeze(1)  # [B, 1, d]
        style_tokens = self.style.expand(b, 1, self.d_model)

        return ConditionTokens(
            audio=audio_tokens,
            au=au_tokens,
            blink=blink_tokens,
            view=view_tokens,
            style=style_tokens,
        )
