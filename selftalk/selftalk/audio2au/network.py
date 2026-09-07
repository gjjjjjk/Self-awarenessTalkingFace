"""Stage 2: Audio-to-AU network (speech-related lower-face AUs only in v1).

    audio window A_{t-w:t+w}
        ↓  (frozen pretrained audio encoder — Wav2Vec2/HuBERT, run in stage 0)
    feature projection
        ↓
    Temporal Transformer
        ↓
    AU intensity (lower-face) + AU confidence

Blink, gaze and head pose are NOT predicted here; they enter stage 3 as
independent condition tokens (see `deformation/tokens.py`).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..configs import Audio2AUConfig


class TemporalTransformer(nn.Module):
    """Causal-free Transformer encoder over the time axis.

    Input x: [B, T, D]. Output: [B, T, D].
    """

    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float) -> None:
        super().__init__()
        if d_model <= 0 or n_heads <= 0 or n_layers <= 0:
            raise ValueError("d_model/n_heads/n_layers must be positive")
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.pos_embed = nn.Parameter(0.02 * torch.randn(1, 1024, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must be [B, T, D], got {tuple(x.shape)}")
        t = x.shape[1]
        if t > self.pos_embed.shape[1]:
            raise ValueError(f"sequence length {t} exceeds positional table {self.pos_embed.shape[1]}")
        return self.encoder(x + self.pos_embed[:, :t])


class Audio2AUNet(nn.Module):
    """Predicts lower-face AU intensities (in [0, 5]) and confidence.

    Args:
        config: Audio2AUConfig.
        num_out_au: number of predicted AUs; defaults to lower-face count.

    Input/Output:
        audio_window A_{t-w:t+w}: [B, T, D_in] pretrained audio features.
        returns dict with `au`: [B, T, num_out_au] and
        `confidence`: [B, T, num_out_au] in (0, 1).
    """

    AU_MAX = 5.0

    def __init__(self, config: Audio2AUConfig, num_out_au: int | None = None) -> None:
        super().__init__()
        self.config = config
        self.num_out_au = int(num_out_au or len(config.lower_face_au_index))
        self.input_proj = nn.Linear(config.audio_feature_dim, config.d_model)
        self.temporal = TemporalTransformer(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            dropout=config.dropout,
        )
        self.head_au = nn.Linear(config.d_model, self.num_out_au)
        self.head_conf = nn.Linear(config.d_model, self.num_out_au)
        nn.init.zeros_(self.head_au.bias)
        nn.init.zeros_(self.head_conf.bias)

    def forward(self, audio_window: torch.Tensor) -> dict[str, torch.Tensor]:
        if audio_window.ndim != 3:
            raise ValueError(f"audio_window must be [B, T, D], got {tuple(audio_window.shape)}")
        if audio_window.shape[-1] != self.input_proj.in_features:
            raise ValueError(
                f"audio feature dim {audio_window.shape[-1]} != "
                f"{self.input_proj.in_features}; check DataConfig.audio_feature_dim"
            )
        h = self.temporal(self.input_proj(audio_window))
        au = self.AU_MAX * torch.sigmoid(self.head_au(h))
        confidence = torch.sigmoid(self.head_conf(h))
        return {"au": au, "confidence": confidence}
