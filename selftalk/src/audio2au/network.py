"""Stage 2: Audio-to-AU network for frame-aligned acoustic features.

The May data stores one DeepSpeech feature map [16, 29] per video frame.
Each map is flattened to a frame token before a non-causal temporal
Transformer predicts lower-face AU intensity and tracking confidence.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..configs import Audio2AUConfig


class TemporalTransformer(nn.Module):
    """Non-causal Transformer encoder over the video-frame time axis."""

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
        """Encode x [B, T, D] and return [B, T, D]."""
        if x.ndim != 3:
            raise ValueError(f"x must be [B, T, D], got {tuple(x.shape)}")
        length = x.shape[1]
        if length > self.pos_embed.shape[1]:
            raise ValueError(
                f"sequence length {length} exceeds positional table "
                f"{self.pos_embed.shape[1]}"
            )
        return self.encoder(x + self.pos_embed[:, :length])


class Audio2AUNet(nn.Module):
    """Predict lower-face AU intensities and OpenFace tracking confidence.

    Args:
        config: Audio2AUConfig. ``audio_feature_dim`` is the flattened
            dimension of one video-frame acoustic feature (464 for May).
        num_out_au: Number of predicted lower-face AUs.

    Inputs:
        audio_window: [B, T, D] or May-format [B, T, 16, 29].

    Returns:
        A dict containing ``au`` [B, T, A] in (0, 5) and ``confidence``
        [B, T, A] in (0, 1).
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

    def initialize_au_prior(self, mean_au: torch.Tensor) -> None:
        """Initialize the AU head bias from training-set means [A]."""
        if mean_au.shape != (self.num_out_au,):
            raise ValueError(
                f"mean_au must be [{self.num_out_au}], got {tuple(mean_au.shape)}"
            )
        probability = (mean_au / self.AU_MAX).clamp(1e-4, 1.0 - 1e-4)
        with torch.no_grad():
            self.head_au.bias.copy_(torch.logit(probability).to(self.head_au.bias))

    def forward(self, audio_window: torch.Tensor) -> dict[str, torch.Tensor]:
        if audio_window.ndim < 3:
            raise ValueError(
                f"audio_window must be [B, T, ...], got {tuple(audio_window.shape)}"
            )
        batch, length = audio_window.shape[:2]
        flattened = audio_window.reshape(batch, length, -1)
        if flattened.shape[-1] != self.config.audio_feature_dim:
            raise ValueError(
                f"flattened audio feature dim {flattened.shape[-1]} != "
                f"{self.config.audio_feature_dim}"
            )
        hidden = self.temporal(torch.nn.functional.gelu(self.input_proj(flattened)))
        au = self.AU_MAX * torch.sigmoid(self.head_au(hidden))
        confidence = torch.sigmoid(self.head_conf(hidden))
        return {"au": au, "confidence": confidence}
