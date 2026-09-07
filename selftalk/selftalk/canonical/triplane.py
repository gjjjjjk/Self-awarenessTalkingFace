"""Stage 1: multi-resolution triplane feature field f(mu).

Points mu in a normalized [-1, 1]^3 head volume are projected onto the three
axis-aligned planes (xy, yz, xz) at multiple resolutions; bilinear samples
are concatenated and projected to a common feature dimension, giving the
spatial feature f(mu_i) attached to every canonical Gaussian.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiResolutionTriplane(nn.Module):
    """Triplane bank at resolutions R_1 < R_2 < ... producing f(mu).

    Args:
        resolutions: plane spatial resolutions per level.
        feat_dim: per-level feature channels (same for all levels here).
        out_dim: dimension of the fused feature f(mu).

    Input/Output:
        mu: [B, N, 3] world-space Gaussian centers (assumed inside bounds).
        returns f(mu): [B, N, out_dim].
    """

    _AXES = ("xy", "yz", "xz")

    def __init__(
        self,
        resolutions: tuple[int, ...] = (64, 128, 256),
        feat_dim: int = 32,
        out_dim: int = 64,
    ) -> None:
        super().__init__()
        if not resolutions or any(r < 2 for r in resolutions):
            raise ValueError(f"invalid triplane resolutions: {resolutions}")
        self.resolutions = tuple(resolutions)
        planes = {}
        for res in self.resolutions:
            for axis in self._AXES:
                planes[f"{axis}_{res}"] = nn.Parameter(0.1 * torch.randn(feat_dim, res, res))
        self.planes = nn.ParameterDict(planes)
        self.fuse = nn.Linear(feat_dim * len(self._AXES) * len(self.resolutions), out_dim)

    def _sample_plane(self, plane: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
        """Bilinear-sample one plane.

        plane: [C, R, R]; uv: [B, N, 2] in [-1, 1]. Returns [B, N, C].
        """
        grid = uv.unsqueeze(2)  # [B, N, 1, 2]
        feat = F.grid_sample(plane.unsqueeze(0).expand(uv.shape[0], -1, -1, -1),
                             grid, mode="bilinear", padding_mode="border", align_corners=True)
        return feat.squeeze(3).permute(0, 2, 1)

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        if mu.ndim != 3 or mu.shape[-1] != 3:
            raise ValueError(f"mu must be [B, N, 3], got {tuple(mu.shape)}")
        uv = torch.clamp(mu, -1.0, 1.0)
        parts = []
        for res in self.resolutions:
            parts.append(self._sample_plane(self.planes[f"xy_{res}"], uv[..., :2]))
            parts.append(self._sample_plane(self.planes[f"yz_{res}"], uv[..., 1:]))
            parts.append(self._sample_plane(self.planes[f"xz_{res}"], uv[..., [0, 2]]))
        return self.fuse(torch.cat(parts, dim=-1))
