"""Stage 1: canonical 3D Gaussian model G_can = {mu_i, r_i, s_i, SH_i, alpha_i, f(mu_i)}.

3DMM/FLAME vertices initialize the Gaussian positions mu; a small MLP maps
the triplane feature f(mu_i) to rotation / scale / SH / opacity of the
canonical Gaussian. Stage-5 deformation later only ever applies constrained
delta updates on top of these canonical parameters.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .triplane import MultiResolutionTriplane


@dataclass
class CanonicalGaussians:
    """Canonical Gaussian set (per batch).

    means: [B, N, 3]; rotations: [B, N, 4] unit quaternions (w, x, y, z);
    scales: [B, N, 3] positive; sh: [B, N, K, 3] with K=(deg+1)^2;
    opacities: [B, N, 1] in (0, 1); features: [B, N, F] triplane f(mu).
    """

    means: torch.Tensor
    rotations: torch.Tensor
    scales: torch.Tensor
    sh: torch.Tensor
    opacities: torch.Tensor
    features: torch.Tensor

    @property
    def num_gaussians(self) -> int:
        return self.means.shape[1]


def quaternion_normalize(q: torch.Tensor) -> torch.Tensor:
    """Project [.., 4] onto the unit-quaternion sphere (w, x, y, z)."""
    return F.normalize(q, dim=-1, eps=1e-8)


class CanonicalGaussianModel(nn.Module):
    """T riplane + MLP predicting canonical Gaussian attributes.

    Args:
        triplane: the multi-resolution feature field f(mu).
        mlp_hidden: hidden width of the attribute MLP.
        mlp_depth: number of hidden linear layers.
        sh_degree: SH degree; K = (sh_degree + 1)^2 coefficients per channel.
    """

    def __init__(
        self,
        triplane: MultiResolutionTriplane,
        mlp_hidden: int = 64,
        mlp_depth: int = 4,
        sh_degree: int = 3,
    ) -> None:
        super().__init__()
        if sh_degree < 0:
            raise ValueError("sh_degree must be >= 0")
        self.triplane = triplane
        self.sh_degree = sh_degree
        self.num_sh = (sh_degree + 1) ** 2
        in_dim = triplane.fuse.out_features

        hidden: list[nn.Module] = []
        dim = in_dim
        for _ in range(mlp_depth):
            hidden += [nn.Linear(dim, mlp_hidden), nn.ReLU()]
            dim = mlp_hidden
        self.trunk = nn.Sequential(*hidden)

        def head(out_dim: int) -> nn.Sequential:
            final = nn.Linear(mlp_hidden, out_dim)
            nn.init.xavier_uniform_(final.weight, gain=0.1)
            nn.init.zeros_(final.bias)
            return nn.Sequential(nn.ReLU(), final)

        self.head_rotation = head(4)
        self.head_scale = head(3)
        self.head_opacity = head(1)
        self.head_sh = head(3 * self.num_sh)

    def init_from_vertices(self, vertices: torch.Tensor) -> None:
        """Initialize canonical means mu from 3DMM/FLAME vertices [N, 3]."""
        if vertices.ndim != 2 or vertices.shape[-1] != 3:
            raise ValueError(f"vertices must be [N, 3], got {tuple(vertices.shape)}")
        self.register_buffer("means", vertices.detach().clone())

    def _default_means(self, num: int, device: torch.device) -> torch.Tensor:
        if hasattr(self, "means"):
            return self.means.to(device)
        return 0.4 * torch.rand(num, 3, device=device) - 0.2

    def forward(self, num_gaussians: int, batch_size: int = 1) -> CanonicalGaussians:
        """Predict canonical attributes for all Gaussians.

        Args:
            num_gaussians: N; must match the vertex count given to
                `init_from_vertices` when that was called.
            batch_size: B (canonical parameters are shared across the batch).

        Returns:
            CanonicalGaussians with all fields [B, N, ...].
        """
        device = next(self.parameters()).device
        means = self._default_means(num_gaussians, device)
        if means.shape[0] != num_gaussians:
            raise ValueError(
                f"num_gaussians={num_gaussians} != initialized vertices {means.shape[0]}"
            )
        mu = means.unsqueeze(0).expand(batch_size, -1, -1)
        feats = self.triplane(mu)
        h = self.trunk(feats)
        rotations = quaternion_normalize(self.head_rotation(h))
        scales = torch.exp(self.head_scale(h))
        opacities = torch.sigmoid(self.head_opacity(h))
        sh = rearrange(self.head_sh(h), "b n (k c) -> b n k c", k=self.num_sh)
        return CanonicalGaussians(
            means=mu,
            rotations=rotations,
            scales=scales,
            sh=sh,
            opacities=opacities,
            features=feats,
        )
