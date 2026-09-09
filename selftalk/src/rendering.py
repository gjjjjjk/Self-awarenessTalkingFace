"""Rasterizer front-end shared by stages 1-8.

Uses gsplat when available (GPU), else falls back to a pure-PyTorch
front-to-back alpha compositing renderer that runs on CPU and keeps the
whole pipeline (and the unit tests) importable without any CUDA extension.
Only the DC (view-independent) SH term is used by the CPU fallback.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from einops import rearrange

from .deformation.types import DeformedGaussians

try:  # gsplat is optional
    import gsplat
    _HAS_GSPLAT = True
except ImportError:  # pragma: no cover - depends on the environment
    gsplat = None
    _HAS_GSPLAT = False


@dataclass
class Camera:
    """Pinhole camera; K intrinsics [3, 3], world-to-cam extrinsic [4, 4]."""

    intrinsics: torch.Tensor
    world_to_camera: torch.Tensor
    height: int
    width: int

    @classmethod
    def look_at(
        cls,
        eye: torch.Tensor,
        target: torch.Tensor,
        up: torch.Tensor,
        fov_deg: float = 30.0,
        height: int = 256,
        width: int = 256,
    ) -> "Camera":
        """Construct a camera at `eye` looking at `target`."""
        device = eye.device
        # OpenCV convention: +z forward (towards the target), x right, y down,
        # so world points in front of the camera get positive depth.
        z = torch.nn.functional.normalize(target - eye, dim=-1)
        x = torch.nn.functional.normalize(torch.cross(z, up, dim=-1), dim=-1)
        y = torch.cross(z, x, dim=-1)
        rotation = torch.stack([x, y, z], dim=0)  # rows: cam axes in world
        translation = -rotation @ eye
        w2c = torch.eye(4, device=device)
        w2c[:3, :3] = rotation
        w2c[:3, 3] = translation
        f = 0.5 * height / math.tan(math.radians(fov_deg) / 2)
        K = torch.tensor([[f, 0.0, width / 2], [0.0, f, height / 2], [0.0, 0.0, 1.0]],
                         device=device)
        return cls(intrinsics=K, world_to_camera=w2c, height=height, width=width)


_SH_C0 = 0.28209479177387814


def _sh_dc_color(sh: torch.Tensor) -> torch.Tensor:
    """DC term of SH coefficients: sh [B?, N, K, 3] -> rgb [.., N, 3]."""
    return torch.clamp(_SH_C0 * sh[..., 0, :] + 0.5, 0.0, 1.0)


def _cpu_render(camera: Camera, gaussians: DeformedGaussians, background: torch.Tensor) -> torch.Tensor:
    """Depth-sorted front-to-back alpha compositing (reference renderer).

    gaussians.* are [N, ...] (single frame). Returns [3, H, W] in [0, 1].
    """
    device = camera.intrinsics.device
    means = gaussians.means.reshape(-1, 3).to(device)
    opacities = gaussians.opacities.reshape(-1).to(device)
    rgb = _sh_dc_color(gaussians.sh.reshape(-1, *gaussians.sh.shape[-2:]))

    K = camera.intrinsics
    w2c = camera.world_to_camera
    cam_points = (w2c[:3, :3] @ means.T).T + w2c[:3, 3]
    depth = cam_points[:, 2]
    if (depth <= 1e-6).all():
        return background.expand(3, camera.height, camera.width).clamp(0, 1)
    uv_h = (K @ cam_points.T).T
    uv = uv_h[:, :2] / uv_h[:, 2:].clamp_min(1e-6)
    visible = (uv_h[:, 2] > 1e-6) \
        & (uv[:, 0] >= 0) & (uv[:, 0] < camera.width) \
        & (uv[:, 1] >= 0) & (uv[:, 1] < camera.height)

    order = torch.argsort(depth)  # near-to-front first: front-to-back compositing
    order = order[visible[order]]

    height, width = camera.height, camera.width
    pixel_uv = uv[order]
    px = pixel_uv[:, 0].round().long().clamp(0, width - 1)
    py = pixel_uv[:, 1].round().long().clamp(0, height - 1)
    alpha_pixel = opacities[order].clamp(0, 1)
    color_pixel = rgb[order]

    canvas = torch.zeros(height * width, 3, device=device)
    transmittance = torch.ones(height * width, device=device)
    flat = py * width + px
    if flat.numel() > 0:
        # Group Gaussians by pixel (stable sort keeps depth order per group)
        # and composite one depth layer at a time; every update is out-of-place
        # so autograd can differentiate through the compositing.
        group = torch.argsort(flat, stable=True)
        flat_g = flat[group]
        alpha_g = alpha_pixel[group]
        color_g = color_pixel[group]
        ar = torch.arange(flat_g.shape[0], device=device)
        new_group = torch.ones_like(ar, dtype=torch.bool)
        new_group[1:] = flat_g[1:] != flat_g[:-1]
        starts = torch.cummax(torch.where(new_group, ar, torch.zeros_like(ar)), 0).values
        rank = ar - starts  # depth index within each pixel group
        for k in range(int(rank.max()) + 1):
            sel = rank == k
            idx = flat_g[sel]
            a = alpha_g[sel]
            update = torch.zeros_like(canvas)
            update[idx] = (a * transmittance[idx]).unsqueeze(-1) * color_g[sel]
            canvas = canvas + update
            factor = torch.ones_like(transmittance)
            factor[idx] = 1.0 - a
            transmittance = transmittance * factor
    canvas = canvas + transmittance.unsqueeze(-1) * background.to(device)
    return rearrange(canvas, "(h w) c -> c h w", h=height)


def render(
    camera: Camera,
    gaussians: DeformedGaussians,
    background: torch.Tensor | None = None,
) -> torch.Tensor:
    """Render one frame. gaussians.* are [N, ...] (batch dim squeezed upstream).

    Returns [3, H, W] in [0, 1].
    """
    if background is None:
        background = torch.zeros(3, device=camera.intrinsics.device)
    if gaussians.means.ndim != 2:
        raise ValueError(f"expected per-frame [N, ...] gaussians, got {tuple(gaussians.means.shape)}")
    if not _HAS_GSPLAT:
        return _cpu_render(camera, gaussians, background)

    device = camera.intrinsics.device
    means = gaussians.means.to(device)
    quats = gaussians.rotations.to(device)
    scales = gaussians.scales.to(device)
    opacities = gaussians.opacities.reshape(-1).to(device)
    sh = gaussians.sh.to(device)
    viewmats = camera.world_to_camera.to(device).unsqueeze(0)
    Ks = camera.intrinsics.to(device).unsqueeze(0)
    colors, _alpha, _info = gsplat.rasterization(
        means=means,
        quats=quats,
        scales=scales.clamp_min(1e-6),
        opacities=opacities,
        colors=_sh_dc_color(sh),
        viewmats=viewmats,
        Ks=Ks,
        width=camera.width,
        height=camera.height,
        backgrounds=background.to(device).reshape(1, 3),
        render_mode="RGB",
        absgrad=False,
    )
    return colors[0].permute(2, 0, 1).clamp(0, 1)
