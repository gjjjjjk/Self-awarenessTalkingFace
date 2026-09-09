"""Stage 6: photometric losses, added incrementally in the plan's order.

    1. L1 + D-SSIM + LPIPS          (lambda_rgb / lambda_ssim / lambda_perc)
    2. lip reconstruction loss       (lambda_lip, masked L1 around the mouth)
    3. geometric + temporal          (losses/regularizers.py)
    4. AU consistency                (selftalk.losses.au_consistency)
    5. audio-visual sync             (interface only — needs a trained SyncNet)

Every term is behind a weight flag so ablations (stage 8) can switch it off.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def l1_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return (pred - gt).abs().mean()


def d_ssim_loss(pred: torch.Tensor, gt: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """1 − SSIM; images [B, 3, H, W] in [0, 1]."""
    if pred.ndim != 4:
        raise ValueError(f"expected [B, 3, H, W], got {tuple(pred.shape)}")
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch {tuple(pred.shape)} vs {tuple(gt.shape)}")
    try:
        from pytorch_msssim import SSIM

        ssim = SSIM(data_range=1.0, win_size=window_size, size_average=True)
        return 1.0 - ssim(pred, gt)
    except ImportError:
        c = window_size // 2
        if pred.shape[-1] <= 2 * c or pred.shape[-2] <= 2 * c:
            return torch.zeros((), device=pred.device)
        p = F.avg_pool2d(pred, kernel_size=window_size, stride=1, padding=c)
        g = F.avg_pool2d(gt, kernel_size=window_size, stride=1, padding=c)
        return 1.0 - (1.0 - (p - g).abs().mean())


_LPIPS = None


def lpips_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """LPIPS perceptual loss (lazy import; returns 0 if lpips unavailable)."""
    global _LPIPS
    if pred.shape[-1] < 32 or pred.shape[-2] < 32:
        return torch.zeros((), device=pred.device)
    try:
        import lpips as lpips_pkg
    except ImportError:
        return torch.zeros((), device=pred.device)
    if _LPIPS is None:
        _LPIPS = lpips_pkg.LPIPS(net="alex")
    device = pred.device
    if next(_LPIPS.parameters()).device != device:
        _LPIPS = _LPIPS.to(device)
    return _LPIPS((pred * 2 - 1).clamp(-1, 1), (gt * 2 - 1).clamp(-1, 1)).mean()


def lip_region_mask(height: int, width: int, mouth_xy: torch.Tensor,
                    radius_frac: float = 0.12, device: torch.device | None = None) -> torch.Tensor:
    """Circular mouth mask [1, 1, H, W] centered at mouth_xy (x, y in pixels)."""
    if mouth_xy.shape[-1] != 2:
        raise ValueError("mouth_xy must be [2] (x, y)")
    device = device or mouth_xy.device
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    cx, cy = mouth_xy.to(device).unbind(-1)
    r = radius_frac * min(height, width)
    return (((xs - cx) ** 2 + (ys - cy) ** 2) < r**2).float()[None, None]


def lip_loss(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked L1 on the mouth region; mask [1, 1, H, W] or broadcastable."""
    if mask.sum() == 0:
        return torch.zeros((), device=pred.device)
    diff = (pred - gt).abs().mean(dim=1, keepdim=True)
    return (diff * mask).sum() / mask.sum().clamp_min(1.0) / 3.0


def photometric_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    lambda_rgb: float = 0.8,
    lambda_ssim: float = 0.2,
    lambda_perc: float = 0.01,
    lip_mask: torch.Tensor | None = None,
    lambda_lip: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Weighted stage-6 terms 1 and 2. Returns per-term dict including `total`."""
    terms = {
        "l1": l1_loss(pred, gt),
        "dssim": d_ssim_loss(pred, gt),
        "lpips": lpips_loss(pred, gt),
    }
    total = lambda_rgb * terms["l1"] + lambda_ssim * terms["dssim"] + lambda_perc * terms["lpips"]
    if lip_mask is not None and lambda_lip > 0:
        terms["lip"] = lip_loss(pred, gt, lip_mask)
        total = total + lambda_lip * terms["lip"]
    terms["total"] = total
    return terms


class SyncLoss(torch.nn.Module):
    """Audio-visual sync loss (stage 6 term 5).

    Requires an externally trained SyncNet scoring (audio_window, lip_crop)
    pairs. Provide `net` with a forward(audio: [B, T, D], lip: [B, 3, H, W])
    -> similarity [B]; until then this returns 0 with a warning once.
    """

    _warned = False

    def __init__(self, net: torch.nn.Module | None = None) -> None:
        super().__init__()
        self.net = net

    def forward(self, audio_window: torch.Tensor, lip_crop: torch.Tensor) -> torch.Tensor:
        if self.net is None:
            if not SyncLoss._warned:
                print("[stage6] SyncLoss: no SyncNet provided, returning 0")
                SyncLoss._warned = True
            return torch.zeros((), device=audio_window.device)
        return -self.net(audio_window, lip_crop).mean()


def au_consistency_loss(
    au_from_render_fn: torch.Tensor, target_au: torch.Tensor, confidence: torch.Tensor | None = None
) -> torch.Tensor:
    """L_AU-render: Huber between AUs extracted from the rendering and U_t.

    au_from_render_fn/target_au: [B, A]. `au_from_render_fn` is expected to be
    produced by an OpenFace pass (or a differentiable surrogate) on rendered
    frames upstream.
    """
    if au_from_render_fn.shape != target_au.shape:
        raise ValueError("AU shape mismatch")
    loss = F.huber_loss(au_from_render_fn, target_au, reduction="none")
    if confidence is not None:
        loss = loss * confidence
        return loss.sum() / confidence.sum().clamp_min(1e-6)
    return loss.mean()
