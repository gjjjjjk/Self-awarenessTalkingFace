"""Stage 8: evaluation metrics.

Image quality: PSNR / SSIM / LPIPS. Expression: AUE (lower/upper face,
computed from OpenFace CSVs of prediction vs GT). Identity: CSIM (needs an
ArcFace embedding; returns NaN when unavailable). Temporal: warping error.
Lip sync: LMD between mouth landmarks (68-point convention: 48..67).

Usage:
    python -m selftalk.evaluation.metrics --pred pred.mp4 --gt gt.mp4
    python -m selftalk.evaluation.metrics --pred_au pred.csv --gt_au gt.csv
"""
from __future__ import annotations

import math

import numpy as np
import torch

from ..losses.regularizers import temporal_warping_error

LOWER_FACE_AU = (7, 8, 9, 10, 11, 12, 13, 14, 15)
UPPER_FACE_AU = (0, 1, 2, 3, 4, 5, 6, 16)
MOUTH_LANDMARKS = tuple(range(48, 68))


def psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Peak signal-to-noise ratio (dB); images [3, H, W] or [B, 3, H, W] in [0, 1]."""
    mse = torch.mean((pred.clamp(0, 1) - gt.clamp(0, 1)) ** 2).item()
    if mse <= 0:
        return float("inf")
    return -10.0 * math.log10(mse)


def ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """SSIM via pytorch-msssim when installed; falls back to 1 - normalized MAE."""
    if pred.ndim != 4:
        pred, gt = pred.unsqueeze(0), gt.unsqueeze(0)
    try:
        from pytorch_msssim import ssim as _ssim

        return float(_ssim(pred.clamp(0, 1), gt.clamp(0, 1), data_range=1.0))
    except ImportError:
        return float(1.0 - (pred - gt).abs().mean().item())


def lpips_metric(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """LPIPS (AlexNet); returns NaN when the lpips package is unavailable."""
    if pred.ndim != 4:
        pred, gt = pred.unsqueeze(0), gt.unsqueeze(0)
    try:
        import lpips as lpips_pkg
    except ImportError:
        return float("nan")
    net = lpips_pkg.LPIPS(net="alex", verbose=False).to(pred.device)
    with torch.no_grad():
        value = net((pred * 2 - 1).clamp(-1, 1), (gt * 2 - 1).clamp(-1, 1))
    return float(value.mean())


def lmd(pred_landmarks: torch.Tensor, gt_landmarks: torch.Tensor,
        subset: tuple[int, ...] = MOUTH_LANDMARKS, normalize: bool = True) -> float:
    """Lip landmark distance. landmarks [T, 68, 2]; normalized by eye distance."""
    if pred_landmarks.shape != gt_landmarks.shape:
        raise ValueError("landmark shape mismatch")
    if pred_landmarks.shape[1] < max(subset) + 1:
        raise ValueError(f"need at least {max(subset) + 1} landmarks")
    p = pred_landmarks[:, list(subset)]
    g = gt_landmarks[:, list(subset)]
    dist = torch.norm(p - g, dim=-1).mean(dim=-1)
    if normalize and pred_landmarks.shape[1] >= 48:
        eye_left = gt_landmarks[:, 36:42].mean(dim=1)
        eye_right = gt_landmarks[:, 42:48].mean(dim=1)
        eye_dist = torch.norm(eye_left - eye_right, dim=-1).clamp_min(1e-6)
        dist = dist / eye_dist
    return float(dist.mean())


def aue(pred_au: np.ndarray, gt_au: np.ndarray) -> dict[str, float]:
    """AU error (mean squared), split into lower/upper face; always per-AU too.

    pred_au/gt_au: [T, 17] OpenFace intensities.
    """
    if pred_au.shape != gt_au.shape or pred_au.shape[-1] != 17:
        raise ValueError(f"expected [T, 17] AU arrays, got {pred_au.shape} vs {gt_au.shape}")
    err = (pred_au - gt_au) ** 2
    per_au = {f"AU{i:02d}": float(err[:, i].mean()) for i in range(17)}
    lower = err[:, list(LOWER_FACE_AU)].mean()
    upper = err[:, list(UPPER_FACE_AU)].mean()
    return {"mean": float(err.mean()), "lower": float(lower),
            "upper": float(upper), "per_au": per_au}


def csim(pred_emb: torch.Tensor, gt_emb: torch.Tensor) -> float:
    """Identity cosine similarity from ArcFace embeddings [D]; NaN if unavailable."""
    if pred_emb.shape != gt_emb.shape:
        raise ValueError("embedding shape mismatch")
    num = torch.dot(pred_emb.flatten(), gt_emb.flatten())
    den = pred_emb.norm() * gt_emb.norm()
    if den.item() == 0:
        return float("nan")
    return float(num / den)


def evaluate_frame_set(pred: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    """Image metrics over a frame stack [T, 3, H, W]."""
    if pred.shape != gt.shape:
        raise ValueError(f"frame stacks must match: {tuple(pred.shape)} vs {tuple(gt.shape)}")
    return {
        "psnr": float(np.mean([psnr(p, g) for p, g in zip(pred, gt)])),
        "ssim": ssim(pred, gt),
        "lpips": lpips_metric(pred, gt),
        "warping": float(temporal_warping_error(pred)),
    }


def evaluate_au_csvs(pred_csv: str, gt_csv: str) -> dict[str, float]:
    """AUE from two OpenFace CSVs (see TalkingGaussian auerror.py convention)."""
    import pandas as pd

    columns = [f"AU{i:02d}_r" for i in (1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45)]
    pred = pd.read_csv(pred_csv)[columns].to_numpy(np.float32)
    gt = pd.read_csv(gt_csv)[columns].to_numpy(np.float32)
    t = min(len(pred), len(gt))
    return {k: v for k, v in aue(pred[:t], gt[:t]).items() if k != "per_au"}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Stage 8 metrics")
    parser.add_argument("--pred", default=None, help="prediction video")
    parser.add_argument("--gt", default=None, help="ground-truth video")
    parser.add_argument("--pred_au", default=None, help="OpenFace CSV of prediction")
    parser.add_argument("--gt_au", default=None, help="OpenFace CSV of GT")
    parser.add_argument("--max_frames", type=int, default=250)
    args = parser.parse_args()

    if args.pred and args.gt:
        import cv2

        def _load(path: str) -> torch.Tensor:
            cap = cv2.VideoCapture(path)
            frames = []
            while len(frames) < args.max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(torch.from_numpy(frame[:, :, ::-1].copy())
                              .permute(2, 0, 1).float() / 255.0)
            cap.release()
            if not frames:
                raise ValueError(f"could not read frames from {path}")
            return torch.stack(frames)

        results = evaluate_frame_set(_load(args.pred), _load(args.gt))
        for key, value in results.items():
            print(f"{key}: {value:.4f}")

    if args.pred_au and args.gt_au:
        for key, value in evaluate_au_csvs(args.pred_au, args.gt_au).items():
            print(f"aue_{key}: {value:.4f}")


if __name__ == "__main__":
    main()
