"""Stage 2 training: Audio-to-AU network on lower-face AUs.

Usage:
    python -m selftalk.training.train_audio2au --binary ../data/binary/subject
    python -m selftalk.training.train_audio2au --selfcheck --epochs 2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..audio2au import Audio2AUNet, audio2au_loss
from ..configs import Audio2AUConfig, DataConfig
from .selfcheck import synthetic_au_sequences


def build_sequences(audio: np.ndarray, au: np.ndarray, window: int, stride: int = 1):
    """Slide a [2w+1] window; targets are the lower-face AUs of each frame.

    Returns (x [M, 2w+1, D], y [M, 2w+1, A_lower]).
    """
    xs, ys = [], []
    for t in range(window, len(audio) - window, stride):
        xs.append(audio[t - window : t + window + 1])
        ys.append(au[t - window : t + window + 1])
    x = torch.from_numpy(np.stack(xs).astype(np.float32))
    y = torch.from_numpy(np.stack(ys).astype(np.float32))
    return x, y


def train(args: argparse.Namespace) -> Path:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    data_cfg = DataConfig(audio_window=args.window)
    if args.selfcheck:
        audio, au_all = synthetic_au_sequences(
            num_frames=args.selfcheck_frames, seed=args.seed
        )
    else:
        audio = np.load(Path(args.binary) / "audio_feat.npy")
        au_all = np.load(Path(args.binary) / "au.npy")
    if len(audio) != len(au_all):
        raise ValueError(f"audio {len(audio)} and AU {len(au_all)} frames misaligned")
    lower = list(data_cfg.lower_face_au)
    au = au_all[:, lower]

    x, y = build_sequences(audio, au, args.window)
    if len(x) == 0:
        raise ValueError("not enough frames for the requested window")
    loader = DataLoader(
        TensorDataset(x, y), batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    cfg = Audio2AUConfig(audio_feature_dim=audio.shape[1])
    model = Audio2AUNet(cfg, num_out_au=len(lower)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = audio2au_loss(
                pred["au"], yb,
                lambda_vel=cfg.lambda_vel, lambda_acc=cfg.lambda_acc,
            )["total"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss)
        print(f"[stage2] epoch {epoch + 1}/{args.epochs} loss {total / max(len(loader), 1):.4f}")

    ckpt = out_dir / "audio2au.pt"
    torch.save({"model": model.state_dict(), "config": cfg.__dict__, "lower_face_au": lower}, ckpt)
    print(f"[stage2] saved {ckpt}")
    return ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: Audio-to-AU training")
    parser.add_argument("--binary", default=None, help="preprocessed subject dir")
    parser.add_argument("--selfcheck", action="store_true",
                        help="train on synthetic sequences (no real data needed)")
    parser.add_argument("--selfcheck_frames", type=int, default=256)
    parser.add_argument("--out", default="runs/audio2au")
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=6666)
    args = parser.parse_args()
    if args.binary is None and not args.selfcheck:
        parser.error("either --binary or --selfcheck is required")
    train(args)


if __name__ == "__main__":
    main()
