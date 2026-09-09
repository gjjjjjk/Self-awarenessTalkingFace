"""Synthetic-data self-check helpers for the stage 2-7 training entry points.

`train_canonical --selfcheck` already synthesizes its own scene; these helpers
extend the same idea to stages 2-7 so the whole pipeline can be validated
without OpenFace, FLAME fits or per-frame cameras (the real-data requirements
of stages 0-1).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SELFCHK_AUDIO_DIM = 64
"""Pseudo audio-feature dimension shared by all self-checks (must match
between the stage-2 checkpoint and the stage-3+ condition encoder)."""

SELFCHK_NUM_AU = 17
SELFCHK_NUM_FRAMES = 24
SELFCHK_IMAGE_SIZE = 256
"""Matches the default `Camera.look_at` training resolution (256x256)."""


def synthetic_au_sequences(num_frames: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Random audio features + AU intensities for the stage-2 self-check.

    Args:
        num_frames: T.
        seed: RNG seed.

    Returns:
        audio: [T, SELFCHK_AUDIO_DIM] float32 pseudo-features.
        au: [T, 17] float32 intensities in [0, 5].
    """
    rng = np.random.default_rng(seed)
    audio = rng.normal(size=(num_frames, SELFCHK_AUDIO_DIM)).astype(np.float32)
    au = (rng.random((num_frames, SELFCHK_NUM_AU)) * 5.0).astype(np.float32)
    return audio, au


def make_synthetic_binary(out_dir: str | Path, num_frames: int = SELFCHK_NUM_FRAMES,
                          seed: int = 0) -> Path:
    """Write a minimal per-subject binary directory for stages 3-7.

    Produces `frames/`, `audio_feat.npy` and `au.npy` compatible with
    `TalkingFaceDataset`; images match the default 256x256 training camera so
    the photometric loss shapes line up.

    Args:
        out_dir: target directory (created if missing).
        num_frames: number of frames to synthesize.
        seed: RNG seed.

    Returns:
        The binary directory path.
    """
    import cv2

    root = Path(out_dir)
    frames = root / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for t in range(num_frames):
        image = (rng.random((SELFCHK_IMAGE_SIZE, SELFCHK_IMAGE_SIZE, 3)) * 255).astype(np.uint8)
        cv2.imwrite(str(frames / f"frame_{t:06d}.jpg"), image)
    audio, au = synthetic_au_sequences(num_frames, seed)
    np.save(root / "audio_feat.npy", audio)
    np.save(root / "au.npy", au)
    return root
