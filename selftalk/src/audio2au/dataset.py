"""Stage 2: May-specific, temporally aligned Audio-to-AU datasets."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


LOWER_FACE_AU_NAMES = (
    "AU10_r",
    "AU12_r",
    "AU14_r",
    "AU15_r",
    "AU17_r",
    "AU20_r",
    "AU23_r",
    "AU25_r",
    "AU26_r",
)


@dataclass(frozen=True)
class FrameRef:
    """Aligned image/AU and audio indices for one video frame."""

    image_id: int
    audio_id: int


@dataclass(frozen=True)
class MayAudioAUData:
    """Loaded May signals and chronological train/validation/test splits."""

    audio: np.ndarray
    au: np.ndarray
    confidence: np.ndarray
    train_refs: tuple[FrameRef, ...]
    val_refs: tuple[FrameRef, ...]
    test_refs: tuple[FrameRef, ...]
    audio_mean: np.ndarray
    audio_std: np.ndarray
    au_mean: np.ndarray
    au_std: np.ndarray

    @property
    def audio_feature_dim(self) -> int:
        """Flattened dimension of one frame-level acoustic feature."""
        return int(np.prod(self.audio.shape[1:]))


def _read_refs(path: Path) -> tuple[FrameRef, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{path} must contain a non-empty frames list")
    refs = tuple(FrameRef(int(frame["img_id"]), int(frame["aud_id"])) for frame in frames)
    image_ids = [ref.image_id for ref in refs]
    if image_ids != sorted(image_ids) or len(set(image_ids)) != len(image_ids):
        raise ValueError(f"{path} frames must be unique and chronological")
    return refs


def load_may_audio_au(
    root: str | Path,
    audio_file: str = "aud_ds.npy",
    train_split: str = "transforms_train.json",
    test_split: str = "transforms_val.json",
    val_fraction: float = 0.1,
) -> MayAudioAUData:
    """Load GaussianTalker's May directory without copying image assets.

    The official training frames are split chronologically into train and
    validation subsets. The official validation frames are retained as the
    final test set so hyperparameter selection never sees them.

    Args:
        root: GaussianTalker subject directory.
        audio_file: Frame-aligned DeepSpeech feature array, normally
            ``aud_ds.npy`` with shape [T, 16, 29].
        train_split: JSON containing the official training frame references.
        test_split: JSON retained for final testing.
        val_fraction: Tail fraction of the official training split used for
            validation.
    """
    root = Path(root)
    if not 0.0 < val_fraction < 0.5:
        raise ValueError("val_fraction must be in (0, 0.5)")

    audio_path = root / audio_file
    au_path = root / "au.csv"
    for path in (audio_path, au_path, root / train_split, root / test_split):
        if not path.exists():
            raise ValueError(f"required May data file missing: {path}")

    audio = np.load(audio_path).astype(np.float32)
    if audio.ndim not in (2, 3) or audio.shape[0] == 0:
        raise ValueError(f"audio must be [T, D] or [T, S, D], got {audio.shape}")
    if not np.isfinite(audio).all():
        raise ValueError(f"audio contains NaN/Inf: {audio_path}")

    table = pd.read_csv(au_path)
    table.columns = table.columns.str.strip()
    required = [*LOWER_FACE_AU_NAMES, "confidence", "success"]
    missing = [name for name in required if name not in table.columns]
    if missing:
        raise ValueError(f"{au_path} missing columns: {missing}")

    au = table[list(LOWER_FACE_AU_NAMES)].to_numpy(dtype=np.float32)
    au = np.clip(au, 0.0, 5.0)
    confidence = table["confidence"].to_numpy(dtype=np.float32)
    success = table["success"].to_numpy(dtype=np.float32)
    confidence = np.clip(confidence, 0.0, 1.0) * (success > 0).astype(np.float32)
    if len(audio) != len(au):
        raise ValueError(f"audio/AU length mismatch: {len(audio)} vs {len(au)}")

    official_train = _read_refs(root / train_split)
    test_refs = _read_refs(root / test_split)
    all_refs = (*official_train, *test_refs)
    if max(ref.image_id for ref in all_refs) >= len(au):
        raise ValueError("split image_id exceeds AU table length")
    if max(ref.audio_id for ref in all_refs) >= len(audio):
        raise ValueError("split aud_id exceeds audio array length")
    overlap = {ref.image_id for ref in official_train} & {ref.image_id for ref in test_refs}
    if overlap:
        raise ValueError(f"official train/test splits overlap at frame {min(overlap)}")

    val_count = max(1, int(round(len(official_train) * val_fraction)))
    train_refs = official_train[:-val_count]
    val_refs = official_train[-val_count:]
    if not train_refs:
        raise ValueError("validation split leaves no training frames")

    train_audio_ids = np.asarray([ref.audio_id for ref in train_refs], dtype=np.int64)
    train_image_ids = np.asarray([ref.image_id for ref in train_refs], dtype=np.int64)
    train_audio = audio[train_audio_ids]
    reduce_axes = tuple(range(train_audio.ndim - 1))
    audio_mean = train_audio.mean(axis=reduce_axes, dtype=np.float64).astype(np.float32)
    audio_std = train_audio.std(axis=reduce_axes, dtype=np.float64).astype(np.float32)
    audio_std = np.maximum(audio_std, 1e-5)
    train_au = au[train_image_ids]
    au_mean = train_au.mean(axis=0, dtype=np.float64).astype(np.float32)
    au_std = np.maximum(train_au.std(axis=0, dtype=np.float64).astype(np.float32), 1e-5)

    return MayAudioAUData(
        audio=audio,
        au=au,
        confidence=confidence,
        train_refs=train_refs,
        val_refs=val_refs,
        test_refs=test_refs,
        audio_mean=audio_mean,
        audio_std=audio_std,
        au_mean=au_mean,
        au_std=au_std,
    )


class WindowedAudioAUDataset(Dataset[dict[str, torch.Tensor]]):
    """Centered temporal windows from one split, with no cross-split context.

    Each sample contains audio [L, ...], AU targets [L, A], confidence
    [L, A], the center frame id, and the center position. With
    ``include_edges=True``, boundary context is replicated so every split
    frame receives exactly one center prediction during evaluation.
    """

    def __init__(
        self,
        audio: np.ndarray,
        au: np.ndarray,
        confidence: np.ndarray,
        refs: Sequence[FrameRef],
        half_window: int,
        audio_mean: np.ndarray,
        audio_std: np.ndarray,
        include_edges: bool = False,
    ) -> None:
        if half_window < 1:
            raise ValueError("half_window must be >= 1")
        if len(refs) < 2 * half_window + 1:
            raise ValueError("split is shorter than one temporal window")
        self.audio = audio
        self.au = au
        self.confidence = confidence
        self.refs = tuple(refs)
        self.half_window = int(half_window)
        self.audio_mean = audio_mean
        self.audio_std = audio_std
        if include_edges:
            self.centers = tuple(range(len(self.refs)))
        else:
            self.centers = tuple(range(half_window, len(self.refs) - half_window))

    def __len__(self) -> int:
        return len(self.centers)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        center = self.centers[index]
        positions = np.arange(
            center - self.half_window,
            center + self.half_window + 1,
            dtype=np.int64,
        )
        positions = np.clip(positions, 0, len(self.refs) - 1)
        selected = [self.refs[int(position)] for position in positions]
        audio_ids = np.asarray([ref.audio_id for ref in selected], dtype=np.int64)
        image_ids = np.asarray([ref.image_id for ref in selected], dtype=np.int64)

        audio = (self.audio[audio_ids] - self.audio_mean) / self.audio_std
        target = self.au[image_ids]
        frame_confidence = self.confidence[image_ids, None]
        confidence = np.broadcast_to(frame_confidence, target.shape).copy()
        return {
            "audio": torch.from_numpy(audio.astype(np.float32, copy=False)),
            "au": torch.from_numpy(target.astype(np.float32, copy=False)),
            "confidence": torch.from_numpy(confidence.astype(np.float32, copy=False)),
            "frame_id": torch.tensor(self.refs[center].image_id, dtype=torch.long),
            "center": torch.tensor(self.half_window, dtype=torch.long),
        }
