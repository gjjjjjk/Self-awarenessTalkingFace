"""CPU tests for the Stage 2 May Audio-to-AU training data path."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from src.audio2au import (
    Audio2AUNet,
    LOWER_FACE_AU_NAMES,
    WindowedAudioAUDataset,
    audio2au_loss,
    load_may_audio_au,
)
from src.configs import Audio2AUConfig
from src.training.train_audio2au import regression_metrics


def _write_split(path, image_ids):
    frames = [
        {"img_id": int(index), "aud_id": int(index), "transform_matrix": np.eye(4).tolist()}
        for index in image_ids
    ]
    path.write_text(json.dumps({"frames": frames}))


def test_may_data_adapter_and_window(tmp_path):
    rng = np.random.default_rng(0)
    num_frames = 40
    np.save(tmp_path / "aud_ds.npy", rng.normal(size=(num_frames, 16, 29)))
    table = {" confidence": np.full(num_frames, 0.98), " success": np.ones(num_frames)}
    for name in LOWER_FACE_AU_NAMES:
        table[f" {name}"] = rng.uniform(0.0, 2.0, num_frames)
    pd.DataFrame(table).to_csv(tmp_path / "au.csv", index=False)
    _write_split(tmp_path / "transforms_train.json", range(30))
    _write_split(tmp_path / "transforms_val.json", range(30, 40))

    data = load_may_audio_au(tmp_path, val_fraction=0.2)
    assert data.audio.shape == (40, 16, 29)
    assert data.audio_feature_dim == 16 * 29
    assert len(data.train_refs) == 24
    assert len(data.val_refs) == 6
    assert len(data.test_refs) == 10
    assert data.audio_mean.shape == (29,)
    assert data.au.shape == (40, 9)

    dataset = WindowedAudioAUDataset(
        data.audio, data.au, data.confidence, data.test_refs, half_window=2,
        audio_mean=data.audio_mean, audio_std=data.audio_std, include_edges=True,
    )
    sample = dataset[0]
    assert len(dataset) == 10
    assert sample["audio"].shape == (5, 16, 29)
    assert sample["au"].shape == (5, 9)
    assert sample["confidence"].shape == (5, 9)
    assert sample["frame_id"].item() == 30
    assert torch.allclose(sample["audio"][0], sample["audio"][1])


def test_may_feature_network_and_weighted_loss():
    config = Audio2AUConfig(
        audio_feature_dim=16 * 29, d_model=32, n_heads=4, n_layers=1
    )
    model = Audio2AUNet(config, num_out_au=9)
    model.initialize_au_prior(torch.full((9,), 0.5))
    output = model(torch.randn(2, 5, 16, 29))
    target = torch.rand(2, 5, 9)
    confidence = torch.full_like(target, 0.98)
    terms = audio2au_loss(
        output["au"], target, confidence=confidence,
        au_weights=torch.linspace(0.5, 1.5, 9),
        pred_confidence=output["confidence"],
    )
    assert output["au"].shape == (2, 5, 9)
    assert output["confidence"].shape == (2, 5, 9)
    assert (output["au"] > 0).all() and (output["au"] < 5).all()
    assert set(terms) == {"total", "huber", "vel", "acc", "confidence"}
    assert all(torch.isfinite(value) for value in terms.values())


def test_constant_baseline_metrics_are_finite():
    target = np.random.default_rng(1).random((20, 9), dtype=np.float32)
    pred = np.broadcast_to(target.mean(axis=0), target.shape).copy()
    confidence = np.ones_like(target)
    metrics = regression_metrics(pred, target, confidence)
    assert np.isfinite(metrics["mean_pearson"])
    assert all(np.isfinite(values["pearson"]) for values in metrics["per_au"].values())
