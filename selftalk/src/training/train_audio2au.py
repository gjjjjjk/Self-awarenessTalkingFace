"""Stage 2: train Audio-to-AU directly on GaussianTalker's May data.

Run from ``selftalk/``:

    python -m src.training.train_audio2au \
        --data ../GaussianTalker1/data/May --audio_file aud_ds.npy
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..audio2au import (
    Audio2AUNet,
    LOWER_FACE_AU_NAMES,
    MayAudioAUData,
    WindowedAudioAUDataset,
    audio2au_loss,
    load_may_audio_au,
)
from ..audio2au.dataset import FrameRef
from ..configs import Audio2AUConfig, DataConfig
from .selfcheck import synthetic_au_sequences


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _synthetic_data(num_frames: int) -> MayAudioAUData:
    if num_frames < 30:
        raise ValueError("selfcheck_frames must be >= 30")
    audio, all_au = synthetic_au_sequences(num_frames=num_frames, seed=0)
    lower = list(DataConfig.lower_face_au)
    au = all_au[:, lower]
    confidence = np.ones(num_frames, dtype=np.float32)
    refs = tuple(FrameRef(index, index) for index in range(num_frames))
    train_end = int(0.8 * num_frames)
    val_end = int(0.9 * num_frames)
    train_refs, val_refs, test_refs = refs[:train_end], refs[train_end:val_end], refs[val_end:]
    train_audio = audio[:train_end]
    audio_mean = train_audio.mean(axis=0, dtype=np.float64).astype(np.float32)
    audio_std = np.maximum(
        train_audio.std(axis=0, dtype=np.float64).astype(np.float32), 1e-5
    )
    au_mean = au[:train_end].mean(axis=0, dtype=np.float64).astype(np.float32)
    au_std = np.maximum(
        au[:train_end].std(axis=0, dtype=np.float64).astype(np.float32), 1e-5
    )
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


def _make_dataset(
    data: MayAudioAUData, refs: tuple[FrameRef, ...], half_window: int, include_edges: bool
) -> WindowedAudioAUDataset:
    return WindowedAudioAUDataset(
        audio=data.audio,
        au=data.au,
        confidence=data.confidence,
        refs=refs,
        half_window=half_window,
        audio_mean=data.audio_mean,
        audio_std=data.audio_std,
        include_edges=include_edges,
    )


def _loader(
    dataset: WindowedAudioAUDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    cfg: Audio2AUConfig,
    au_weights: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return audio2au_loss(
        output["au"],
        batch["au"],
        confidence=batch["confidence"],
        lambda_vel=cfg.lambda_vel,
        lambda_acc=cfg.lambda_acc,
        au_weights=au_weights,
        pred_confidence=output["confidence"],
        lambda_confidence=cfg.lambda_confidence,
    )


def train_epoch(
    model: Audio2AUNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    cfg: Audio2AUConfig,
    au_weights: torch.Tensor,
    device: torch.device,
    amp_enabled: bool,
) -> dict[str, float]:
    """Train for one epoch and return example-weighted loss terms."""
    model.train()
    totals = {name: 0.0 for name in ("total", "huber", "vel", "acc", "confidence")}
    examples = 0
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            output = model(batch["audio"])
            terms = _loss(output, batch, cfg, au_weights)
        scaler.scale(terms["total"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()

        size = int(batch["audio"].shape[0])
        examples += size
        for name in totals:
            totals[name] += float(terms[name].detach()) * size
    return {name: value / max(examples, 1) for name, value in totals.items()}


def _weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    denominator = float(weights.sum())
    return float((values * weights).sum() / max(denominator, 1e-8))


def regression_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
) -> dict[str, Any]:
    """Return aggregate and per-AU accuracy/temporal metrics."""
    if pred.shape != target.shape or pred.shape != confidence.shape:
        raise ValueError("prediction, target, and confidence shapes must match")
    per_au: dict[str, dict[str, float]] = {}
    for index, name in enumerate(LOWER_FACE_AU_NAMES):
        p = pred[:, index]
        y = target[:, index]
        weight = confidence[:, index]
        error = p - y
        mae = _weighted_average(np.abs(error), weight)
        rmse = math.sqrt(_weighted_average(error**2, weight))
        valid = weight > 0
        pv, yv = p[valid], y[valid]
        if len(pv) < 2:
            pearson = 0.0
            ccc = 0.0
        else:
            centered_p = pv.astype(np.float64) - float(pv.mean())
            centered_y = yv.astype(np.float64) - float(yv.mean())
            correlation_denom = float(
                np.sqrt(np.dot(centered_p, centered_p) * np.dot(centered_y, centered_y))
            )
            pearson = (
                0.0
                if correlation_denom < 1e-12
                else float(np.dot(centered_p, centered_y) / correlation_denom)
            )
            mean_p, mean_y = float(pv.mean()), float(yv.mean())
            var_p, var_y = float(pv.var()), float(yv.var())
            covariance = float(np.mean((pv - mean_p) * (yv - mean_y)))
            ccc_denom = var_p + var_y + (mean_p - mean_y) ** 2
            ccc = 0.0 if ccc_denom < 1e-8 else 2.0 * covariance / ccc_denom
        per_au[name.removesuffix("_r")] = {
            "mae": mae,
            "rmse": rmse,
            "pearson": pearson,
            "ccc": float(ccc),
        }

    velocity_mae = float(np.abs(np.diff(pred, axis=0) - np.diff(target, axis=0)).mean())
    acceleration_mae = float(
        np.abs(np.diff(pred, n=2, axis=0) - np.diff(target, n=2, axis=0)).mean()
    )
    return {
        "mean_mae": float(np.mean([value["mae"] for value in per_au.values()])),
        "mean_rmse": float(np.mean([value["rmse"] for value in per_au.values()])),
        "mean_pearson": float(np.mean([value["pearson"] for value in per_au.values()])),
        "mean_ccc": float(np.mean([value["ccc"] for value in per_au.values()])),
        "velocity_mae": velocity_mae,
        "acceleration_mae": acceleration_mae,
        "per_au": per_au,
    }


@torch.no_grad()
def evaluate(
    model: Audio2AUNet,
    loader: DataLoader,
    cfg: Audio2AUConfig,
    au_weights: torch.Tensor,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate unique center-frame predictions in chronological order."""
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    predicted_confidences: list[np.ndarray] = []
    frame_ids: list[np.ndarray] = []
    total_loss = 0.0
    examples = 0
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            output = model(batch["audio"])
            terms = _loss(output, batch, cfg, au_weights)
        row = torch.arange(batch["audio"].shape[0], device=device)
        center = batch["center"]
        predictions.append(output["au"][row, center].float().cpu().numpy())
        targets.append(batch["au"][row, center].float().cpu().numpy())
        confidences.append(batch["confidence"][row, center].float().cpu().numpy())
        predicted_confidences.append(
            output["confidence"][row, center].float().cpu().numpy()
        )
        frame_ids.append(batch["frame_id"].cpu().numpy())
        size = int(batch["audio"].shape[0])
        examples += size
        total_loss += float(terms["total"].detach()) * size

    arrays = {
        "prediction": np.concatenate(predictions),
        "target": np.concatenate(targets),
        "confidence": np.concatenate(confidences),
        "predicted_confidence": np.concatenate(predicted_confidences),
        "frame_id": np.concatenate(frame_ids),
    }
    order = np.argsort(arrays["frame_id"])
    arrays = {name: value[order] for name, value in arrays.items()}
    metrics = regression_metrics(
        arrays["prediction"], arrays["target"], arrays["confidence"]
    )
    metrics["loss"] = total_loss / max(examples, 1)
    confidence_error = np.abs(
        arrays["predicted_confidence"] - arrays["confidence"]
    )
    metrics["confidence_mae"] = float(confidence_error.mean())
    return metrics, arrays


def _baseline_metrics(
    arrays: dict[str, np.ndarray], train_au_mean: np.ndarray
) -> dict[str, dict[str, Any]]:
    target = arrays["target"]
    confidence = arrays["confidence"]
    mean_prediction = np.broadcast_to(train_au_mean, target.shape).copy()
    persistence = np.empty_like(target)
    persistence[0] = train_au_mean
    persistence[1:] = target[:-1]
    return {
        "train_mean": regression_metrics(mean_prediction, target, confidence),
        "previous_gt_frame": regression_metrics(persistence, target, confidence),
    }


def _checkpoint_payload(
    model: Audio2AUNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    cfg: Audio2AUConfig,
    data: MayAudioAUData,
    args: argparse.Namespace,
    epoch: int,
    best_val_mae: float,
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_mae": best_val_mae,
        "config": asdict(cfg),
        "lower_face_au": list(DataConfig.lower_face_au),
        "au_names": list(LOWER_FACE_AU_NAMES),
        "audio_mean": torch.from_numpy(data.audio_mean),
        "audio_std": torch.from_numpy(data.audio_std),
        "au_mean": torch.from_numpy(data.au_mean),
        "au_std": torch.from_numpy(data.au_std),
        "window": args.window,
        "audio_file": args.audio_file,
        "data_root": str(args.data),
        "split": {
            "train_frames": len(data.train_refs),
            "val_frames": len(data.val_refs),
            "test_frames": len(data.test_refs),
            "val_fraction": args.val_fraction,
        },
    }


def _write_predictions(path: Path, arrays: dict[str, np.ndarray]) -> None:
    columns: dict[str, np.ndarray] = {"frame_id": arrays["frame_id"]}
    for index, name in enumerate(LOWER_FACE_AU_NAMES):
        short = name.removesuffix("_r")
        columns[f"pred_{short}"] = arrays["prediction"][:, index]
        columns[f"gt_{short}"] = arrays["target"][:, index]
        columns[f"confidence_{short}"] = arrays["confidence"][:, index]
    pd.DataFrame(columns).to_csv(path, index=False)


def _write_curves(path: Path, arrays: dict[str, np.ndarray]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[stage2] matplotlib unavailable; skipping prediction curves")
        return
    figure, axes = plt.subplots(3, 3, figsize=(16, 10), sharex=True)
    for index, (axis, name) in enumerate(zip(axes.flat, LOWER_FACE_AU_NAMES)):
        axis.plot(arrays["frame_id"], arrays["target"][:, index], label="GT", linewidth=1)
        axis.plot(
            arrays["frame_id"], arrays["prediction"][:, index], label="pred", linewidth=1
        )
        axis.set_title(name.removesuffix("_r"))
        axis.set_ylim(0.0, 5.0)
    axes.flat[0].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def train(args: argparse.Namespace) -> Path:
    """Train, validate, checkpoint, and evaluate an Audio-to-AU model."""
    _seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but torch.cuda.is_available() is false")
    amp_enabled = bool(args.amp and device.type == "cuda")

    if args.selfcheck:
        data = _synthetic_data(args.selfcheck_frames)
    else:
        data = load_may_audio_au(
            args.data,
            audio_file=args.audio_file,
            train_split=args.train_split,
            test_split=args.test_split,
            val_fraction=args.val_fraction,
        )

    train_dataset = _make_dataset(data, data.train_refs, args.window, include_edges=False)
    val_dataset = _make_dataset(data, data.val_refs, args.window, include_edges=True)
    test_dataset = _make_dataset(data, data.test_refs, args.window, include_edges=True)
    train_loader = _loader(
        train_dataset, args.batch_size, True, args.num_workers, device, args.seed
    )
    val_loader = _loader(
        val_dataset, args.eval_batch_size, False, args.num_workers, device, args.seed
    )
    test_loader = _loader(
        test_dataset, args.eval_batch_size, False, args.num_workers, device, args.seed
    )
    all_refs = tuple(
        sorted((*data.train_refs, *data.val_refs, *data.test_refs), key=lambda ref: ref.image_id)
    )
    all_dataset = _make_dataset(data, all_refs, args.window, include_edges=True)
    all_loader = _loader(
        all_dataset, args.eval_batch_size, False, args.num_workers, device, args.seed
    )

    cfg = Audio2AUConfig(
        audio_feature_dim=data.audio_feature_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        lambda_vel=args.lambda_vel,
        lambda_acc=args.lambda_acc,
        lambda_confidence=args.lambda_confidence,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
    )
    model = Audio2AUNet(cfg, num_out_au=len(LOWER_FACE_AU_NAMES)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(1, args.patience // 3)
    )
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    else:  # pragma: no cover - compatibility with early PyTorch 2.x
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    inverse_std = 1.0 / torch.from_numpy(data.au_std).to(device)
    au_weights = (inverse_std / inverse_std.mean()).clamp(0.25, 4.0)

    start_epoch = 1
    best_val_mae = float("inf")
    if args.resume:
        resume = torch.load(args.resume, map_location=device)
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        start_epoch = int(resume["epoch"]) + 1
        best_val_mae = float(resume["best_val_mae"])
        print(f"[stage2] resumed {args.resume} at epoch {start_epoch}")
    else:
        model.initialize_au_prior(torch.from_numpy(data.au_mean).to(device))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"
    history_path = out_dir / "history.json"
    if args.resume and history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
    else:
        history: list[dict[str, Any]] = []
    stale_epochs = 0

    print(
        f"[stage2] audio={tuple(data.audio.shape)} flat_dim={data.audio_feature_dim} "
        f"train/val/test={len(data.train_refs)}/{len(data.val_refs)}/{len(data.test_refs)} "
        f"window={2 * args.window + 1}"
    )
    for epoch in range(start_epoch, args.epochs + 1):
        train_terms = train_epoch(
            model, train_loader, optimizer, scaler, cfg, au_weights, device, amp_enabled
        )
        val_metrics, _ = evaluate(
            model, val_loader, cfg, au_weights, device, amp_enabled
        )
        val_mae = float(val_metrics["mean_mae"])
        scheduler.step(val_mae)
        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train": train_terms,
            "val": val_metrics,
        }
        history.append(row)

        improved = val_mae < best_val_mae - args.min_delta
        if improved:
            best_val_mae = val_mae
            stale_epochs = 0
        else:
            stale_epochs += 1
        payload = _checkpoint_payload(
            model, optimizer, scheduler, cfg, data, args, epoch, best_val_mae
        )
        torch.save(payload, last_path)
        if improved:
            torch.save(payload, best_path)
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(
            f"[stage2] epoch {epoch}/{args.epochs} "
            f"train={train_terms['total']:.4f} val_mae={val_mae:.4f} "
            f"val_ccc={val_metrics['mean_ccc']:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        if stale_epochs >= args.patience:
            print(f"[stage2] early stopping after {stale_epochs} stale epochs")
            break

    evaluation_checkpoint = best_path
    if not evaluation_checkpoint.exists() and args.resume:
        evaluation_checkpoint = Path(args.resume)
    if not evaluation_checkpoint.exists():
        raise RuntimeError("training did not produce a best checkpoint")
    best = torch.load(evaluation_checkpoint, map_location=device)
    model.load_state_dict(best["model"])
    test_metrics, test_arrays = evaluate(
        model, test_loader, cfg, au_weights, device, amp_enabled
    )
    report = {
        "checkpoint_epoch": int(best["epoch"]),
        "model": test_metrics,
        "baselines": _baseline_metrics(test_arrays, data.au_mean),
    }
    (out_dir / "test_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    _write_predictions(out_dir / "test_predictions.csv", test_arrays)
    np.savez_compressed(
        out_dir / "test_predictions.npz",
        **test_arrays,
        au_names=np.asarray(LOWER_FACE_AU_NAMES),
    )
    _, all_arrays = evaluate(model, all_loader, cfg, au_weights, device, amp_enabled)
    _write_predictions(out_dir / "all_predictions.csv", all_arrays)
    np.savez_compressed(
        out_dir / "all_predictions.npz",
        **all_arrays,
        au_names=np.asarray(LOWER_FACE_AU_NAMES),
    )
    if args.plots:
        _write_curves(out_dir / "test_curves.png", test_arrays)
    print(
        f"[stage2] test_mae={test_metrics['mean_mae']:.4f} "
        f"test_ccc={test_metrics['mean_ccc']:.4f} -> {out_dir}"
    )
    return evaluation_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: May Audio-to-AU training")
    parser.add_argument(
        "--data", "--binary", dest="data", default="../GaussianTalker1/data/May",
        help="GaussianTalker subject directory",
    )
    parser.add_argument("--audio_file", default="aud_ds.npy")
    parser.add_argument("--train_split", default="transforms_train.json")
    parser.add_argument("--test_split", default="transforms_val.json")
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--window", type=int, default=4, help="half-window in video frames")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_vel", type=float, default=0.2)
    parser.add_argument("--lambda_acc", type=float, default=0.1)
    parser.add_argument("--lambda_confidence", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--out", default="runs/audio2au/May")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--selfcheck", action="store_true")
    parser.add_argument("--selfcheck_frames", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=6666)
    args = parser.parse_args()
    if args.window < 1 or args.batch_size < 1 or args.eval_batch_size < 1:
        parser.error("window and batch sizes must be positive")
    if args.epochs < 1 or args.patience < 1:
        parser.error("epochs and patience must be positive")
    if args.d_model % args.n_heads != 0:
        parser.error("d_model must be divisible by n_heads")
    train(args)


if __name__ == "__main__":
    main()
