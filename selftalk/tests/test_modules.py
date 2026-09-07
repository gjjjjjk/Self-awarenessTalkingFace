"""CPU-only smoke tests for all selftalk modules (stages 0-8)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from selftalk.audio2au import Audio2AUNet, audio2au_loss
from selftalk.audio2au.loss import _first_difference, _second_difference
from selftalk.canonical import CanonicalGaussianModel, MultiResolutionTriplane
from selftalk.configs import Audio2AUConfig, DeformationConfig
from selftalk.deformation import (
    ConditionEncoder,
    DirectConditionDeformation,
    MotionTokenDeformation,
    constrained_gaussian_update,
    default_region_weights,
    quaternion_multiply,
)
from selftalk.deformation.region_weights import NUM_REGIONS
from selftalk.evaluation import aue, psnr
from selftalk.losses import photometric_loss
from selftalk.losses.regularizers import (
    acceleration_regularizer,
    displacement_regularizer,
    knn_neighbors,
    neighborhood_consistency,
    velocity_regularizer,
)
from selftalk.rendering import Camera, render


@pytest.fixture()
def canonical() -> torch.nn.Module:
    n = 128
    triplane = MultiResolutionTriplane(resolutions=(16, 32), feat_dim=8, out_dim=32)
    model = CanonicalGaussianModel(triplane, mlp_hidden=32, mlp_depth=2, sh_degree=1)
    model.init_from_vertices(0.3 * torch.rand(n, 3) - 0.15)
    model.eval()
    return model


@pytest.fixture()
def condition() -> dict:
    torch.manual_seed(0)
    encoder = ConditionEncoder(d_model=64, audio_feature_dim=64, num_au=17)
    return {
        "encoder": encoder,
        "audio": torch.randn(2, 5, 64),
        "au": torch.rand(2, 17) * 5,
        "blink": torch.rand(2, 1),
        "pose": torch.randn(2, 6),
    }


def _encode(condition: dict):
    return condition["encoder"](
        condition["audio"], condition["au"], condition["blink"], condition["pose"]
    )


def test_triplane_shapes():
    triplane = MultiResolutionTriplane(resolutions=(16, 32), feat_dim=8, out_dim=24)
    mu = torch.rand(2, 50, 3) * 2 - 1
    f = triplane(mu)
    assert f.shape == (2, 50, 24)
    assert torch.isfinite(f).all()


def test_canonical_model_invariants(canonical):
    out = canonical(num_gaussians=128, batch_size=3)
    assert out.means.shape == (3, 128, 3)
    assert out.rotations.shape == (3, 128, 4)
    assert torch.allclose(out.rotations.norm(dim=-1), torch.ones(3, 128), atol=1e-4)
    assert (out.scales > 0).all()
    assert (out.opacities > 0).all() and (out.opacities < 1).all()
    assert out.sh.shape == (3, 128, 4, 3)
    assert out.features.shape == (3, 128, 32)


def test_audio2au_forward_and_loss():
    torch.manual_seed(0)
    cfg = Audio2AUConfig(audio_feature_dim=64, d_model=64, n_layers=2)
    net = Audio2AUNet(cfg, num_out_au=9)
    x = torch.randn(2, 8, 64)
    out = net(x)
    assert out["au"].shape == (2, 8, 9)
    assert out["confidence"].shape == (2, 8, 9)
    assert (out["au"] >= 0).all() and (out["au"] <= 5).all()
    gt = torch.rand(2, 8, 9) * 5
    losses = audio2au_loss(out["au"], gt, lambda_vel=0.2, lambda_acc=0.1)
    for value in losses.values():
        assert torch.isfinite(value)
    assert _first_difference(torch.rand(2, 8, 9)).shape == (2, 7, 9)
    assert _second_difference(torch.rand(2, 8, 9)).shape == (2, 6, 9)


def test_condition_encoder_shapes(condition):
    tokens = _encode(condition)
    assert tokens.audio.shape == (2, 5, 64)
    assert tokens.au.shape == (2, 17, 64)
    assert tokens.blink.shape == (2, 1, 64)
    assert tokens.view.shape == (2, 1, 64)
    assert tokens.style.shape == (2, 1, 64)
    seq = tokens.as_sequence()
    assert seq.shape == (2, 5 + 17 + 3, 64)
    zeroed = tokens.without(("au", "blink", "view", "style"))
    assert torch.all(zeroed.au == 0)


def test_direct_deformation_zero_init_is_identity(canonical, condition):
    torch.manual_seed(0)
    cfg = DeformationConfig(d_model=64, cross_attn_layers=1)
    net = DirectConditionDeformation(cfg, triplane_dim=32, num_sh=4)
    can = canonical(num_gaussians=128, batch_size=2)
    deltas = net(can, _encode(condition), region_ids=torch.zeros(128, dtype=torch.long))
    assert deltas.means.shape == (2, 128, 3)
    assert deltas.sh.shape == (2, 128, 4, 3)
    assert torch.allclose(deltas.means, torch.zeros_like(deltas.means), atol=1e-6)
    assert torch.allclose(deltas.logit_opacities, torch.zeros_like(deltas.logit_opacities), atol=1e-6)


def test_motion_tokenizer(canonical, condition):
    torch.manual_seed(0)
    cfg = DeformationConfig(d_model=64, num_motion_tokens=16, decoder_layers=1,
                            cross_attn_layers=1)
    net = MotionTokenDeformation(cfg, triplane_dim=32, num_sh=4)
    motion = net.tokenizer(_encode(condition))
    assert motion.shape == (2, 16, 64)
    can = canonical(num_gaussians=128, batch_size=2)
    deltas = net(can, _encode(condition), region_ids=torch.zeros(128, dtype=torch.long))
    assert deltas.means.shape == (2, 128, 3)


def test_constrained_update_invariants(canonical):
    can = canonical(num_gaussians=128, batch_size=2)
    from selftalk.deformation.types import DeformationDeltas

    deltas = DeformationDeltas(
        means=torch.randn(2, 128, 3) * 0.01,
        rotations=torch.randn(2, 128, 4),
        log_scales=torch.randn(2, 128, 3) * 0.1,
        sh=torch.randn(2, 128, 4, 3) * 0.1,
        logit_opacities=torch.randn(2, 128, 1) * 0.5,
    )
    out = constrained_gaussian_update(can, deltas)
    quat_norm = out.rotations.norm(dim=-1)
    assert torch.allclose(quat_norm, torch.ones_like(quat_norm), atol=1e-5)
    assert (out.scales > 0).all()
    assert (out.opacities > 0).all() and (out.opacities < 1).all()
    assert out.means.shape == can.means.shape

    zero = DeformationDeltas(
        means=torch.zeros(2, 128, 3),
        rotations=torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(2, 128, 4),
        log_scales=torch.zeros(2, 128, 3),
        sh=torch.zeros(2, 128, 4, 3),
        logit_opacities=torch.zeros(2, 128, 1),
    )
    ident = constrained_gaussian_update(can, zero)
    assert torch.allclose(ident.means, can.means)
    assert torch.allclose(ident.scales, can.scales, atol=1e-5)
    assert torch.allclose(ident.opacities, can.opacities, atol=1e-4)
    assert torch.allclose(ident.rotations, can.rotations, atol=1e-4)


def test_quaternion_multiply_identity():
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(4, 4)
    q = torch.nn.functional.normalize(torch.randn(4, 4), dim=-1)
    prod = quaternion_multiply(identity, q)
    assert torch.allclose(prod, q, atol=1e-6)


def test_region_weights_ranges():
    rw = default_region_weights()
    for field in ("displacement", "neighborhood", "rigidity", "velocity", "acceleration"):
        table = getattr(rw, field)
        assert table.shape == (NUM_REGIONS,)
        assert (table > 0).all()
    assert rw.rigidity[3] > rw.rigidity[0]  # hair stronger than mouth


def test_regularizers_finite(canonical):
    can = canonical(num_gaussians=128, batch_size=3)
    delta = torch.randn(3, 128, 3) * 0.01
    w = torch.ones(3, 128)
    neigh = knn_neighbors(can.means[0].detach(), k=8)
    assert neigh.shape == (128, 8)
    values = {
        "disp": displacement_regularizer(delta, w),
        "neigh": neighborhood_consistency(delta, neigh, w),
        "vel": velocity_regularizer(delta, w),
        "acc": acceleration_regularizer(delta, w),
    }
    for name, value in values.items():
        assert torch.isfinite(value), name


def test_phometric_loss_terms():
    torch.manual_seed(0)
    pred = torch.rand(2, 3, 64, 64)
    gt = torch.rand(2, 3, 64, 64)
    terms = photometric_loss(pred, gt, lambda_rgb=0.8, lambda_ssim=0.2, lambda_perc=0.01)
    for name in ("l1", "dssim", "lpips", "total"):
        assert torch.isfinite(terms[name]), name
    assert terms["l1"] >= 0 and terms["dssim"] >= 0


def test_cpu_render_shape(canonical):
    from selftalk.deformation.types import DeformedGaussians

    can = canonical(num_gaussians=128, batch_size=1)
    frame = DeformedGaussians(
        means=can.means[0], rotations=can.rotations[0], scales=can.scales[0],
        sh=can.sh[0], opacities=can.opacities[0],
    )
    camera = Camera.look_at(
        torch.tensor([0.0, 0.0, 2.0]), torch.zeros(3), torch.tensor([0.0, 1.0, 0.0]),
        height=32, width=32,
    )
    image = render(camera, frame)
    assert image.shape == (3, 32, 32)
    assert (image >= 0).all() and (image <= 1).all()
    assert torch.isfinite(image).all()
    assert (image > 0).any()  # camera convention: in-view Gaussians must project


def test_cpu_render_differentiable(canonical):
    from selftalk.deformation.types import DeformedGaussians

    can = canonical(num_gaussians=128, batch_size=1)
    opacities = can.opacities[0].clamp(0.1, 0.9)
    sh = can.sh[0]
    opacities.retain_grad()
    sh.retain_grad()
    frame = DeformedGaussians(
        means=can.means[0], rotations=can.rotations[0], scales=can.scales[0],
        sh=sh, opacities=opacities,
    )
    camera = Camera.look_at(
        torch.tensor([0.0, 0.0, 2.0]), torch.zeros(3), torch.tensor([0.0, 1.0, 0.0]),
        height=32, width=32,
    )
    image = render(camera, frame)
    image.sum().backward()
    assert opacities.grad is not None and torch.isfinite(opacities.grad).all()
    assert sh.grad is not None and torch.isfinite(sh.grad).all()


def test_dataset_roundtrip(tmp_path):
    cv2 = pytest.importorskip("cv2")

    frames = tmp_path / "frames"
    frames.mkdir()
    for t in range(12):
        cv2.imwrite(str(frames / f"frame_{t:06d}.jpg"),
                    (np.random.rand(16, 16, 3) * 255).astype(np.uint8))
    np.save(tmp_path / "audio_feat.npy", np.random.rand(12, 8).astype(np.float32))
    np.save(tmp_path / "au.npy", np.random.rand(12, 17).astype(np.float32) * 5)

    from selftalk.data import TalkingFaceDataset

    ds = TalkingFaceDataset(tmp_path, audio_window=2)
    assert len(ds) == 12
    sample = ds[5]
    assert sample.image.shape == (3, 16, 16)
    assert sample.audio_window.shape == (5, 8)
    assert sample.au.shape == (17,)
    with pytest.raises(IndexError):
        _ = ds[12]


def test_openface_parsing(tmp_path):
    import pandas as pd

    from selftalk.data.preprocess import parse_openface_au, parse_openface_extras

    rng = np.random.default_rng(0)
    num = 5
    data = {"frame": np.arange(num), "confidence": rng.uniform(0.5, 1.0, num)}
    for i in (1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45):
        data[f"AU{i:02d}_r"] = rng.uniform(0.0, 5.0, num)
    for i in range(68):
        data[f"x_{i}"] = rng.uniform(0.0, 512.0, num)
        data[f"y_{i}"] = rng.uniform(0.0, 512.0, num)
    for c in ("pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry", "pose_Rz"):
        data[c] = rng.uniform(-1.0, 1.0, num)
    # OpenFace CSVs often carry a leading space in column names.
    csv_path = tmp_path / "au.csv"
    pd.DataFrame(data).to_csv(csv_path, index=False)

    au, conf = parse_openface_au(csv_path)
    assert au.shape == (num, 17)
    assert au.min() >= 0.0 and au.max() <= 5.0
    assert conf.shape == (num,)
    assert np.allclose(au[:, -1], np.clip(data["AU45_r"], 0.0, 5.0))

    pose, landmarks, blink = parse_openface_extras(csv_path)
    assert pose.shape == (num, 6)
    assert landmarks.shape == (num, 68, 2)
    assert blink.shape == (num, 1)
    assert blink.min() >= 0.0 and blink.max() <= 1.0


def test_au_pyfeat_row_conversion():
    from selftalk.data.au_pyfeat import (
        blink_from_ear,
        eye_aspect_ratio,
        to_openface_row,
    )

    # Open-eye landmarks: wide horizontal extent -> high EAR -> no blink.
    open_eye = np.array([[0.0, 0.0], [0.05, 0.05], [0.10, 0.06], [0.15, 0.0],
                         [0.10, -0.06], [0.05, -0.05]] * 2, np.float32)
    # Closed-eye: vertical extent collapses -> low EAR -> full blink.
    closed_eye = np.array([[0.0, 0.0], [0.05, 0.005], [0.10, 0.006], [0.15, 0.0],
                           [0.10, -0.006], [0.05, -0.005]] * 2, np.float32)
    landmarks = np.zeros((68, 2), np.float32)
    landmarks[36:42] = open_eye[:6]
    landmarks[42:48] = closed_eye[6:]
    ear = eye_aspect_ratio(landmarks)
    assert 0.0 < ear < 0.5
    assert blink_from_ear(0.30) == pytest.approx(0.0)
    assert blink_from_ear(0.13) == pytest.approx(1.0)

    row = to_openface_row({6: 1.2, 12: 3.9, 26: 9.0}, landmarks, (2.0, 3.0, 4.0),
                          xyz=(0.1, -0.2, 3.0))
    assert row["AU06_r"] == pytest.approx(1.2)
    assert row["AU12_r"] == pytest.approx(3.9)
    assert row["AU26_r"] == 5.0  # clipped to OpenFace range
    assert row["AU01_r"] == 0.0  # missing AUs are zero-filled
    assert 0.0 <= row["AU45_r"] <= 5.0  # EAR fallback for blink
    assert row["x_0"] == float(landmarks[0, 0]) and row["y_67"] == float(landmarks[67, 1])
    assert row["pose_Rx"] == 3.0 and row["pose_Ry"] == 4.0 and row["pose_Rz"] == 2.0
    assert row["pose_Tx"] == 0.1 and row["pose_Ty"] == -0.2 and row["pose_Tz"] == 3.0

    no_xyz = to_openface_row({6: 1.0}, None, (1.0, 2.0, 3.0))
    assert no_xyz["pose_Tx"] == 0.0

    minimal = to_openface_row({45: 2.5}, None, None)
    assert minimal["AU45_r"] == pytest.approx(2.5)
    assert "x_0" not in minimal and "pose_Rx" not in minimal


def test_metrics():
    img = torch.rand(3, 32, 32)
    assert psnr(img, img) == float("inf")
    noisy = (img + 0.1).clamp(0, 1)
    assert psnr(noisy, img) > 10.0
    au_pred = np.random.rand(20, 17).astype(np.float32)
    result = aue(au_pred, au_pred.copy())
    assert result["mean"] == pytest.approx(0.0)
    assert set(result["per_au"]) == {f"AU{i:02d}" for i in range(17)}


def test_selfcheck_synthetic_data(tmp_path):
    pytest.importorskip("cv2")

    from selftalk.training.selfcheck import (
        SELFCHK_AUDIO_DIM,
        make_synthetic_binary,
        synthetic_au_sequences,
    )

    audio, au = synthetic_au_sequences(12)
    assert audio.shape == (12, SELFCHK_AUDIO_DIM)
    assert au.shape == (12, 17)
    assert au.min() >= 0.0 and au.max() <= 5.0

    root = make_synthetic_binary(tmp_path, num_frames=4)
    assert (root / "audio_feat.npy").exists()
    assert (root / "au.npy").exists()
    assert len(list((root / "frames").glob("frame_*.jpg"))) == 4
