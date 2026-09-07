"""Stages 3-7 training: AU-conditioned Gaussian deformation.

Variants (--variant):
    audio          audio tokens only                        (stage 3 ablation)
    audio_au       audio + AU tokens                        (stage 3 main)
    audio_au_av    audio + AU + view/blink                  (stage 3 full)
    motion_token   stage-4 model with global motion tokens

Training strategy per the plan:
    1. freeze canonical Gaussians + audio encoder + AU net,
    2. train only the attention modules and deformation heads,
    3. optionally low-LR fine-tune the triplane afterwards
       (--triplane_finetune_from_iter),
    4. stage 7 scheduled sampling: alternate GT/predicted AU with a
       probability ramp and Gaussian noise injection on AU inputs.

Losses are added in the documented order, each behind a weight flag.

Usage:
    python -m selftalk.training.train_deform --binary ../data/binary/subject \
        --canonical_ckpt runs/canonical/canonical.pt --variant motion_token
    python -m selftalk.training.train_deform --selfcheck \
        --canonical_ckpt runs/canonical/canonical.pt --variant motion_token
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..audio2au import Audio2AUNet
from ..canonical import CanonicalGaussianModel, MultiResolutionTriplane
from ..configs import Audio2AUConfig, DataConfig, DeformationConfig, TrainDeformConfig
from ..data.dataset import FrameSample, TalkingFaceDataset, collate_samples
from ..deformation import (
    ConditionEncoder,
    DirectConditionDeformation,
    MotionTokenDeformation,
    constrained_gaussian_update,
    default_region_weights,
)
from ..deformation.types import DeformedGaussians
from ..losses import photometric_loss
from ..losses.regularizers import (
    acceleration_regularizer,
    displacement_regularizer,
    knn_neighbors,
    neighborhood_consistency,
    velocity_regularizer,
)
from ..rendering import Camera, render
from .selfcheck import make_synthetic_binary


def load_frozen_modules(
    canonical_ckpt: str, audio2au_ckpt: str | None, device: torch.device
) -> tuple[CanonicalGaussianModel, Audio2AUNet | None, int, int]:
    """Load frozen stage-1/2 checkpoints."""
    payload = torch.load(canonical_ckpt, map_location=device)
    triplane_out = payload["config"]["triplane_out"]
    triplane = MultiResolutionTriplane(out_dim=triplane_out)
    model = CanonicalGaussianModel(
        triplane, sh_degree=payload["config"]["sh_degree"]
    ).to(device)
    if "means" in payload["model"]:
        # The canonical means live in the checkpoint as a buffer (registered
        # via init_from_vertices); re-register it before the strict load.
        model.init_from_vertices(payload["model"]["means"].to(device))
    model.load_state_dict(payload["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    au_net = None
    if audio2au_ckpt is not None:
        au_payload = torch.load(audio2au_ckpt, map_location=device)
        cfg = Audio2AUConfig(**{
            k: v for k, v in au_payload["config"].items()
            if k in Audio2AUConfig.__dataclass_fields__
        })
        au_net = Audio2AUNet(cfg, num_out_au=len(au_payload["lower_face_au"])).to(device)
        au_net.load_state_dict(au_payload["model"])
        au_net.eval()
        for p in au_net.parameters():
            p.requires_grad_(False)
    return model, au_net, payload["num_gaussians"], triplane_out


def build_deformation(variant: str, triplane_dim: int, num_sh: int, device: torch.device):
    dcfg = DeformationConfig()
    drop: tuple[str, ...] = ()
    if variant == "audio":
        drop = ("au", "blink", "view")
        model = DirectConditionDeformation(dcfg, triplane_dim, num_sh, drop_tokens=drop)
    elif variant == "audio_au":
        drop = ("blink", "view")
        model = DirectConditionDeformation(dcfg, triplane_dim, num_sh, drop_tokens=drop)
    elif variant == "audio_au_av":
        model = DirectConditionDeformation(dcfg, triplane_dim, num_sh)
    elif variant == "motion_token":
        model = MotionTokenDeformation(dcfg, triplane_dim, num_sh)
    else:
        raise ValueError(f"unknown variant {variant}")
    return model.to(device)


def gt_ratio_schedule(it: int, total: int, cfg: TrainDeformConfig) -> float:
    """Stage 7: linearly decay the ground-truth-AU probability."""
    frac = min(1.0, max(0.0, it / max(total - 1, 1)))
    return cfg.au_gt_ratio_start + frac * (cfg.au_gt_ratio_end - cfg.au_gt_ratio_start)


def run_deform_step(
    samples: list[FrameSample],
    canonical: CanonicalGaussianModel,
    au_net: Audio2AUNet | None,
    deform_net: torch.nn.Module,
    condition_encoder: ConditionEncoder,
    camera: Camera,
    region_ids: torch.Tensor,
    train_cfg: TrainDeformConfig,
    num_gaussians: int,
    device: torch.device,
    gt_ratio: float = 1.0,
) -> dict[str, torch.Tensor]:
    """One forward/backward over a small list of consecutive frames."""
    batch = collate_samples(samples)
    batch = {k: v.to(device) for k, v in batch.items()}
    b = batch["au"].shape[0]

    au_input = batch["au"].clone()
    if au_net is not None and gt_ratio < 1.0:
        with torch.no_grad():
            pred = au_net(batch["audio_window"])
        au_lower_idx = torch.tensor(list(DataConfig.lower_face_au), device=device)
        proj = pred["au"][:, -1]
        use_pred = (torch.rand(b, device=device) > gt_ratio).unsqueeze(-1)
        au_input[:, au_lower_idx] = torch.where(
            use_pred, proj, au_input[:, au_lower_idx]
        )
    if train_cfg.au_noise_std > 0:
        au_input = au_input + train_cfg.au_noise_std * torch.randn_like(au_input)

    condition = condition_encoder(
        batch["audio_window"], au_input.clamp(0, 5), batch["blink"], batch["pose"]
    )

    canonical_out = canonical(num_gaussians=num_gaussians, batch_size=b)
    deltas = deform_net(canonical_out, condition, region_ids)
    deformed = constrained_gaussian_update(canonical_out, deltas)

    pred_images = []
    for i in range(b):
        frame = DeformedGaussians(
            means=deformed.means[i], rotations=deformed.rotations[i],
            scales=deformed.scales[i], sh=deformed.sh[i], opacities=deformed.opacities[i],
        )
        pred_images.append(render(camera, frame))
    pred = torch.stack(pred_images)

    terms = photometric_loss(
        pred, batch["image"], lambda_lip=train_cfg.lambda_lip, lip_mask=None
    )

    region_weights = default_region_weights(device)
    w_disp = region_weights.displacement[region_ids].unsqueeze(0).expand(b, -1)
    reg_disp = displacement_regularizer(deltas.means, w_disp)
    neigh_idx = knn_neighbors(canonical_out.means[0].detach(), k=8)
    w_neigh = region_weights.neighborhood[region_ids].unsqueeze(0).expand(b, -1)
    reg_neigh = neighborhood_consistency(deltas.means, neigh_idx, w_neigh)
    reg_vel = velocity_regularizer(deltas.means, w_disp) if b >= 2 \
        else torch.zeros((), device=device)
    reg_acc = acceleration_regularizer(deltas.means, w_disp) if b >= 3 \
        else torch.zeros((), device=device)

    total = terms["total"] + train_cfg.lambda_geo * (reg_disp + reg_neigh) \
        + train_cfg.lambda_temp * (reg_vel + reg_acc)
    return {
        "total": total, "l1": terms["l1"], "dssim": terms["dssim"],
        "reg_disp": reg_disp, "reg_neigh": reg_neigh,
        "reg_vel": reg_vel, "reg_acc": reg_acc,
    }


def train(args: argparse.Namespace) -> Path:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_cfg = TrainDeformConfig(variant=args.variant)
    train_cfg.iterations = args.iterations

    if args.selfcheck:
        selfcheck_binary = Path(args.out) / "selfcheck_binary"
        make_synthetic_binary(selfcheck_binary, seed=args.seed)
        print(f"[deform] selfcheck binary -> {selfcheck_binary}")
        args.binary = str(selfcheck_binary)

    canonical, au_net, num_gaussians, triplane_dim = load_frozen_modules(
        args.canonical_ckpt, args.audio2au_ckpt, device
    )
    dataset = TalkingFaceDataset(args.binary, audio_window=args.window)
    num_sh = (canonical.sh_degree + 1) ** 2

    deform_net = build_deformation(args.variant, triplane_dim, num_sh, device)
    condition_encoder = ConditionEncoder(
        d_model=DeformationConfig().d_model,
        audio_feature_dim=dataset.audio.shape[1],
    ).to(device)
    region_ids = torch.zeros(num_gaussians, dtype=torch.long, device=device)

    params = list(deform_net.parameters()) + list(condition_encoder.parameters())
    opt = torch.optim.AdamW(params, lr=train_cfg.lr)
    if args.triplane_finetune:
        opt.add_param_group({
            "params": canonical.triplane.parameters(),
            "lr": train_cfg.triplane_finetune_lr,
        })

    eye = torch.tensor([0.0, 0.0, 1.6], device=device)
    camera = Camera.look_at(eye, torch.zeros(3, device=device),
                            torch.tensor([0.0, 1.0, 0.0], device=device))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    deform_net.train()
    condition_encoder.train()
    for it in range(1, train_cfg.iterations + 1):
        start = int(torch.randint(0, max(len(dataset) - train_cfg.batch_frames, 1), (1,)))
        samples = [dataset[i] for i in range(start, start + train_cfg.batch_frames)]
        gt_ratio = gt_ratio_schedule(it, train_cfg.iterations, train_cfg)
        terms = run_deform_step(
            samples, canonical, au_net, deform_net, condition_encoder, camera,
            region_ids, train_cfg, num_gaussians, device, gt_ratio=gt_ratio,
        )
        opt.zero_grad(set_to_none=True)
        terms["total"].backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        if it % max(1, train_cfg.iterations // 10) == 0 or it == 1:
            print(f"[deform:{args.variant}] iter {it}/{train_cfg.iterations} "
                  f"l1 {float(terms['l1']):.4f} gt_au {gt_ratio:.2f}")

    ckpt = out_dir / f"deform_{args.variant}.pt"
    torch.save({
        "deform": deform_net.state_dict(),
        "condition_encoder": condition_encoder.state_dict(),
        "variant": args.variant,
        "num_gaussians": num_gaussians,
        "region_ids": region_ids.cpu(),
    }, ckpt)
    print(f"[deform] saved {ckpt}")
    return ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description="Stages 3-7: deformation training")
    parser.add_argument("--binary", default=None, help="preprocessed subject dir")
    parser.add_argument("--selfcheck", action="store_true",
                        help="synthesize a tiny binary dir and train briefly on it")
    parser.add_argument("--canonical_ckpt", required=True)
    parser.add_argument("--audio2au_ckpt", default=None)
    parser.add_argument("--variant", default="audio_au",
                        choices=["audio", "audio_au", "audio_au_av", "motion_token"])
    parser.add_argument("--out", default="runs/deform")
    parser.add_argument("--window", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=30_000)
    parser.add_argument("--batch_frames", type=int, default=3,
                        help="consecutive frames per step (>=3 for the acceleration term)")
    parser.add_argument("--triplane_finetune", action="store_true",
                        help="low-LR triplane fine-tuning (stage 3 step 3)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=6666)
    args = parser.parse_args()
    if args.binary is None and not args.selfcheck:
        parser.error("either --binary or --selfcheck is required")
    train(args)


if __name__ == "__main__":
    main()
