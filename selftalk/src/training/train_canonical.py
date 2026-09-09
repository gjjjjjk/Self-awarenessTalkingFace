"""Stage 1 training: canonical 3D Gaussians on per-frame data.

    L_can = lambda_1 L1 + lambda_s D-SSIM + lambda_p LPIPS

Real data needs per-frame cameras (`cams.npy`, [T, 4, 4] world-to-camera,
inside the binary dir). Without cameras use --selfcheck, which synthesizes a
small scene (ground-truth Gaussians -> renders) and verifies the pipeline
reduces L_can.

Usage:
    python -m selftalk.training.train_canonical --binary ../data/binary/subject
    python -m selftalk.training.train_canonical --selfcheck
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ..canonical import CanonicalGaussianModel, MultiResolutionTriplane
from ..configs import CanonicalConfig
from ..losses import photometric_loss
from ..rendering import Camera, render
from ..deformation.types import DeformedGaussians


def _to_renderable(canon) -> DeformedGaussians:
    """Squeeze the shared batch dim of canonical output for single-frame render."""
    return DeformedGaussians(
        means=canon.means[0],
        rotations=canon.rotations[0],
        scales=canon.scales[0],
        sh=canon.sh[0],
        opacities=canon.opacities[0],
    )


def _synthetic_reference(num: int, device: torch.device, sh_k: int):
    ref = DeformedGaussians(
        means=0.4 * torch.rand(num, 3, device=device) - 0.2,
        rotations=torch.nn.functional.normalize(torch.randn(num, 4, device=device), dim=-1),
        scales=0.02 + 0.02 * torch.rand(num, 3, device=device),
        sh=0.1 * torch.randn(num, sh_k, 3, device=device),
        opacities=torch.rand(num, 1, device=device) * 0.5 + 0.5,
    )
    return ref


def train(args: argparse.Namespace) -> Path:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    cfg = CanonicalConfig()
    cfg.iterations = args.iterations

    triplane = MultiResolutionTriplane(
        resolutions=cfg.triplane_resolutions,
        feat_dim=cfg.triplane_feat_dim,
        out_dim=cfg.mlp_hidden,
    )
    model = CanonicalGaussianModel(
        triplane, mlp_hidden=cfg.mlp_hidden, mlp_depth=cfg.mlp_depth, sh_degree=cfg.sh_degree
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    eye = torch.tensor([0.0, 0.0, 1.6], device=device)
    target = torch.zeros(3, device=device)
    up = torch.tensor([0.0, 1.0, 0.0], device=device)
    camera = Camera.look_at(eye, target, up)

    if args.selfcheck:
        num = args.num_gaussians
        ref = _synthetic_reference(num, device, (cfg.sh_degree + 1) ** 2)
        model.init_from_vertices(ref.means.detach().clone())
        gt_image = render(camera, ref)
    else:
        from ..data import TalkingFaceDataset

        dataset = TalkingFaceDataset(args.binary)
        num = len(dataset)
        vertices = None
        flame_vertices = Path(args.binary) / "flame" / "vertices.npy"
        if flame_vertices.exists():
            vertices = torch.from_numpy(np.load(flame_vertices)).float()
            num = vertices.shape[0]
            model.init_from_vertices(vertices.to(device))
        cams_path = Path(args.binary) / "cams.npy"
        if not cams_path.exists():
            raise ValueError(
                f"{cams_path} missing: per-frame cameras required for real training "
                "(use --selfcheck for a pipeline check)"
            )
        gt_image = dataset[0].image.to(device)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    for it in range(1, cfg.iterations + 1):
        canon = model(num_gaussians=num, batch_size=1)
        pred = render(camera, _to_renderable(canon))
        terms = photometric_loss(
            pred.unsqueeze(0), gt_image.unsqueeze(0),
            lambda_rgb=cfg.lambda_1, lambda_ssim=cfg.lambda_ssim,
            lambda_perc=cfg.lambda_lpips,
        )
        loss = terms["total"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it % max(1, cfg.iterations // 10) == 0 or it == 1:
            print(f"[stage1] iter {it}/{cfg.iterations} l1 {float(terms['l1']):.4f} "
                  f"dssim {float(terms['dssim']):.4f}")

    ckpt = out_dir / "canonical.pt"
    torch.save({"model": model.state_dict(), "num_gaussians": num,
                "config": {"sh_degree": cfg.sh_degree, "triplane_out": cfg.mlp_hidden}}, ckpt)
    print(f"[stage1] saved {ckpt}")
    return ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: canonical Gaussians")
    parser.add_argument("--binary", default=None, help="preprocessed subject dir")
    parser.add_argument("--selfcheck", action="store_true")
    parser.add_argument("--out", default="runs/canonical")
    parser.add_argument("--iterations", type=int, default=30_000)
    parser.add_argument("--num_gaussians", type=int, default=2048)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=6666)
    args = parser.parse_args()
    if args.binary is None and not args.selfcheck:
        parser.error("either --binary or --selfcheck is required")
    train(args)


if __name__ == "__main__":
    main()
