"""Stage 7: AU editability at inference time.

Edits one AU intensity (e.g. AU26 x1.5), re-renders, and reports the change
confined to the target region (non-target Gaussians must stay near-static
because the stage-5 constrained update plus region regularization keep them
in place).

Usage:
    python -m selftalk.training.au_editing \
        --ckpt runs/deform/deform_motion_token.pt \
        --canonical_ckpt runs/canonical/canonical.pt \
        --binary ../data/binary/subject --au AU26 --scale 1.5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..data.dataset import TalkingFaceDataset
from ..deformation import (
    ConditionEncoder,
    constrained_gaussian_update,
)
from ..deformation.types import DeformedGaussians
from ..configs import DeformationConfig
from ..rendering import Camera, render
from .train_deform import build_deformation, load_frozen_modules

AU_NAME_TO_INDEX = {
    name: idx for idx, name in enumerate(
        ["AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10", "AU12",
         "AU14", "AU15", "AU17", "AU20", "AU23", "AU25", "AU26", "AU45"]
    )
}


def edit_au(
    au_vector: torch.Tensor, au_name: str, scale: float, clamp_range: tuple[float, float] = (0.0, 5.0)
) -> torch.Tensor:
    """Scale one AU intensity, clamped to the valid OpenFace range.

    au_vector: [A] or [B, A] AU intensities; the last dimension is edited.
    """
    if au_name not in AU_NAME_TO_INDEX:
        raise ValueError(f"unknown AU {au_name}; valid: {sorted(AU_NAME_TO_INDEX)}")
    if au_vector.shape[-1] != len(AU_NAME_TO_INDEX):
        raise ValueError(
            f"au_vector last dim {au_vector.shape[-1]} != {len(AU_NAME_TO_INDEX)}"
        )
    out = au_vector.clone()
    idx = AU_NAME_TO_INDEX[au_name]
    out[..., idx] = (scale * out[..., idx]).clamp(*clamp_range)
    return out


def render_edited_frame(
    ckpt: str,
    canonical_ckpt: str,
    binary: str,
    frame_index: int,
    au_name: str,
    scale: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Render (original, edited, |difference|) for one frame."""
    payload = torch.load(ckpt, map_location=device)
    variant = payload["variant"]
    canonical, _au_net, num_gaussians, triplane_dim = load_frozen_modules(
        canonical_ckpt, None, device
    )
    dataset = TalkingFaceDataset(binary)
    sample = dataset[frame_index]
    num_sh = (canonical.sh_degree + 1) ** 2

    deform_net = build_deformation(variant, triplane_dim, num_sh, device)
    deform_net.load_state_dict(payload["deform"])
    deform_net.eval()

    condition_encoder = ConditionEncoder(
        d_model=DeformationConfig().d_model,
        audio_feature_dim=dataset.audio.shape[1],
    ).to(device)
    condition_encoder.load_state_dict(payload["condition_encoder"])
    condition_encoder.eval()

    region_ids = payload["region_ids"].to(device)
    b = 1
    audio = sample.audio_window.unsqueeze(0).to(device)
    blink = sample.blink.unsqueeze(0).to(device)
    pose = sample.pose.unsqueeze(0).to(device)
    au_orig = sample.au.unsqueeze(0).to(device)
    au_edit = edit_au(au_orig, au_name, scale)

    camera = Camera.look_at(
        torch.tensor([0.0, 0.0, 1.6], device=device), torch.zeros(3, device=device),
        torch.tensor([0.0, 1.0, 0.0], device=device),
    )

    def _render(au: torch.Tensor) -> torch.Tensor:
        cond = condition_encoder(audio, au, blink, pose)
        canon = canonical(num_gaussians=num_gaussians, batch_size=b)
        deltas = deform_net(canon, cond, region_ids)
        deformed = constrained_gaussian_update(canon, deltas)
        return render(camera, DeformedGaussians(
            means=deformed.means[0], rotations=deformed.rotations[0],
            scales=deformed.scales[0], sh=deformed.sh[0], opacities=deformed.opacities[0],
        ))

    with torch.no_grad():
        img_orig = _render(au_orig)
        img_edit = _render(au_edit)
    diff = (img_edit - img_orig).abs()
    changed = diff.sum(dim=0)
    target_change = float(changed[changed > 0.01].numel())
    print(f"[stage7] {au_name} x{scale}: changed pixels {target_change} "
          f"({100.0 * target_change / changed.numel():.2f}% of frame)")
    return {"original": img_orig, "edited": img_edit, "difference": diff}


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 7: AU editing")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--canonical_ckpt", required=True)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--au", default="AU26")
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument("--out", default=None, help="optional .pt output path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    result = render_edited_frame(
        args.ckpt, args.canonical_ckpt, args.binary, args.frame,
        args.au, args.scale, torch.device(args.device),
    )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.cpu() for k, v in result.items()}, out)
        print(f"[stage7] saved {out}")


if __name__ == "__main__":
    main()
