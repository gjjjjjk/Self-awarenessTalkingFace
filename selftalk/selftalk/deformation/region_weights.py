"""Stage 3: region ids and per-region regularizer weights.

Plan: dynamic regions (mouth, eyes) get weaker geometric constraints; static
regions (hair, ears, background/torso) get stronger ones. Region ids come
from the 3DMM/FLAME mesh segmentation (see `canonical/model.py`), or from a
distance heuristic on landmarks.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

MOUTH = 0
EYES = 1
FACE = 2
HAIR = 3
EAR = 4
TORSO = 5
BACKGROUND = 6
NUM_REGIONS = 7


@dataclass
class RegionWeights:
    """Per-region multipliers for each stage-5 regularizer.

    weight_*: [NUM_REGIONS] tensors indexed by region id.
    """

    displacement: torch.Tensor
    neighborhood: torch.Tensor
    rigidity: torch.Tensor
    velocity: torch.Tensor
    acceleration: torch.Tensor

    def select(self, region_ids: torch.Tensor, field: str) -> torch.Tensor:
        """Gather per-Gaussian weights: region_ids [N] -> [N] or [B, N]."""
        table = getattr(self, field)
        return table[region_ids]


def default_region_weights(device: torch.device | str = "cpu") -> RegionWeights:
    """Weak constraints in MOUTH/EYES, strong static ones in HAIR/EAR/BG."""
    ones = torch.ones(NUM_REGIONS, device=device)
    weak = torch.tensor([2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0], device=device) * 0.5
    strong = torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0], device=device)

    displacement = torch.tensor([0.5, 0.5, 1.0, 2.0, 2.0, 2.0, 3.0], device=device)
    neighborhood = weak.clone()
    neighborhood[MOUTH] = 0.25
    neighborhood[EYES] = 0.25
    rigidity = strong.clone()
    rigidity[MOUTH] = 0.25
    rigidity[EYES] = 0.5
    velocity = ones.clone()
    velocity[MOUTH] = 0.5
    acceleration = ones.clone()
    acceleration[MOUTH] = 0.5
    return RegionWeights(
        displacement=displacement,
        neighborhood=neighborhood,
        rigidity=rigidity,
        velocity=velocity,
        acceleration=acceleration,
    )


def infer_region_ids_from_landmarks(means: torch.Tensor, mouth_center: torch.Tensor) -> torch.Tensor:
    """Heuristic region ids when mesh segmentation is unavailable.

    Gaussians within `mouth_radius` of the (3D) mouth center are MOUTH; the
    rest default to FACE. means: [N, 3]; mouth_center: [3].

    Returns region ids [N].
    """
    if means.ndim != 2 or means.shape[-1] != 3:
        raise ValueError(f"means must be [N, 3], got {tuple(means.shape)}")
    dist = torch.norm(means - mouth_center.to(means.device), dim=-1)
    ids = torch.full((means.shape[0],), FACE, dtype=torch.long, device=means.device)
    ids[dist < 0.05] = MOUTH
    return ids
