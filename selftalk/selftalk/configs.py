"""Central configuration dataclasses, one group per pipeline stage.

Mirrors the hyperparameters named in 自感知说话头.md (stages 0-8). Every
training entry point takes a `SelfTalkConfig` (or a YAML file that is mapped
onto one via `from_yaml`).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Tuple

import yaml


@dataclass
class DataConfig:
    """Stage 0: per-frame data format {I_t, A_{t-w:t+w}, AU_t, pi_t, L_t}."""

    binary_dir: Path = Path("../data/binary")
    audio_window: int = 16
    """Half-width w of the context audio window A_{t-w:t+w} (frames)."""
    audio_feature_dim: int = 768
    """Dimension of the pretrained audio-encoder features (HuBERT/Wav2Vec2)."""
    num_au: int = 17
    """OpenFace AU intensities (AU01,02,04,05,06,07,09,10,12,14,15,17,20,23,25,26,45)."""
    lower_face_au: Tuple[int, ...] = (7, 8, 9, 10, 11, 12, 13, 14, 15)
    """0-based indices of speech-related lower-face AUs (AU10,12,14,15,17,20,23,25,26)."""
    upper_face_au: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 16)
    """0-based indices of upper-face AUs (AU01,02,04,05,06,07,09,45)."""
    image_size: Tuple[int, int] = (512, 512)


@dataclass
class CanonicalConfig:
    """Stage 1: canonical 3D Gaussian modeling via triplane + MLP."""

    triplane_resolutions: Tuple[int, ...] = (64, 128, 256)
    triplane_feat_dim: int = 32
    mlp_hidden: int = 64
    mlp_depth: int = 4
    sh_degree: int = 3
    lambda_1: float = 0.8
    """lambda_1 in L_can = lambda_1 L1 + lambda_s D-SSIM + lambda_p LPIPS."""
    lambda_ssim: float = 0.2
    lambda_lpips: float = 0.01
    lr: float = 1.6e-4
    iterations: int = 30_000


@dataclass
class Audio2AUConfig:
    """Stage 2: Audio-to-AU network (lower-face AUs only in v1)."""

    audio_feature_dim: int = 768
    """Dimension of the pretrained audio features fed to the AU net."""
    encoder_name: str = "facebook/wav2vec2-base"
    freeze_encoder: bool = True
    lower_face_au_index: tuple[int, ...] = (7, 8, 9, 10, 11, 12, 13, 14, 15)
    """0-based indices of the predicted lower-face AUs."""
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.1
    lambda_vel: float = 0.2
    """lambda_v: weight of the first-difference L1 term."""
    lambda_acc: float = 0.1
    """lambda_a: weight of the second-difference L1 term."""
    lr: float = 1e-4
    epochs: int = 50


@dataclass
class DeformationConfig:
    """Stages 3-5: condition tokens, motion tokens, constrained update."""

    d_model: int = 256
    """Token/hidden dimension d."""
    n_heads: int = 8
    num_motion_tokens: int = 32
    """K in Q_motion ∈ R^{K×d}; start small, ablate."""
    cross_attn_layers: int = 2
    decoder_layers: int = 2
    region_feat_dim: int = 16
    lambda_reg_disp: float = 0.01
    lambda_reg_neigh: float = 0.05
    lambda_reg_rigid: float = 0.05
    lambda_reg_vel: float = 0.01
    lambda_reg_acc: float = 0.01


@dataclass
class TrainDeformConfig:
    """Stage 3 training strategy + stage 7 AU-schedule flags."""

    variant: str = "audio_au"
    """audio | audio_au | audio_au_av | motion_token."""
    lr: float = 1e-4
    triplane_finetune_lr: float = 1e-6
    triplane_finetune_from_iter: int = 15_000
    iterations: int = 30_000
    batch_frames: int = 3
    au_gt_ratio_start: float = 1.0
    """Stage 7 scheduled sampling: probability of ground-truth AU at start."""
    au_gt_ratio_end: float = 0.2
    au_noise_std: float = 0.05
    lambda_lip: float = 0.4
    lambda_au_consistency: float = 0.1
    lambda_sync: float = 0.0
    lambda_temp: float = 0.05
    lambda_geo: float = 0.05


@dataclass
class SelfTalkConfig:
    """Top-level config grouping all stages."""

    data: DataConfig = field(default_factory=DataConfig)
    canonical: CanonicalConfig = field(default_factory=CanonicalConfig)
    audio2au: Audio2AUConfig = field(default_factory=Audio2AUConfig)
    deform: DeformationConfig = field(default_factory=DeformationConfig)
    train_deform: TrainDeformConfig = field(default_factory=TrainDeformConfig)
    device: str = "cuda"
    seed: int = 6666

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SelfTalkConfig":
        raw = yaml.safe_load(Path(path).read_text())
        if raw is None:
            return cls()
        cfg = cls()
        for section, values in raw.items():
            if not hasattr(cfg, section):
                raise ValueError(f"Unknown config section: {section}")
            sub = getattr(cfg, section)
            for key, value in values.items():
                if not hasattr(sub, key):
                    raise ValueError(f"Unknown config key: {section}.{key}")
                setattr(sub, key, value)
        return cfg
