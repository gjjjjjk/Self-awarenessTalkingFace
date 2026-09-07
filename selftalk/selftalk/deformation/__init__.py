"""Stages 3-5: AU-conditioned Gaussian deformation."""

from .baseline import CrossAttentionBlock, DirectConditionDeformation
from .constrained_update import constrained_gaussian_update, quaternion_multiply, safe_logit
from .heads import DeformationHeads, GaussianQueryMLP, canonical_attribute_summary
from .motion_tokens import GatedAudioAUFusion, MotionTokenDeformation, MotionTokenizer
from .region_weights import (
    BACKGROUND,
    EAR,
    EYES,
    FACE,
    HAIR,
    MOUTH,
    NUM_REGIONS,
    TORSO,
    RegionWeights,
    default_region_weights,
    infer_region_ids_from_landmarks,
)
from .tokens import ConditionEncoder
from .types import ConditionTokens, DeformedGaussians, DeformationDeltas

__all__ = [
    "CrossAttentionBlock",
    "DirectConditionDeformation",
    "GatedAudioAUFusion",
    "GaussianQueryMLP",
    "DeformationHeads",
    "MotionTokenDeformation",
    "MotionTokenizer",
    "ConditionEncoder",
    "ConditionTokens",
    "DeformationDeltas",
    "DeformedGaussians",
    "constrained_gaussian_update",
    "quaternion_multiply",
    "safe_logit",
    "canonical_attribute_summary",
    "RegionWeights",
    "default_region_weights",
    "infer_region_ids_from_landmarks",
    "MOUTH",
    "EYES",
    "FACE",
    "HAIR",
    "EAR",
    "TORSO",
    "BACKGROUND",
    "NUM_REGIONS",
]
