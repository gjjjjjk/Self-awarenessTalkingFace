"""Stage 1: canonical 3D Gaussian modeling (triplane + attribute MLP)."""

from .model import CanonicalGaussianModel, CanonicalGaussians
from .triplane import MultiResolutionTriplane

__all__ = ["CanonicalGaussianModel", "CanonicalGaussians", "MultiResolutionTriplane"]
