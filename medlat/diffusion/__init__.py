# Backward-compatibility re-exports.
# The implementation now lives in medlat.scheduling.
# This stub keeps existing imports like `from medlat.diffusion import create_gaussian_diffusion` working.
from medlat.scheduling.gaussian import GaussianDiffusionScheduler, create_gaussian_diffusion  # noqa: F401
