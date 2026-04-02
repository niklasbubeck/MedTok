# Backward-compatibility re-exports.
# The implementation now lives in medlat.scheduling.
# This stub keeps existing imports like `from medlat.transport import create_transport` working.
from medlat.scheduling.flow import (  # noqa: F401
    Transport, ModelType, WeightType, PathType, Sampler,
    FlowMatchingScheduler, create_transport,
)
