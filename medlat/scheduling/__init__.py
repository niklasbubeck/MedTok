"""
medlat.scheduling — unified generative scheduler package.

Replaces the separate medlat.diffusion and medlat.transport packages.
Both paradigms now subclass BaseScheduler and are created through a
single create_scheduler() factory.

    from medlat.scheduling import create_scheduler

    # Gaussian diffusion
    sched = create_scheduler("diffusion", steps=1000, noise_schedule="cosine")

    # Flow matching
    sched = create_scheduler("flow", path_type="Linear", prediction="velocity")

    # Same two calls for both:
    terms   = sched.training_losses(model, x_start, model_kwargs=cond)
    samples = sched.p_sample_loop(model, shape, model_kwargs=cond)
"""
from .base import BaseScheduler
from .gaussian import GaussianDiffusionScheduler, create_gaussian_diffusion
from .flow import FlowMatchingScheduler, create_transport, Transport, Sampler, ModelType, WeightType, PathType
from .self_flow import DualTimestepScheduler, create_dual_timestep_scheduler

from typing import Any


def create_scheduler(scheduler_type: str = "diffusion", **kwargs: Any) -> BaseScheduler:
    """Instantiate a generative scheduler by type.

    Args:
        scheduler_type: 'diffusion' for Gaussian diffusion,
                        'flow' (or 'transport') for flow matching,
                        'self_flow' for dual-timestep Self-Flow scheduling.
        **kwargs:       Forwarded verbatim to the scheduler factory.

    Returns:
        A BaseScheduler-compatible object.
    """
    if scheduler_type == "diffusion":
        return create_gaussian_diffusion(**kwargs)
    elif scheduler_type in ("flow", "transport"):
        return create_transport(**kwargs)
    elif scheduler_type == "self_flow":
        return create_dual_timestep_scheduler(**kwargs)
    else:
        raise ValueError(
            f"Unknown scheduler_type '{scheduler_type}'. "
            "Choose 'diffusion', 'flow', 'transport', or 'self_flow'."
        )


__all__ = [
    "BaseScheduler",
    "GaussianDiffusionScheduler",
    "FlowMatchingScheduler",
    "DualTimestepScheduler",
    "create_gaussian_diffusion",
    "create_transport",
    "create_dual_timestep_scheduler",
    "create_scheduler",
    "Transport",
    "Sampler",
    "ModelType",
    "WeightType",
    "PathType",
]
