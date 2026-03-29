"""
Abstract base classes that define the contracts for first-stage and generator models.
All registered models should inherit from the appropriate base class.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple
import torch
import torch.nn as nn

__all__ = [
    "FirstStageModel",
    "ContinuousFirstStage",
    "DiscreteFirstStage",
    "TokenFirstStage",
    "GeneratorModel",
    "AutoregressiveGenerator",
    "NonAutoregressiveGenerator",
]


class FirstStageModel(nn.Module, ABC):
    """Base class for all first-stage (tokenizer/autoencoder) models."""

    @property
    @abstractmethod
    def vae_stride(self) -> Tuple[int, ...]:
        """Spatial downsampling factor as a tuple, e.g. (8, 8) for f8."""

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """Dimensionality of the latent / token embedding."""

    @abstractmethod
    def encode(self, x: torch.Tensor) -> Any:
        """Encode input image to latent representation."""

    @abstractmethod
    def decode(self, z: Any) -> torch.Tensor:
        """Decode latent back to image space."""


class ContinuousFirstStage(FirstStageModel, ABC):
    """
    First-stage model with a continuous (e.g. Gaussian) latent space.

    encode(x) must return (z, loss, extra)
    decode(z) must return reconstructed image tensor
    """

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """Number of latent channels (z_channels)."""


class DiscreteFirstStage(FirstStageModel, ABC):
    """
    First-stage model with a discrete (codebook) latent space.

    encode(x) must return (quant, loss, info) where info = (_, _, indices)
    decode(quant) must return reconstructed image tensor
    decode_code(indices, out_shape) must return reconstructed image tensor
    encode_to_prequant(x) must return (h, None, None)
    decode_from_prequant(h) must return reconstructed image tensor
    """

    @property
    @abstractmethod
    def n_embed(self) -> int:
        """Codebook size."""

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """Embedding dimension per codebook entry."""

    @abstractmethod
    def encode_to_prequant(self, x: torch.Tensor) -> Any:
        """Encode to pre-quantization features."""

    @abstractmethod
    def decode_from_prequant(self, h: torch.Tensor) -> torch.Tensor:
        """Quantize and decode from pre-quantization features."""

    @abstractmethod
    def decode_code(self, indices: torch.Tensor, out_shape: Optional[Tuple] = None) -> torch.Tensor:
        """Decode from codebook indices."""


class TokenFirstStage(FirstStageModel, ABC):
    """
    Learned tokenizer (TiTok, MAETok, VMAE, ViTA, etc.).
    Discrete codebook but different interface from VQ-VAE.
    """

    @property
    @abstractmethod
    def n_embed(self) -> int:
        """Codebook size."""

    @abstractmethod
    def decode_code(self, indices: torch.Tensor, out_shape: Optional[Tuple] = None) -> torch.Tensor:
        """Decode from codebook indices."""


class GeneratorModel(nn.Module, ABC):
    """Base class for all generator models."""

    @property
    def vae_stride(self) -> Optional[Tuple[int, ...]]:
        """Expected spatial downsampling of the first-stage model. None if unconstrained."""
        return None


class AutoregressiveGenerator(GeneratorModel, ABC):
    """Generator that produces outputs token-by-token or in masked fashion."""


class NonAutoregressiveGenerator(GeneratorModel, ABC):
    """Generator based on diffusion / score-matching over continuous latents."""
