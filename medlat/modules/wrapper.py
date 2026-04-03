import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from medlat.utils import init_from_ckpt, get_model_type, validate_compatibility

logger = logging.getLogger(__name__)

__all__ = ["GenWrapper"]


class GenWrapper(nn.Module):
    """Glue layer that pairs a frozen first-stage tokenizer with a trainable generator.

    GenWrapper handles four encode/decode routing modes automatically based on
    the types of the models you pass in:

    ┌─────────────────┬──────────────────┬──────────────────────────────┬────────────────────────────┐
    │  First Stage    │  Generator       │  encode                      │  decode                    │
    ├─────────────────┼──────────────────┼──────────────────────────────┼────────────────────────────┤
    │  continuous     │  non-autoregr.   │  first_stage.encode          │  first_stage.decode        │
    │  continuous     │  autoregressive  │  first_stage.encode          │  first_stage.decode        │
    │  discrete       │  non-autoregr.   │  first_stage.encode_to_prequant │ first_stage.decode_from_prequant │
    │  discrete       │  autoregressive  │  first_stage.encode          │  first_stage.decode_code   │
    └─────────────────┴──────────────────┴──────────────────────────────┴────────────────────────────┘

    The first stage is **always frozen** (``requires_grad=False``, kept in
    ``eval()``).  Only the generator's parameters are trainable.

    Scale factor
    ────────────
    The latent tensor is multiplied by ``scale_factor`` after encoding and
    divided back before decoding — this normalises the latent distribution to
    unit variance, which stabilises diffusion training.

    * ``scale_factor=None`` (default): auto-estimated over the first
      ``scale_steps`` training batches.  A log message is emitted at INFO
      level when estimation starts and when the value locks in.  Common
      values: 0.18215 (SD 1.x VAEs), 0.13025 (SD 2.x / SDXL VAEs).
    * ``scale_factor=<float>``: fixed forever; auto-estimation is disabled.

    Discrete + autoregressive note
    ────────────────────────────────
    When routing through ``discrete + autoregressive``, ``vae_encode`` stores
    the quantisation shape internally as ``self._quant_shape``.  If you call
    ``vae_decode`` before ever calling ``vae_encode``, pass the explicit
    ``out_shape=(B, H, W, C)`` argument to avoid a RuntimeError.

    Quick-start
    ───────────
    ::

        from medlat import get_model, GenWrapper, suggest_generator_params

        tok = get_model("continuous.aekl.f8_d16", img_size=256)
        gen = get_model("dit.xl_2", img_size=256, num_classes=1000,
                        **suggest_generator_params(tok))
        wrapper = GenWrapper(gen, tok)

        z      = wrapper.vae_encode(images)          # encode (no grad)
        loss   = scheduler.training_losses(wrapper, z, model_kwargs=cond)
        recon  = wrapper.vae_decode(z)               # decode (no grad)
    """

    def __init__(
        self,
        generator: Optional[Dict[str, Any]],
        first_stage: Optional[Dict[str, Any]],
        scale_factor: float = None,
        ckpt_path: Optional[str] = None,
        scale_steps: int = 100,
    ):
        super().__init__()
        self.generator = generator
        self.generator_type = get_model_type(generator)
        self.first_stage = first_stage
        self.first_stage_type = get_model_type(first_stage)
        logger.info(f"generator_type: {self.generator_type}, first_stage_type: {self.first_stage_type}")
        self.scale_steps = scale_steps

        if self.first_stage_type == "discrete" and self.generator_type == "non-autoregressive":
            self.fcn_encode = self.first_stage.encode_to_prequant
            self.fcn_decode = self.first_stage.decode_from_prequant

        elif self.first_stage_type == "continuous" and self.generator_type == "non-autoregressive":
            self.fcn_encode = self.first_stage.encode
            self.fcn_decode = self.first_stage.decode

        elif self.first_stage_type == "discrete" and self.generator_type == "autoregressive":
            self.fcn_encode = self.first_stage.encode
            self.fcn_decode = self.first_stage.decode_code

        elif self.first_stage_type == "continuous" and self.generator_type == "autoregressive":
            self.fcn_encode = self.first_stage.encode
            self.fcn_decode = self.first_stage.decode

        else:
            raise ValueError(f"Unsupported combination of generator and first stage types: {self.generator_type} and {self.first_stage_type}")

        self._validate_at_construction()

        # Cross-check that tokenizer and generator are dimensionally compatible.
        # validate_compatibility raises ValueError with an actionable message if not.
        if self.first_stage is not None and self.generator is not None:
            try:
                validate_compatibility(self.first_stage, self.generator)
            except ValueError as exc:
                raise ValueError(
                    f"GenWrapper: incompatible first_stage / generator pair.\n{exc}\n\n"
                    "Tip: use suggest_generator_params(tokenizer) to get the correct kwargs."
                ) from exc

        # Determine if we should do automatic scale_factor estimation
        self._auto_scale_factor = scale_factor is None

        # Register scale factor as buffer
        # If None, initialize to 1.0 (will be updated automatically)
        # If provided, use the given value (will not be updated)
        initial_scale = 1.0 if scale_factor is None else scale_factor
        self.register_buffer("scale_factor", torch.tensor(initial_scale))

        # Track step counter and running statistics for automatic scale_factor determination
        # Only used when _auto_scale_factor is True
        self._scale_step_counter = 0
        self.register_buffer("_running_std_sum", torch.tensor(0.0))
        self.register_buffer("_running_std_count", torch.tensor(0))

        if ckpt_path is not None:
            init_from_ckpt(self, ckpt_path)

        # Freeze first stage
        if self.first_stage is not None:
            for p in self.first_stage.parameters():
                p.requires_grad = False
            self.first_stage.eval()

    # ---------------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------------
    def _validate_at_construction(self):
        fs_type = self.first_stage_type
        gen_type = self.generator_type
        fs_cls = self.first_stage.__class__.__name__
        gen_cls = self.generator.__class__.__name__

        # ── first-stage required attributes ───────────────────────────────
        if fs_type == "continuous":
            if not hasattr(self.first_stage, "embed_dim"):
                raise AttributeError(
                    f"{fs_cls} must expose an 'embed_dim' property "
                    f"(number of latent channels). "
                    f"Ensure it inherits from ContinuousFirstStage."
                )
        elif fs_type == "discrete":
            for attr in ("n_embed", "embed_dim"):
                if not hasattr(self.first_stage, attr):
                    raise AttributeError(
                        f"{fs_cls} must expose a '{attr}' property. "
                        f"Ensure it inherits from DiscreteFirstStage."
                    )

        # ── generator required attributes, with did-you-mean hints ────────
        if gen_type == "non-autoregressive":
            if not hasattr(self.generator, "in_channels"):
                hint = ""
                if hasattr(self.first_stage, "embed_dim"):
                    hint = (
                        f"\n  → Pass in_channels={self.first_stage.embed_dim} "
                        f"to {gen_cls} (matches tokenizer.embed_dim). "
                        f"Tip: use suggest_generator_params(tokenizer) to get this automatically."
                    )
                raise AttributeError(
                    f"{gen_cls} must expose an 'in_channels' attribute "
                    f"(number of latent channels it expects as input).{hint}"
                )
            else:
                # Proactively warn when there is a channel mismatch
                gen_in = self.generator.in_channels
                fs_dim = getattr(self.first_stage, "embed_dim", None)
                if fs_dim is not None and gen_in != fs_dim:
                    raise ValueError(
                        f"Channel mismatch: {gen_cls}.in_channels={gen_in} "
                        f"but {fs_cls}.embed_dim={fs_dim}.\n"
                        f"  → Rebuild the generator with in_channels={fs_dim}, or pass "
                        f"**suggest_generator_params(tokenizer) to get_model()."
                    )

        elif gen_type == "autoregressive" and fs_type == "discrete":
            has_cs = hasattr(self.generator, "codebook_size")
            has_nt = hasattr(self.generator, "num_tokens")
            if not has_cs and not has_nt:
                hint = ""
                if hasattr(self.first_stage, "n_embed"):
                    hint = (
                        f"\n  → Pass codebook_size={self.first_stage.n_embed} (or num_tokens) "
                        f"to {gen_cls} (matches tokenizer.n_embed). "
                        f"Tip: use suggest_generator_params(tokenizer)."
                    )
                raise AttributeError(
                    f"{gen_cls} must expose a 'codebook_size' or 'num_tokens' attribute.{hint}"
                )
            else:
                # Proactively check codebook size matches
                gen_cs = getattr(self.generator, "codebook_size",
                                 getattr(self.generator, "num_tokens", None))
                fs_ne = getattr(self.first_stage, "n_embed", None)
                if gen_cs is not None and fs_ne is not None and gen_cs != fs_ne:
                    raise ValueError(
                        f"Codebook size mismatch: {gen_cls}.codebook_size={gen_cs} "
                        f"but {fs_cls}.n_embed={fs_ne}.\n"
                        f"  → Rebuild the generator with codebook_size={fs_ne}, or pass "
                        f"**suggest_generator_params(tokenizer) to get_model()."
                    )

    # ---------------------------------------------------------------------
    # Training mode handling
    # ---------------------------------------------------------------------
    def train(self, mode: bool = True):
        super().train(mode)
        if self.generator is not None:
            self.generator.train(mode)

        if self.first_stage is not None:
            self.first_stage.eval()
            for p in self.first_stage.parameters():
                p.requires_grad = False
        return self

    # ---------------------------------------------------------------------
    # Scale factor determination
    # ---------------------------------------------------------------------
    def _update_scale_factor(self, quant: torch.Tensor) -> None:
        """
        Automatically determine scale_factor during the first scale_steps steps.
        Updates scale_factor based on the standard deviation of quantized latents.
        After scale_steps steps, the scale_factor remains fixed.
        Only updates if scale_factor was initially None.
        """
        # Only do automatic estimation if scale_factor was None
        if not self._auto_scale_factor:
            return

        if not self.training:
            return  # Don't mutate scale factor in eval mode

        if self._scale_step_counter >= self.scale_steps:
            return  # Already frozen

        if self._scale_step_counter == 0:
            logger.info(
                f"scale_factor auto-estimation started (will run for {self.scale_steps} steps)."
            )
        if self._scale_step_counter == self.scale_steps - 1:
            logger.info(f"Scale factor fixed at {self.scale_factor.item():.6f}")

        with torch.no_grad():
            # Compute standard deviation of quantized latents
            quant_std = quant.std()

            # Accumulate std values using registered buffers
            self._running_std_sum += quant_std
            self._running_std_count += 1
            self._scale_step_counter += 1

            # Compute average std and update scale_factor in-place, fully on device
            avg_std = self._running_std_sum / self._running_std_count
            self.scale_factor.copy_(1.0 / (avg_std + 1e-8))

    # ---------------------------------------------------------------------
    # Encoding
    # ---------------------------------------------------------------------
    def vae_encode(self, image: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if self.first_stage is None:
            return image

        with torch.no_grad():
            quant, loss, info = self.fcn_encode(image)
        self._quant_shape = quant.permute(0, 2, 3, 1).shape  # (B, H, W, C) for indices decoding later

        # Automatically determine scale_factor during the first scale_steps steps
        if self.training:
            self._update_scale_factor(quant)
        quant = quant * self.scale_factor

        # ---- autoregressive path ----
        if info is None:
            return quant
        else:
            _, _, indices = info
            if isinstance(indices, torch.Tensor):  # normal
                return indices.reshape(image.shape[0], -1)
            elif isinstance(indices, (list, tuple)):  # residual quantizer
                indices = [ind.reshape(image.shape[0], -1) for ind in indices]
                return torch.cat(indices, dim=1)
            return quant

    # ---------------------------------------------------------------------
    # Decoding
    # ---------------------------------------------------------------------
    def vae_decode(self, z: torch.Tensor, out_shape=None) -> torch.Tensor:
        if self.first_stage is None:
            return z

        if self.generator_type == "autoregressive" and self.first_stage_type == "discrete":
            if out_shape is None and not hasattr(self, "_quant_shape"):
                raise RuntimeError(
                    "vae_decode() with discrete autoregressive routing requires a prior call to "
                    "vae_encode() to establish the quantization shape. "
                    "Call vae_encode() at least once before vae_decode()."
                )
            with torch.no_grad():
                return self.fcn_decode(z, out_shape=out_shape if out_shape is not None else self._quant_shape)
        else:
            with torch.no_grad():
                return self.fcn_decode(z / self.scale_factor)

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    # Repr
    # ---------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover
        sf_val = self.scale_factor.item()
        sf_mode = "auto" if self._auto_scale_factor else "fixed"
        tok_name = self.first_stage.__class__.__name__ if self.first_stage else "None"
        gen_name = self.generator.__class__.__name__ if self.generator else "None"
        return (
            f"GenWrapper(\n"
            f"  routing      = {self.first_stage_type} + {self.generator_type},\n"
            f"  tokenizer    = {tok_name},\n"
            f"  generator    = {gen_name},\n"
            f"  scale_factor = {sf_val:.6f}  [{sf_mode}],\n"
            f")"
        )

    # ---------------------------------------------------------------------
    # Forward
    # ---------------------------------------------------------------------
    def forward(self, x, *args, **kwargs):
        """Pass latent tokens/features directly to the generator.

        ``x`` is expected to already be in latent space (i.e. the output of
        ``vae_encode``).  This method does *not* call ``vae_encode`` or
        ``vae_decode`` — use those explicitly when you need the full
        encode → generate → decode pipeline.

        Args:
            x: latent input to the generator (e.g. ``(B, N, D)`` tokens or ``(B, C, H, W)`` maps).
            args: positional args forwarded verbatim to ``generator.forward``.
            kwargs: keyword args forwarded verbatim to ``generator.forward``.
        """
        return self.generator.forward(x, *args, **kwargs)
