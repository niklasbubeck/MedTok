"""
medlat.scheduling.self_flow
============================
Self-Flow scheduler: dual-timestep flow matching with per-token timesteps.

Part 1 — DualTimestepScheduler
    Each sample's N tokens are split into anchor (low noise) and masked
    (high noise) groups.  The masking strategy controls which tokens are
    masked:

    ``"random"``      Uniform random selection (original Self-Flow).
    ``"complexity"``  Bias toward high-variance patches — tokens whose
                      spatial content has more local variation are more
                      likely to be masked, forcing the model to reconstruct
                      informative regions.  Particularly beneficial for
                      medical images where diagnostically relevant structures
                      (lesions, edges) concentrate in high-frequency patches.

Part 2 — SelfFlowLoss
    For the full EMA teacher self-distillation objective see
    medlat.generators.non_autoregressive.self_flow.model.SelfFlowLoss.

Reference: Chefer et al., ICML 2025, https://arxiv.org/abs/2603.06507
"""
from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple

import torch
from einops import rearrange

from .base import BaseScheduler


class DualTimestepScheduler(BaseScheduler):
    """Flow matching with per-token timesteps (linear OT path).

    Implements Part 1 of Self-Flow: dual-timestep scheduling (DTS).
    Drops in as a replacement for FlowMatchingScheduler — the training
    loop and GenWrapper are unchanged.

    **Token groups** — each sample's N tokens are split into anchor tokens
    ``(1 − mask_ratio)`` sampled at low noise ``t ~ U(t_anchor_range)``, and
    masked tokens ``(mask_ratio)`` sampled at high noise ``t ~ U(t_masked_range)``.

    The ``masking_strategy`` controls which tokens are masked:
    ``"random"`` uses uniform random selection (standard Self-Flow behaviour);
    ``"complexity"`` weights by per-patch variance so high-frequency patches are
    preferentially masked (falls back to random when spatial input is unavailable).

    **Spatial ↔ token conversion** — if ``x_start`` is a spatial latent
    ``(B, C, H, W)``, the scheduler patchifies and unpatchifies internally using
    ``patch_size``, so the GenWrapper training loop needs no changes.
    """

    def __init__(
        self,
        patch_size: int = 2,
        mask_ratio: float = 0.25,
        t_anchor_range: Tuple[float, float] = (0.0, 0.4),
        t_masked_range: Tuple[float, float] = (0.6, 1.0),
        use_lognorm: bool = False,
        masking_strategy: Literal["random", "complexity"] = "random",
    ):
        """
        Args:
            patch_size:        spatial patch size for patchify/unpatchify.
            mask_ratio:        fraction of tokens that are masked (high-noise).
            t_anchor_range:    [lo, hi] for anchor token timesteps (low noise).
            t_masked_range:    [lo, hi] for masked token timesteps (high noise).
            use_lognorm:       sample t from logit-normal rather than uniform.
            masking_strategy:  ``"random"`` (default) or ``"complexity"``.
                               With ``"complexity"``, tokens are sampled
                               proportional to their per-patch variance so
                               high-information regions are masked more often.
                               Requires ``x_start`` to be a spatial tensor
                               (B, C, H, W) in ``training_losses``; falls back
                               to random when only token input (B, N, D) is
                               provided.  Existing code using positional args
                               is unaffected — ``masking_strategy`` is
                               keyword-only with default ``"random"``.
        """
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.t_anchor_range = t_anchor_range
        self.t_masked_range = t_masked_range
        self.use_lognorm = use_lognorm
        self.masking_strategy = masking_strategy

    # ------------------------------------------------------------------
    # Spatial ↔ token helpers
    # ------------------------------------------------------------------

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, N, patch_dim)  with N = (H/p)·(W/p)."""
        p = self.patch_size
        return rearrange(x, "b c (gh p) (gw q) -> b (gh gw) (p q c)", p=p, q=p)

    def _unpatchify(self, tokens: torch.Tensor, C: int, H: int, W: int) -> torch.Tensor:
        """(B, N, patch_dim) → (B, C, H, W)."""
        p = self.patch_size
        return rearrange(
            tokens, "b (gh gw) (p q c) -> b c (gh p) (gw q)",
            gh=H // p, gw=W // p, p=p, q=p, c=C,
        )

    # ------------------------------------------------------------------
    # Masking strategies
    # ------------------------------------------------------------------

    def _random_masked_idx(self, B: int, N: int, n_masked: int, device: torch.device) -> torch.Tensor:
        """Uniform random selection — returns (B, n_masked) indices."""
        perm = torch.argsort(torch.rand(B, N, device=device), dim=1)
        return perm[:, N - n_masked:]

    def _complexity_masked_idx(
        self, imgs: torch.Tensor, n_masked: int
    ) -> torch.Tensor:
        """Variance-weighted selection — returns (B, n_masked) indices.

        Each token is scored by its per-patch variance across all pixels and
        channels.  Tokens are then sampled *without replacement* proportional
        to these scores, so high-variance (diagnostically informative) patches
        are masked more often.

        Args:
            imgs:     spatial input (B, C, H, W) — used only for scoring.
            n_masked: number of tokens to select per sample.

        Returns:
            (B, n_masked) long tensor of masked token indices.
        """
        patches = self._patchify(imgs)               # (B, N, patch_dim)
        scores = patches.var(dim=-1).float()         # (B, N) — variance per token
        scores = scores + 1e-6                       # avoid zero weights
        return torch.multinomial(scores, num_samples=n_masked, replacement=False)

    # ------------------------------------------------------------------
    # Timestep sampling
    # ------------------------------------------------------------------

    def _sample_t(
        self, shape: Tuple[int, ...], lo: float, hi: float, device: torch.device
    ) -> torch.Tensor:
        if self.use_lognorm:
            t = torch.sigmoid(torch.randn(shape, device=device))
        else:
            t = torch.rand(shape, device=device)
        return t * (hi - lo) + lo

    def sample_per_token_timesteps(
        self,
        B: int,
        N: int,
        device: torch.device,
        imgs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample independent per-token timesteps; return (t, mask).

        Args:
            B:     batch size.
            N:     number of tokens per sample.
            device: target device.
            imgs:  spatial input (B, C, H, W) required for
                   ``masking_strategy="complexity"``.  Ignored for ``"random"``.

        Returns:
            t    (B, N) float — per-token timestep in [0, 1].
            mask (B, N) bool  — True at masked (high-noise) positions.
        """
        n_masked = max(1, int(self.mask_ratio * N))
        n_anchor = N - n_masked

        if self.masking_strategy == "complexity" and imgs is not None:
            masked_idx = self._complexity_masked_idx(imgs, n_masked)
        else:
            masked_idx = self._random_masked_idx(B, N, n_masked, device)

        # anchor indices = all tokens not in masked_idx
        # build via complement: create full index then remove masked ones
        all_idx = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)  # (B, N)
        mask_bool = torch.zeros(B, N, dtype=torch.bool, device=device)
        mask_bool.scatter_(1, masked_idx.to(device), True)
        anchor_idx = all_idx[~mask_bool].reshape(B, n_anchor)

        lo_a, hi_a = self.t_anchor_range
        lo_m, hi_m = self.t_masked_range

        t = torch.empty(B, N, device=device)
        t.scatter_(1, anchor_idx, self._sample_t((B, n_anchor), lo_a, hi_a, device))
        t.scatter_(1, masked_idx.to(device), self._sample_t((B, n_masked), lo_m, hi_m, device))

        return t, mask_bool

    # ------------------------------------------------------------------
    # Training interface — BaseScheduler
    # ------------------------------------------------------------------

    def training_losses(
        self,
        model: Any,
        x_start: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """Flow-matching loss with per-token timesteps.

        Args:
            model:        SelfFlowPerTokenDiT or compatible callable.
            x_start:      clean data; (B, C, H, W) spatial or (B, N, D) tokens.
                          Spatial inputs are patchified automatically, and when
                          ``masking_strategy="complexity"`` the raw spatial values
                          are used to score patch informativeness.
            t:            ignored — per-token timesteps are sampled internally.
            noise:        optional pre-drawn N(0,I) noise; sampled if None.
            model_kwargs: forwarded to the model.  Must contain key ``"y"``
                          (integer class labels, shape (B,)).

        Returns:
            dict with keys:
                "loss"  — (B,) per-sample flow MSE (call .mean() for backward).
                "mse"   — (B,) per-sample MSE (same as "loss").
                "t"     — (B, N) per-token timesteps.
                "mask"  — (B, N) bool, True at masked positions.
        """
        if model_kwargs is None:
            model_kwargs = {}
        y = model_kwargs.get("y")
        extra = {k: v for k, v in model_kwargs.items() if k != "y"}

        # Keep raw spatial tensor for complexity scoring before patchifying
        imgs_spatial: Optional[torch.Tensor] = None
        if x_start.ndim == 4:
            imgs_spatial = x_start
            x1 = self._patchify(x_start)
        else:
            x1 = x_start

        B, N, _ = x1.shape
        device = x1.device

        x0 = torch.randn_like(x1) if noise is None else noise
        t_tok, mask = self.sample_per_token_timesteps(B, N, device, imgs=imgs_spatial)

        t_exp = t_tok.unsqueeze(-1)
        x_t = t_exp * x1 + (1.0 - t_exp) * x0
        ut = x1 - x0           # OT velocity target; model returns +v_pred

        v = model(x_t, t_tok, y, **extra)
        mse = ((v - ut) ** 2).mean(dim=(1, 2))
        return {"loss": mse, "mse": mse, "t": t_tok, "mask": mask}

    # ------------------------------------------------------------------
    # Inference interface — BaseScheduler
    # ------------------------------------------------------------------

    def p_sample_loop(
        self,
        model: Any,
        shape: Tuple,
        noise: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict] = None,
        device: Optional[torch.device] = None,
        progress: bool = False,
        sampler: str = "heun",
        **sampler_kwargs,
    ) -> torch.Tensor:
        """ODE integration from t=0 (noise) to t=1 (data).

        Args:
            model:        trained SelfFlowPerTokenDiT (or callable).
            shape:        output shape — spatial (B,C,H,W) or token (B,N,D).
            noise:        optional starting noise (spatial or token).
            model_kwargs: forwarded to model; must contain ``"y"`` (class labels).
            device:       inferred from model if None.
            progress:     log progress every 10 % of steps.
            sampler:      ``"euler"`` (1 call/step) or ``"heun"`` (2 calls/step).
            **sampler_kwargs: ``num_steps`` (default 50).

        Returns:
            Tensor in the same domain as ``shape``.
        """
        if model_kwargs is None:
            model_kwargs = {}
        y = model_kwargs.get("y")
        extra = {k: v for k, v in model_kwargs.items() if k != "y"}
        num_steps = sampler_kwargs.get("num_steps", 50)

        spatial_shape: Optional[Tuple[int, int, int]] = None
        if len(shape) == 4:
            B, C, H, W = shape
            spatial_shape = (C, H, W)
            p = self.patch_size
            token_shape = (B, (H // p) * (W // p), p * p * C)
        else:
            B = shape[0]
            token_shape = tuple(shape)

        if device is None:
            try:
                device = next(model.parameters()).device
            except (StopIteration, AttributeError):
                device = y.device if y is not None else torch.device("cpu")

        if noise is None:
            x = torch.randn(*token_shape, device=device)
        elif noise.ndim == 4:
            x = self._patchify(noise.to(device))
        else:
            x = noise.to(device)

        if y is not None:
            y = y.to(device)

        dt = 1.0 / num_steps

        for i in range(num_steps):
            t_val = i * dt
            t_batch = torch.full((B,), t_val, device=device)

            with torch.no_grad():
                v = model(x, t_batch, y, **extra)

            if sampler == "heun" and i < num_steps - 1:
                t_next = torch.full((B,), min(1.0, t_val + dt), device=device)
                x_pred = x + dt * v
                with torch.no_grad():
                    v2 = model(x_pred, t_next, y, **extra)
                x = x + dt * (v + v2) / 2.0
            else:
                x = x + dt * v

            if progress and (i % max(1, num_steps // 10) == 0 or i == num_steps - 1):
                print(f"  [{i + 1:>4}/{num_steps}] t={t_val:.3f}")

        if spatial_shape is not None:
            C, H, W = spatial_shape
            x = self._unpatchify(x, C, H, W)

        return x


def create_dual_timestep_scheduler(**kwargs) -> DualTimestepScheduler:
    """Factory; called by create_scheduler('self_flow', ...)."""
    return DualTimestepScheduler(**kwargs)
