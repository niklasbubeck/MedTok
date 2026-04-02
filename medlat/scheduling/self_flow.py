"""
medlat.scheduling.self_flow
============================
Self-Flow scheduler: dual-timestep flow matching with per-token timesteps.

Part 1 — DualTimestepScheduler
    Each token draws an independent t ∈ [0,1].  Anchor tokens are nearly
    clean (low t), masked tokens are nearly pure noise (high t).  The
    resulting information asymmetry drives implicit representation learning
    without any external objective.

Part 2 — SelfFlowLoss
    For the full EMA teacher self-distillation objective see
    medlat.generators.non_autoregressive.self_flow.model.SelfFlowLoss.

Reference: Chefer et al., ICML 2025, https://arxiv.org/abs/2603.06507
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from einops import rearrange

from .base import BaseScheduler


class DualTimestepScheduler(BaseScheduler):
    """Flow matching with per-token timesteps (linear OT path).

    Implements Part 1 of Self-Flow: dual-timestep scheduling (DTS).
    Drops in as a replacement for FlowMatchingScheduler — the training
    loop and GenWrapper are unchanged.

    Token groups
    ────────────
    Each sample's N tokens are split randomly into:
        anchor  (1 − mask_ratio)   t ~ U(t_anchor_range)   low noise
        masked  (mask_ratio)       t ~ U(t_masked_range)   high noise

    Sign convention
    ───────────────
    SelfFlowPerTokenDiT.forward() returns −v_pred.  training_losses corrects
    for this via negate_target=True (default), flipping the MSE target from
    x1−x0 to x0−x1.

    Spatial ↔ token conversion
    ──────────────────────────
    The backbone expects patchified tokens (B, N, patch_dim).  If x_start
    is a spatial latent (B, C, H, W), the scheduler handles patchify /
    unpatchify internally via patch_size, so the GenWrapper training loop
    needs no changes.
    """

    def __init__(
        self,
        patch_size: int = 2,
        mask_ratio: float = 0.25,
        t_anchor_range: Tuple[float, float] = (0.0, 0.4),
        t_masked_range: Tuple[float, float] = (0.6, 1.0),
        use_lognorm: bool = False,
    ):
        """
        Args:
            patch_size:      spatial patch size used to patchify/unpatchify
                             (B,C,H,W) ↔ (B,N,patch_dim).
            mask_ratio:      fraction of tokens that are masked (high-noise).
            t_anchor_range:  [lo, hi] for anchor token timesteps (low noise).
            t_masked_range:  [lo, hi] for masked token timesteps (high noise).
            use_lognorm:     sample t from logit-normal rather than uniform.
        """
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.t_anchor_range = t_anchor_range
        self.t_masked_range = t_masked_range
        self.use_lognorm = use_lognorm

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
        self, B: int, N: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample independent per-token timesteps; return (t, mask).

        Returns:
            t    (B, N) float — per-token timestep in [0, 1].
            mask (B, N) bool  — True at masked (high-noise) positions.
        """
        n_masked = max(1, int(self.mask_ratio * N))
        n_anchor = N - n_masked

        perm = torch.argsort(torch.rand(B, N, device=device), dim=1)
        anchor_idx = perm[:, :n_anchor]
        masked_idx = perm[:, n_anchor:]

        lo_a, hi_a = self.t_anchor_range
        lo_m, hi_m = self.t_masked_range

        t = torch.empty(B, N, device=device)
        t.scatter_(1, anchor_idx, self._sample_t((B, n_anchor), lo_a, hi_a, device))
        t.scatter_(1, masked_idx, self._sample_t((B, n_masked), lo_m, hi_m, device))

        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        mask.scatter_(1, masked_idx, True)
        return t, mask

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
                          Spatial inputs are patchified automatically.
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

        spatial_shape: Optional[Tuple[int, int, int]] = None
        if x_start.ndim == 4:
            B, C, H, W = x_start.shape
            spatial_shape = (C, H, W)
            x1 = self._patchify(x_start)
        else:
            x1 = x_start

        B, N, _ = x1.shape
        device = x1.device

        x0 = torch.randn_like(x1) if noise is None else noise
        t_tok, mask = self.sample_per_token_timesteps(B, N, device)

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
        """ODE integration from t=1 (pure noise) to t=0 (clean data).

        Args:
            model:        trained SelfFlowPerTokenDiT (or callable).
            shape:        output shape — spatial (B,C,H,W) or token (B,N,D).
                          Spatial inputs are converted to tokens internally.
            noise:        optional starting noise (spatial or token).
            model_kwargs: forwarded to model; must contain ``"y"`` (class labels).
            device:       inferred from model if None.
            progress:     log progress every 10% of steps.
            sampler:      "euler" (1 call/step) or "heun" (2 calls/step).
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

        # Integrate the ODE from t=0 (noise) → t=1 (data).
        # dx/dt = v_pred  →  x_{t+dt} = x_t + dt * v_pred  (Euler)
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
