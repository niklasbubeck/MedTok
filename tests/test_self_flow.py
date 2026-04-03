"""Tests for DualTimestepScheduler (self_flow)."""

import pytest
import torch


BATCH = 2
N = 16   # token count (must be compatible with patch_size below)
D = 8    # token feature dim
C, H, W = 3, 8, 8   # spatial: patch_size=2 → (H/2)*(W/2)=16 tokens


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scheduler(**kwargs):
    from medlat.scheduling.self_flow import DualTimestepScheduler
    return DualTimestepScheduler(patch_size=2, **kwargs)


# ---------------------------------------------------------------------------
# Basic construction
# ---------------------------------------------------------------------------

def test_constructs_defaults():
    s = _scheduler()
    assert s.masking_strategy == "random"


def test_constructs_complexity_strategy():
    s = _scheduler(masking_strategy="complexity")
    assert s.masking_strategy == "complexity"


# ---------------------------------------------------------------------------
# sample_per_token_timesteps — random strategy
# ---------------------------------------------------------------------------

def test_random_timesteps_shape():
    s = _scheduler(mask_ratio=0.25)
    t, mask = s.sample_per_token_timesteps(BATCH, N, torch.device("cpu"))
    assert t.shape == (BATCH, N)
    assert mask.shape == (BATCH, N)


def test_random_mask_ratio():
    s = _scheduler(mask_ratio=0.25)
    _, mask = s.sample_per_token_timesteps(BATCH, N, torch.device("cpu"))
    n_masked = mask.sum(dim=1)
    expected = max(1, int(0.25 * N))
    assert (n_masked == expected).all(), f"Expected {expected} masked tokens, got {n_masked}"


def test_random_timesteps_in_range():
    s = _scheduler(mask_ratio=0.5, t_anchor_range=(0.0, 0.4), t_masked_range=(0.6, 1.0))
    t, mask = s.sample_per_token_timesteps(BATCH, N, torch.device("cpu"))
    anchor_t = t[~mask]
    masked_t = t[mask]
    assert (anchor_t >= 0.0).all() and (anchor_t <= 0.4).all(), "Anchor t out of range"
    assert (masked_t >= 0.6).all() and (masked_t <= 1.0).all(), "Masked t out of range"


# ---------------------------------------------------------------------------
# sample_per_token_timesteps — complexity strategy
# ---------------------------------------------------------------------------

def test_complexity_timesteps_shape():
    s = _scheduler(mask_ratio=0.25, masking_strategy="complexity")
    imgs = torch.randn(BATCH, C, H, W)
    t, mask = s.sample_per_token_timesteps(BATCH, N, torch.device("cpu"), imgs=imgs)
    assert t.shape == (BATCH, N)
    assert mask.shape == (BATCH, N)


def test_complexity_mask_ratio():
    s = _scheduler(mask_ratio=0.5, masking_strategy="complexity")
    imgs = torch.randn(BATCH, C, H, W)
    _, mask = s.sample_per_token_timesteps(BATCH, N, torch.device("cpu"), imgs=imgs)
    n_masked = mask.sum(dim=1)
    expected = max(1, int(0.5 * N))
    assert (n_masked == expected).all()


def test_complexity_falls_back_to_random_without_imgs():
    """complexity strategy must fall back to random when imgs is None."""
    s = _scheduler(mask_ratio=0.25, masking_strategy="complexity")
    t, mask = s.sample_per_token_timesteps(BATCH, N, torch.device("cpu"), imgs=None)
    assert t.shape == (BATCH, N)


def test_complexity_degenerate_uniform_image():
    """Uniform-variance image (all-same pixels) must not cause multinomial to crash."""
    s = _scheduler(mask_ratio=0.5, masking_strategy="complexity")
    imgs = torch.zeros(BATCH, C, H, W)  # all-zero → all patches have zero variance
    # The +1e-6 floor in _complexity_masked_idx should make this safe
    t, mask = s.sample_per_token_timesteps(BATCH, N, torch.device("cpu"), imgs=imgs)
    assert t.shape == (BATCH, N)


# ---------------------------------------------------------------------------
# Spatial input auto-patchify in training_losses
# ---------------------------------------------------------------------------

def test_training_losses_spatial_input():
    """Spatial (B,C,H,W) input is patchified internally; loss is (B,) shaped."""
    s = _scheduler(mask_ratio=0.25)

    class _DummyModel(torch.nn.Module):
        def forward(self, x_t, t, y, **kw):
            return torch.zeros_like(x_t)

    model = _DummyModel()
    x_start = torch.randn(BATCH, C, H, W)
    out = s.training_losses(model, x_start, model_kwargs={"y": torch.zeros(BATCH, dtype=torch.long)})
    assert "loss" in out
    assert out["loss"].shape == (BATCH,)


def test_training_losses_token_input():
    """Token (B,N,D) input passes through without patchify."""
    s = _scheduler(mask_ratio=0.25)

    class _DummyModel(torch.nn.Module):
        def forward(self, x_t, t, y, **kw):
            return torch.zeros_like(x_t)

    model = _DummyModel()
    x_start = torch.randn(BATCH, N, D)
    out = s.training_losses(model, x_start, model_kwargs={"y": torch.zeros(BATCH, dtype=torch.long)})
    assert out["loss"].shape == (BATCH,)
