#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from medlat import get_model, create_scheduler, DualTimestepScheduler
from medlat.generators.non_autoregressive.self_flow import SelfFlowLoss, patchify

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import STL10
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Subset
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ── Constants ──────────────────────────────────────────────────────────────
IMG_SIZE     = 96    # native STL10 resolution — no resize needed
PATCH_SIZE   = 8     # → (96/8)² = 144 tokens, patch_dim = 8×8×3 = 192
NUM_CLASSES  = 10
BATCH_SIZE   = 64
NUM_EPOCHS   = 1000
NUM_WORKERS  = 4
ODE_STEPS    = 100   # integration steps at FID eval time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

# ── STL10 ──────────────────────────────────────────────────────────────────
# train: 5000 labelled images   test: 8000 labelled images
tfm = transforms.ToTensor()   # STL10 images are already 96×96, values in [0,1]

train_set = STL10(root=".", split="train", transform=tfm, download=True)
test_set  = STL10(root=".", split="test",  transform=tfm, download=True)

train_set = Subset(train_set, np.random.choice(len(train_set), 1000, replace=False))
test_set = Subset(test_set, np.random.choice(len(test_set), 100, replace=False))

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)


print(f"train: {len(train_set)}  |  test: {len(test_set)}")
print(f"tokens per image: {(IMG_SIZE // PATCH_SIZE) ** 2}")


# In[ ]:


# ── Shared utilities ───────────────────────────────────────────────────────

def make_backbone():
    """Same architecture for all three experiments — fair comparison."""
    return get_model(
        "self_flow.b_8",
        input_size=IMG_SIZE,
        in_channels=3,
        num_classes=NUM_CLASSES,
    ).to(device)


# ── Sampler shared by all three models ─────────────────────────────────────
# DualTimestepScheduler.p_sample_loop works for any SelfFlowPerTokenDiT
# regardless of how it was trained (FM, DTS, or SelfFlowLoss).
_sampler = DualTimestepScheduler(patch_size=PATCH_SIZE)

@torch.no_grad()
def sample_images(model_fn, n):
    """Sample n images in batches; return (n, 3, H, W) float tensor in [0,1]."""
    out = []
    for start in range(0, n, BATCH_SIZE):
        bs = min(BATCH_SIZE, n - start)
        y  = torch.randint(0, NUM_CLASSES, (bs,), device=device)
        imgs = _sampler.p_sample_loop(
            model_fn, shape=(bs, 3, IMG_SIZE, IMG_SIZE),
            model_kwargs={"y": y}, sampler="euler", num_steps=ODE_STEPS,
        )
        out.append(imgs.clamp(0, 1).cpu())
    return torch.cat(out, dim=0)


def compute_fid(model_fn):
    """FID of model_fn against the STL10 test set (8000 images)."""
    fid = FrechetInceptionDistance(normalize=True).to(device)

    # Real images
    for imgs, _ in tqdm(test_loader, desc="FID real", leave=False):
        fid.update(imgs.to(device), real=True)

    # Generated images — same count as test set
    n_fake = len(test_set)
    for start in tqdm(range(0, n_fake, BATCH_SIZE), desc="FID fake", leave=False):
        bs   = min(BATCH_SIZE, n_fake - start)
        y    = torch.randint(0, NUM_CLASSES, (bs,), device=device)
        fake = _sampler.p_sample_loop(
            model_fn, shape=(bs, 3, IMG_SIZE, IMG_SIZE),
            model_kwargs={"y": y}, sampler="euler", num_steps=ODE_STEPS,
        )
        fid.update(fake.clamp(0, 1).to(device), real=False)

    return fid.compute().item()


def show_samples(imgs, title, n=8):
    fig, axes = plt.subplots(1, n, figsize=(2 * n, 2))
    for ax, img in zip(axes, imgs[:n]):
        ax.imshow(img.permute(1, 2, 0).cpu().numpy())
        ax.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# accumulate results across cells
results = {}   # name → FID score


# In[ ]:


# ══════════════════════════════════════════════════════════════════════════
# Experiment 1 — Standard OT Flow Matching  (via create_scheduler("flow"))
# ──────────────────────────────────────────────────────────────────────────
# Single timestep t ~ U(0,1) per sample, broadcast to all tokens.
# x_t = (1-t)*noise + t*data  →  velocity target v = data - noise.
# create_scheduler("flow") handles sampling, interpolation, and loss —
# the backbone just needs to accept (x_t, t, y=labels) and return +v_pred.
# ══════════════════════════════════════════════════════════════════════════

model_fm = make_backbone()
opt_fm   = optim.AdamW(model_fm.parameters(), lr=1e-4)
sched_fm = create_scheduler("flow", path_type="Linear", prediction="velocity", loss_weight=None)

n_params = sum(p.numel() for p in model_fm.parameters())
print(f"backbone: {n_params:,} params  |  {model_fm.x_embedder.num_patches} tokens/image")

for epoch in tqdm(range(NUM_EPOCHS), desc="Exp 1 · OT-FM"):
    model_fm.train()
    ep_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        # FlowMatchingScheduler returns per-sample losses → .mean() for backward
        loss = sched_fm.training_losses(model_fm, imgs,
                                        model_kwargs={"y": labels})["loss"].mean()
        loss.backward()
        opt_fm.step()
        opt_fm.zero_grad()
        ep_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"  epoch {epoch+1:3d}/{NUM_EPOCHS} | loss={ep_loss/len(train_loader):.4f}")

# ── Qualitative sample grid ─────────────────────────────────────────────
model_fm.eval()
show_samples(sample_images(model_fm, 8), "Exp 1 · OT-FM samples")

# ── FID ─────────────────────────────────────────────────────────────────
fid_fm = compute_fid(model_fm)
results["1 · OT-FM"] = fid_fm
print(f"\nFID (OT-FM) = {fid_fm:.2f}")


# In[ ]:


# ══════════════════════════════════════════════════════════════════════════
# Experiment 2 — Self-Flow Part 1: Dual-Timestep Scheduling (DTS)
# ──────────────────────────────────────────────────────────────────────────
# Each token independently draws t from one of two ranges:
#   anchor tokens  (75 %)  t ~ U(0.0, 0.4)  — nearly clean
#   masked  tokens (25 %)  t ~ U(0.6, 1.0)  — nearly pure noise
#
# MASKING_STRATEGY controls which tokens get masked:
#   "random"      — uniform random (original Self-Flow)
#   "complexity"  — biased toward high-variance patches; diagnostically
#                   relevant structures (lesions, edges) are masked more
#                   often, forcing the model to reconstruct informative regions
# ══════════════════════════════════════════════════════════════════════════

MASKING_STRATEGY = "complexity"   # ← swap to "random" for standard Self-Flow

model_dts = make_backbone()
opt_dts   = optim.AdamW(model_dts.parameters(), lr=1e-4)

sched_dts = create_scheduler(
    "self_flow",
    patch_size=PATCH_SIZE,
    mask_ratio=0.25,
    t_anchor_range=(0.0, 0.4),
    t_masked_range=(0.6, 1.0),
    masking_strategy=MASKING_STRATEGY,
)
print(f"DTS scheduler — masking_strategy={MASKING_STRATEGY!r}")

for epoch in tqdm(range(NUM_EPOCHS), desc=f"Exp 2 · DTS ({MASKING_STRATEGY})"):
    model_dts.train()
    ep_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        # Returns per-sample (B,) loss — .mean() before backward
        loss = sched_dts.training_losses(model_dts, imgs,
                                         model_kwargs={"y": labels})["loss"].mean()
        loss.backward()
        opt_dts.step()
        opt_dts.zero_grad()
        ep_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"  epoch {epoch+1:3d}/{NUM_EPOCHS} | loss={ep_loss/len(train_loader):.4f}")

# ── Qualitative sample grid ─────────────────────────────────────────────
model_dts.eval()
show_samples(sample_images(model_dts, 8), f"Exp 2 · DTS ({MASKING_STRATEGY}) samples")

# ── FID ─────────────────────────────────────────────────────────────────
fid_dts = compute_fid(model_dts)
results[f"2 · DTS ({MASKING_STRATEGY})"] = fid_dts
print(f"\nFID (DTS {MASKING_STRATEGY}) = {fid_dts:.2f}")


# In[ ]:


# ══════════════════════════════════════════════════════════════════════════
# Experiment 3 — Full Self-Flow: DTS + EMA Teacher Self-Distillation
# ──────────────────────────────────────────────────────────────────────────
# Adds a cosine self-supervised loss at masked (high-noise) token positions.
# An EMA momentum copy of the student provides stable feature targets —
# no external foundation model required.
#
#   Total loss: L = L_flow  +  λ · L_self
#
#   L_self = −mean_{n ∈ masked} cosine_sim(
#                normalize(feat_student[n]),
#                normalize(stop_grad(feat_teacher[n])) )
#
# Sampling uses the EMA teacher (more stable than the student at test time).
# ══════════════════════════════════════════════════════════════════════════

model_sf  = make_backbone()
sched_sf  = DualTimestepScheduler(
    patch_size=PATCH_SIZE, mask_ratio=0.25,
    t_anchor_range=(0.0, 0.4), t_masked_range=(0.6, 1.0),
)
loss_fn   = SelfFlowLoss(
    backbone=model_sf,
    scheduler=sched_sf,
    feature_layer=6,       # ~0.5 × depth; paper uses 20/28 for XL
    ema_decay=0.99,
    self_sup_weight=1.0,
).to(device)
opt_sf    = optim.AdamW(loss_fn.student.parameters(), lr=1e-4)

for epoch in tqdm(range(NUM_EPOCHS), desc="Exp 3 · Self-Flow"):
    loss_fn.student.train()
    ep_flow = ep_self = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        x1  = patchify(imgs, PATCH_SIZE)   # (B, 144, 192)
        out = loss_fn(x1, labels)

        out["loss"].backward()
        opt_sf.step()
        loss_fn.update_ema()               # ← momentum update after every step
        opt_sf.zero_grad()

        ep_flow += out["flow_loss"].item()
        ep_self += out["self_sup_loss"].item()

    if (epoch + 1) % 10 == 0:
        n = len(train_loader)
        print(f"  epoch {epoch+1:3d}/{NUM_EPOCHS} | "
              f"flow={ep_flow/n:.4f}  self_sup={ep_self/n:.4f}")

# ── Qualitative sample grid (EMA teacher) ───────────────────────────────
loss_fn.teacher.eval()
show_samples(sample_images(loss_fn.teacher, 8), "Exp 3 · Self-Flow samples (EMA teacher)")

# ── FID (EMA teacher) ────────────────────────────────────────────────────
fid_sf = compute_fid(loss_fn.teacher)
results["3 · Self-Flow"] = fid_sf
print(f"\nFID (Self-Flow) = {fid_sf:.2f}")


# In[ ]:


# ══════════════════════════════════════════════════════════════════════════
# Results
# ══════════════════════════════════════════════════════════════════════════
print("─" * 35)
print(f"{'Method':<20}  {'FID ↓':>10}")
print("─" * 35)
for name, fid in results.items():
    print(f"{name:<20}  {fid:>10.2f}")
print("─" * 35)

fig, ax = plt.subplots(figsize=(6, 4))
names = list(results.keys())
fids  = [results[n] for n in names]
bars  = ax.bar(names, fids, color=["#4C72B0", "#DD8452", "#55A868"])
ax.bar_label(bars, fmt="%.1f", padding=3)
ax.set_ylabel("FID ↓  (lower is better)")
ax.set_title(f"STL10 · {NUM_EPOCHS} epochs · same backbone (self_flow.b_8)")
ax.set_ylim(0, max(fids) * 1.2)
plt.tight_layout()
plt.show()

