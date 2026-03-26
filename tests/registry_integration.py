"""
Comprehensive test suite for all medlat registry models.

Run with:  python tests/registry_integration.py
            (not collected by pytest — use for full registry / forward-pass checks)

Covers:
  1. Registry presence  – every registered name is queryable
  2. Instantiation      – one representative per model family
  3. Forward pass       – key families with small synthetic inputs

Skip policies:
  - adm.*         : return config dicts, not nn.Module
  - fractal.*     : model architecture too large for CPU forward test (32-block, embed_dim=1024)
  - token.softvq.* / token.maetok.* alignment heads : download DINO/CLIP → network
"""

import math
import sys
import os
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import medlat
from medlat import get_model, available_models

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"

_results = {"pass": 0, "fail": 0, "skip": 0}


def run(name, fn, skip=False, skip_reason=""):
    if skip:
        print(f"  {SKIP}  {name}  ({skip_reason})")
        _results["skip"] += 1
        return
    try:
        fn()
        print(f"  {PASS}  {name}")
        _results["pass"] += 1
    except Exception:
        print(f"  {FAIL}  {name}")
        traceback.print_exc()
        _results["fail"] += 1


# ---------------------------------------------------------------------------
# 1. Registry presence — every model we know about should be registered
# ---------------------------------------------------------------------------

EXPECTED_MODELS = [
    # Continuous VAE
    "continuous.aekl.f4_d3", "continuous.aekl.f4_d8", "continuous.aekl.f4_d16", "continuous.aekl.f4_d32",
    "continuous.aekl.f8_d4", "continuous.aekl.f8_d8", "continuous.aekl.f8_d16", "continuous.aekl.f8_d32",
    "continuous.aekl.f16_d8", "continuous.aekl.f16_d16", "continuous.aekl.f16_d32", "continuous.aekl.f16_d64",
    "continuous.aekl.f32_d64",
    "continuous.medvae.f8_d16", "continuous.medvae.f8_d32",
    "continuous.vavae.f8_d16_dinov2", "continuous.vavae.f8_d32_dinov2",
    "continuous.vavae.f16_d16_mae", "continuous.vavae.f16_d32_mae", "continuous.vavae.f16_d64_mae",
    "continuous.vavae.f16_d16_dinov2", "continuous.vavae.f16_d32_dinov2", "continuous.vavae.f16_d64_dinov2",
    # Discrete VQ
    "discrete.vq.f4_d3_e8192", "discrete.vq.f4_d8_e8192", "discrete.vq.f4_d16_e8192", "discrete.vq.f4_d32_e8192",
    "discrete.vq.f8_d4_e16384",
    # Discrete quantizers
    "discrete.quantizer.vector_quantizer", "discrete.quantizer.vector_quantizer2",
    "discrete.quantizer.lookup_free_quantizer", "discrete.quantizer.binary_spherical_quantizer",
    "discrete.quantizer.finite_scalar_quantizer", "discrete.quantizer.soft_vector_quantizer",
    "discrete.quantizer.residual_quantizer", "discrete.quantizer.grouped_residual_quantizer",
    # Token models
    "token.titok.s_128", "token.titok.s_128_e2e",
    "token.maetok.s_128", "token.maetok.b_128_p16",
    "token.softvq.s_t32_d32", "token.softvq.b_t32_d32",
    "token.detok.ss", "token.detok.sb", "token.detok.bb", "token.detok.ll",
    # Generators – non-autoregressive
    "dit.s_1", "dit.s_2", "dit.s_4", "dit.s_8",
    "dit.b_1", "dit.b_2", "dit.b_4", "dit.b_8",
    "dit.l_1", "dit.l_2", "dit.l_4", "dit.l_8",
    "dit.xl_1", "dit.xl_2", "dit.xl_4", "dit.xl_8",
    "mdt.xl_2", "mdt.l_2", "mdt.b_2", "mdt.s_2",
    "mdt.xl_4", "mdt.l_4", "mdt.b_4", "mdt.s_4",
    "uvit.small", "uvit.small_deep", "uvit.mid", "uvit.large", "uvit.huge",
    "adm.diffusion.64c",  # registry normalizes to lowercase
    # Generators – autoregressive
    "mar.b", "mar.l", "mar.h",
    "maskgit.b", "maskgit.l", "maskgit.h",
    "mage.xs_16", "mage.b_8",
    "maskbit.s", "maskbit.b", "maskbit.l", "maskbit.bert_b",
    "rar.b", "rar.l", "rar.xl", "rar.h",
    "taming.gpt_b", "taming.gpt_l", "taming.gpt_h",
    "fractal.ar_64", "fractal.mar_64", "fractal.mar_base_256",
]

print("\n=== 1. Registry presence ===")
all_names = set(available_models())
for name in EXPECTED_MODELS:
    run(f"registry:{name}", lambda n=name: (
        None if n in all_names else (_ for _ in ()).throw(AssertionError(f"'{n}' not in registry"))
    ))

print(f"\n  (total registered: {len(all_names)} models)")


# ---------------------------------------------------------------------------
# 2. Instantiation
# ---------------------------------------------------------------------------

print("\n=== 2. Instantiation ===")

# ---- Continuous VAE ----
run("continuous.aekl.f4_d3",
    lambda: get_model("continuous.aekl.f4_d3", img_size=32))
run("continuous.aekl.f8_d4",
    lambda: get_model("continuous.aekl.f8_d4", img_size=64))
run("continuous.medvae.f8_d16",
    lambda: get_model("continuous.medvae.f8_d16", img_size=64))
run("continuous.vavae.f8_d16_dinov2",
    lambda: get_model("continuous.vavae.f8_d16_dinov2", img_size=64))
run("continuous.maisi.f4_d3",
    lambda: get_model("continuous.maisi.f4_d3"))

# ---- Discrete VQ ----
run("discrete.vq.f4_d3_e8192",
    lambda: get_model("discrete.vq.f4_d3_e8192", img_size=32))
run("discrete.vq.f8_d4_e16384",
    lambda: get_model("discrete.vq.f8_d4_e16384", img_size=64,
                      attn_resolutions=[]))  # skip attn for small size

# ---- Discrete quantizers (standalone, registered as classes) ----
run("discrete.quantizer.vector_quantizer2",
    lambda: get_model("discrete.quantizer.vector_quantizer2", n_e=512, e_dim=4, beta=0.25))
run("discrete.quantizer.lookup_free_quantizer",
    lambda: get_model("discrete.quantizer.lookup_free_quantizer"))
run("discrete.quantizer.binary_spherical_quantizer",
    lambda: get_model("discrete.quantizer.binary_spherical_quantizer"))
run("discrete.quantizer.finite_scalar_quantizer",
    lambda: get_model("discrete.quantizer.finite_scalar_quantizer"))
run("discrete.quantizer.soft_vector_quantizer",
    lambda: get_model("discrete.quantizer.soft_vector_quantizer", n_e=512, e_dim=4))
run("discrete.quantizer.residual_quantizer",
    lambda: get_model("discrete.quantizer.residual_quantizer", n_e=512, e_dim=4, beta=0.25))

# ---- adm.* returns a config dict, not a nn.Module ----
def _adm_is_dict():
    result = get_model("adm.diffusion.64C")
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"

run("adm.diffusion.64C (config dict)", _adm_is_dict)

# ---- Token models ----
run("token.detok.ss",
    lambda: get_model("token.detok.ss", img_size=32))
run("token.detok.sb",
    lambda: get_model("token.detok.sb", img_size=32))
run("token.titok.s_128",
    lambda: get_model("token.titok.s_128", img_size=32))
run("token.titok.s_128_e2e",
    lambda: get_model("token.titok.s_128_e2e", img_size=32))
# MAETok and SoftVQ: disable aux decoders that download DINOv2/CLIP
run("token.maetok.s_128",
    lambda: get_model("token.maetok.s_128", img_size=64,
                      aux_hog_dec=False, aux_dino_dec=False, aux_clip_dec=False))
run("token.softvq.s_t32_d32",
    lambda: get_model("token.softvq.s_t32_d32", img_size=64),
    skip=True, skip_reason="DinoAlignment downloads DINOv2 weights at init")

# ---- Generator – DIT ----
run("dit.s_1", lambda: get_model("dit.s_1", img_size=32, vae_stride=4, num_classes=10))
run("dit.s_2", lambda: get_model("dit.s_2", img_size=32, vae_stride=4, num_classes=10))
run("dit.b_2", lambda: get_model("dit.b_2", img_size=32, vae_stride=4, num_classes=10))
run("dit.xl_2", lambda: get_model("dit.xl_2", img_size=32, vae_stride=4, num_classes=10))

# ---- Generator – MAR ----
run("mar.b", lambda: get_model("mar.b"))
run("mar.l", lambda: get_model("mar.l"))

# ---- Generator – MaskGIT ----
run("maskgit.b", lambda: get_model("maskgit.b", num_tokens=512, seq_len=16, num_classes=10))
run("maskgit.l", lambda: get_model("maskgit.l", num_tokens=512, seq_len=16, num_classes=10))

# ---- Generator – MAGE ----
run("mage.xs_16", lambda: get_model("mage.xs_16", num_tokens=512, img_size=64))
run("mage.b_8",   lambda: get_model("mage.b_8",   num_tokens=1024, img_size=256))

# ---- Generator – MaskBit ----
run("maskbit.s",
    lambda: get_model("maskbit.s", img_size=64, input_stride=8, codebook_size=512, nclass=10))
run("maskbit.b",   lambda: get_model("maskbit.b"))
run("maskbit.bert_b", lambda: get_model("maskbit.bert_b"))

# ---- Generator – RAR ----
run("rar.b", lambda: get_model("rar.b"))
run("rar.l", lambda: get_model("rar.l"))

# ---- Generator – Taming ----
run("taming.gpt_b", lambda: get_model("taming.gpt_b", vocab_size=512, block_size=64))
run("taming.gpt_l", lambda: get_model("taming.gpt_l", vocab_size=512, block_size=64))
run("taming.gpt_h", lambda: get_model("taming.gpt_h", vocab_size=512, block_size=64))

# ---- Generator – MDT ----
run("mdt.s_2",  lambda: get_model("mdt.s_2",  input_size=8, num_classes=10))
run("mdt.b_2",  lambda: get_model("mdt.b_2",  input_size=8, num_classes=10))
run("mdt.xl_2", lambda: get_model("mdt.xl_2", input_size=8, num_classes=10))
run("mdt.s_4",  lambda: get_model("mdt.s_4",  input_size=8, num_classes=10))

# ---- Generator – UViT ----
run("uvit.small",      lambda: get_model("uvit.small",      img_size=32, patch_size=4))
run("uvit.small_deep", lambda: get_model("uvit.small_deep", img_size=32, patch_size=4))
run("uvit.mid",        lambda: get_model("uvit.mid",        img_size=32, patch_size=4))

# ---- Generator – Fractal ----
run("fractal.ar_64",        lambda: get_model("fractal.ar_64"))
run("fractal.mar_64",       lambda: get_model("fractal.mar_64"))
run("fractal.mar_base_256", lambda: get_model("fractal.mar_base_256"))


# ---------------------------------------------------------------------------
# 3. Forward pass
# ---------------------------------------------------------------------------

print("\n=== 3. Forward pass ===")


# ---- Continuous VAE (AutoencoderKL): forward(x) -> (dec, loss) ----
def _continuous_vae_forward(model_name, img_size=32, z_channels=3, in_channels=3, **model_kwargs):
    B = 2
    model = get_model(model_name, img_size=img_size, **model_kwargs)
    model.eval()
    x = torch.randn(B, in_channels, img_size, img_size)
    with torch.no_grad():
        dec, loss = model(x)
    assert dec.shape == (B, in_channels, img_size, img_size), f"Bad shape {dec.shape}"

run("continuous.aekl.f4_d3 forward",
    lambda: _continuous_vae_forward("continuous.aekl.f4_d3", img_size=32, z_channels=3))
run("continuous.aekl.f8_d4 forward",
    lambda: _continuous_vae_forward("continuous.aekl.f8_d4", img_size=64, z_channels=4))
run("continuous.aekl.f16_d8 forward",
    lambda: _continuous_vae_forward("continuous.aekl.f16_d8", img_size=64, z_channels=8,
                                    attn_resolutions=[]))  # disable attn to avoid size mismatch


# ---- Discrete VQ (VQModel): forward(x) -> (dec, diff) ----
def _discrete_vq_forward(model_name, img_size=32, in_channels=3, **model_kwargs):
    B = 2
    model = get_model(model_name, img_size=img_size, **model_kwargs)
    model.eval()
    x = torch.randn(B, in_channels, img_size, img_size)
    with torch.no_grad():
        out = model(x)
    dec = out[0]
    assert dec.shape == (B, in_channels, img_size, img_size), f"Bad shape {dec.shape}"

run("discrete.vq.f4_d3_e8192 forward",
    lambda: _discrete_vq_forward("discrete.vq.f4_d3_e8192", img_size=32))
run("discrete.vq.f4_d8_e8192 forward",
    lambda: _discrete_vq_forward("discrete.vq.f4_d8_e8192", img_size=32))


# ---- Discrete quantizer: forward(z) -> (z_q, loss, info) ----
def _quantizer_forward():
    n_e, e_dim, B, H, W = 512, 4, 2, 8, 8
    model = get_model("discrete.quantizer.vector_quantizer2", n_e=n_e, e_dim=e_dim, beta=0.25)
    model.eval()
    z = torch.randn(B, e_dim, H, W)
    with torch.no_grad():
        z_q, loss, info = model(z)
    assert z_q.shape == (B, e_dim, H, W), f"Bad shape {z_q.shape}"

run("discrete.quantizer.vector_quantizer2 forward", _quantizer_forward)


# ---- Token DeTok: forward(x) -> (decoded, kl_loss) ----
def _detok_forward(model_name, img_size=32, in_channels=3):
    B = 2
    model = get_model(model_name, img_size=img_size)
    model.eval()
    x = torch.randn(B, in_channels, img_size, img_size)
    with torch.no_grad():
        decoded, kl_loss = model(x)
    assert decoded.shape == (B, in_channels, img_size, img_size), f"Bad shape {decoded.shape}"

run("token.detok.ss forward", lambda: _detok_forward("token.detok.ss", img_size=32))
run("token.detok.sb forward", lambda: _detok_forward("token.detok.sb", img_size=32))


# ---- Token MAETok (returns VQModel): forward(x) -> (dec, diff) ----
def _maetok_forward():
    B, img_size, in_channels = 2, 64, 3
    model = get_model("token.maetok.s_128", img_size=img_size,
                      aux_hog_dec=False, aux_dino_dec=False, aux_clip_dec=False)
    model.eval()
    x = torch.randn(B, in_channels, img_size, img_size)
    with torch.no_grad():
        out = model(x)
    dec = out[0]
    assert dec.shape[0] == B, f"Bad batch dim {dec.shape}"

run("token.maetok.s_128 forward", _maetok_forward)


# ---- TiTok: forward(x) -> (rec, loss_dict) ----
def _titok_forward():
    B, img_size, in_channels = 2, 32, 3
    model = get_model("token.titok.s_128", img_size=img_size)
    model.eval()
    x = torch.randn(B, in_channels, img_size, img_size)
    with torch.no_grad():
        out = model(x)
    # TiTok returns (rec, loss_dict) or similar
    assert out[0].shape[0] == B, f"Bad batch dim {out[0].shape}"

run("token.titok.s_128 forward", _titok_forward)


# ---- DIT: forward(x, t, y) -> out ----
def _dit_forward(model_name, patch_size=2):
    img_size, vae_stride, in_channels, num_classes, B = 32, 4, 4, 10, 2
    model = get_model(model_name, img_size=img_size, vae_stride=vae_stride,
                      in_channels=in_channels, num_classes=num_classes, learn_sigma=False)
    model.eval()
    latent_h = img_size // vae_stride  # 8
    x = torch.randn(B, in_channels, latent_h, latent_h)
    t = torch.randint(0, 1000, (B,))
    y = torch.randint(0, num_classes, (B,))
    with torch.no_grad():
        out = model(x, t, y)
    assert out.shape == (B, in_channels, latent_h, latent_h), f"Bad shape {out.shape}"

run("dit.s_1 forward", lambda: _dit_forward("dit.s_1", patch_size=1))
run("dit.s_2 forward", lambda: _dit_forward("dit.s_2", patch_size=2))
run("dit.b_2 forward", lambda: _dit_forward("dit.b_2", patch_size=2))


# ---- MaskGIT: forward_encoder(x, y) -> logits ----
def _maskgit_forward():
    num_tokens, seq_len, num_classes, B = 512, 16, 10, 2
    model = get_model("maskgit.b", num_tokens=num_tokens, seq_len=seq_len, num_classes=num_classes)
    model.eval()
    x = torch.randint(0, num_tokens, (B, seq_len))
    y = torch.randint(0, num_classes, (B,))
    with torch.no_grad():
        x_out, gt_indices, token_drop_mask, token_all_mask = model.forward_encoder(x, y)
    assert x_out.shape[0] == B, f"Bad batch dim {x_out.shape}"

run("maskgit.b forward_encoder", _maskgit_forward)


# ---- MAR forward ----
def _mar_forward():
    img_size, vae_stride, in_channels, num_classes, B = 16, 4, 4, 10, 2
    model = get_model("mar.b", img_size=img_size, vae_stride=vae_stride,
                      in_channels=in_channels, class_num=num_classes,
                      diffloss_d=2, diffloss_w=64)
    model.eval()
    x = torch.randn(B, in_channels, img_size // vae_stride, img_size // vae_stride)
    y = torch.randint(0, num_classes, (B,))
    with torch.no_grad():
        loss = model(x, y)
    assert loss.ndim == 0, f"Expected scalar loss, got shape {loss.shape}"

run("mar.b forward", _mar_forward)


# ---- MDT forward (from existing suite) ----
def _mdt_forward(model_name):
    input_size, in_channels, num_classes, B = 8, 4, 10, 2
    model = get_model(model_name, input_size=input_size,
                      in_channels=in_channels, num_classes=num_classes, learn_sigma=False)
    model.eval()
    x = torch.randn(B, in_channels, input_size, input_size)
    t = torch.randint(0, 1000, (B,))
    y = torch.randint(0, num_classes, (B,))
    with torch.no_grad():
        out = model(x, t, y)
    assert out.shape == (B, in_channels, input_size, input_size), f"Bad shape {out.shape}"

run("mdt.s_2 forward",  lambda: _mdt_forward("mdt.s_2"))
run("mdt.b_2 forward",  lambda: _mdt_forward("mdt.b_2"))
run("mdt.s_4 forward",  lambda: _mdt_forward("mdt.s_4"))
run("mdt.b_4 forward",  lambda: _mdt_forward("mdt.b_4"))


# ---- UViT forward (from existing suite) ----
def _uvit_forward(model_name):
    img_size, patch_size, in_chans, num_classes, B = 32, 4, 3, 10, 2
    model = get_model(model_name, img_size=img_size, patch_size=patch_size,
                      in_chans=in_chans, num_classes=num_classes)
    model.eval()
    x = torch.randn(B, in_chans, img_size, img_size)
    t = torch.randint(0, 1000, (B,))
    y = torch.randint(0, num_classes, (B,))
    with torch.no_grad():
        out = model(x, t, y)
    assert out.shape == (B, in_chans, img_size, img_size), f"Bad shape {out.shape}"

run("uvit.small forward",      lambda: _uvit_forward("uvit.small"))
run("uvit.small_deep forward", lambda: _uvit_forward("uvit.small_deep"))
run("uvit.mid forward",        lambda: _uvit_forward("uvit.mid"))


# ---- MaskBit forward ----
def _lfqbert_forward(model_name, **init_kwargs):
    img_size      = init_kwargs.get("img_size", 64)
    input_stride  = init_kwargs.get("input_stride", 8)
    codebook_size = init_kwargs.get("codebook_size", 512)
    splits        = init_kwargs.get("codebook_splits", 1)
    nclass        = init_kwargs.get("nclass", 10)
    B = 2
    seq_len = (img_size // input_stride) ** 2
    eff_size = int(2 ** (int(math.log2(codebook_size)) // splits))
    model = get_model(model_name, **init_kwargs)
    model.eval()
    img_tokens   = torch.randint(0, eff_size, (B, seq_len, splits))
    class_labels = torch.randint(0, nclass, (B,))
    drop_mask    = torch.zeros(B, dtype=torch.bool)
    with torch.no_grad():
        logits = model(img_tokens, class_labels, drop_mask)
    assert logits.shape[0] == B and logits.shape[1] == seq_len, f"Bad shape {logits.shape}"

_mkw = dict(img_size=64, input_stride=8, codebook_size=512, codebook_splits=1, nclass=10)
run("maskbit.s forward", lambda: _lfqbert_forward("maskbit.s", **_mkw))
run("maskbit.b forward", lambda: _lfqbert_forward("maskbit.b", **_mkw))
run("maskbit.l forward", lambda: _lfqbert_forward("maskbit.l", **_mkw))

def _bert_forward():
    img_size, input_stride, cs, splits, nclass, B = 64, 8, 512, 1, 10, 2
    seq_len = (img_size // input_stride) ** 2
    model = get_model("maskbit.bert_b", img_size=img_size, input_stride=input_stride,
                      codebook_size=cs, codebook_splits=splits, nclass=nclass)
    model.eval()
    img_tokens   = torch.randint(0, cs, (B, seq_len, splits))
    class_labels = torch.randint(0, nclass, (B,))
    drop_mask    = torch.zeros(B, dtype=torch.bool)
    with torch.no_grad():
        logits = model(img_tokens, class_labels, drop_mask)
    assert logits.shape[0] == B and logits.shape[1] == seq_len, f"Bad shape {logits.shape}"

run("maskbit.bert_b forward", _bert_forward)


# ---- Taming GPT forward ----
def _taming_forward(model_name):
    vocab_size, block_size, B, T = 512, 64, 2, 32
    model = get_model(model_name, vocab_size=vocab_size, block_size=block_size)
    model.eval()
    idx     = torch.randint(0, vocab_size, (B, T))
    targets = torch.randint(0, vocab_size, (B, T))
    with torch.no_grad():
        logits, loss = model(idx, targets=targets)
    assert logits.shape == (B, T, vocab_size), f"Bad logits shape {logits.shape}"
    assert loss is not None

run("taming.gpt_b forward", lambda: _taming_forward("taming.gpt_b"))
run("taming.gpt_l forward", lambda: _taming_forward("taming.gpt_l"))
run("taming.gpt_h forward", lambda: _taming_forward("taming.gpt_h"))


# ---- MAGE forward ----
def _mage_forward():
    num_tokens, img_size, B = 512, 32, 2
    model = get_model("mage.xs_16", num_tokens=num_tokens, img_size=img_size)
    model.eval()
    seq_len = (img_size // 16) ** 2  # patch_size=16 for xs_16
    imgs = torch.randint(0, num_tokens, (B, seq_len))
    with torch.no_grad():
        loss, _, _ = model(imgs)
    assert loss.ndim == 0, f"Expected scalar loss, got {loss.shape}"

run("mage.xs_16 forward", _mage_forward)


# ---- RAR forward ----
def _rar_forward():
    # Use force_one_d_seq to bypass PatchEmbed's img_size assertion and feed
    # a pre-flattened token sequence directly (as done at inference time).
    B, num_classes, seq_len, token_channels = 2, 10, 8, 8
    model = get_model("rar.b",
                      token_channels=token_channels,
                      num_classes=num_classes, diffloss_d=2, diffloss_w=64,
                      force_one_d_seq=seq_len)
    model.eval()
    # token_embed_dim = token_channels * prod(patch_size=(1,1)) = token_channels
    x = torch.randn(B, seq_len, token_channels)
    labels = torch.randint(0, num_classes, (B,))
    with torch.no_grad():
        loss = model(x, labels)
    assert loss.ndim == 0, f"Expected scalar loss, got {loss.shape}"

run("rar.b forward", _rar_forward)


# ---- Fractal forward (skip: 32-block, embed_dim=1024 — too slow on CPU) ----
for name in ["fractal.ar_64", "fractal.mar_64"]:
    run(f"{name} forward", lambda: None,
        skip=True, skip_reason="32-block embed_dim=1024 architecture too slow for CPU forward test")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

total = sum(_results.values())
print(f"\n=== Summary: {_results['pass']}/{total} passed, "
      f"{_results['fail']} failed, {_results['skip']} skipped ===\n")

if _results["fail"] > 0:
    raise SystemExit(1)
