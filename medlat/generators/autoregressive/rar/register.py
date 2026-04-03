from medlat.registry import register_model
from .rardiff import RAR_B, RAR_L, RAR_XL, RAR_H

__all__ = []


@register_model("rar.b", paper_url="https://arxiv.org/abs/2411.00776")
def rar_b(**kwargs):
    """Base RAR model (width=768, depth=12, heads=12)."""
    kwargs.setdefault('img_size', (256, 256))
    kwargs.setdefault('patch_size', (1, 1))
    kwargs.setdefault('tokenizer_patch_size', (16, 16))
    kwargs.setdefault('token_channels', 16)
    return RAR_B(**kwargs)


@register_model("rar.l", paper_url="https://arxiv.org/abs/2411.00776")
def rar_l(**kwargs):
    """Large RAR model (width=1024, depth=24, heads=16)."""
    kwargs.setdefault('img_size', (256, 256))
    kwargs.setdefault('patch_size', (1, 1))
    kwargs.setdefault('tokenizer_patch_size', (16, 16))
    kwargs.setdefault('token_channels', 16)
    return RAR_L(**kwargs)


@register_model("rar.xl", paper_url="https://arxiv.org/abs/2411.00776")
def rar_xl(**kwargs):
    """XL RAR model (width=1152, depth=28, heads=16)."""
    kwargs.setdefault('img_size', (256, 256))
    kwargs.setdefault('patch_size', (1, 1))
    kwargs.setdefault('tokenizer_patch_size', (16, 16))
    kwargs.setdefault('token_channels', 16)
    return RAR_XL(**kwargs)


@register_model("rar.h", paper_url="https://arxiv.org/abs/2411.00776")
def rar_h(**kwargs):
    """Huge RAR model (width=1280, depth=32, heads=16)."""
    kwargs.setdefault('img_size', (256, 256))
    kwargs.setdefault('patch_size', (1, 1))
    kwargs.setdefault('tokenizer_patch_size', (16, 16))
    kwargs.setdefault('token_channels', 16)
    return RAR_H(**kwargs)
