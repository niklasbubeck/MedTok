import torch
import torch.nn as nn
from medlat.registry import register_model
from .transformer import MaskGIT
from functools import partial

__all__ = []


@register_model("maskgit.b", paper_url="https://arxiv.org/abs/2202.04200")
def MaskGIT_B(**kwargs):
    model = MaskGIT(
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

@register_model("maskgit.l", paper_url="https://arxiv.org/abs/2202.04200")
def MaskGIT_L(**kwargs):
    model = MaskGIT(
        embed_dim=1024, depth=16, num_heads=16, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

@register_model("maskgit.h", paper_url="https://arxiv.org/abs/2202.04200")
def MaskGIT_H(**kwargs):
    model = MaskGIT(
        embed_dim=1280, depth=20, num_heads=16, mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model