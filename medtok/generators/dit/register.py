import torch 
import torch.nn as nn
from medtok.registry import register_model
from medtok.generators.dit.models import DiT

@register_model("dit.xl_1")
def DiT_XL_1(dims=2,**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=[1]*dims, num_heads=16, **kwargs)

@register_model("dit.xl_2")
def DiT_XL_2(dims=2,**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=[2]*dims, num_heads=16, **kwargs)

@register_model("dit.xl_4")
def DiT_XL_4(dims=2,**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=[4]*dims, num_heads=16, **kwargs)

@register_model("dit.xl_8")
def DiT_XL_8(dims=2,**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=[8]*dims, num_heads=16, **kwargs)

@register_model("dit.l_1")
def DiT_L_1(dims=2,**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=[1]*dims, num_heads=16, **kwargs)

@register_model("dit.l_2")
def DiT_L_2(dims=2,**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=[2]*dims, num_heads=16, **kwargs)

@register_model("dit.l_4")
def DiT_L_4(dims=2,**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=[4]*dims, num_heads=16, **kwargs)

@register_model("dit.l_8")
def DiT_L_8(dims=2,**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=[8]*dims, num_heads=16, **kwargs)

@register_model("dit.b_1")
def DiT_B_1(dims=2,**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=[1]*dims, num_heads=12, **kwargs)

@register_model("dit.b_2")
def DiT_B_2(dims=2,**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=[2]*dims, num_heads=12, **kwargs)

@register_model("dit.b_4")
def DiT_B_4(dims=2,**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=[4]*dims, num_heads=12, **kwargs)

@register_model("dit.b_8")
def DiT_B_8(dims=2,**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=[8]*dims, num_heads=12, **kwargs)

@register_model("dit.s_1")
def DiT_S_1(dims=2, **kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=[1]*dims, num_heads=6, **kwargs)

@register_model("dit.s_2")
def DiT_S_2(dims=2, **kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=[2]*dims, num_heads=6, **kwargs)

@register_model("dit.s_4")
def DiT_S_4(dims=2, **kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=[4]*dims, num_heads=6, **kwargs)

@register_model("dit.s_8")
def DiT_S_8(dims=2, **kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=[8]*dims, num_heads=6, **kwargs)