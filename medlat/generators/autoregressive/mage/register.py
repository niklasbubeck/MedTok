from medlat.registry import register_model
from .mage import (
    MAGE_ViT_XS_4, MAGE_ViT_XS_8, MAGE_ViT_XS_16,
    MAGE_ViT_S_4,  MAGE_ViT_S_8,  MAGE_ViT_S_16,
    MAGE_ViT_B_4,  MAGE_ViT_B_8,  MAGE_ViT_B_16,
    MAGE_ViT_L_4,  MAGE_ViT_L_8,  MAGE_ViT_L_16,
)

__all__ = []


@register_model("mage.xs_4")
def mage_xs_4(**kwargs):
    return MAGE_ViT_XS_4(**kwargs)


@register_model("mage.xs_8")
def mage_xs_8(**kwargs):
    return MAGE_ViT_XS_8(**kwargs)


@register_model("mage.xs_16")
def mage_xs_16(**kwargs):
    return MAGE_ViT_XS_16(**kwargs)


@register_model("mage.s_4")
def mage_s_4(**kwargs):
    return MAGE_ViT_S_4(**kwargs)


@register_model("mage.s_8")
def mage_s_8(**kwargs):
    return MAGE_ViT_S_8(**kwargs)


@register_model("mage.s_16")
def mage_s_16(**kwargs):
    return MAGE_ViT_S_16(**kwargs)


@register_model("mage.b_4")
def mage_b_4(**kwargs):
    return MAGE_ViT_B_4(**kwargs)


@register_model("mage.b_8")
def mage_b_8(**kwargs):
    return MAGE_ViT_B_8(**kwargs)


@register_model("mage.b_16")
def mage_b_16(**kwargs):
    return MAGE_ViT_B_16(**kwargs)


@register_model("mage.l_4")
def mage_l_4(**kwargs):
    return MAGE_ViT_L_4(**kwargs)


@register_model("mage.l_8")
def mage_l_8(**kwargs):
    return MAGE_ViT_L_8(**kwargs)


@register_model("mage.l_16")
def mage_l_16(**kwargs):
    return MAGE_ViT_L_16(**kwargs)
