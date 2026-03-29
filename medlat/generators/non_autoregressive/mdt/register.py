from medlat.registry import register_model
from .models import (
    MDTv2_XL_2, MDTv2_L_2, MDTv2_B_2, MDTv2_S_2,
    MDTv2_XL_4, MDTv2_L_4, MDTv2_B_4, MDTv2_S_4,
)

__all__ = []


@register_model("mdt.xl_2")
def mdt_xl_2(**kwargs):
    return MDTv2_XL_2(**kwargs)


@register_model("mdt.l_2")
def mdt_l_2(**kwargs):
    return MDTv2_L_2(**kwargs)


@register_model("mdt.b_2")
def mdt_b_2(**kwargs):
    return MDTv2_B_2(**kwargs)


@register_model("mdt.s_2")
def mdt_s_2(**kwargs):
    return MDTv2_S_2(**kwargs)


@register_model("mdt.xl_4")
def mdt_xl_4(**kwargs):
    return MDTv2_XL_4(**kwargs)


@register_model("mdt.l_4")
def mdt_l_4(**kwargs):
    return MDTv2_L_4(**kwargs)


@register_model("mdt.b_4")
def mdt_b_4(**kwargs):
    return MDTv2_B_4(**kwargs)


@register_model("mdt.s_4")
def mdt_s_4(**kwargs):
    return MDTv2_S_4(**kwargs)
