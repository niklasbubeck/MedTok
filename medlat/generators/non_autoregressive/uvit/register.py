from medlat.registry import register_model
from .models import (
    U_ViT_Small, U_ViT_Small_Deep, U_ViT_Mid, U_ViT_Large, U_ViT_Huge,
)

__all__ = []


@register_model("uvit.small")
def uvit_small(**kwargs):
    return U_ViT_Small(**kwargs)


@register_model("uvit.small_deep")
def uvit_small_deep(**kwargs):
    return U_ViT_Small_Deep(**kwargs)


@register_model("uvit.mid")
def uvit_mid(**kwargs):
    return U_ViT_Mid(**kwargs)


@register_model("uvit.large")
def uvit_large(**kwargs):
    return U_ViT_Large(**kwargs)


@register_model("uvit.huge")
def uvit_huge(**kwargs):
    return U_ViT_Huge(**kwargs)
