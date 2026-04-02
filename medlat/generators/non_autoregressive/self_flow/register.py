from medlat.registry import register_model
from .model import SelfFlowPerTokenDiT

__all__ = []

# Shared defaults for all self_flow variants.
# Users can override any of these via get_model() kwargs, e.g.:
#   get_model("self_flow.b_2", input_size=32, in_channels=4, num_classes=1000)
_DEFAULTS = dict(
    input_size=32,
    in_channels=4,
    num_classes=1000,
    learn_sigma=False,
    mlp_ratio=4.0,
)


def _build(depth, hidden_size, num_heads, patch_size, **user_kwargs):
    kwargs = dict(_DEFAULTS)
    kwargs.update(user_kwargs)
    return SelfFlowPerTokenDiT(
        depth=depth,
        hidden_size=hidden_size,
        num_heads=num_heads,
        patch_size=patch_size,
        **kwargs,
    )


@register_model("self_flow.xl_2")
def SelfFlow_XL_2(**kwargs):
    return _build(depth=28, hidden_size=1152, num_heads=16, patch_size=2, **kwargs)


@register_model("self_flow.l_2")
def SelfFlow_L_2(**kwargs):
    return _build(depth=24, hidden_size=1024, num_heads=16, patch_size=2, **kwargs)


@register_model("self_flow.b_2")
def SelfFlow_B_2(**kwargs):
    return _build(depth=12, hidden_size=768, num_heads=12, patch_size=2, **kwargs)


@register_model("self_flow.s_2")
def SelfFlow_S_2(**kwargs):
    return _build(depth=12, hidden_size=384, num_heads=6, patch_size=2, **kwargs)


# patch_size=8 variants — suitable for direct pixel-space training on 64×64 images
# (gives 64 tokens per image instead of 1024, much faster to prototype)

@register_model("self_flow.xl_8")
def SelfFlow_XL_8(**kwargs):
    return _build(depth=28, hidden_size=1152, num_heads=16, patch_size=8, **kwargs)


@register_model("self_flow.l_8")
def SelfFlow_L_8(**kwargs):
    return _build(depth=24, hidden_size=1024, num_heads=16, patch_size=8, **kwargs)


@register_model("self_flow.b_8")
def SelfFlow_B_8(**kwargs):
    return _build(depth=12, hidden_size=768, num_heads=12, patch_size=8, **kwargs)


@register_model("self_flow.s_8")
def SelfFlow_S_8(**kwargs):
    return _build(depth=12, hidden_size=384, num_heads=6, patch_size=8, **kwargs)
