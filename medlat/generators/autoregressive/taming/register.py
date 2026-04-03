from medlat.registry import register_model
from .transformer import GPT_B, GPT_L, GPT_H

__all__ = []


@register_model("taming.gpt_b", paper_url="https://arxiv.org/abs/2012.09841")
def taming_gpt_b(**kwargs):
    """Base Taming Transformer GPT (n_embd=768, n_layer=12, n_head=12).
    Requires: vocab_size, block_size.
    """
    return GPT_B(**kwargs)


@register_model("taming.gpt_l", paper_url="https://arxiv.org/abs/2012.09841")
def taming_gpt_l(**kwargs):
    """Large Taming Transformer GPT (n_embd=1024, n_layer=16, n_head=16).
    Requires: vocab_size, block_size.
    """
    return GPT_L(**kwargs)


@register_model("taming.gpt_h", paper_url="https://arxiv.org/abs/2012.09841")
def taming_gpt_h(**kwargs):
    """Huge Taming Transformer GPT (n_embd=1280, n_layer=20, n_head=16).
    Requires: vocab_size, block_size.
    """
    return GPT_H(**kwargs)
