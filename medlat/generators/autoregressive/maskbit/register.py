from medlat.registry import register_model
from .maskbit import LFQBert, Bert

__all__ = []


@register_model("maskbit.s", paper_url="https://arxiv.org/abs/2409.16211")
def maskbit_s(**kwargs):
    """Small MaskBit generator (LFQ-BERT)."""
    return LFQBert(hidden_dim=384, depth=12, heads=6, mlp_dim=1536, **kwargs)


@register_model("maskbit.b", paper_url="https://arxiv.org/abs/2409.16211")
def maskbit_b(**kwargs):
    """Base MaskBit generator (LFQ-BERT)."""
    return LFQBert(hidden_dim=768, depth=24, heads=8, mlp_dim=3072, **kwargs)


@register_model("maskbit.l", paper_url="https://arxiv.org/abs/2409.16211")
def maskbit_l(**kwargs):
    """Large MaskBit generator (LFQ-BERT)."""
    return LFQBert(hidden_dim=1024, depth=24, heads=16, mlp_dim=4096, **kwargs)


@register_model("maskbit.bert_s", paper_url="https://arxiv.org/abs/2409.16211")
def maskbit_bert_s(**kwargs):
    """Small standard BERT generator."""
    return Bert(hidden_dim=384, depth=12, heads=6, mlp_dim=1536, **kwargs)


@register_model("maskbit.bert_b", paper_url="https://arxiv.org/abs/2409.16211")
def maskbit_bert_b(**kwargs):
    """Base standard BERT generator."""
    return Bert(hidden_dim=768, depth=24, heads=8, mlp_dim=3072, **kwargs)


@register_model("maskbit.bert_l", paper_url="https://arxiv.org/abs/2409.16211")
def maskbit_bert_l(**kwargs):
    """Large standard BERT generator."""
    return Bert(hidden_dim=1024, depth=24, heads=16, mlp_dim=4096, **kwargs)
