from .maetok import MaskAEModel
from medtok.registry import register_model

@register_model("token.maetok.b_128",
                code_url="https://github.com/Hhhhhhao/continuous_tokenizer",
                paper_url="https://arxiv.org/pdf/2502.03444",)
def MAETok_B_128(image_size: int = 256, 
                 base_image_size: int = 256,
                 patch_size: int = 16,
                 codebook_embed_dim: int = 32,
                 num_latent_tokens: int = 128,
                 aux_hog_dec: bool = True,
                 aux_dino_dec: bool = True,
                 aux_clip_dec: bool = True,
                 aux_biomed_clip_dec: bool = False,
                 **kwargs):
    return MaskAEModel(image_size=image_size, base_image_size=base_image_size, enc_patch_size=patch_size, dec_patch_size=patch_size, codebook_embed_dim=codebook_embed_dim, num_latent_tokens=num_latent_tokens, aux_hog_dec=aux_hog_dec, aux_dino_dec=aux_dino_dec, aux_clip_dec=aux_clip_dec, aux_biomed_clip_dec=aux_biomed_clip_dec, **kwargs)