import torch 
import torch.nn as nn 
import torch.nn.functional as F
from timm import create_model

from .modules import MAETokViTEncoder, MAETokViTDecoder, HOGGenerator
from src.registry import register_model

__all__ = ['MAETok_B_128', 'MaskAEModel']

class Normalize(nn.Module):
    def __init__(self, mean, std, device=None):
        super(Normalize, self).__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mean = torch.tensor(mean).view(1, -1, 1, 1).to(device)
        self.std = torch.tensor(std).view(1, -1, 1, 1).to(device)

    def forward(self, x):
        return (x - self.mean) / self.std

class Denormalize(nn.Module):
    def __init__(self, mean, std, device=None):
        super(Denormalize, self).__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mean = torch.tensor(mean).view(1, -1, 1, 1).to(device)
        self.std = torch.tensor(std).view(1, -1, 1, 1).to(device)

    def forward(self, x):
        return x * self.std + self.mean

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

class MaskAEModel(nn.Module):
    def __init__(self, 
                 image_size: int = 256,
                 base_image_size: int = 256,
                 codebook_embed_dim: int = 32,
                 dec_use_movq: bool = False,
                 num_latent_tokens: int = 128,
                 to_pixel: str = 'linear',
                 enc_embed_dim: int = 768,
                 dec_embed_dim: int = 768,
                 enc_depth: int = 12,
                 dec_depth: int = 12,
                 enc_num_heads: int = 12,
                 dec_num_heads: int = 12,
                 enc_mlp_ratio: float = 4.,
                 dec_mlp_ratio: float = 4.,
                 enc_patch_size: int = 16,
                 dec_patch_size: int = 16,
                 enc_drop_path_rate: float = 0.0,
                 dec_drop_path_rate: float = 0.05,
                 dec_cls_token: bool = False,
                 use_ape: bool = False,
                 use_rope: bool = True,
                 rope_mixed: bool = True,
                 rope_theta: float = 10.0,
                 repa_patch_size: int = 16,
                 repa_model: str = 'vit_large_patch14_dinov2.lvd142m',
                 repa_proj_dim: int = 1024,
                 enc_token_drop: float = 0.4,
                 enc_token_drop_max: float = 0.6,
                 aux_dec_model: str = 'vit_tinytiny_patch14_dinov2_movq2',
                 aux_loss_mask: bool = True,
                 aux_dec_cls_token: bool = True,
                 aux_hog_dec: bool = True,
                 aux_dino_dec: bool = True,
                 aux_clip_dec: bool = True,
                 aux_biomed_clip_dec: bool = False,
                 ):
        super().__init__()
        self.image_size = image_size
        self.base_image_size = base_image_size
        self.dec_use_movq = dec_use_movq
        self.codebook_embed_dim = codebook_embed_dim
        self.num_latent_tokens = num_latent_tokens
        self.to_pixel = to_pixel
        self.enc_embed_dim = enc_embed_dim
        self.dec_embed_dim = dec_embed_dim
        self.enc_depth = enc_depth
        self.dec_depth = dec_depth
        self.enc_num_heads = enc_num_heads
        self.dec_num_heads = dec_num_heads
        self.enc_mlp_ratio = enc_mlp_ratio
        self.dec_mlp_ratio = dec_mlp_ratio
        self.enc_patch_size = enc_patch_size
        self.dec_patch_size = dec_patch_size
        self.enc_drop_path_rate = enc_drop_path_rate
        self.dec_drop_path_rate = dec_drop_path_rate
        self.dec_cls_token = dec_cls_token
        self.use_ape = use_ape
        self.use_rope = use_rope
        self.rope_mixed = rope_mixed
        self.rope_theta = rope_theta
        self.repa_patch_size = repa_patch_size
        self.repa_model = repa_model
        self.repa_proj_dim = repa_proj_dim
        self.enc_token_drop = enc_token_drop
        self.enc_token_drop_max = enc_token_drop_max
        self.aux_dec_model = aux_dec_model
        self.aux_loss_mask = aux_loss_mask
        self.aux_dec_cls_token = aux_dec_cls_token
        self.aux_hog_dec = aux_hog_dec
        self.aux_dino_dec = aux_dino_dec
        self.aux_clip_dec = aux_clip_dec
        self.aux_biomed_clip_dec = aux_biomed_clip_dec
        # Initialize the repa model for DINO
        if self.aux_dino_dec:
            self.repa_model = create_model(self.repa_model, pretrained=True, img_size=self.image_size, patch_size=self.repa_patch_size)
            for param in self.repa_model.parameters():
                param.requires_grad = False
            self.repa_model.eval()
            self.scale = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.de_scale = Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.encoder = MAETokViTEncoder(
            in_channels=3,
            img_size=self.image_size,
            patch_size=self.enc_patch_size,
            embed_dim=self.enc_embed_dim,
            depth=self.enc_depth,
            num_heads=self.enc_num_heads,
            mlp_ratio=self.enc_mlp_ratio,
            num_latent_tokens=self.num_latent_tokens,
            use_ape=self.use_ape, use_rope=self.use_rope, rope_mixed=self.rope_mixed, rope_theta=self.rope_theta,
            token_drop=self.enc_token_drop,
            token_drop_max=self.enc_token_drop_max,
            base_img_size=self.base_image_size,
            drop_path_rate=self.enc_drop_path_rate
        )

        self.decoder = MAETokViTDecoder(
            in_channels=3,
            img_size=self.image_size,
            patch_size=self.dec_patch_size,
            embed_dim=self.dec_embed_dim,
            depth=self.dec_depth,
            num_heads=self.dec_num_heads,
            mlp_ratio=self.dec_mlp_ratio,
            num_latent_tokens=self.num_latent_tokens,
            use_ape=self.use_ape, use_rope=self.use_rope, rope_mixed=self.rope_mixed, rope_theta=self.rope_theta,
            cls_token=self.dec_cls_token,
            codebook_embed_dim=self.codebook_embed_dim,
            to_pixel=self.to_pixel,
            base_img_size=self.base_image_size,
            use_movq=self.dec_use_movq,
            drop_path_rate=self.dec_drop_path_rate
        )           
        # Add missing quant_conv layer
        self.quant_conv = nn.Linear(self.encoder.embed_dim, self.codebook_embed_dim)
        
        # Add missing post_quant_conv for main decoder
        self.post_quant_conv = nn.Linear(self.codebook_embed_dim, self.decoder.embed_dim)
        
        self.aux_hog_decoder = self.aux_hog_dec
        # if self.aux_hog_decoder:
        #     print('Using HOG decoder: ', self.aux_dec_model)
        #     self.decoder_hog = MAETokViTDecoder(
        #         in_channels=3, 
        #         num_latent_tokens=self.num_latent_tokens,
        #         model_name=self.aux_dec_model,
        #         model_kwargs={'img_size': self.image_size, 'patch_size': self.dec_patch_size, 'drop_path_rate': 0.0, 'latent_dim': self.codebook_embed_dim},
        #         pretrained=False,
        #         tuning_method='full',
        #         tuning_kwargs={'r': 8},
        #         use_ape=self.use_ape, use_rope=self.use_rope, rope_mixed=self.rope_mixed, rope_theta=self.rope_theta,
        #         cls_token=self.aux_dec_cls_token,
        #         codebook_embed_dim=self.codebook_embed_dim,
        #         to_pixel='identity',
        #         base_img_size=self.base_image_size
        #     )
        #     self.post_quant_conv_hog = nn.Linear(self.codebook_embed_dim, self.decoder_hog.embed_dim)
        #     self.to_pixel_hog = nn.Linear(self.decoder_hog.embed_dim, 108)
        #     self.hog_generator = HOGGenerator()
        #     if 'movq' in self.aux_dec_model:
        #         self.hog_use_movq = True 
        #     else:
        #         self.hog_use_movq = False
        
        self.aux_dino_decoder = self.aux_dino_dec
        # if self.aux_dino_decoder:
        #     print('Using DINO decoder: ', self.aux_dec_model)
        #     self.decoder_dino = MAETokViTDecoder(
        #         in_channels=3, 
        #         num_latent_tokens=self.num_latent_tokens,
        #         model_name=self.aux_dec_model,
        #         model_kwargs={'img_size': self.repa_model.img_size, 'patch_size': self.repa_model.patch_size, 'drop_path_rate': 0.0, 'latent_dim': self.codebook_embed_dim},
        #         pretrained=False,
        #         tuning_method='full',
        #         tuning_kwargs={'r': 8},
        #         use_ape=self.use_ape, use_rope=self.use_rope, rope_mixed=self.rope_mixed, rope_theta=self.rope_theta,
        #         cls_token=self.aux_dec_cls_token,
        #         codebook_embed_dim=self.codebook_embed_dim,
        #         to_pixel='identity',
        #         base_img_size=self.base_image_size
        #     )
        #     self.post_quant_conv_dino = nn.Linear(self.codebook_embed_dim, self.decoder_dino.embed_dim)
        #     self.to_pixel_dino = nn.Linear(self.decoder_dino.embed_dim, self.repa_model.embed_dim)
        #     if 'movq' in self.aux_dec_model:
        #         self.dino_use_movq = True 
        #     else:
        #         self.dino_use_movq = False
        
        self.aux_clip_decoder = self.aux_clip_dec
        # if self.aux_clip_decoder:
        #     self.clip_model = create_model('vit_so400m_patch14_siglip_gap_224', pretrained=True, img_size=self.image_size, patch_size=self.repa_patch_size)
        #     for param in self.clip_model.parameters():
        #         param.requires_grad = False
        #     self.clip_model.dynamic_img_size = True
        #     self.clip_model.eval()
        #     self.clip_de_scale = Denormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        #     self.clip_scale = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
        #     print('Using CLIP decoder: ', self.aux_dec_model)
        #     self.decoder_clip = MAETokViTDecoder(
        #         in_channels=3, 
        #         num_latent_tokens=self.num_latent_tokens,
        #         model_name=self.aux_dec_model,
        #         model_kwargs={'img_size': self.clip_model.img_size, 'patch_size': self.clip_model.patch_size, 'drop_path_rate': 0.0, 'latent_dim': self.codebook_embed_dim},
        #         pretrained=False,
        #         tuning_method='full',
        #         tuning_kwargs={'r': 8},
        #         use_ape=self.use_ape, use_rope=self.use_rope, rope_mixed=self.rope_mixed, rope_theta=self.rope_theta,
        #         cls_token=self.aux_dec_cls_token,
        #         codebook_embed_dim=self.codebook_embed_dim,
        #         to_pixel='identity',
        #         base_img_size=self.base_image_size
        #     )
        #     self.post_quant_conv_clip = nn.Linear(self.codebook_embed_dim, self.decoder_clip.embed_dim)
        #     self.to_pixel_clip = nn.Linear(self.decoder_clip.embed_dim, self.clip_model.embed_dim)
        #     if 'movq' in self.aux_dec_model:
        #         self.clip_use_movq = True 
        #     else:
        #         self.clip_use_movq = False
            
        self.aux_biomed_clip_decoder = self.aux_biomed_clip_dec
        # if self.aux_biomed_clip_decoder:
        #     self.biomed_clip_model, _, _ = open_clip.create_model_and_transforms(
        #     model_name="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        # )
        #     for param in self.biomed_clip_model.parameters():
        #         param.requires_grad = False
        #     self.biomed_clip_model.dynamic_img_size = True
        #     self.biomed_clip_model.eval()

        #     print('Using Biomed CLIP decoder: ', self.aux_dec_model)
        #     self.decoder_biomed_clip = MAETokViTDecoder(
        #         in_channels=3,
        #         img_size=[224,352],
        #         patch_size=repa_patch_size,
        #         embed_dim=384,
        #         depth=6,
        #         num_heads=6,
        #         mlp_ratio=4.,
        #         num_latent_tokens=self.num_latent_tokens,
        #         use_ape=self.use_ape, use_rope=self.use_rope, rope_mixed=self.rope_mixed, rope_theta=self.rope_theta,
        #         cls_token=self.aux_dec_cls_token,
        #         codebook_embed_dim=self.codebook_embed_dim,
        #         to_pixel='identity',
        #         base_img_size=self.base_image_size,
        #         use_movq=self.dec_use_movq,
        #     )           
        #     self.post_quant_conv_clip = nn.Linear(self.codebook_embed_dim, self.decoder_biomed_clip.embed_dim)
        #     self.to_pixel_clip = nn.Linear(self.decoder_biomed_clip.embed_dim, self.biomed_clip_model.embed_dim)



    def encode(self, x):
        if self.training:
            h, mask = self.encoder(x, return_mask=True)
        else:
            h = self.encoder(x)
        quant = self.quant_conv(h)
        emb_loss = torch.tensor(0.)
        info = None
        if self.training:
            return quant, emb_loss, info, mask
        else:
            return quant, emb_loss, info

    def decode_hog(self, quant, x=None, h=None, w=None):
        tmp_quant = quant 
        quant = self.post_quant_conv_hog(quant)
        if self.hog_use_movq:
            dec = self.decoder_hog(quant, tmp_quant, h, w)
        else:
            dec = self.decoder_hog(quant, None, h, w)
        return dec
    
    def decode_dino(self, quant, x=None, h=None, w=None):
        tmp_quant = quant 
        quant = self.post_quant_conv_dino(quant)
        if self.dino_use_movq:
            dec = self.decoder_dino(quant, tmp_quant, h, w)
        else:
            dec = self.decoder_dino(quant, None, h, w)
        return dec

    def decode_clip(self, quant, x=None, h=None, w=None):
        tmp_quant = quant 
        quant = self.post_quant_conv_clip(quant)
        if self.clip_use_movq:
            dec = self.decoder_clip(quant, tmp_quant, h, w)
        else:
            dec = self.decoder_clip(quant, None, h, w)
        return dec

    def decode(self, quant, x=None, h=None, w=None):
        tmp_quant = quant 
        quant = self.post_quant_conv(quant)
        if self.dec_use_movq:
            dec = self.decoder(quant, tmp_quant, h, w)
        else:
            dec = self.decoder(quant, None, h, w)
        return dec

    def forward(self, input):
        b, _, h, w, *_  = input.size()
        if self.training:
            quant, diff, info, mask = self.encode(input)
        else:
            quant, diff, info = self.encode(input)
        self.quant = quant
        dec = self.decode(quant, x=input)
        # decode hog
        if self.training:
            # decode hog feature
            if self.aux_hog_decoder:
                dec_hog = self.decode_hog(quant, x=input, h=h, w=w)   
                dec_hog = self.to_pixel_hog(dec_hog)
                # get hog_target
                z_hog = self.hog_generator(input) 
                if self.aux_loss_mask:
                    hog_rec_loss = F.mse_loss(dec_hog, z_hog, reduction='none')
                    hog_rec_loss = (hog_rec_loss * mask).sum() / mask.sum() / z_hog.size(-1)
                else:
                    hog_rec_loss = F.mse_loss(dec_hog, z_hog)
            else:
                hog_rec_loss = 0.0
        
            # decode dinov2 feature
            if self.aux_dino_decoder:
                dec_dino = self.decode_dino(quant, x=input, h=h, w=w)
                dec_dino = self.to_pixel_dino(dec_dino)
                
                # get z from repa_encoder
                rescale_x = self.scale(self.de_scale(input))
                z_dino = self.repa_model.forward_features(rescale_x)[:, self.repa_model.num_prefix_tokens:]

                z_dino = F.normalize(z_dino, dim=-1)
                dec_dino = F.normalize(dec_dino, dim=-1)

                if self.aux_loss_mask:
                    dino_rec_loss = -(dec_dino * z_dino).sum(dim=-1, keepdim=True)
                    dino_rec_loss = (dino_rec_loss * mask).sum() / mask.sum()
                else:
                    dino_rec_loss = mean_flat(-(dec_dino * z_dino).sum(dim=-1))
                    dino_rec_loss = dino_rec_loss.mean()
            else:
                dino_rec_loss = 0.0
            
            # deocde clip feature
            if self.aux_clip_decoder:
                dec_clip = self.decode_clip(quant, x=input, h=h, w=w)
                dec_clip = self.to_pixel_clip(dec_clip)
                # get clip_target
                rescale_x = self.clip_scale(self.clip_de_scale(input))
                z_clip = self.clip_model.forward_features(rescale_x)[:, self.clip_model.num_prefix_tokens:]
                
                z_clip = F.normalize(z_clip, dim=-1)
                dec_clip = F.normalize(dec_clip, dim=-1)
                
                if self.aux_loss_mask:
                    clip_rec_loss = -(dec_clip * z_clip).sum(dim=-1, keepdim=True)
                    clip_rec_loss = (clip_rec_loss * mask).sum() / mask.sum()
                else:
                    clip_rec_loss = mean_flat(-(dec_clip * z_clip).sum(dim=-1))
                    clip_rec_loss = clip_rec_loss.mean()   
            else:
                clip_rec_loss = 0.0
        
        # Combine auxiliary losses with main loss
        if self.training:
            total_aux_loss = hog_rec_loss + dino_rec_loss + clip_rec_loss
            diff = diff + total_aux_loss
        
        return dec, diff, info

    def get_last_layer(self):
        return self.decoder.to_pixel.model[-1]

@register_model("token.maetok.b_128")
def MAETok_B_128(image_size: int = 256, base_image_size: int = 256, num_latent_tokens: int = 128, **kwargs):
    return MaskAEModel(image_size=image_size, base_image_size=base_image_size, num_latent_tokens=num_latent_tokens, **kwargs)