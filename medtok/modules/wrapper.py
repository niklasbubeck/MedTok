import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from medtok.utils import init_from_ckpt

class GenWrapper(nn.Module):
    def __init__(
        self,
        generator: Optional[Dict[str, Any]],
        first_stage: Optional[Dict[str, Any]],
        scale_factor: float = None,
        ckpt_path: Optional[str] = None
    ):
        super().__init__()
        self.generator = generator
        self.first_stage = first_stage
        self.scale_factor = scale_factor

        if ckpt_path is not None: 
            init_from_ckpt(self, ckpt_path)


        for param in self.first_stage.parameters():
            param.requires_grad = False
        self.first_stage.eval()

    def train(self, mode: bool = True):
        # Set wrapper and generator to desired train/eval mode,
        # but ALWAYS force first_stage to remain in eval and keep frozen.
        super().train(mode)
        if self.generator is not None:
            self.generator.train(mode)

        # Force first_stage to remain in eval mode and frozen, always, regardless of mode
        if self.first_stage is not None:
            self.first_stage.eval()
            for p in self.first_stage.parameters():
                p.requires_grad = False

        return self

    def vae_encode(self, image: torch.Tensor, sample: bool=True) -> torch.Tensor:
        temp = self.first_stage.encode(image).mode()
        return temp

    def vae_decode(self, z: torch.Tensor) -> torch.Tensor:
        decoded = self.first_stage.decode(z)
        return decoded

    def forward(self, x, *args, **kwargs):
        loss = self.generator.forward(x, *args, **kwargs)
        return loss
