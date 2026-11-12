"""Use a trained diffusion model through the same predict() interface as the baselines and the GAN."""
from pathlib import Path

import numpy as np
import torch

from .models.diffusion import Diffusion, UNet


class DiffusionModel:
    name = "Diffusion"

    def __init__(self, checkpoint: Path = None, device: str = None, unet: UNet = None, content_bank=None,
                 n_steps: int = 25, guidance: float = 1.0, seed: int = 0):
        """Load from a checkpoint file, or wrap a U-Net that is still training. n_steps = DDIM
        steps per generation; guidance > 1 pushes the result further towards the person's style."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if unet is None:
            ckpt = torch.load(checkpoint, map_location=self.device)
            unet = UNet(**ckpt["config"]).to(self.device)
            unet.load_state_dict(ckpt["unet_ema"])
            content_bank = ckpt["content_bank"]
        self.unet = unet.eval()
        self.content = torch.as_tensor(content_bank, dtype=torch.float32, device=self.device) * 2 - 1
        self.diffusion = Diffusion(device=self.device)
        self.n_steps, self.guidance, self.seed = n_steps, guidance, seed

    @torch.no_grad()
    def predict(self, ref_images, ref_idx, target_idx):
        refs = torch.as_tensor(np.asarray(ref_images, np.float32) / 255, device=self.device)[None, :, None]
        mask = torch.ones(1, refs.shape[1], dtype=torch.bool, device=self.device)
        # Always all 62 characters in one batch: one fixed shape (the GPU tunes its routines once,
        # instead of for every new batch size), and every character always starts from the same noise.
        content = self.content.unsqueeze(1)
        generator = torch.Generator(device=self.device).manual_seed(self.seed)  # same noise every call: repeatable
        with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device == "cuda"):
            style = self.unet.style_embedding(refs, mask).expand(len(content), -1)  # once per person
            x0 = self.diffusion.sample(self.unet, content, style, self.unet.null_style, self.n_steps,
                                       self.guidance, generator)
        ink = ((x0.float() + 1) / 2).clamp(0, 1).squeeze(1)  # -1..1 -> ink 0..1
        return ink[torch.as_tensor(target_idx, device=self.device)].cpu().numpy()
