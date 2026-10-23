"""Use a trained GAN generator through the same predict() interface as the baselines."""
from pathlib import Path

import numpy as np
import torch

from .models.gan import Generator


class GanModel:
    name = "GAN"

    def __init__(self, checkpoint: Path, device: str = None, generator: Generator = None, content_bank=None):
        """Load from a checkpoint file, or wrap a generator that is still training."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if generator is None:
            ckpt = torch.load(checkpoint, map_location=self.device)
            generator = Generator(**ckpt["config"]).to(self.device)
            generator.load_state_dict(ckpt["generator_ema"])
            content_bank = ckpt["content_bank"]  # a tensor, so the safe weights_only loading works
        self.generator = generator.eval()
        self.content = torch.as_tensor(content_bank, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def predict(self, ref_images, ref_idx, target_idx):
        refs = torch.as_tensor(np.asarray(ref_images, np.float32) / 255, device=self.device)[None, :, None]
        mask = torch.ones(1, refs.shape[1], dtype=torch.bool, device=self.device)
        content = self.content[torch.as_tensor(target_idx, device=self.device)].unsqueeze(1)
        with torch.autocast(self.device, dtype=torch.bfloat16, enabled=self.device == "cuda"):
            code = self.generator.style_code(refs, mask)                     # once per writer
            out = self.generator.draw(content, code.expand(len(content), -1))
        return out.float().squeeze(1).cpu().numpy()
