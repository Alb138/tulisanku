"""Legibility score: can the trained character classifier read a glyph as the intended character?

A generated letter that the classifier cannot read is bad, however good its pixel
scores look. Two versions are reported:
  exact     the classifier's top answer is exactly the intended character
  merged    look-alikes count as correct (case, 0/O/o, 1/I/l), which even a person
            often cannot tell apart from a single glyph
"""
from pathlib import Path

import numpy as np
import torch

from .charset import LOOK_ALIKE
from .models.classifier import CharClassifier

DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints" / "classifier.pt"
GROUP = np.unique(LOOK_ALIKE, return_inverse=True)[1]  # look-alike group id per character index


class Reader:
    def __init__(self, checkpoint: Path = DEFAULT_CHECKPOINT, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CharClassifier().to(self.device).eval()
        self.model.load_state_dict(torch.load(checkpoint, map_location=self.device)["model"])

    @torch.no_grad()
    def read(self, images: np.ndarray) -> np.ndarray:
        """Predicted character index for each glyph; images [m, H, W] with values in 0..1."""
        x = torch.as_tensor(np.asarray(images, np.float32), device=self.device).unsqueeze(1)
        with torch.autocast(self.device, dtype=torch.float16, enabled=self.device == "cuda"):
            return self.model(x).argmax(1).cpu().numpy()

    def scores(self, images: np.ndarray, intended: np.ndarray) -> dict:
        read = self.read(images)
        return {"legible": float((read == intended).mean()),
                "legible_merged": float((GROUP[read] == GROUP[intended]).mean())}
