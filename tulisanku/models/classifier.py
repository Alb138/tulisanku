"""A small CNN that reads a single 64x64 glyph and says which of the 62 characters it is.

Four stages; each halves the image size and doubles the number of feature channels:
  64x64x1 -> 32x32x32 -> 16x16x64 -> 8x8x128 -> 4x4x256 -> average -> 62 scores
Early stages detect edges and stroke pieces; later stages combine them into
letter parts and whole letters.
"""
import torch
import torch.nn as nn


def stage(c_in: int, c_out: int) -> nn.Sequential:
    """Two 3x3 convolutions (each followed by batch norm + ReLU), then halve the size."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CharClassifier(nn.Module):
    def __init__(self, n_classes: int = 62, width: int = 32, dropout: float = 0.3):
        super().__init__()
        self.body = nn.Sequential(stage(1, width), stage(width, width * 2),
                                  stage(width * 2, width * 4), stage(width * 4, width * 8))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(width * 8, n_classes))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """A 256-number description of the glyph (reused later to compare styles)."""
        return self.body(x).mean(dim=(2, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))  # raw scores ("logits"), one per character
