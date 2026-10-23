"""Style encoder: turns a glyph into 128 numbers describing the writer's habits.

Trained to recognize WHO wrote a glyph (one class per training writer). The layer
before the final decision becomes the "embedding": glyphs by the same person end up
close together even when they are different characters. Because new (test) writers
are never among the training classes, only the embedding is used afterwards.

Training uses a CosFace head: scores are cosine similarities between the embedding
and one learned direction per writer, and the correct writer must win by a margin.
This pushes each writer's glyphs into a tight cluster, which also helps for writers
the network has never seen.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .classifier import stage


class StyleEncoder(nn.Module):
    def __init__(self, dim: int = 128, width: int = 32):
        super().__init__()
        self.body = nn.Sequential(stage(1, width), stage(width, width * 2),
                                  stage(width * 2, width * 4), stage(width * 4, width * 8))
        self.project = nn.Linear(width * 8, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Unit-length style embeddings [B, dim]."""
        return F.normalize(self.project(self.body(x).mean(dim=(2, 3))), dim=1)


class CosFaceHead(nn.Module):
    """Training-only head: cosine scores to each writer, with a margin for the true writer."""

    def __init__(self, dim: int, n_writers: int, scale: float = 30.0, margin: float = 0.2):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_writers, dim) * 0.01)
        self.scale, self.margin = scale, margin

    def forward(self, embedding: torch.Tensor, writer: torch.Tensor) -> torch.Tensor:
        cosine = embedding @ F.normalize(self.weight, dim=1).T
        cosine = cosine - self.margin * F.one_hot(writer, cosine.shape[1])
        return self.scale * cosine
