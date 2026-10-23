"""Random distortions applied on the GPU, a little differently every time.

The model never sees exactly the same image twice, so it has to learn what makes an
"A" an "A" instead of memorizing pixels. Thicker/thinner strokes also help it bridge
the gap between neat fonts and real handwriting.
"""
import math

import torch
import torch.nn.functional as F


def random_affine(x: torch.Tensor, rotate_deg=8.0, scale=(0.9, 1.1), shift=0.06, shear=0.15) -> torch.Tensor:
    """Rotate, scale, shear and shift a batch [B, 1, H, W] (ink = high values)."""
    b = x.shape[0]
    angle = (torch.rand(b, device=x.device) * 2 - 1) * math.radians(rotate_deg)
    s = torch.empty(b, device=x.device).uniform_(*scale)
    sh = (torch.rand(b, device=x.device) * 2 - 1) * shear
    cos, sin = torch.cos(angle), torch.sin(angle)
    theta = torch.zeros(b, 2, 3, device=x.device)
    theta[:, 0, 0], theta[:, 0, 1] = cos / s, (-sin + sh) / s
    theta[:, 1, 0], theta[:, 1, 1] = sin / s, cos / s
    theta[:, :, 2] = (torch.rand(b, 2, device=x.device) * 2 - 1) * shift * 2  # grid units are -1..1
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


def random_thickness(x: torch.Tensor, p: float = 0.3) -> torch.Tensor:
    """Make strokes thicker (max-pool = dilation) or thinner (min-pool = erosion) for some images."""
    choice = torch.rand(x.shape[0], 1, 1, 1, device=x.device)
    thicker = F.max_pool2d(x, 3, stride=1, padding=1)
    thinner = -F.max_pool2d(-x, 3, stride=1, padding=1)
    return torch.where(choice < p / 2, thicker, torch.where(choice < p, thinner, x))


def augment(x: torch.Tensor) -> torch.Tensor:
    return random_affine(random_thickness(x)).clamp(0, 1)
