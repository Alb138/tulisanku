"""Few-shot handwriting diffusion model: a U-Net that predicts the noise in a noisy glyph.

Training: take a writer's real glyph x0 (ink scaled to -1..1), pick a random time t, mix it
with Gaussian noise (x_t = sqrt(a_t) x0 + sqrt(1 - a_t) eps) and train the U-Net to predict
eps. Generating: start from pure noise and remove the predicted noise step by step.

Conditioning - the same two clues the GAN gets:
  content  the target character drawn in a plain font, fed in as a second image channel,
           so every layer can see which letter it is drawing
  style    each reference glyph -> 256 numbers (same encoder design as the GAN), averaged
           over the references; added to the time embedding, which scales and shifts every
           residual block ("adaptive group norm", the diffusion cousin of AdaIN)
For classifier-free guidance (Ho & Salimans, 2022) the style is replaced by a learned
"no style" vector for 10% of training examples. At generation time the model predicts with
and without the style and pushes further towards the style: guidance > 1 = more like the person.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gan import down


def timestep_embedding(t, dim: int):
    """Sinusoidal embedding of the diffusion time (like the position encoding of transformers)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=1)


class ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, emb_dim: int):
        super().__init__()
        self.norm1, self.conv1 = nn.GroupNorm(16, c_in), nn.Conv2d(c_in, c_out, 3, padding=1)
        self.emb = nn.Linear(emb_dim, 2 * c_out)
        self.norm2, self.conv2 = nn.GroupNorm(16, c_out), nn.Conv2d(c_out, c_out, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)  # every block starts as "do nothing": stabler training
        nn.init.zeros_(self.conv2.bias)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.emb(F.silu(emb))[..., None, None].chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        return self.skip(x) + self.conv2(F.silu(h))


class Attention(nn.Module):
    """Self-attention over all positions of a (small) feature map: lets distant parts of a
    letter agree with each other, e.g. the two bowls of a B."""

    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        self.norm, self.heads = nn.GroupNorm(16, channels), heads
        self.qkv, self.out = nn.Conv2d(channels, 3 * channels, 1), nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, self.heads, c // self.heads, h * w).transpose(-1, -2).unbind(1)
        y = F.scaled_dot_product_attention(q, k, v).transpose(-1, -2).reshape(b, c, h, w)
        return x + self.out(y)


class StyleEncoder(nn.Module):
    """Reference glyphs [B, K, 1, H, W] + mask [B, K] -> one style vector [B, style_dim]."""

    def __init__(self, width: int = 64, style_dim: int = 256):
        super().__init__()
        w = width
        self.body = nn.Sequential(down(1, w // 2), down(w // 2, w), down(w, 2 * w), down(2 * w, 4 * w),
                                  nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4 * w, style_dim))

    def forward(self, refs, ref_mask):
        b, k = refs.shape[:2]
        codes = self.body(refs.flatten(0, 1)).view(b, k, -1)
        m = ref_mask.float().unsqueeze(-1)
        return (codes * m).sum(1) / m.sum(1).clamp(min=1)


class UNet(nn.Module):
    def __init__(self, channels=(64, 128, 192, 256), attention=(False, False, True, True),
                 blocks: int = 2, style_dim: int = 256):
        super().__init__()
        emb = 4 * channels[0]
        self.time = nn.Sequential(nn.Linear(channels[0], emb), nn.SiLU(), nn.Linear(emb, emb))
        self.style = StyleEncoder(style_dim=style_dim)
        self.style_proj = nn.Sequential(nn.Linear(style_dim, emb), nn.SiLU(), nn.Linear(emb, emb))
        self.null_style = nn.Parameter(torch.zeros(emb))  # "no style given", for guidance
        self.inp = nn.Conv2d(2, channels[0], 3, padding=1)  # noisy glyph + content glyph

        self.down, skips, c = nn.ModuleList(), [channels[0]], channels[0]
        for level, (c_out, attn) in enumerate(zip(channels, attention)):
            for _ in range(blocks):
                self.down.append(nn.ModuleList([ResBlock(c, c_out, emb), Attention(c_out) if attn else nn.Identity()]))
                c = c_out
                skips.append(c)
            if level < len(channels) - 1:
                self.down.append(nn.Conv2d(c, c, 3, stride=2, padding=1))  # 64 -> 32 -> 16 -> 8
                skips.append(c)
        self.mid = nn.ModuleList([ResBlock(c, c, emb), Attention(c), ResBlock(c, c, emb)])
        self.up = nn.ModuleList()
        for level in reversed(range(len(channels))):
            c_out, attn = channels[level], attention[level]
            for _ in range(blocks + 1):
                self.up.append(nn.ModuleList([ResBlock(c + skips.pop(), c_out, emb),
                                              Attention(c_out) if attn else nn.Identity()]))
                c = c_out
            if level > 0:
                self.up.append(nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c, c, 3, padding=1)))
        self.out = nn.Sequential(nn.GroupNorm(16, c), nn.SiLU(), nn.Conv2d(c, 1, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)
        self.time_dim = channels[0]

    def style_embedding(self, refs, ref_mask):
        return self.style_proj(self.style(refs, ref_mask))

    def forward(self, x, t, content, style_emb):
        """x, content [B, 1, 64, 64] in -1..1; t [B]; style_emb [B, emb] (or the null style)."""
        emb = self.time(timestep_embedding(t, self.time_dim)) + style_emb
        h = self.inp(torch.cat([x, content], dim=1))
        hs = [h]
        for layer in self.down:
            if isinstance(layer, nn.ModuleList):
                h = layer[1](layer[0](h, emb))
            else:
                h = layer(h)
            hs.append(h)
        h = self.mid[2](self.mid[1](self.mid[0](h, emb)), emb)
        for layer in self.up:
            if isinstance(layer, nn.ModuleList):
                h = layer[1](layer[0](torch.cat([h, hs.pop()], dim=1), emb))
            else:
                h = layer(h)
        return self.out(h)


class Diffusion:
    """Noise schedule (cosine, Nichol & Dhariwal 2021), training targets and DDIM sampling."""

    def __init__(self, steps: int = 1000, device="cpu"):
        s = 0.008
        f = torch.cos((torch.arange(steps + 1, dtype=torch.float64) / steps + s) / (1 + s) * math.pi / 2) ** 2
        self.alpha_bar = (f[1:] / f[0]).clamp(min=1e-5, max=0.99999).float().to(device)  # share of signal left at t
        self.steps = steps

    def add_noise(self, x0, t, noise):
        a = self.alpha_bar[t].view(-1, 1, 1, 1)
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    @torch.no_grad()
    def sample(self, model, content, style_emb, null_emb, n_steps: int = 25, guidance: float = 1.0, generator=None):
        """DDIM (deterministic) from pure noise; guidance mixes the predictions with and without style."""
        x = torch.randn(content.shape, device=content.device, generator=generator)
        times = torch.linspace(self.steps - 1, 0, n_steps, device=content.device).long()
        for i, t in enumerate(times):
            tt = t.expand(len(x))
            if guidance == 1:
                eps = model(x, tt, content, style_emb)
            else:
                both = model(torch.cat([x, x]), torch.cat([tt, tt]), torch.cat([content, content]),
                             torch.cat([style_emb, null_emb.expand_as(style_emb)]))
                eps_style, eps_none = both.float().chunk(2)
                eps = eps_none + guidance * (eps_style - eps_none)
            eps = eps.float()
            a = self.alpha_bar[t]
            x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-1, 1)  # the model's guess of the clean glyph
            eps = (x - a.sqrt() * x0) / (1 - a).sqrt()                  # noise consistent with the clipped guess
            a_next = self.alpha_bar[times[i + 1]] if i + 1 < n_steps else torch.tensor(1.0, device=x.device)
            x = a_next.sqrt() * x0 + (1 - a_next).sqrt() * eps
        return x0
