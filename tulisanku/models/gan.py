"""Few-shot handwriting GAN: generator (content + style -> glyph) and discriminator (critic).

Generator
  content encoder  the target character drawn in a plain "content font" (64x64)
                   -> shape features at 16x16
  style encoder    each reference glyph -> 256 numbers; averaged over the references
                   (any number of them) -> one style code -> AdaIN parameters
  decoder          AdaIN layers re-scale and shift every feature channel using the style
                   code ("same letter shape, your version"), then upsample back to 64x64

Discriminator
  Looks at one glyph and outputs a realism score. It also knows which character the glyph
  should be ("projection" conditioning): a sharp glyph of the wrong letter still scores low.
  Optionally a second head also knows WHO wrote it (one learned embedding per training
  writer or font): "is this a real glyph of writer #1234?". Without it, the critic only
  asks "is this a realistic B?", so nothing but the L1 loss rewards matching the person
  (as in DM-Font / LF-Font, which condition the critic on both character and font).
  Spectral normalization keeps the critic from changing too abruptly, which makes GAN
  training much more stable.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class AdaIN(nn.Module):
    """Normalize each channel of each image, then apply a style-given scale and shift."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.channels = channels

    def forward(self, x, params):  # params [B, 2*C]: first half scale, second half shift
        gamma, beta = params[:, :self.channels], params[:, self.channels:]
        return self.norm(x) * (1 + gamma[..., None, None]) + beta[..., None, None]


class AdaINResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.ada1, self.ada2 = AdaIN(channels), AdaIN(channels)
        self.n_params = 4 * channels

    def forward(self, x, params):
        c = self.ada1.channels
        h = F.relu(self.ada1(self.conv1(x), params[:, :2 * c]))
        h = self.ada2(self.conv2(h), params[:, 2 * c:])
        return x + h


class UpBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.ada = AdaIN(c_out)
        self.n_params = 2 * c_out

    def forward(self, x, params):
        return F.relu(self.ada(self.conv(F.interpolate(x, scale_factor=2, mode="nearest")), params))


def down(c_in, c_out, stride=2):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 4 if stride == 2 else 3, stride, 1),
                         nn.InstanceNorm2d(c_out, affine=True), nn.ReLU(inplace=True))


class Generator(nn.Module):
    def __init__(self, width: int = 64, style_dim: int = 256):
        super().__init__()
        w = width
        self.content = nn.Sequential(down(1, w, stride=1), down(w, 2 * w), down(2 * w, 4 * w),  # 64 -> 16
                                     down(4 * w, 4 * w, stride=1))
        self.style_body = nn.Sequential(down(1, w // 2), down(w // 2, w), down(w, 2 * w), down(2 * w, 4 * w),
                                        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4 * w, style_dim))
        self.blocks = nn.ModuleList([AdaINResBlock(4 * w), AdaINResBlock(4 * w)])
        self.ups = nn.ModuleList([UpBlock(4 * w, 2 * w), UpBlock(2 * w, w)])   # 16 -> 32 -> 64
        self.out = nn.Conv2d(w, 1, 7, padding=3)
        n_params = sum(b.n_params for b in self.blocks) + sum(u.n_params for u in self.ups)
        self.to_params = nn.Sequential(nn.Linear(style_dim, style_dim), nn.ReLU(inplace=True),
                                       nn.Linear(style_dim, n_params))

    def style_code(self, refs, ref_mask):
        """refs [B, K, 1, H, W], ref_mask [B, K] -> style code [B, style_dim] (mean over real refs)."""
        b, k = refs.shape[:2]
        codes = self.style_body(refs.flatten(0, 1)).view(b, k, -1)
        m = ref_mask.float().unsqueeze(-1)
        return (codes * m).sum(1) / m.sum(1).clamp(min=1)

    def forward(self, content, refs, ref_mask):
        return self.draw(content, self.style_code(refs, ref_mask))

    def draw(self, content, style_code):
        """Draw the content characters in the given style (style_code [B, style_dim])."""
        params = self.to_params(style_code)
        h, used = self.content(content), 0
        for layer in list(self.blocks) + list(self.ups):
            h = layer(h, params[:, used:used + layer.n_params])
            used += layer.n_params
        return torch.sigmoid(self.out(h))  # ink intensity 0..1


class Discriminator(nn.Module):
    def __init__(self, n_chars: int = 62, width: int = 64, n_styles: int = 0):
        super().__init__()
        w = width
        sn = spectral_norm
        self.body = nn.Sequential(
            sn(nn.Conv2d(1, w, 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True),          # 32
            sn(nn.Conv2d(w, 2 * w, 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True),      # 16
            sn(nn.Conv2d(2 * w, 4 * w, 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True),  # 8
            sn(nn.Conv2d(4 * w, 8 * w, 4, 2, 1)), nn.LeakyReLU(0.2, inplace=True),  # 4
        )
        self.linear = sn(nn.Linear(8 * w, 1))
        self.embed = sn(nn.Embedding(n_chars, 8 * w))
        self.style_embed = sn(nn.Embedding(n_styles, 8 * w)) if n_styles else None

    def forward(self, x, char, style=None):
        """List of scores, one per head: [realism + "is it this char?"] (+ [realism + "is it this writer?"])."""
        h = self.body(x).sum(dim=(2, 3))
        real = self.linear(h).squeeze(1)
        scores = [real + (self.embed(char) * h).sum(1)]
        if self.style_embed is not None:
            scores.append(real + (self.style_embed(style) * h).sum(1))
        return scores
