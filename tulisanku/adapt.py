"""Per-person adaptation at generation time, using only the letters the person drew.

1. Fine-tuning: before generating, train the model for a few steps to redraw the person's
   own reference letters (content glyph + style code -> their real glyph, L1 loss).
     part="code"     adjust only the 256-number style code; the drawing network stays
                     frozen, so other letters cannot break, but it stays within known styles
     part="decoder"  also adjust the network that draws (AdaIN parameters, decoder layers);
                     it can learn new quirks, but may "forget" how to draw other letters
   The decoder only sees the style code, never the reference images themselves, so it
   cannot simply copy a reference letter into another character.
2. Calibration: the model also redraws the reference letters, so we can measure how its
   versions differ from the real ones in size and stroke thickness - and correct every
   generated letter by the same amount. Size is corrected per group (digits, capitals,
   lowercase): the model's digits are much too small (NIST writers wrote digits 12% smaller
   than capitals, and size is where the model is least sure, so it hedges smaller), while
   capitals are only slightly off. A group without a reference letter uses the overall factor.
"""
import numpy as np
import torch
from scipy.ndimage import affine_transform, distance_transform_edt, zoom

from .gan_model import GanModel
from .nist import BASELINE_ROW

THRESHOLD = 0.5
GROUPS = np.repeat([0, 1, 2], [10, 26, 26])  # CHARSET order: digits, capitals, lowercase


def finetune(generator, content, refs, ref_idx, steps: int, lr: float, part: str = "decoder"):
    """refs [k, H, W] float 0..1 on the generator's device. Adapts the generator in place
    (part="decoder") and returns the style code to draw with [1, style_dim]."""
    target = refs.unsqueeze(1)
    glyphs = content[torch.as_tensor(ref_idx, device=refs.device)].unsqueeze(1)
    with torch.no_grad():
        code = generator.style_code(refs[None, :, None], torch.ones(1, len(refs), dtype=torch.bool, device=refs.device))
    if part == "code":
        code = code.clone().requires_grad_(True)
        params = [code]
    else:
        params = [p for name, p in generator.named_parameters() if not name.startswith(("content.", "style_body."))]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        loss = (generator.draw(glyphs, code.expand(len(glyphs), -1)) - target).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    generator.requires_grad_(False)
    return code.detach()


def _ink_rows_cols(img):
    rows, cols = np.flatnonzero(img.max(axis=1) > THRESHOLD), np.flatnonzero(img.max(axis=0) > THRESHOLD)
    return (rows, cols) if len(rows) else (None, None)


def scale_glyph(img, s: float):
    """Scale a glyph by s around the middle of its ink on the baseline (the baseline stays put),
    at most until its top reaches the top of the image."""
    rows, cols = _ink_rows_cols(img)
    if cols is None or s == 1:
        return img
    s = min(s, (BASELINE_ROW - 1) / max(BASELINE_ROW - rows[0], 1)) if s > 1 else s
    center = np.array([BASELINE_ROW, (cols[0] + cols[-1]) / 2])
    return affine_transform(img, np.diag([1 / s, 1 / s]), offset=center - center / s, order=1)


def thicken(img, delta: float, up: int = 4):
    """Move the stroke edges outwards by delta pixels (negative = thinner), anti-aliased."""
    ink = zoom(img, up, order=1) > THRESHOLD
    sdf = distance_transform_edt(ink) - distance_transform_edt(~ink)  # > 0 inside the ink
    big = (sdf > -delta * up).astype(np.float32)
    return big.reshape(img.shape[0], up, img.shape[1], up).mean(axis=(1, 3))


def calibration(generated_refs, real_refs, ref_idx, per_group: bool = True, deltas=np.linspace(-1, 1.5, 11)):
    """Size factor per group [digits, capitals, lowercase] and thickness delta (pixels) that
    make the generated reference letters match the real ones. per_group=False: one factor."""
    ratios = np.full(len(ref_idx), np.nan)
    for i, (g, r) in enumerate(zip(generated_refs, real_refs)):
        (gr, _), (rr, _) = _ink_rows_cols(g), _ink_rows_cols(r)
        if gr is not None and rr is not None and len(gr) >= 3:
            ratios[i] = len(rr) / len(gr)
    groups = GROUPS[np.asarray(ref_idx)] if per_group else np.full(len(ref_idx), -1)
    overall = np.nanmedian(ratios) if np.isfinite(ratios).any() else 1.0
    scales = np.array([np.nanmedian(ratios[groups == g]) if np.isfinite(ratios[groups == g]).any() else overall
                       for g in range(3)])
    scales = np.clip(scales, 0.7, 1.8)
    scaled = [scale_glyph(g, scales[GROUPS[c]]) for g, c in zip(generated_refs, ref_idx)]
    real_ink = sum((r > THRESHOLD).sum() for r in real_refs)
    errors = [abs(sum((thicken(g, d) > THRESHOLD).sum() for g in scaled) - real_ink) for d in deltas]
    return scales, float(deltas[int(np.argmin(errors))])


def apply_calibration(images, char_idx, scales, delta: float):
    return np.stack([thicken(scale_glyph(img, scales[GROUPS[c]]), delta) for img, c in zip(images, char_idx)])


class AdaptedGan(GanModel):
    """The GAN plus optional per-person fine-tuning and calibration (same predict() interface)."""

    def __init__(self, checkpoint, steps: int = 0, part: str = "decoder", lr: float = 1e-4,
                 calibrate: str = None, name: str = None, device: str = None):
        """calibrate: None, "global" (one size factor) or "group" (digits / capitals / lowercase)."""
        super().__init__(checkpoint, device)
        self.generator.requires_grad_(False)
        self.base = {k: v.clone() for k, v in self.generator.state_dict().items()}
        self.steps, self.part, self.lr, self.calibrate = steps, part, lr, calibrate
        self.name = name or "GAN" + (f" + ft-{part}-{steps}" if steps else "") + (f" + cal-{calibrate}" if calibrate else "")
        self.last = {}  # details of the last prediction (scale, thickness)

    def predict(self, ref_images, ref_idx, target_idx):
        self.generator.load_state_dict(self.base)  # every person starts from the same model
        refs = torch.as_tensor(np.asarray(ref_images, np.float32) / 255, device=self.device)
        mask = torch.ones(1, len(refs), dtype=torch.bool, device=self.device)
        if self.steps:
            code = finetune(self.generator, self.content, refs, ref_idx, self.steps, self.lr, self.part)
        else:
            with torch.no_grad():
                code = self.generator.style_code(refs[None, :, None], mask)
        chars = np.concatenate([np.asarray(target_idx), np.asarray(ref_idx)])  # refs too, for calibration
        with torch.no_grad():
            out = self.generator.draw(self.content[torch.as_tensor(chars, device=self.device)].unsqueeze(1),
                                      code.expand(len(chars), -1)).squeeze(1).cpu().numpy()
        pred, pred_refs = out[:len(target_idx)], out[len(target_idx):]
        if self.calibrate:
            scales, delta = calibration(pred_refs, np.asarray(ref_images) / 255, ref_idx, self.calibrate == "group")
            self.last = {"scale (digits, capitals, lowercase)": scales.round(2).tolist(), "thicken px": delta}
            pred = apply_calibration(pred, target_idx, scales, delta)
        return pred
