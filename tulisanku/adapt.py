"""Per-person adaptation at generation time, using only the letters the person drew.

1. Fine-tuning: before generating, train the model for a few steps to redraw the person's
   own reference letters (content glyph + style code -> their real glyph, L1 loss).
     part="code"     adjust only the 256-number style code; the drawing network stays
                     frozen, so other letters cannot break, but it stays within known styles
     part="decoder"  also adjust the network that draws (AdaIN parameters, decoder layers);
                     it can learn new quirks, but may "forget" how to draw other letters
   The decoder only sees the style code, never the reference images themselves, so it
   cannot simply copy a reference letter into another character.
2. Guarding the other letters: the L1 loss only watches the k reference letters, so the
   other letters can quietly get speckled or broken. Every step, 16 random OTHER letters are
   also drawn in the person's style and checked by
     anchor=w   L1 distance to what the model drew for them BEFORE fine-tuning ("learning
                without forgetting"): the shared network can still pick up quirks, but no
                single letter can drift far from a clean version
     critic=..  the trained critic ("is this a realistic B?"), which needs no answer key.
                A FROZEN critic can be gamed: with weight 1 it turned almost every letter
                into a box with a frame that this critic scores as "very real". So it gets a
                small weight and a hinge (no more push once a letter scores as real).
3. Clean-up (clean=True): remove specks, stray marks and the faint haze away from strokes,
   with the same blob rules used for cleaning the NIST scans.
4. Sharpen (sharpen=True): re-draw the edges crisply (a thicken() of 0 pixels), like the
   uniform pen of a drawing app.
5. Calibration (calibrate=...): the model also redraws the reference letters, so we can
   measure how its versions differ from the real ones in size and stroke thickness - and
   correct every generated letter by the same amount, optionally per group (digits,
   capitals, lowercase): the model's digits are too small (NIST writers wrote digits 12%
   smaller than capitals, and size is where the model is least sure, so it hedges smaller).
   On validation writers this made results worse: 5 letters are too few to measure size.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import affine_transform, binary_dilation, distance_transform_edt, zoom

from .charset import CHARSET
from .gan_model import GanModel
from .models.gan import Discriminator
from .nist import BASELINE_ROW, clean_scan

THRESHOLD = 0.5
GROUPS = np.repeat([0, 1, 2], [10, 26, 26])  # CHARSET order: digits, capitals, lowercase
MIN_SPECK = 4  # pixels at 64 px (the NIST scans use 15 at 128 px)


def finetune(generator, content, refs, ref_idx, steps: int, lr: float, part: str = "decoder",
             critic=None, critic_weight: float = 0.1, anchor: float = 0.0,
             l1_weight: float = 50.0, others: int = 16, seed: int = 0):
    """refs [k, H, W] float 0..1 on the generator's device. Adapts the generator in place
    (part="decoder") and returns the style code to draw with [1, style_dim]."""
    device = refs.device
    target = refs.unsqueeze(1)
    k = len(refs)
    ref_chars = torch.as_tensor(ref_idx, device=device)
    with torch.no_grad():
        code = generator.style_code(refs[None, :, None], torch.ones(1, k, dtype=torch.bool, device=device))
        before = generator.draw(content.unsqueeze(1), code.expand(len(content), -1)) if anchor else None
    if part == "code":
        code = code.clone().requires_grad_(True)
        params = [code]
    else:
        params = [p for name, p in generator.named_parameters() if not name.startswith(("content.", "style_body."))]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr)
    rng = torch.Generator(device=device).manual_seed(seed)
    for _ in range(steps):
        chars = ref_chars
        if critic is not None or anchor:  # the refs plus random other letters
            chars = torch.cat([ref_chars, torch.randint(0, len(content), (others,), generator=rng, device=device)])
        out = generator.draw(content[chars].unsqueeze(1), code.expand(len(chars), -1))
        loss = l1_weight * (out[:k] - target).abs().mean()
        if anchor:
            loss = loss + anchor * (out[k:] - before[chars[k:]]).abs().mean()
        if critic is not None:
            loss = loss + critic_weight * F.relu(1 - critic(out[k:], chars[k:])[0]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    generator.requires_grad_(False)
    return code.detach()


def load_critic(checkpoint, device) -> Discriminator:
    """The trained critic from a training checkpoint (last.pt), frozen."""
    state = torch.load(checkpoint, map_location=device)["discriminator"]
    n_styles = state["style_embed.weight_orig"].shape[0] if "style_embed.weight_orig" in state else 0
    critic = Discriminator(n_styles=n_styles).to(device)
    critic.load_state_dict(state)
    return critic.eval().requires_grad_(False)


def clean_glyph(img, char: str):
    """Remove specks and stray marks (blob rules from nist.clean_scan), then the faint haze
    that is not next to a kept stroke."""
    cleaned, _ = clean_scan((np.clip(img, 0, 1) * 255).astype(np.uint8), char, min_speck=MIN_SPECK)
    return img * binary_dilation(cleaned > 128, iterations=1)


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
    """The GAN plus optional per-person fine-tuning and post-processing (same predict() interface).

    cache: pass the same dict to variants that share fine-tuning settings and differ only in
    post-processing, so each person is fine-tuned once instead of once per variant."""

    def __init__(self, checkpoint, steps: int = 0, part: str = "decoder", lr: float = 1e-4,
                 anchor: float = 0.0, critic=None, critic_weight: float = 0.1,
                 calibrate: str = None, clean: bool = False, sharpen: bool = False,
                 cache: dict = None, name: str = None, device: str = None):
        """anchor / critic / critic_weight: how the other letters are guarded while fine-tuning
        (critic = a training checkpoint, last.pt); calibrate: None, "global" or "group"."""
        super().__init__(checkpoint, device)
        self.generator.requires_grad_(False)
        self.base = {k: v.clone() for k, v in self.generator.state_dict().items()}
        self.critic = load_critic(critic, self.device) if critic and steps else None
        self.steps, self.part, self.lr = steps, part, lr
        self.anchor, self.critic_weight = anchor, critic_weight
        self.calibrate, self.clean, self.sharpen = calibrate, clean, sharpen
        self.cache = cache
        parts = [f"ft-{part}-{steps}" if steps else "", f"anchor-{anchor:g}" if anchor and steps else "",
                 f"critic-{critic_weight:g}" if self.critic else "",
                 f"cal-{calibrate}" if calibrate else "", "clean" if clean else "", "sharpen" if sharpen else ""]
        self.name = name or " + ".join(["GAN", *filter(None, parts)])
        self.last = {}  # details of the last prediction (calibration factors)

    def _generate(self, ref_images, ref_idx, chars):
        """Fine-tune (if asked) and draw `chars` in the person's style."""
        self.generator.load_state_dict(self.base)  # every person starts from the same model
        refs = torch.as_tensor(np.asarray(ref_images, np.float32) / 255, device=self.device)
        if self.steps:
            code = finetune(self.generator, self.content, refs, ref_idx, self.steps, self.lr, self.part,
                            self.critic, self.critic_weight, self.anchor)
        else:
            with torch.no_grad():
                code = self.generator.style_code(refs[None, :, None],
                                                 torch.ones(1, len(refs), dtype=torch.bool, device=self.device))
        with torch.no_grad():
            return self.generator.draw(self.content[torch.as_tensor(chars, device=self.device)].unsqueeze(1),
                                       code.expand(len(chars), -1)).squeeze(1).cpu().numpy()

    def predict(self, ref_images, ref_idx, target_idx):
        target_idx, ref_idx = np.asarray(target_idx), np.asarray(ref_idx)
        chars = np.concatenate([target_idx, ref_idx])  # refs too, for calibration
        key = (np.asarray(ref_images).tobytes(), tuple(ref_idx), tuple(target_idx))
        if self.cache is not None and key in self.cache:
            out = self.cache[key].astype(np.float32)
        else:
            out = self._generate(ref_images, ref_idx, chars)
            if self.cache is not None:
                self.cache[key] = out.astype(np.float16)
        pred, pred_refs = out[:len(target_idx)], out[len(target_idx):]
        if self.calibrate:
            scales, delta = calibration(pred_refs, np.asarray(ref_images) / 255, ref_idx, self.calibrate == "group")
            self.last = {"scale (digits, capitals, lowercase)": scales.round(2).tolist(), "thicken px": delta}
            pred = apply_calibration(pred, target_idx, scales, delta)
        if self.clean:
            pred = np.stack([clean_glyph(p, CHARSET[c]) for p, c in zip(pred, target_idx)])
        if self.sharpen:
            pred = np.stack([thicken(p, 0) for p in pred])
        return pred
