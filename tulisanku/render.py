"""Turn a font file into a stack of small grayscale letter images ("glyphs").

Key idea: all letters of one font share ONE scale and ONE baseline. We do not
stretch each letter to fill its box, because the relative size and position of
letters (a "g" hanging below the line, lowercase shorter than capitals) is part
of a handwriting style, and we need it later to assemble a real font file.

Images use ink = 255 (white) on paper = 0 (black). Models usually find it
easier when the "interesting" pixels are the non-zero ones.
"""
import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

from .charset import CHARSET


def missing_characters(path, chars=CHARSET) -> list:
    """Characters the font has no glyph for (checked in the font's character map)."""
    with TTFont(str(path), lazy=True) as font:
        cmap = font.getBestCmap() or {}
    return [c for c in chars if ord(c) not in cmap]


def _load_font(path, size: int, weight: int):
    font = ImageFont.truetype(str(path), size, layout_engine=ImageFont.Layout.BASIC)
    try:  # variable fonts: pick the regular weight instead of the file's default
        axes = font.get_variation_axes()
    except OSError:
        return font  # not a variable font
    values = []
    for axis in axes:
        name = axis["name"].decode() if isinstance(axis["name"], bytes) else axis["name"]
        value = weight if name.lower() == "weight" else axis["default"]
        values.append(min(max(value, axis["minimum"]), axis["maximum"]))
    font.set_variation_by_axes(values)
    return font


def render_font(path, chars=CHARSET, size: int = 64, supersample: int = 4,
                margin: float = 0.06, weight: int = 400):
    """Render every character of a font into a (len(chars), size, size) uint8 array.

    We draw at `supersample` times the target size and shrink at the end;
    this gives smooth (anti-aliased) edges.
    Returns (images, stats) and raises ValueError if a character is blank.
    """
    canvas = size * supersample

    def measure(font):
        boxes = [font.getbbox(c, anchor="ls") for c in chars]  # relative to (left, baseline)
        blank = [c for c, (l, t, r, b) in zip(chars, boxes) if r <= l or b <= t]
        if blank:
            raise ValueError(f"blank glyphs: {''.join(blank)}")
        return boxes

    # 1) Measure at a probe size, then choose one scale so the tallest stack
    #    (highest ascender to lowest descender) and widest letter both fit.
    probe = _load_font(path, canvas, weight)
    boxes = measure(probe)
    top, bottom = min(b[1] for b in boxes), max(b[3] for b in boxes)
    widest = max(b[2] - b[0] for b in boxes)
    usable = canvas * (1 - 2 * margin)
    scale = min(usable / (bottom - top), usable / widest)

    # 2) Re-measure at the final size and put the baseline at the same height for every letter.
    font = _load_font(path, max(1, round(canvas * scale)), weight)
    boxes = measure(font)
    top, bottom = min(b[1] for b in boxes), max(b[3] for b in boxes)
    baseline_y = (canvas - (bottom - top)) / 2 - top

    images = []
    for c, (left, _, right, _) in zip(chars, boxes):
        img = Image.new("L", (canvas, canvas), 0)
        x = (canvas - (right - left)) / 2 - left  # center horizontally
        ImageDraw.Draw(img).text((x, baseline_y), c, fill=255, font=font, anchor="ls")
        images.append(np.asarray(img.resize((size, size), Image.Resampling.BOX)))

    heights = [b[3] - b[1] for b in boxes]
    stats = {"median_glyph_height": float(np.median(heights) / canvas)}
    return np.stack(images), stats
