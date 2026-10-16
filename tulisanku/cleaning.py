"""Data cleaning for the font glyph dataset.

The raw dataset (fonts_<size>.npz/.csv) is never modified. Cleaning only decides,
for every font, whether to keep it and why not - so every removal is documented,
reversible, and can be reported.

Rules, in order of priority:
  1. symbol font   - not letters at all (barcodes, placeholder blocks, math symbols)
  2. faint letters - strokes so thin that at this resolution they become pale gray
  3. duplicate     - near-identical to another font (e.g. Noto Sans vs. Noto Sans
                     Devanagari share the same Latin letters); keeping both would
                     leak the same style into train and test
Caps-only fonts (lowercase drawn as capitals) are only flagged, not removed.

Note: an earlier version removed fonts with a low median glyph height. That
measured proportions, not quality - it also removed clear casual handwriting with
tall loops and deep descenders. "Solid ink" measures the actual problem (faintness).
"""
import re

import numpy as np
from PIL import Image, ImageDraw

from .charset import CHAR_TO_INDEX, LOWER, UPPER

SYMBOL_FONTS = re.compile(
    r"^(Libre Barcode|Flow (Block|Circular|Rounded)$|Redacted|Linefont$|Wavefont$"
    r"|Noto Sans Symbols|jsMath cm(ex|sy)10$)"
)


def is_symbol_font(family: str) -> bool:
    return bool(SYMBOL_FONTS.match(family))


def solid_ink(images: np.ndarray) -> np.ndarray:
    """Share of pixels that are clearly ink (> 50% intensity), per font."""
    return (images > 128).mean(axis=(1, 2, 3))


def caps_only(images: np.ndarray, tolerance: float = 0.01) -> np.ndarray:
    """True for fonts whose lowercase letters are identical to their capitals."""
    lower = images[:, [CHAR_TO_INDEX[c] for c in LOWER]].astype(np.float32)
    upper = images[:, [CHAR_TO_INDEX[c] for c in UPPER]].astype(np.float32)
    return np.abs(lower - upper).mean(axis=(1, 2, 3)) / 255 < tolerance


def font_vectors(images: np.ndarray, size: int = 16) -> np.ndarray:
    """Shrink each font's 62 glyphs to size x size and flatten into one unit vector."""
    n, c, h, w = images.shape
    f = h // size
    small = images.reshape(n, c, size, f, size, f).astype(np.float32).mean(axis=(3, 5))
    vectors = small.reshape(n, -1)
    return vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)


def near_duplicate_pairs(images: np.ndarray, threshold: float) -> list:
    """All font pairs (i, j, similarity) with cosine similarity above the threshold.

    Cosine similarity compares the "direction" of two vectors: 1.0 means the two
    fonts' letters are drawn identically, lower values mean different shapes.
    """
    vectors = font_vectors(images)
    similarity = vectors @ vectors.T
    i, j = np.where(np.triu(similarity, k=1) > threshold)
    return sorted(zip(i.tolist(), j.tolist(), similarity[i, j].tolist()), key=lambda p: -p[2])


def removal_reasons(families: list, images: np.ndarray,
                    min_ink: float = 0.015, dup_threshold: float = 0.98):
    """Return (reasons, duplicate_pairs); reasons[i] is "" if font i is kept."""
    reasons = [""] * len(families)
    ink = solid_ink(images)
    for i, family in enumerate(families):
        if is_symbol_font(family):
            reasons[i] = "symbol font"
        elif ink[i] < min_ink:
            reasons[i] = "faint letters"

    pairs = near_duplicate_pairs(images, dup_threshold)
    for i, j, _ in pairs:  # most similar pairs first; keep the earlier font, drop the later
        if not reasons[i] and not reasons[j]:
            reasons[j] = f"duplicate of {families[i]}"
    return reasons, pairs


def contact_sheet(images: np.ndarray, rows: list, labels: list, path, label_width: int = 260):
    """Black-on-white grid with a text label in front of each font's row."""
    if not rows:
        return
    _, c, s, _ = images.shape
    sheet = Image.new("L", (label_width + c * s, len(rows) * s), 255)
    draw = ImageDraw.Draw(sheet)
    for r, (index, label) in enumerate(zip(rows, labels)):
        row = 255 - images[index].transpose(1, 0, 2).reshape(s, c * s)
        sheet.paste(Image.fromarray(row), (label_width, r * s))
        draw.text((4, r * s + s // 2 - 6), label[:40], fill=0)
    sheet.save(path)
