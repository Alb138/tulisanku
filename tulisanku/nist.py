"""Real handwriting from NIST Special Database 19, in the same format as the fonts.

SD19 comes in two versions of the same scans:
  by_write/hsf_X/fWWWW_TT/{d,u,l,c}WWWW_TT/*.png  -> we know the WRITER, not the letter
  by_class/<hex code>/.../*.png                  -> we know the LETTER, not the writer
Identical scans have identical bytes, so we fingerprint every file and join the
two versions on that fingerprint (like a database join on a shared key).

Each scan is a 128x128 black-on-white box with one character. Unlike a font, it
has no baseline, so we rebuild one so that handwriting looks like the font data:
  - one scale per writer: their typical capital height becomes CAP_HEIGHT pixels
    (their own lowercase/capital proportions are kept - that is part of the style)
  - letters sit on the baseline; descender letters (g j p q y) hang below it, with
    their top at the writer's lowercase (x-) height
The targets below were measured on the cleaned handwriting fonts at 64x64.
"""
import hashlib
import io
import zipfile

import numpy as np
from PIL import Image
from scipy import ndimage

from .charset import CHARSET, CHAR_TO_INDEX, UPPER

SIZE = 64
BASELINE_ROW = 43       # bottom of capitals in handwriting fonts (64x64)
CAP_HEIGHT = 36         # median capital height in handwriting fonts
SUPERSAMPLE = 4
DESCENDERS = set("gjpqy")
X_HEIGHT_LETTERS = set("acemnorsuvwxz")  # lowercase letters without ascenders/descenders


def fingerprint(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def label_from_class_folder(hex_code: str) -> str:
    return chr(int(hex_code, 16))  # e.g. "4a" -> "J"


def read_ink(archive: zipfile.ZipFile, name: str) -> np.ndarray:
    """Load a scan as uint8 with ink = 255 on paper = 0 (like the font images)."""
    with archive.open(name) as fh:
        img = Image.open(io.BytesIO(fh.read())).convert("L")
    return 255 - np.asarray(img)


def clean_scan(ink: np.ndarray, char: str, min_speck: int = 15) -> tuple:
    """Remove scanning artifacts; returns (cleaned ink, what was removed).

    The scan is split into separate ink blobs ("connected components"):
      - tiny specks are removed, except for i and j whose dots are small separate blobs
      - small marks BESIDE the letter (no horizontal overlap with it) are stray dots;
        an i/j dot sits above its letter instead, so position tells them apart
      - thin, wide blobs BELOW the letter are pieces of the printed box line. Thin bars
        above the letter are kept on purpose: a T's top bar may be a separate stroke.
    """
    blobs, n = ndimage.label(ink > 128, structure=np.ones((3, 3)))
    if n <= 1:
        return ink, []
    areas = ndimage.sum_labels(np.ones_like(blobs), blobs, index=range(1, n + 1))
    boxes = ndimage.find_objects(blobs)
    main = int(np.argmax(areas))
    main_rows, main_cols = boxes[main]
    remove, removed = np.zeros(n + 1, bool), []
    for k, (rows, cols) in enumerate(boxes):
        if k == main:
            continue
        h, w = rows.stop - rows.start, cols.stop - cols.start
        beside = cols.stop <= main_cols.start or cols.start >= main_cols.stop
        if areas[k] < min_speck and char not in "ij":
            remove[k + 1] = True
            removed.append("speck")
        elif beside and areas[k] < 0.1 * areas[main] and char not in "ij":
            remove[k + 1] = True
            removed.append("stray mark")
        elif h <= 6 and w >= 3 * h and w >= 12 and rows.start >= main_rows.stop - 2:
            remove[k + 1] = True
            removed.append("underline")
    cleaned = ink.copy()
    cleaned[remove[blobs]] = 0
    return cleaned, removed


def ink_box(ink: np.ndarray, threshold: int = 128):
    rows = np.where((ink > threshold).any(axis=1))[0]
    cols = np.where((ink > threshold).any(axis=0))[0]
    if len(rows) == 0:
        return None
    return rows[0], rows[-1] + 1, cols[0], cols[-1] + 1


def place_writer(glyphs: dict, stats=None) -> tuple:
    """Turn one writer's {char: ink scan} into ([62, 64, 64] images, [62] mask).

    `stats` (optional Counter) counts the cleaning steps applied.
    """
    crops = {}
    for char, ink in glyphs.items():
        ink, removed = clean_scan(ink, char)
        if stats is not None:
            stats.update(removed)
        box = ink_box(ink)
        if box is not None:
            top, bottom, left, right = box
            crops[char] = ink[top:bottom, left:right]
    caps = [c.shape[0] for ch, c in crops.items() if ch in UPPER]
    if not caps:
        return None
    canvas = SIZE * SUPERSAMPLE
    scale = CAP_HEIGHT * SUPERSAMPLE / np.median(caps)
    xh = [c.shape[0] for ch, c in crops.items() if ch in X_HEIGHT_LETTERS]
    x_height = (np.median(xh) if xh else 0.6 * np.median(caps)) * scale
    baseline = BASELINE_ROW * SUPERSAMPLE

    images = np.zeros((len(CHARSET), SIZE, SIZE), np.uint8)
    mask = np.zeros(len(CHARSET), bool)
    for char, crop in crops.items():
        h, w = crop.shape
        s = min(scale, 0.95 * canvas / max(h, w))  # shrink the rare letter that would not fit
        new_h, new_w = max(1, round(h * s)), max(1, round(w * s))
        glyph = Image.fromarray(crop).resize((new_w, new_h), Image.Resampling.LANCZOS)
        top = round(baseline - x_height) if char in DESCENDERS else baseline - new_h
        top = min(max(top, 0), canvas - new_h)
        left = (canvas - new_w) // 2
        big = Image.new("L", (canvas, canvas), 0)
        big.paste(glyph, (left, top))
        i = CHAR_TO_INDEX[char]
        images[i] = np.asarray(big.resize((SIZE, SIZE), Image.Resampling.BOX))
        mask[i] = True
    return images, mask
