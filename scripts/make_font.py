"""Make a handwriting font: a few handwritten letters -> generated alphabet -> .ttf file.

Default: the main model (style critic, step 15k) gets a few letters (--refs) of one drawn
session and generates all 62 characters; tulisanku/font.py turns them into a font.
--real builds the font from everything the person really drew instead (no model): a check
of the font-making step on its own, and the "what it should look like" comparison.

Writes outputs/fonts/<name>.ttf and <name>_preview.png. The preview is rendered FROM the
.ttf by FreeType (the text engine behind most apps), so it also proves the font works.
To install on Windows: double-click the .ttf -> Install; it appears in Word under its name.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tulisanku.charset import CHAR_TO_INDEX, CHARSET  # noqa: E402
from tulisanku.data import load_handwriting  # noqa: E402
from tulisanku.font import build_font  # noqa: E402
from tulisanku.gan_model import GanModel  # noqa: E402

PREVIEW = ["The quick brown fox jumps over the lazy dog.", "TULISANKU 2026: Halo, semuanya.",
           "0123456789  ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz  ' ; -"]


def preview(font_path: Path, out: Path, size: int = 64):
    font = ImageFont.truetype(str(font_path), size)
    line_height = int(size * 1.35)
    width = max(int(font.getlength(t)) for t in PREVIEW) + 40
    img = Image.new("L", (width, line_height * len(PREVIEW) + 30), 255)
    draw = ImageDraw.Draw(img)
    for i, text in enumerate(PREVIEW):
        draw.text((20, 15 + i * line_height), text, font=font, fill=0)
    img.save(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", required=True, help="drawn session id, e.g. from collect/session_review.csv")
    parser.add_argument("--refs", default="AgkR7", help="the letters the model sees")
    parser.add_argument("--real", action="store_true", help="use the person's own drawings, no model")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/gan_style/ema_015000.pt"))
    parser.add_argument("--name", help="font family name (default: Tulisanku <session>)")
    parser.add_argument("--out", type=Path, default=Path("outputs/fonts"))
    args = parser.parse_args()

    drawn, mask, meta = load_handwriting("drawn")
    rows = np.flatnonzero(meta.session == args.session)
    if not len(rows):
        sys.exit(f"unknown session {args.session}")
    glyphs, available = np.asarray(drawn[rows[0]]), mask[rows[0]]

    if args.real:
        chars = "".join(c for c, ok in zip(CHARSET, available) if ok)
        images = glyphs[available] / 255
        name = args.name or f"Tulisanku {args.session} real"
    else:
        refs = np.array([CHAR_TO_INDEX[c] for c in args.refs if available[CHAR_TO_INDEX[c]]])
        chars = CHARSET
        images = GanModel(args.checkpoint).predict(glyphs[refs], refs, np.arange(len(CHARSET)))
        name = args.name or f"Tulisanku {args.session}"
        print(f"generated {len(CHARSET)} characters from {len(refs)} letters: {''.join(CHARSET[r] for r in refs)}")

    stem = name.replace(" ", "_")
    path = build_font(images, args.out / f"{stem}.ttf", name, chars)
    preview(path, args.out / f"{stem}_preview.png")
    print(f"font: {path} ({path.stat().st_size / 1024:.0f} KB), preview: {args.out / f'{stem}_preview.png'}")


if __name__ == "__main__":
    main()
