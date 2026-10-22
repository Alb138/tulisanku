"""Build the dataset of handwriting drawn on the collection page (same format as NIST).

Reads  data/raw/handwriting/drawn/<session>/*.png   (512x512, black on white)
       collect/session_review.csv                   (which sessions to use, writer ids)
Writes drawn_<size>.npz   images: uint8 [n_sessions, 62, size, size] (ink = 255)
                          mask:   bool  [n_sessions, 62]
       drawn_<size>.csv   session, writer, device, n_chars
       drawn_<size>_preview.png

The page shows a guide line, but most people did not write on it (letters floated
43 px above it on average, and wandered ~78 px within one writer). So the baseline
is rebuilt from the letters themselves, exactly like for NIST (tulisanku.nist.place_writer).
That also means all datasets share one normalization.

All sessions are test data: they answer "does it work on Indonesian handwriting?".
The `writer` column groups sessions by the same person (e.g. a team member's three).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tulisanku.charset import CHARSET  # noqa: E402
from tulisanku.nist import SIZE, place_writer  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/handwriting/drawn"))
    parser.add_argument("--review", type=Path, default=Path("collect/session_review.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    review = pd.read_csv(args.review, dtype=str)
    review = review[review["use"] == "yes"]

    rows, images, masks = [], [], []
    for session, writer in zip(review["session"], review["writer"]):
        folder = args.raw_dir / session
        glyphs = {}
        for char in CHARSET:
            path = folder / f"{ord(char):02x}.png"
            if path.exists():
                glyphs[char] = 255 - np.asarray(Image.open(path).convert("L"))  # ink = 255
        result = place_writer(glyphs)
        if result is None:
            print(f"skipped {session}: no capital letters to set the scale")
            continue
        images.append(result[0])
        masks.append(result[1])
        info = json.loads((folder / "session.json").read_text(encoding="utf-8"))
        rows.append({"index": len(rows), "session": session, "writer": writer,
                     "device": info["device"], "n_chars": int(result[1].sum())})

    images, masks = np.stack(images), np.stack(masks)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"drawn_{SIZE}"
    np.savez_compressed(f"{stem}.npz", images=images, mask=masks, chars=np.array(list(CHARSET)))
    pd.DataFrame(rows).to_csv(f"{stem}.csv", index=False)

    shown = list(range(min(12, len(rows))))
    n, c, s, _ = images[shown].shape
    Image.fromarray(255 - images[shown].transpose(0, 2, 1, 3).reshape(n * s, c * s)).save(f"{stem}_preview.png")
    print(f"Saved {images.shape}: {len(rows)} sessions from {len(set(r['writer'] for r in rows))} writers")


if __name__ == "__main__":
    main()
