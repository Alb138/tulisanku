"""Decide which fonts to keep, without modifying the raw dataset.

Reads  fonts_<size>.npz / fonts_<size>.csv  (from build_font_dataset.py)
Writes fonts_<size>_quality.csv   the raw table + caps_only + reason ("" = keep)
       fonts_<size>_removed.png   every removed font, labelled with the reason
       fonts_<size>_duplicates.png the most similar font pairs, to check the threshold

Example:
  python scripts/clean_font_dataset.py --data-dir data/processed
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tulisanku.cleaning import caps_only, contact_sheet, removal_reasons  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--min-height", type=float, default=0.40)
    parser.add_argument("--dup-threshold", type=float, default=0.98)
    args = parser.parse_args()

    stem = args.data_dir / f"fonts_{args.size}"
    images = np.load(f"{stem}.npz")["images"]
    meta = pd.read_csv(f"{stem}.csv")

    meta["caps_only"] = caps_only(images)
    reasons, pairs = removal_reasons(meta["family"].tolist(), meta["median_glyph_height"].to_numpy(),
                                     images, args.min_height, args.dup_threshold)
    meta["reason"] = reasons
    meta.to_csv(f"{stem}_quality.csv", index=False)

    kept = meta[meta["reason"] == ""]
    short_reason = meta["reason"].str.replace(r"duplicate of .*", "duplicate", regex=True)
    print(f"Kept {len(kept)} of {len(meta)} fonts")
    print("Removed:", short_reason[short_reason != ""].value_counts().to_dict())
    print("Caps-only (kept, flagged):", int(kept["caps_only"].sum()))
    print("\nKept fonts by category and split:")
    print(pd.crosstab(kept["category"], kept["split"], margins=True))

    removed = meta.index[meta["reason"] != ""].tolist()
    contact_sheet(images, removed, [f"{meta.family[i]}: {meta.reason[i]}" for i in removed],
                  f"{stem}_removed.png")
    shown = pairs[:30]  # the most similar pairs; look at where "same font" turns into "different font"
    rows = [k for i, j, _ in shown for k in (i, j)]
    labels = [f"{s:.3f} {meta.family[k]}" for i, j, s in shown for k in (i, j)]
    contact_sheet(images, rows, labels, f"{stem}_duplicates.png")
    print(f"\nSaved {stem}_quality.csv and contact sheets")


if __name__ == "__main__":
    main()
