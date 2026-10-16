"""Build the font glyph dataset: Google Fonts -> one image per character per font.

Output (in --out-dir):
  fonts_<size>.npz   images: uint8 [n_fonts, 62, size, size]  (ink = 255)
                     chars:  the 62 characters, in image order
  fonts_<size>.csv   one row per font: family, category, license, split, ...
  preview_<size>.png a grid of a few fonts, to check the result by eye

Example (quick local test on 24 handwriting fonts):
  python scripts/build_font_dataset.py --categories HANDWRITING --limit 24
"""
import argparse
import csv
import os
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tulisanku import google_fonts  # noqa: E402
from tulisanku.charset import CHARSET  # noqa: E402
from tulisanku.render import missing_characters, render_font  # noqa: E402


def process_family(job):
    """Check coverage and render one family. Runs in a worker process."""
    path, size = job
    if not Path(path).exists():
        return None, "file not downloaded"
    try:
        missing = missing_characters(path)
        if missing:
            return None, "missing characters"
        images, stats = render_font(path, size=size)
        return (images, stats), "ok"
    except Exception as e:  # broken or unusual font files: skip, but count why
        return None, f"render error: {type(e).__name__}"


def split_by_family(families, seed, fractions=(0.8, 0.1, 0.1)):
    """Assign whole font families to train/val/test, stratified by category.

    Splitting by family (not by image) prevents data leakage: the test set
    contains styles the model has never seen in any form.
    """
    rng = random.Random(seed)
    by_category = defaultdict(list)
    for i, f in enumerate(families):
        by_category[f.category].append(i)
    split = [""] * len(families)
    for indices in by_category.values():
        rng.shuffle(indices)
        n_train = round(len(indices) * fractions[0])
        n_val = round(len(indices) * fractions[1])
        for rank, i in enumerate(indices):
            split[i] = "train" if rank < n_train else "val" if rank < n_train + n_val else "test"
    return split


def save_preview(images, rows, path):
    """Black-on-white grid: one row per font, one column per character."""
    n, c, s, _ = images[rows].shape
    grid = 255 - images[rows].transpose(0, 2, 1, 3).reshape(n * s, c * s)
    Image.fromarray(grid).save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-dir", type=Path, default=Path("data/raw/google-fonts"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--size", type=int, default=64, help="image size in pixels")
    parser.add_argument("--categories", nargs="*", default=None,
                        help="e.g. HANDWRITING SERIF (default: all categories)")
    parser.add_argument("--limit", type=int, default=None, help="only use N random families (for testing)")
    parser.add_argument("--families", nargs="*", default=None,
                        help='only these family names, e.g. "Caveat" "Flow Block" (for testing)')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    # 1) Metadata only -> choose families
    google_fonts.clone_metadata(args.repo_dir)
    families = [f for f in google_fonts.list_families(args.repo_dir) if google_fonts.is_latin(f)]
    if args.categories:
        wanted = {c.upper() for c in args.categories}
        families = [f for f in families if wanted & set(f.categories)]
    if args.families:
        names = {n.lower() for n in args.families}
        families = [f for f in families if f.name.lower() in names]
    if args.limit:
        families = random.Random(args.seed).sample(families, min(args.limit, len(families)))
    print(f"Selected {len(families)} Latin font families")

    # 2) Download one font file per family
    google_fonts.download_font_files(args.repo_dir, families)

    # 3) Render (in parallel) and keep only fonts that cover all 62 characters
    jobs = [(str(args.repo_dir / f.relpath), args.size) for f in families]
    kept, images, reasons = [], [], Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for family, (result, reason) in zip(families, tqdm(pool.map(process_family, jobs, chunksize=4),
                                                           total=len(jobs), desc="Rendering")):
            reasons[reason] += 1
            if result is not None:
                kept.append((family, result[1]))
                images.append(result[0])
    print("Results:", dict(reasons))
    if not kept:
        sys.exit("No fonts rendered - check the messages above.")

    # 4) Split by family and save
    images = np.stack(images)
    split = split_by_family([f for f, _ in kept], args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"fonts_{args.size}"
    np.savez_compressed(f"{stem}.npz", images=images, chars=np.array(list(CHARSET)))
    with open(f"{stem}.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["index", "family", "category", "all_categories", "license", "file",
                         "split", "median_glyph_height"])
        for i, ((f, stats), s) in enumerate(zip(kept, split)):
            writer.writerow([i, f.name, f.category, "|".join(f.categories), f.license, f.relpath,
                             s, round(stats["median_glyph_height"], 4)])

    preview_rows = random.Random(args.seed).sample(range(len(kept)), min(12, len(kept)))
    save_preview(images, preview_rows, f"{stem}_preview.png")
    print(f"Saved {images.shape} to {stem}.npz  ({Counter(split)})")
    print(f"Categories: {dict(Counter(f.category for f, _ in kept))}")


if __name__ == "__main__":
    main()
