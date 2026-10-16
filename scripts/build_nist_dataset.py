"""Build the real-handwriting dataset from NIST SD19 (same format as the fonts).

Reads  data/raw/nist/by_write.zip and by_class.zip  (see download_nist.py)
Writes nist_<size>.npz   images: uint8 [n_writers, 62, size, size] (ink = 255)
                         mask:   bool  [n_writers, 62] - which characters the writer has
       nist_<size>.csv   one row per writer: partition, split, number of characters
       nist_<size>_preview.png

Split: partition hsf_4 (written by high-school students) is NIST's recommended test
set and a different population from the rest (Census Bureau employees) - a natural,
honest test of whether a model generalizes to new kinds of writers.
"""
import argparse
import random
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tulisanku.charset import CHARSET  # noqa: E402
from tulisanku.nist import fingerprint, label_from_class_folder, place_writer, read_ink  # noqa: E402

# Prefer each character from the field where it was prompted (d/u/l) over the free-text box (c).
FIELD_PRIORITY = {"d": 0, "u": 0, "l": 0, "c": 1}


def png_names(archive):
    return [n for n in archive.namelist() if n.endswith(".png")]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/nist"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    by_write = zipfile.ZipFile(args.raw_dir / "by_write.zip")
    by_class = zipfile.ZipFile(args.raw_dir / "by_class.zip")

    # 1) Fingerprint every labelled scan: fingerprint -> set of labels
    # (by_class also has "train_xx" folders that repeat the hsf_* images - skip the copies)
    labels = defaultdict(set)
    class_names = [n for n in png_names(by_class) if n.split("/")[2].startswith("hsf_")]
    for name in tqdm(class_names, desc="Fingerprinting by_class"):
        labels[fingerprint(by_class.read(name))].add(label_from_class_folder(name.split("/")[1]))

    # 2) Fingerprint every scan with a known writer and look up its label (the join)
    samples = defaultdict(dict)  # writer -> {char: (priority, name)}
    stats, partition_of = Counter(), {}
    for name in tqdm(png_names(by_write), desc="Joining by_write"):
        _, partition, writer, field, _ = name.split("/")
        found = labels.get(fingerprint(by_write.read(name)))
        if not found:
            stats["no label found"] += 1
            continue
        if len(found) > 1:
            stats["ambiguous label"] += 1
            continue
        char = next(iter(found))
        stats["labelled"] += 1
        partition_of[writer] = partition
        candidate = (FIELD_PRIORITY.get(field[0], 2), name)
        if char in CHARSET and (char not in samples[writer] or candidate < samples[writer][char]):
            samples[writer][char] = candidate
    print("Join results:", dict(stats))

    # 3) Normalize each writer's characters into the font layout
    writers, images, masks, cleaning = [], [], [], Counter()
    for writer in tqdm(sorted(samples), desc="Placing glyphs"):
        glyphs = {c: read_ink(by_write, name) for c, (_, name) in samples[writer].items()}
        result = place_writer(glyphs, cleaning)
        if result is not None:
            writers.append(writer)
            images.append(result[0])
            masks.append(result[1])
    images, masks = np.stack(images), np.stack(masks)
    print("Artifacts removed (blobs):", dict(cleaning))

    # 4) Split by writer: hsf_4 = test, a random slice of the rest = val
    rng = random.Random(args.seed)
    split = []
    for w in writers:
        if partition_of[w] == "hsf_4":
            split.append("test")
        else:
            split.append("val" if rng.random() < args.val_fraction else "train")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"nist_{images.shape[-1]}"
    np.savez_compressed(f"{stem}.npz", images=images, mask=masks, chars=np.array(list(CHARSET)))
    pd.DataFrame({"index": range(len(writers)), "writer": writers,
                  "partition": [partition_of[w] for w in writers], "split": split,
                  "n_chars": masks.sum(axis=1)}).to_csv(f"{stem}.csv", index=False)

    rows = rng.sample(range(len(writers)), min(12, len(writers)))
    n, c, s, _ = images[rows].shape
    grid = 255 - images[rows].transpose(0, 2, 1, 3).reshape(n * s, c * s)
    Image.fromarray(grid).save(f"{stem}_preview.png")
    print(f"Saved {images.shape} for {len(writers)} writers ({Counter(split)})")
    print(f"Characters per writer: median {np.median(masks.sum(axis=1)):.0f} of {len(CHARSET)}")


if __name__ == "__main__":
    main()
