"""Evaluate the two baselines (mean glyph, nearest font) on every test set.

Test sets: held-out fonts, NIST test writers (high-school students), and the
handwriting drawn on the collection page. The font library is the clean training fonts.

Writes outputs/results/baselines.csv (one row per writer and trial),
       outputs/results/baselines_summary.csv, and
       outputs/figures/baselines_examples.png (references, truth, both predictions).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tulisanku.baselines import MeanGlyph, NearestFont  # noqa: E402
from tulisanku.data import load_fonts, load_handwriting  # noqa: E402
from tulisanku.evaluation import evaluate, pick_references  # noqa: E402
from tulisanku.legibility import Reader  # noqa: E402
from tulisanku.style import StyleGallery, StyleJudge  # noqa: E402


def example_figure(models, images, mask, writers, k, path, seed=0, n_targets=14):
    """Per writer: k references | real characters | one row per model."""
    tiles = []
    for w in writers:
        glyphs = np.asarray(images[w])
        refs, targets = pick_references(np.flatnonzero(mask[w]), k, seed, 0, w)
        targets = targets[:n_targets]
        blank = np.zeros((64, 64), np.float32)
        ref_row = np.concatenate([glyphs[r] / 255 for r in refs] + [blank] * (n_targets - len(refs)), axis=1)
        rows = [ref_row, np.concatenate(glyphs[targets] / 255, axis=1)]
        rows += [np.concatenate(m.predict(glyphs[refs], refs, targets), axis=1) for m in models]
        tiles += rows + [np.full((6, rows[0].shape[1]), 0.5)]
    grid = 1 - np.concatenate(tiles, axis=0)  # black ink on white
    Image.fromarray((grid * 255).astype(np.uint8)).save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 20])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    library, _, _ = load_fonts("train")
    models = [MeanGlyph(library), NearestFont(library)]
    reader, judge = Reader(), StyleJudge()
    metrics = ["l1", "ssim", "iou", "legible", "legible_merged", "style_top1", "style_pct"]
    test_sets = {"fonts (held out)": load_fonts("test"), "NIST (high school)": load_handwriting("nist", "test"),
                 "drawn (ours)": load_handwriting("drawn")}

    results = []
    for set_name, (images, mask, meta) in test_sets.items():
        gallery = StyleGallery(judge, images, mask, meta["writer"] if "writer" in meta else None)
        for model in models:
            for k in args.ks:
                r = evaluate(model, images, mask, k, trials=args.trials, reader=reader, style=gallery)
                results.append(r.assign(test_set=set_name, model=model.name))
                print(f"{set_name:20s} {model.name:13s} k={k:2d}  "
                      + "  ".join(f"{m}={r[m].mean():.3f}" for m in metrics), flush=True)
        print(f"{set_name:20s} real handwriting (ceiling): legible={r['real_legible'].mean():.3f}  "
              f"legible_merged={r['real_legible_merged'].mean():.3f}", flush=True)
    results = pd.concat(results, ignore_index=True)

    (args.out / "results").mkdir(parents=True, exist_ok=True)
    (args.out / "figures").mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out / "results" / "baselines.csv", index=False)
    summary = results.groupby(["test_set", "model", "k"])[metrics].mean().round(3)
    summary.to_csv(args.out / "results" / "baselines_summary.csv")
    print("\n", summary.unstack("model").to_string())

    images, mask, _ = test_sets["NIST (high school)"]
    example_figure(models, images, mask, [0, 1, 2], 5, args.out / "figures" / "baselines_examples_nist.png")
    images, mask, _ = test_sets["drawn (ours)"]
    example_figure(models, images, mask, [3, 6, 9], 5, args.out / "figures" / "baselines_examples_drawn.png")


if __name__ == "__main__":
    main()
