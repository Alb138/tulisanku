"""Evaluate trained GAN checkpoints exactly like the baselines, plus a memorization check.

1. Memorization check ("would more data help?"): the same few-shot test on NIST writers
   the GAN trained on vs. writers it never saw. A model that memorized its training
   writers scores clearly better on them; a model that learned general rules scores about
   the same on both. The style score is left out here: the style encoder was trained on
   the same training writers, so it is not a fair judge for them.
2. The baseline test sets (held-out fonts, NIST test writers, drawn) with the same k,
   trials and reference letters as scripts/evaluate_baselines.py, so the numbers are
   directly comparable with outputs/results/baselines_summary.csv.

Writes outputs/results/gan.csv, gan_summary.csv, gan_memorization.csv and
       outputs/figures/gan_examples_{nist,drawn}.png (references, truth, nearest font, GANs).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_baselines import example_figure  # noqa: E402
from tulisanku.baselines import NearestFont  # noqa: E402
from tulisanku.data import load_fonts, load_handwriting  # noqa: E402
from tulisanku.evaluation import evaluate  # noqa: E402
from tulisanku.gan_model import GanModel  # noqa: E402
from tulisanku.legibility import Reader  # noqa: E402
from tulisanku.style import StyleGallery, StyleJudge  # noqa: E402

METRICS = ["l1", "ssim", "iou", "legible", "legible_merged", "style_top1", "style_pct"]


def load_gan(path: Path) -> GanModel:
    model = GanModel(path)
    model.name = f"GAN {int(path.stem.split('_')[-1]) // 1000}k"  # ema_015000.pt -> "GAN 15k"
    return model


def memorization_check(models, reader, n_writers: int, k: int = 5) -> pd.DataFrame:
    results = []
    for split, label in [("train", "seen in training"), ("val", "never seen")]:
        images, mask, _ = load_handwriting("nist", split)
        rows = np.sort(np.random.default_rng(0).choice(len(images), min(n_writers, len(images)), replace=False))
        images, mask = np.asarray(images[rows]), mask[rows]
        for model in models:
            r = evaluate(model, images, mask, k, trials=1, reader=reader)
            results.append(r.assign(writers=label, model=model.name))
            print(f"memorization  {model.name:8s} {label:17s} "
                  + "  ".join(f"{m}={r[m].mean():.3f}" for m in ["l1", "ssim", "iou", "legible"]), flush=True)
    return pd.concat(results, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", type=Path, nargs="*",
                        default=[Path("checkpoints/gan/ema_015000.pt"), Path("checkpoints/gan/ema_040000.pt")])
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 20])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--memorization-writers", type=int, default=300)
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    (args.out / "results").mkdir(parents=True, exist_ok=True)
    (args.out / "figures").mkdir(parents=True, exist_ok=True)

    models = [load_gan(p) for p in args.checkpoints]
    reader, judge = Reader(), StyleJudge()

    memo = memorization_check(models, reader, args.memorization_writers)
    memo.to_csv(args.out / "results" / "gan_memorization.csv", index=False)

    test_sets = {"fonts (held out)": load_fonts("test"), "NIST (high school)": load_handwriting("nist", "test"),
                 "drawn (ours)": load_handwriting("drawn")}
    results = []
    for set_name, (images, mask, meta) in test_sets.items():
        gallery = StyleGallery(judge, images, mask, meta["writer"] if "writer" in meta else None)
        for model in models:
            for k in args.ks:
                r = evaluate(model, images, mask, k, trials=args.trials, reader=reader, style=gallery)
                results.append(r.assign(test_set=set_name, model=model.name))
                print(f"{set_name:20s} {model.name:8s} k={k:2d}  "
                      + "  ".join(f"{m}={r[m].mean():.3f}" for m in METRICS), flush=True)
    results = pd.concat(results, ignore_index=True)
    results.to_csv(args.out / "results" / "gan.csv", index=False)
    summary = results.groupby(["test_set", "model", "k"])[METRICS].mean().round(3)
    summary.to_csv(args.out / "results" / "gan_summary.csv")

    baselines = pd.read_csv(args.out / "results" / "baselines_summary.csv", index_col=["test_set", "model", "k"])
    table = pd.concat([baselines, summary]).xs(5, level="k")
    print("\nAll models, k=5:\n", table.to_string())

    library, _, _ = load_fonts("train")
    figure_models = [NearestFont(library), *models]
    images, mask, _ = test_sets["NIST (high school)"]
    example_figure(figure_models, images, mask, [0, 1, 2], 5, args.out / "figures" / "gan_examples_nist.png")
    images, mask, _ = test_sets["drawn (ours)"]
    example_figure(figure_models, images, mask, [3, 6, 9], 5, args.out / "figures" / "gan_examples_drawn.png")


if __name__ == "__main__":
    main()
