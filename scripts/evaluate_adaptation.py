"""Does per-person adaptation (tulisanku/adapt.py) make generated letters closer to the writer?

Every variant uses only the k reference letters of a person - never the letters it is
scored on - exactly like the other models (same references as evaluate_baselines.py).

  --stage select  try a grid of variants on NIST VALIDATION writers and pick settings there,
                  so the test sets stay untouched by the choice
  --stage test    score the chosen variants on the drawn (ours) and NIST test writers

Writes outputs/results/adaptation_<stage>.csv and prints a summary with seconds per person.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tulisanku.adapt import AdaptedGan  # noqa: E402
from tulisanku.data import load_handwriting  # noqa: E402
from tulisanku.evaluation import evaluate  # noqa: E402
from tulisanku.legibility import Reader  # noqa: E402
from tulisanku.style import StyleGallery, StyleJudge  # noqa: E402

METRICS = ["l1", "ssim", "iou", "legible", "style_top1", "style_pct"]
GRID = [dict(), dict(calibrate="global"), dict(calibrate="group"),
        dict(steps=100, part="code", lr=1e-2), dict(steps=300, part="code", lr=1e-2),
        dict(steps=50, part="decoder", lr=1e-4), dict(steps=150, part="decoder", lr=1e-4),
        dict(steps=300, part="decoder", lr=1e-4), dict(steps=150, part="decoder", lr=3e-4)]


def parse_variant(text: str) -> dict:
    """'decoder-150-1e-4+cal-group' -> dict(part='decoder', steps=150, lr=1e-4, calibrate='group');
    also 'base', 'cal-global'."""
    v = {}
    for piece in text.split("+"):
        if piece.startswith("cal-"):
            v["calibrate"] = piece[4:]
        elif piece != "base":
            part, steps, lr = piece.split("-", 2)
            v.update(part=part, steps=int(steps), lr=float(lr))
    return v


def run(variants, test_sets, checkpoint, k, trials, judge, reader):
    rows = []
    for set_name, (images, mask, writer_ids) in test_sets.items():
        gallery = StyleGallery(judge, images, mask, writer_ids)
        for v in variants:
            model = AdaptedGan(checkpoint, **v)
            if v.get("steps"):
                model.name += f" (lr {v['lr']:g})"
            t0 = time.time()
            r = evaluate(model, images, mask, k, trials=trials, reader=reader, style=gallery)
            seconds = (time.time() - t0) / len(r)
            rows.append(r.assign(test_set=set_name, model=model.name, seconds=seconds))
            print(f"{set_name:22s} {model.name:44s} " + "  ".join(f"{m}={r[m].mean():.3f}" for m in METRICS)
                  + f"  ({seconds:.2f} s/person)", flush=True)
    return pd.concat(rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["select", "test"], required=True)
    parser.add_argument("--variants", nargs="+", default=["base", "cal-group"],
                        help="test stage, e.g. base cal-global decoder-150-1e-4+cal-group")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/gan_style/ema_015000.pt"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--writers", type=int, default=100, help="NIST writers per set (fine-tuning takes time)")
    args = parser.parse_args()

    reader, judge = Reader(), StyleJudge()
    rng = np.random.default_rng(0)
    if args.stage == "select":
        images, mask, _ = load_handwriting("nist", "val")
        rows = np.sort(rng.choice(len(images), args.writers, replace=False))
        test_sets = {"NIST val": (np.asarray(images[rows]), mask[rows], None)}
        results = run(GRID, test_sets, args.checkpoint, args.k, 1, judge, reader)
    else:
        images, mask, _ = load_handwriting("nist", "test")
        rows = np.sort(rng.choice(len(images), args.writers, replace=False))
        drawn, drawn_mask, drawn_meta = load_handwriting("drawn")
        test_sets = {"drawn (ours)": (drawn, drawn_mask, drawn_meta["writer"]),
                     "NIST test": (np.asarray(images[rows]), mask[rows], None)}
        results = run([parse_variant(v) for v in args.variants], test_sets, args.checkpoint, args.k, 3, judge, reader)

    Path("outputs/results").mkdir(parents=True, exist_ok=True)
    results.to_csv(f"outputs/results/adaptation_{args.stage}.csv", index=False)
    summary = results.groupby(["test_set", "model"], sort=False)[METRICS + ["seconds"]].mean().round(3)
    print("\n", summary.to_string())


if __name__ == "__main__":
    main()
