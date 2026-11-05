"""Does per-person adaptation (tulisanku/adapt.py) make generated letters closer to the writer?

Every variant uses only the k reference letters of a person - never the letters it is
scored on - exactly like the other models (same references as evaluate_baselines.py).

  --stage select  a grid of variants on NIST VALIDATION writers, to pick settings there
  --stage drawn   given variants on the drawn handwriting (ours), 3 trials per person
  --stage cv      choose among the drawn results without peeking: 2-fold cross-validation
                  over people - choose on one half, score the choice on the other half, swap
  --stage nist    given variants on NIST test writers

drawn and nist append to outputs/results/adaptation_<stage>.csv after every variant and
skip variants already there, so an interrupted run continues where it stopped. Variants
are written like  base  decoder-150-1e-4+anchor-10+clean  (tokens: <part>-<steps>-<lr>,
anchor-<w>, critic-<w>, cal-<global|group>, clean, sharpen). Variants that only differ
in post-processing (clean, sharpen, cal) share one fine-tuning per person.
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
CRITIC = "checkpoints/gan_style/last.pt"
GRID = ["base", "cal-global", "cal-group", "code-100-1e-2", "code-300-1e-2", "decoder-50-1e-4",
        "decoder-150-1e-4", "decoder-300-1e-4", "decoder-150-3e-4"]
FINETUNE_KEYS = ("steps", "part", "lr", "anchor", "critic", "critic_weight")


def parse_variant(text: str) -> dict:
    """'decoder-150-1e-4+anchor-10+clean' -> dict(part='decoder', steps=150, lr=1e-4, anchor=10, clean=True)."""
    v = {}
    for piece in text.split("+"):
        name, _, value = piece.partition("-")
        if piece == "base":
            continue
        elif piece in ("clean", "sharpen"):
            v[piece] = True
        elif name == "cal":
            v["calibrate"] = value
        elif name == "anchor":
            v["anchor"] = float(value)
        elif name == "critic":
            v.update(critic=CRITIC, critic_weight=float(value))
        else:
            part, steps, lr = piece.split("-", 2)
            v.update(part=part, steps=int(steps), lr=float(lr))
    return v


def run(variants, test_sets, checkpoint, k, trials, judge, reader, out_csv: Path = None):
    done = pd.read_csv(out_csv) if out_csv is not None and out_csv.exists() else None
    rows, cache_key, cache = [], None, {}
    for set_name, (images, mask, writer_ids) in test_sets.items():
        gallery = StyleGallery(judge, images, mask, writer_ids)
        for text in variants:
            v = parse_variant(text)
            key = (set_name, *(v.get(name) for name in FINETUNE_KEYS))
            if key != cache_key:  # keep only the fine-tuning results of the current settings
                cache_key, cache = key, {}
            model = AdaptedGan(checkpoint, cache=cache, **v)
            if v.get("steps"):
                model.name += f" (lr {v['lr']:g})"
            if done is not None and ((done.test_set == set_name) & (done.model == model.name)).any():
                print(f"{set_name:16s} {model.name:60s} already done", flush=True)
                continue
            t0 = time.time()
            r = evaluate(model, images, mask, k, trials=trials, reader=reader, style=gallery)
            r = r.assign(test_set=set_name, model=model.name, variant=text, seconds=(time.time() - t0) / len(r))
            rows.append(r)
            print(f"{set_name:16s} {model.name:60s} " + "  ".join(f"{m}={r[m].mean():.3f}" for m in METRICS)
                  + f"  ({r.seconds.iloc[0]:.2f} s/person)", flush=True)
            if out_csv is not None:
                r.to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
    return pd.concat(rows, ignore_index=True) if rows else None


def cross_validate(path: Path, max_legibility_drop: float = 0.03):
    """2-fold CV over people: on each half, pick the variant with the best style score that
    loses at most `max_legibility_drop` legibility against the plain GAN; score it on the other half."""
    r = pd.read_csv(path)
    _, _, meta = load_handwriting("drawn")
    r["person"] = meta["writer"].to_numpy()[r.writer]
    people = np.sort(r.person.unique())
    r["fold"] = r.person.map({p: i % 2 for i, p in enumerate(people)})
    print("All variants, all 32 sessions (for reading, not for choosing):")
    print(r.groupby("model", sort=False)[METRICS + ["seconds"]].mean().round(3).to_string(), "\n")
    per_fold = r.groupby(["fold", "model"])[METRICS].mean()
    base = r.model[r.variant == "base"].iloc[0]
    held_out = []
    for test_fold in (0, 1):
        choose = per_fold.loc[1 - test_fold]
        ok = choose[choose.legible >= choose.loc[base, "legible"] - max_legibility_drop]
        best = ok.style_pct.idxmax()
        score = per_fold.loc[test_fold]
        held_out.append(score.loc[[base, best]].assign(chosen=best))
        sessions = r[r.fold == 1 - test_fold].writer.nunique()
        print(f"choose on half {1 - test_fold} ({sessions} sessions), score on half {test_fold}: {best}")
        print(score.loc[[base, best]].round(3).to_string(), "\n")
    both = pd.concat(held_out).reset_index()
    both["role"] = np.where(both.model == base, "plain GAN", "chosen variant")
    print("Held-out average (each half scored by a choice made on the other half):")
    print(both.groupby("role")[METRICS].mean().round(3).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["select", "drawn", "cv", "nist"], required=True)
    parser.add_argument("--variants", nargs="+", default=["base"])
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/gan_style/ema_015000.pt"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--writers", type=int, default=100, help="NIST writers per set (fine-tuning takes time)")
    args = parser.parse_args()
    results_dir = Path("outputs/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / f"adaptation_{args.stage}.csv"
    if args.stage == "cv":
        return cross_validate(results_dir / "adaptation_drawn.csv")

    reader, judge = Reader(), StyleJudge()
    rng = np.random.default_rng(0)
    if args.stage == "drawn":
        drawn, drawn_mask, drawn_meta = load_handwriting("drawn")
        run(args.variants, {"drawn (ours)": (drawn, drawn_mask, drawn_meta["writer"])},
            args.checkpoint, args.k, 3, judge, reader, out_csv)
        return
    split, trials, variants = ("val", 1, GRID) if args.stage == "select" else ("test", 3, args.variants)
    images, mask, _ = load_handwriting("nist", split)
    rows = np.sort(rng.choice(len(images), args.writers, replace=False))
    run(variants, {f"NIST {split}": (np.asarray(images[rows]), mask[rows], None)},
        args.checkpoint, args.k, trials, judge, reader, out_csv)


if __name__ == "__main__":
    main()
