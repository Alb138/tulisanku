"""Choose the diffusion model's generation settings on NIST VALIDATION writers.

  guidance    1 = plain conditional model; > 1 pushes each denoising step further towards the
              person's style (classifier-free guidance; each step then runs the U-Net twice)
  ddim steps  how many denoising steps one generation takes: fewer = faster, maybe rougher

Settings are written as <guidance>:<steps>, e.g.  1:25 2:25 3:25  or  2:10 2:50.
Results are appended to outputs/results/diffusion_select.csv after every setting, and
settings already there are skipped, so an interrupted run continues where it stopped.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tulisanku.data import load_handwriting  # noqa: E402
from tulisanku.diffusion_model import DiffusionModel  # noqa: E402
from tulisanku.evaluation import evaluate  # noqa: E402
from tulisanku.legibility import Reader  # noqa: E402
from tulisanku.style import StyleGallery, StyleJudge  # noqa: E402

METRICS = ["l1", "ssim", "iou", "legible", "style_top1", "style_pct"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("settings", nargs="+", help="<guidance>:<ddim steps>, e.g. 1:25 2:25")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/diffusion/ema_030000.pt"))
    parser.add_argument("--writers", type=int, default=50)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    images, mask, _ = load_handwriting("nist", "val")
    rows = np.sort(np.random.default_rng(0).choice(len(images), args.writers, replace=False))
    images, mask = np.asarray(images[rows]), mask[rows]
    reader = Reader()
    gallery = StyleGallery(StyleJudge(), images, mask)
    out = Path("outputs/results/diffusion_select.csv")
    done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["checkpoint", "guidance", "ddim_steps"])
    model = DiffusionModel(args.checkpoint)
    model.predict(images[0][:5], np.arange(5), np.arange(62))  # one-time GPU tuning, not counted in the timing

    for setting in args.settings:
        guidance, steps = float(setting.split(":")[0]), int(setting.split(":")[1])
        if ((done.checkpoint == str(args.checkpoint)) & (done.guidance == guidance) & (done.ddim_steps == steps)).any():
            print(f"guidance {guidance:g}, {steps} steps: already done")
            continue
        model.guidance, model.n_steps = guidance, steps
        if guidance != 1:  # the doubled batch needs its own one-time tuning
            model.predict(images[0][:5], np.arange(5), np.arange(62))
        t0 = time.time()
        r = evaluate(model, images, mask, args.k, trials=1, reader=reader, style=gallery)
        r = r.assign(checkpoint=str(args.checkpoint), guidance=guidance, ddim_steps=steps,
                     seconds=(time.time() - t0) / len(r))
        r.to_csv(out, mode="a", header=not out.exists(), index=False)
        print(f"guidance {guidance:g}, {steps:2d} steps: " + "  ".join(f"{m}={r[m].mean():.3f}" for m in METRICS)
              + f"  ({r.seconds.iloc[0]:.1f} s/person)", flush=True)

    results = pd.read_csv(out)
    results = results[results.checkpoint == str(args.checkpoint)]
    print("\n", results.groupby(["guidance", "ddim_steps"])[METRICS + ["seconds"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
