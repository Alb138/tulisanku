"""Train the few-shot handwriting diffusion model (tulisanku/models/diffusion.py).

Data: the same examples as the GAN - clean training fonts + NIST training writers, each
example k = 1..8 reference glyphs and one target glyph of the same writer.

One training step:
  1. Pick a random time t for every example and mix the real target glyph with noise:
     the larger t, the more noise (at the last t it is pure noise).
  2. The U-Net sees the noisy glyph, the plain content glyph and the style of the
     references (for 10% of the examples: the "no style" vector instead) and predicts the
     noise that was added. Loss: mean squared error between predicted and real noise.
  3. Update the EMA copy of the U-Net (a smoothed average of its weights, used for sampling).
Unlike a GAN there is no opponent: the loss goes down steadily and is a fair progress signal.

The evaluation networks (character classifier, style encoder) are NOT used for training.

Outputs: <out>/last.pt (for --resume, every --sample-every steps), <out>/ema_<step>.pt,
outputs/<run>/samples/step_<step>.png, outputs/results/<run>_history.json  (<run> = name of --out)
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_gan import CONTENT_FONT, sample_sheet, sample_writers  # noqa: E402
from tulisanku.data import load_fonts, load_handwriting  # noqa: E402
from tulisanku.diffusion_model import DiffusionModel  # noqa: E402
from tulisanku.evaluation import evaluate  # noqa: E402
from tulisanku.fewshot_data import FewShotBatches  # noqa: E402
from tulisanku.legibility import Reader  # noqa: E402
from tulisanku.models.diffusion import Diffusion, UNet  # noqa: E402
from tulisanku.style import StyleGallery, StyleJudge  # noqa: E402

DEVICE = "cuda"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-refs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=1000, help="steps to ramp the learning rate up from 0")
    parser.add_argument("--p-uncond", type=float, default=0.1, help="share of examples trained without style")
    # Sized for a 4 GB laptop GPU: (64, 128, 192, 256) with 2 blocks needs 6.4 GB at batch 64 and
    # spills into system memory (9 s/step); this fits in 2.5 GB at 0.33 s/step.
    parser.add_argument("--channels", type=int, nargs="+", default=[32, 64, 128, 192])
    parser.add_argument("--blocks", type=int, default=1, help="residual blocks per level on the way down")
    parser.add_argument("--ema", type=float, default=0.999)
    parser.add_argument("--sample-every", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=10000, help="generating is slow (~3 s per writer)")
    parser.add_argument("--val-writers", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/diffusion"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True

    fonts, font_mask, font_meta = load_fonts("train")
    nist, nist_mask, _ = load_handwriting("nist", "train")
    data = FewShotBatches([(fonts, font_mask), (nist, nist_mask)], batch_size=args.batch_size, max_refs=args.max_refs)
    content_bank = np.asarray(fonts[int(font_meta.index[font_meta.family == CONTENT_FONT][0])], np.float32) / 255
    bank = torch.from_numpy(content_bank).to(DEVICE) * 2 - 1  # ink -> -1..1, like the glyphs

    config = {"channels": tuple(args.channels), "blocks": args.blocks,
              "attention": tuple(i >= len(args.channels) - 2 for i in range(len(args.channels)))}  # attention at 16 and 8 px
    unet = UNet(**config).to(DEVICE)
    unet_ema = copy.deepcopy(unet).eval().requires_grad_(False)
    diffusion = Diffusion(device=DEVICE)
    opt = torch.optim.AdamW(unet.parameters(), lr=args.lr, weight_decay=0.0)
    schedule = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup))
    print(f"U-Net {sum(p.numel() for p in unet.parameters()):,} parameters", flush=True)

    start, history = 0, []
    args.out.mkdir(parents=True, exist_ok=True)
    if args.resume and (args.out / "last.pt").exists():
        ck = torch.load(args.out / "last.pt", map_location=DEVICE)
        unet.load_state_dict(ck["unet"]); unet_ema.load_state_dict(ck["unet_ema"])
        opt.load_state_dict(ck["opt"]); schedule.load_state_dict(ck["schedule"])
        start, history = ck["step"], ck["history"]
        print(f"resumed from step {start}", flush=True)

    val_images, val_mask, _ = load_handwriting("nist", "val")
    show = sample_writers(val_images, val_mask)
    run = args.out.name
    samples_dir = Path("outputs") / run / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    reader, judge = Reader(), StyleJudge()
    n_val = args.val_writers
    val_gallery = StyleGallery(judge, val_images[:n_val], val_mask[:n_val])
    ema_model = DiffusionModel(unet=unet_ema, content_bank=content_bank)
    save_bank = torch.from_numpy(content_bank)

    batches = data.batches(seed=start)
    t0, losses = time.time(), []
    for step in range(start + 1, args.steps + 1):
        refs, ref_mask, char, target, _ = next(batches)
        refs = torch.from_numpy(refs).to(DEVICE).float().div_(255).unsqueeze(2)           # [B, K, 1, H, W]
        ref_mask = torch.from_numpy(ref_mask).to(DEVICE)
        x0 = torch.from_numpy(target).to(DEVICE).float().div_(127.5).sub_(1).unsqueeze(1)  # [B, 1, H, W], -1..1
        content = bank[torch.from_numpy(char).to(DEVICE)].unsqueeze(1)

        # 1. noise the real glyphs at random times
        t = torch.randint(0, diffusion.steps, (len(x0),), device=DEVICE)
        noise = torch.randn_like(x0)
        x_t = diffusion.add_noise(x0, t, noise)
        # 2. predict the noise (style hidden for p_uncond of the examples, for guidance later)
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            style = unet.style_embedding(refs, ref_mask)
            drop = torch.rand(len(x0), device=DEVICE) < args.p_uncond
            style = torch.where(drop[:, None], unet.null_style.to(style.dtype).expand_as(style), style)
            pred = unet(x_t, t, content, style)
        loss = F.mse_loss(pred.float(), noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        opt.step()
        schedule.step()
        # 3. EMA
        with torch.no_grad():
            for p_ema, p in zip(unet_ema.parameters(), unet.parameters()):
                p_ema.lerp_(p, 1 - args.ema)

        losses.append(loss.item())
        if step % 200 == 0:
            rate = 200 / (time.time() - t0)
            print(f"step {step:6d}  noise-prediction loss {np.mean(losses):.4f}  "
                  f"({rate:.1f} steps/s, ~{(args.steps - step) / rate / 60:.0f} min left)", flush=True)
            t0, losses = time.time(), []
        if step % args.sample_every == 0:
            sample_sheet(ema_model, show, samples_dir / f"step_{step:06d}.png", step)
        if step % args.eval_every == 0 or step == args.steps:
            r = evaluate(ema_model, val_images[:n_val], val_mask[:n_val], k=5, trials=1, reader=reader, style=val_gallery)
            scores = {m: float(r[m].mean()) for m in ["l1", "ssim", "iou", "legible", "legible_merged",
                                                      "style_top1", "style_pct"]}
            history.append({"step": step, **scores})
            print(f"  validation ({n_val} NIST val writers, k=5): " + "  ".join(f"{m}={v:.3f}" for m, v in scores.items()),
                  flush=True)
            torch.save({"unet_ema": unet_ema.state_dict(), "config": config, "content_bank": save_bank,
                        "step": step, "val": scores}, args.out / f"ema_{step:06d}.pt")
            Path("outputs/results").mkdir(parents=True, exist_ok=True)
            json.dump(history, open(f"outputs/results/{run}_history.json", "w"), indent=2)
        if step % args.sample_every == 0 or step == args.steps:  # resume point
            torch.save({"unet": unet.state_dict(), "unet_ema": unet_ema.state_dict(), "opt": opt.state_dict(),
                        "schedule": schedule.state_dict(), "step": step, "history": history, "config": config,
                        "content_bank": save_bank}, args.out / "last.pt")


if __name__ == "__main__":
    main()
