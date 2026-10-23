"""Train the few-shot handwriting GAN.

Data: clean training fonts + NIST training writers (half the examples each). Every example:
k = 1..8 reference glyphs of one writer and one target glyph of the same writer.

One training step (two players, taking turns):
  1. Critic (discriminator) step: show it real target glyphs and generated ones; it learns
     to score real ones high and generated ones low (hinge loss). With --style-critic it
     also gets the writer's id and judges "is this a real glyph of THIS writer?".
  2. Generator step: generate again; it learns to (a) make the critic score its glyphs as
     real, and (b) match the real target glyph (L1 loss, weight --l1).
  3. Update the EMA copy of the generator (a smoothed average of its weights).
GAN losses do not decrease steadily - the two players push against each other - so
progress is judged from the sample images and the periodic validation scores.

The evaluation networks (character classifier, style encoder) are NOT used for training:
a model trained to please its judges would get inflated scores.

Outputs: <out>/last.pt (everything, for resuming; every --sample-every steps), <out>/ema_<step>.pt,
outputs/<run>/samples/step_<step>.png, outputs/results/<run>_history.json  (<run> = name of the --out folder)
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
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tulisanku.charset import CHAR_TO_INDEX  # noqa: E402
from tulisanku.data import load_fonts, load_handwriting  # noqa: E402
from tulisanku.evaluation import evaluate  # noqa: E402
from tulisanku.fewshot_data import FewShotBatches  # noqa: E402
from tulisanku.gan_model import GanModel  # noqa: E402
from tulisanku.legibility import Reader  # noqa: E402
from tulisanku.models.gan import Discriminator, Generator  # noqa: E402
from tulisanku.style import StyleGallery, StyleJudge  # noqa: E402

DEVICE = "cuda"
CONTENT_FONT = "Inter"
SAMPLE_REFS = "AgkR7"  # the same 5 letters for every sample sheet, as if typed into the app


def sample_sheet(model, writers, path, step):
    """For each writer: the 5 reference glyphs, the generated alphabet, the real alphabet."""
    refs = np.array([CHAR_TO_INDEX[c] for c in SAMPLE_REFS])
    rows = []
    for name, glyphs, mask in writers:
        use = refs[mask[refs]]
        pred = model.predict(glyphs[use], use, np.arange(62))
        blank = np.zeros((64, 64), np.float32)
        rows += [np.concatenate([glyphs[r] / 255 for r in use] + [blank] * (62 - len(use)), axis=1),
                 np.concatenate(list(pred), axis=1), np.concatenate(list(glyphs / 255), axis=1)]
    img = Image.fromarray((255 - np.concatenate(rows) * 255).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)
    for i, (name, _, _) in enumerate(writers):
        for j, label in enumerate(["refs", "generated", "real"]):
            draw.text((2, (3 * i + j) * 64 + 2), f"{name} {label}" if j == 0 else label, fill=(220, 0, 0))
    draw.text((img.width - 120, 2), f"step {step}", fill=(220, 0, 0))
    img.save(path)


def critic_loss(real_scores, fake_scores):
    """Hinge loss, averaged over the critic's heads (character, and writer with --style-critic)."""
    return sum(F.relu(1 - r).mean() + F.relu(1 + f).mean() for r, f in zip(real_scores, fake_scores)) / len(real_scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--max-refs", type=int, default=8)
    parser.add_argument("--lr-g", type=float, default=2e-4)
    parser.add_argument("--lr-d", type=float, default=2e-4)
    parser.add_argument("--l1", type=float, default=50.0, help="weight of 'match the real glyph' vs 'fool the critic'")
    parser.add_argument("--style-critic", action="store_true",
                        help="critic also checks whose handwriting it is (one embedding per training writer/font)")
    parser.add_argument("--ema", type=float, default=0.999)
    parser.add_argument("--sample-every", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/gan"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True

    fonts, font_mask, font_meta = load_fonts("train")
    nist, nist_mask, _ = load_handwriting("nist", "train")
    data = FewShotBatches([(fonts, font_mask), (nist, nist_mask)], batch_size=args.batch_size, max_refs=args.max_refs)
    content_bank = np.asarray(fonts[int(font_meta.index[font_meta.family == CONTENT_FONT][0])], np.float32) / 255
    bank = torch.from_numpy(content_bank).to(DEVICE)

    config = {"width": 64, "style_dim": 256}
    G = Generator(**config).to(DEVICE)
    D = Discriminator(n_styles=data.n_writers if args.style_critic else 0).to(DEVICE)
    G_ema = copy.deepcopy(G).eval().requires_grad_(False)
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.0, 0.99))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.0, 0.99))
    print(f"generator {sum(p.numel() for p in G.parameters()):,} parameters, "
          f"discriminator {sum(p.numel() for p in D.parameters()):,}")

    start, history = 0, []
    args.out.mkdir(parents=True, exist_ok=True)
    if args.resume and (args.out / "last.pt").exists():
        ck = torch.load(args.out / "last.pt", map_location=DEVICE)
        G.load_state_dict(ck["generator"]); D.load_state_dict(ck["discriminator"])
        G_ema.load_state_dict(ck["generator_ema"])
        opt_g.load_state_dict(ck["opt_g"]); opt_d.load_state_dict(ck["opt_d"])
        start, history = ck["step"], ck["history"]
        print(f"resumed from step {start}")

    # Fixed writers for the sample sheets: 4 unseen NIST writers + 4 drawn sessions (incl. the team's).
    val_images, val_mask, _ = load_handwriting("nist", "val")
    drawn, drawn_mask, drawn_meta = load_handwriting("drawn")
    show = [(f"NIST val {i}", np.asarray(val_images[i]), val_mask[i]) for i in range(4)]
    for session in ["27dddc07", *drawn_meta.session[drawn_meta.writer != "team-1"][:3]]:
        i = int(drawn_meta.index[drawn_meta.session == session][0])
        show.append((f"drawn {session}", np.asarray(drawn[i]), drawn_mask[i]))
    run = args.out.name  # "gan" for the default run; other runs keep their own samples and history
    samples_dir = Path("outputs") / run / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    reader, judge = Reader(), StyleJudge()
    val_gallery = StyleGallery(judge, val_images[:100], val_mask[:100])
    ema_model = GanModel(None, generator=G_ema, content_bank=content_bank)

    batches = data.batches(seed=start)
    t0, logs = time.time(), []
    for step in range(start + 1, args.steps + 1):
        refs, ref_mask, char, target, writer = next(batches)
        refs = torch.from_numpy(refs).to(DEVICE).float().div_(255).unsqueeze(2)       # [B, K, 1, H, W]
        ref_mask = torch.from_numpy(ref_mask).to(DEVICE)
        char, writer = torch.from_numpy(char).to(DEVICE), torch.from_numpy(writer).to(DEVICE)
        target = torch.from_numpy(target).to(DEVICE).float().div_(255).unsqueeze(1)  # [B, 1, H, W]
        content = bank[char].unsqueeze(1)

        # 1. Critic step (the critic runs in full 32-bit precision: small, and more stable)
        with torch.no_grad(), torch.autocast(DEVICE, dtype=torch.bfloat16):
            fake = G(content, refs, ref_mask)
        d_loss = critic_loss(D(target, char, writer), D(fake.float(), char, writer))
        opt_d.zero_grad(set_to_none=True)
        d_loss.backward()
        opt_d.step()

        # 2. Generator step
        with torch.autocast(DEVICE, dtype=torch.bfloat16):
            fake = G(content, refs, ref_mask)
        fake = fake.float()
        scores = D(fake, char, writer)
        adv = -sum(s.mean() for s in scores) / len(scores)
        l1 = (fake - target).abs().mean()
        g_loss = adv + args.l1 * l1
        opt_g.zero_grad(set_to_none=True)
        g_loss.backward()
        opt_g.step()

        # 3. EMA: the smoothed copy moves 0.1% of the way towards the current weights
        with torch.no_grad():
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.lerp_(p, 1 - args.ema)
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

        logs.append((d_loss.item(), adv.item(), l1.item()))
        if step % 200 == 0:
            d, a, l = np.mean(logs, axis=0)
            rate = 200 / (time.time() - t0)
            print(f"step {step:6d}  critic loss {d:.3f}  generator: fool-critic {a:+.3f}  L1 {l:.4f}  "
                  f"({rate:.1f} steps/s, ~{(args.steps - step) / rate / 60:.0f} min left)", flush=True)
            t0, logs = time.time(), []
        if step % args.sample_every == 0:
            sample_sheet(ema_model, show, samples_dir / f"step_{step:06d}.png", step)
        if step % args.eval_every == 0 or step == args.steps:
            r = evaluate(ema_model, val_images[:100], val_mask[:100], k=5, trials=1, reader=reader, style=val_gallery)
            scores = {m: float(r[m].mean()) for m in ["l1", "ssim", "iou", "legible", "legible_merged",
                                                      "style_top1", "style_pct"]}
            history.append({"step": step, **scores})
            print(f"  validation (100 NIST val writers, k=5): " + "  ".join(f"{m}={v:.3f}" for m, v in scores.items()),
                  flush=True)
            torch.save({"generator_ema": G_ema.state_dict(), "config": config, "content_bank": torch.from_numpy(content_bank),
                        "step": step, "val": scores}, args.out / f"ema_{step:06d}.pt")
            Path("outputs/results").mkdir(parents=True, exist_ok=True)
            json.dump(history, open(f"outputs/results/{run}_history.json", "w"), indent=2)
        # Resume point, saved often: an interrupted run loses at most --sample-every steps
        if step % args.sample_every == 0 or step == args.steps:
            torch.save({"generator": G.state_dict(), "discriminator": D.state_dict(),
                        "generator_ema": G_ema.state_dict(), "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                        "step": step, "history": history, "config": config, "content_bank": torch.from_numpy(content_bank)},
                       args.out / "last.pt")


if __name__ == "__main__":
    main()
