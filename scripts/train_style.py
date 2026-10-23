"""Train the style encoder: learn to recognize WHO wrote a glyph.

Training data: NIST training writers (each writer is one class, ~2,800 classes).
Validation (every epoch): split-half identification on NIST validation writers - can
each writer's first half of characters be matched to their own second half (different
letters) among 301 writers?
Test (once): the same on NIST high-school writers and drawn handwriting, compared with
two simpler "embeddings": raw pixels and the character classifier's features.

Augmentation is deliberately gentle (tiny shifts and size changes only): slant and stroke
thickness ARE style, so distorting them would teach the network to ignore its best clues.

Writes checkpoints/style.pt and outputs/results/style.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tulisanku.augment import random_affine  # noqa: E402
from tulisanku.baselines import shrink  # noqa: E402
from tulisanku.data import load_handwriting  # noqa: E402
from tulisanku.legibility import Reader  # noqa: E402
from tulisanku.models.style import CosFaceHead, StyleEncoder  # noqa: E402
from tulisanku.style import StyleJudge, split_half_identification  # noqa: E402
from tulisanku.training_data import GlyphBatches  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def embed_all(encoder, images, mask) -> np.ndarray:
    encoder.eval()
    rows, chars = np.nonzero(mask)
    emb = np.zeros(np.asarray(mask).shape + (encoder.project.out_features,), np.float32)
    for s in range(0, len(rows), 2048):
        r, c = rows[s:s + 2048], chars[s:s + 2048]
        x = torch.from_numpy(np.asarray(images[r, c])).to(DEVICE).float().div_(255).unsqueeze(1)
        with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
            emb[r, c] = encoder(x).float().cpu().numpy()
    return emb


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/style.pt"))
    args = parser.parse_args()
    torch.manual_seed(0)

    train_images, train_mask, _ = load_handwriting("nist", "train")
    val_images, val_mask, _ = load_handwriting("nist", "val")
    batches = GlyphBatches([(train_images, train_mask)], batch_size=args.batch_size, labels="row")
    n_writers = len(train_images)
    print(f"{batches.n_samples:,} glyphs from {n_writers:,} training writers, {len(batches)} batches per epoch")

    encoder, head = StyleEncoder().to(DEVICE), CosFaceHead(128, n_writers).to(DEVICE)
    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epochs * len(batches),
                                                    pct_start=0.15)
    scaler = torch.amp.GradScaler(DEVICE, enabled=DEVICE == "cuda")
    loss_fn = nn.CrossEntropyLoss()

    best, history = -1.0, []
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        start, total, correct, seen = time.time(), 0.0, 0, 0
        encoder.train()
        for x, writer in batches.epoch(shuffle=True, seed=epoch):
            x = torch.from_numpy(x).to(DEVICE).float().div_(255).unsqueeze(1)
            x = random_affine(x, rotate_deg=0, scale=(0.95, 1.05), shift=0.03, shear=0)  # gentle on purpose
            writer = torch.from_numpy(writer).to(DEVICE)
            with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
                logits = head(encoder(x).float(), writer)
                loss = loss_fn(logits, writer)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += loss.item() * len(writer)
            correct += (logits.argmax(1) == writer).sum().item()
            seen += len(writer)
        val = split_half_identification(embed_all(encoder, val_images, val_mask), val_mask)
        history.append({"epoch": epoch, "loss": total / seen, "train_writer_acc": correct / seen, **val})
        saved = ""
        if val["style_pct"] > best:
            best = val["style_pct"]
            torch.save({"encoder": encoder.state_dict(), "epoch": epoch, "val": val}, args.checkpoint)
            saved = "  <- best so far, saved"
        print(f"epoch {epoch:2d}  loss {total / seen:.3f}  single-glyph writer acc {correct / seen:.3f}  "
              f"val: top1 {val['style_top1']:.3f}  top5 {val['style_top5']:.3f}  pct {val['style_pct']:.3f}  "
              f"({time.time() - start:.0f}s){saved}", flush=True)

    # Test once, against two simpler embeddings, to show what a dedicated style network adds.
    judge, reader = StyleJudge(args.checkpoint), Reader()
    tests = {"NIST (high school)": load_handwriting("nist", "test"), "drawn (ours)": load_handwriting("drawn")}
    results = {}
    for name, (images, mask, meta) in tests.items():
        writers = meta["writer"] if "writer" in meta else None
        rows, chars = np.nonzero(mask)
        glyphs = np.asarray(images[rows, chars])
        pixels = np.zeros(mask.shape + (32 * 32,), np.float32)
        pixels[rows, chars] = shrink(glyphs)
        clf = np.zeros(mask.shape + (256,), np.float32)
        with torch.no_grad():
            for s in range(0, len(rows), 2048):
                x = torch.from_numpy(glyphs[s:s + 2048]).to(DEVICE).float().div_(255).unsqueeze(1)
                clf[rows[s:s + 2048], chars[s:s + 2048]] = reader.model.features(x).float().cpu().numpy()
        results[name] = {
            "raw pixels": split_half_identification(pixels, mask, writers),
            "character classifier features": split_half_identification(clf, mask, writers),
            "style encoder": split_half_identification(judge.embed_dataset(images, mask), mask, writers),
        }
        print(f"\n{name} - split-half writer identification among {len(mask)} sessions:")
        for method, r in results[name].items():
            print(f"  {method:30s} top1 {r['style_top1']:.3f}  top5 {r['style_top5']:.3f}  pct {r['style_pct']:.3f}")

    Path("outputs/results").mkdir(parents=True, exist_ok=True)
    json.dump({"history": history, "test": results}, open("outputs/results/style.json", "w"), indent=2)


if __name__ == "__main__":
    main()
