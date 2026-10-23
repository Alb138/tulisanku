"""Train the character classifier (64x64 glyph -> one of 62 characters).

Training data: clean training fonts (without caps-only fonts) + NIST training writers.
Validation:    font and NIST validation splits, checked after every epoch.
Test (once, at the end): held-out fonts, NIST high-school writers, drawn handwriting.

Writes checkpoints/classifier.pt (best validation epoch), outputs/results/classifier.json
and outputs/figures/classifier_curves.png.

The training loop, step by step (see `train_one_epoch`):
  1. take a batch of glyphs and move it to the GPU
  2. augment it (random distortions)
  3. forward pass: the model scores all 62 characters for every glyph
  4. loss: cross-entropy - how wrong the scores are compared with the true characters
  5. backward pass: compute how each weight should change to lower the loss
  6. optimizer step: change the weights a little in that direction
Mixed precision runs most of this in 16-bit numbers (~4x faster on this GPU).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tulisanku.augment import augment  # noqa: E402
from tulisanku.charset import CHARSET  # noqa: E402
from tulisanku.data import load_fonts, load_handwriting  # noqa: E402
from tulisanku.models.classifier import CharClassifier  # noqa: E402
from tulisanku.training_data import GlyphBatches  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).to(DEVICE).float().div_(255).unsqueeze(1)  # [B, 1, H, W], 0..1


def look_alike(char: str) -> str:
    """Merge characters that are often identical in handwriting: case, 0/O/o, 1/I/l."""
    if char in "0Oo":
        return "o"
    if char in "1Il":
        return "l"
    return char.lower()


LOOK_ALIKE = np.array([look_alike(c) for c in CHARSET])


def train_one_epoch(model, batches, optimizer, scheduler, scaler, loss_fn, seed):
    model.train()
    total_loss = correct = seen = 0
    for x, y in batches.epoch(shuffle=True, seed=seed):
        x, y = augment(to_tensor(x)), torch.from_numpy(y).to(DEVICE)          # 1-2
        with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
            logits = model(x)                                                # 3
            loss = loss_fn(logits, y)                                        # 4
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()                                        # 5
        scaler.step(optimizer)                                               # 6
        scaler.update()
        scheduler.step()
        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        seen += len(y)
    return total_loss / seen, correct / seen


@torch.no_grad()
def predict(model, images, mask, batch_size: int = 1024):
    """Predicted and true character indices for every available glyph (no augmentation)."""
    model.eval()
    preds, truths = [], []
    for x, y in GlyphBatches([(images, mask)], batch_size=batch_size).epoch(shuffle=False):
        with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
            preds.append(model(to_tensor(x)).argmax(1).cpu().numpy())
        truths.append(y)
    return np.concatenate(preds), np.concatenate(truths)


def accuracy(model, images, mask) -> dict:
    p, t = predict(model, images, mask)
    return {"exact": float((p == t).mean()), "look_alike_merged": float((LOOK_ALIKE[p] == LOOK_ALIKE[t]).mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/classifier.pt"))
    args = parser.parse_args()
    torch.manual_seed(0)

    fonts_train, fm_train, _ = load_fonts("train", caps_only=False)
    nist_train, nm_train, _ = load_handwriting("nist", "train")
    train = GlyphBatches([(fonts_train, fm_train), (nist_train, nm_train)], batch_size=args.batch_size)
    val = {"fonts": load_fonts("val", caps_only=False)[:2], "NIST": load_handwriting("nist", "val")[:2]}
    print(f"{train.n_samples:,} training glyphs, {len(train)} batches per epoch, device: {DEVICE}")

    model = CharClassifier().to(DEVICE)
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    # Warm up, then slowly lower the learning rate: big steps early, careful steps at the end.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epochs * len(train),
                                                    pct_start=0.15)
    scaler = torch.amp.GradScaler(DEVICE, enabled=DEVICE == "cuda")
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)  # don't let it become 100% sure: helps with ambiguous glyphs

    history, best = [], -1.0
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        loss, train_acc = train_one_epoch(model, train, optimizer, scheduler, scaler, loss_fn, seed=epoch)
        val_acc = {name: accuracy(model, *data)["exact"] for name, data in val.items()}
        score = float(np.mean(list(val_acc.values())))
        history.append({"epoch": epoch, "loss": loss, "train_acc": train_acc, **{f"val_{k}": v for k, v in val_acc.items()}})
        saved = ""
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": val_acc}, args.checkpoint)
            saved = "  <- best so far, saved"
        print(f"epoch {epoch:2d}  loss {loss:.3f}  train acc {train_acc:.3f}  "
              f"val fonts {val_acc['fonts']:.3f}  val NIST {val_acc['NIST']:.3f}  ({time.time() - start:.0f}s){saved}",
              flush=True)

    # Test sets: touched only once, with the best checkpoint.
    model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE)["model"])
    tests = {"fonts (held out)": load_fonts("test", caps_only=False)[:2],
             "NIST (high school)": load_handwriting("nist", "test")[:2], "drawn (ours)": load_handwriting("drawn")[:2]}
    test_acc = {name: accuracy(model, *data) for name, data in tests.items()}
    print("\nTest accuracy (exact / look-alikes merged):")
    for name, acc in test_acc.items():
        print(f"  {name:20s} {acc['exact']:.3f} / {acc['look_alike_merged']:.3f}")

    p, t = predict(model, *tests["NIST (high school)"])
    pairs = {}
    for a, b in zip(t[p != t], p[p != t]):
        pairs[(CHARSET[a], CHARSET[b])] = pairs.get((CHARSET[a], CHARSET[b]), 0) + 1
    top = sorted(pairs.items(), key=lambda kv: -kv[1])[:12]
    print("Most common NIST mistakes (true -> predicted):", ", ".join(f"{a}->{b} ({n})" for (a, b), n in top))

    Path("outputs/results").mkdir(parents=True, exist_ok=True)
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    json.dump({"history": history, "test": test_acc, "top_confusions": [[a, b, n] for (a, b), n in top]},
              open("outputs/results/classifier.json", "w"), indent=2)
    epochs = [h["epoch"] for h in history]
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, [h["train_acc"] for h in history], label="train (augmented)")
    plt.plot(epochs, [h["val_fonts"] for h in history], label="val fonts")
    plt.plot(epochs, [h["val_NIST"] for h in history], label="val NIST")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("outputs/figures/classifier_curves.png", dpi=120)


if __name__ == "__main__":
    main()
