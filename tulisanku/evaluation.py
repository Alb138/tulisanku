"""The few-shot evaluation used for every model.

For each test writer: pick k reference characters at random, let the model generate
all the writer's other characters, and compare them with what the writer really wrote.
The random choice depends only on (seed, trial, k, writer), so every model sees exactly
the same references - otherwise one model could simply be luckier.

A model is any object with
    predict(ref_images [k, H, W] uint8, ref_idx [k], target_idx [m]) -> [m, H, W] floats in 0..1
where ref_idx / target_idx are character indices into CHARSET.

Scores (averaged over a writer's generated characters):
  l1    mean absolute pixel difference            (lower is better)
  ssim  structural similarity: local brightness,
        contrast and structure, not single pixels (higher is better, max 1)
  iou   overlap of the inked areas                 (higher is better, max 1)
"""
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


def ssim(a: np.ndarray, b: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """SSIM per image for stacks [n, H, W] with values in 0..1 (Wang et al., 2004)."""
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    blur = lambda x: gaussian_filter(x, sigma=(0, sigma, sigma))  # noqa: E731
    mu_a, mu_b = blur(a), blur(b)
    var_a, var_b = blur(a * a) - mu_a ** 2, blur(b * b) - mu_b ** 2
    cov = blur(a * b) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))
    return s.mean(axis=(1, 2))


def ink_iou(a: np.ndarray, b: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    a, b = a > threshold, b > threshold
    union = (a | b).sum(axis=(1, 2))
    return np.where(union > 0, (a & b).sum(axis=(1, 2)) / np.maximum(union, 1), 1.0)


def scores(pred: np.ndarray, truth: np.ndarray) -> dict:
    pred, truth = np.clip(pred, 0, 1).astype(np.float32), truth.astype(np.float32) / 255
    return {"l1": float(np.abs(pred - truth).mean()),
            "ssim": float(ssim(pred, truth).mean()),
            "iou": float(ink_iou(pred, truth).mean())}


def pick_references(available: np.ndarray, k: int, seed: int, trial: int, writer: int):
    rng = np.random.default_rng([seed, trial, k, writer])
    refs = np.sort(rng.choice(available, k, replace=False))
    return refs, np.setdiff1d(available, refs)


def evaluate(model, images, mask, k: int, trials: int = 3, seed: int = 0) -> pd.DataFrame:
    """One row per (writer, trial) with the writer's average scores."""
    rows = []
    for writer in range(len(images)):
        available = np.flatnonzero(mask[writer])
        if len(available) <= k:
            continue
        glyphs = np.asarray(images[writer])  # one writer's 62 glyphs, read from disk
        for trial in range(trials):
            refs, targets = pick_references(available, k, seed, trial, writer)
            pred = model.predict(glyphs[refs], refs, targets)
            rows.append({"writer": writer, "trial": trial, "k": k, **scores(pred, glyphs[targets])})
    return pd.DataFrame(rows)
