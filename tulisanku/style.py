"""Style score: does a generated alphabet look like it was written by this specific person?

A trained style encoder (models/style.py) turns glyphs into embeddings. For a test writer,
the generated characters are compared with the REAL versions of the same characters
from every writer in the test set ("whose handwriting is this most like?"):

  style_top1  the true writer is the most similar one        (chance: 1 / number of writers)
  style_top5  the true writer is among the 5 most similar
  style_pct   share of other writers the alphabet resembles less than the true writer
              (0.5 = chance, 1.0 = perfect)

Comparing the same characters keeps the comparison about style, not about which letters
happen to be in the set. Other sessions of the same person (e.g. a team member's three
drawn sessions) are left out of the candidates.
"""
from pathlib import Path

import numpy as np
import torch

from .models.style import StyleEncoder

DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints" / "style.pt"


class StyleJudge:
    def __init__(self, checkpoint: Path = DEFAULT_CHECKPOINT, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = StyleEncoder().to(self.device).eval()
        self.encoder.load_state_dict(torch.load(checkpoint, map_location=self.device)["encoder"])

    @torch.no_grad()
    def embed(self, images: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """images [m, H, W] with values in 0..1 -> unit embeddings [m, dim]."""
        out = []
        for s in range(0, len(images), batch_size):
            x = torch.as_tensor(np.asarray(images[s:s + batch_size], np.float32), device=self.device).unsqueeze(1)
            with torch.autocast(self.device, dtype=torch.float16, enabled=self.device == "cuda"):
                out.append(self.encoder(x).float().cpu().numpy())
        return np.concatenate(out)

    def embed_dataset(self, images, mask) -> np.ndarray:
        """Embeddings of every available real glyph: [n, 62, dim] (zeros where missing)."""
        rows, chars = np.nonzero(mask)
        emb = np.zeros(mask.shape + (self.encoder.project.out_features,), np.float32)
        for s in range(0, len(rows), 4096):
            r, c = rows[s:s + 4096], chars[s:s + 4096]
            emb[r, c] = self.embed(np.asarray(images[r, c], np.float32) / 255)
        return emb


def _normalize(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def rank_scores(similarity: np.ndarray, true: int, candidates: np.ndarray) -> dict:
    """Top-1 / top-5 / percentile of the true writer among the candidates."""
    others = similarity[candidates & (np.arange(len(similarity)) != true)]
    better = int((others >= similarity[true]).sum())
    return {"style_top1": float(better == 0), "style_top5": float(better < 5),
            "style_pct": float((others < similarity[true]).mean()) if len(others) else float("nan")}


class StyleGallery:
    """The real glyphs of one test set, ready to compare generated alphabets against."""

    def __init__(self, judge: StyleJudge, images, mask, writer_ids=None):
        self.judge, self.mask = judge, np.asarray(mask)
        self.real = judge.embed_dataset(images, self.mask)
        self.writer_ids = np.asarray(writer_ids if writer_ids is not None else np.arange(len(self.mask)))

    def score(self, writer: int, generated: np.ndarray, targets: np.ndarray) -> dict:
        gen = self.judge.embed(generated)                        # [t, dim]
        m = self.mask[:, targets].astype(np.float32)             # which candidates have each target char
        count = m.sum(axis=1)
        real = _normalize((self.real[:, targets] * m[..., None]).sum(axis=1))
        fake = _normalize(m @ gen)                               # generated set, same characters per candidate
        similarity = (real * fake).sum(axis=1)
        same_person = (self.writer_ids == self.writer_ids[writer]) & (np.arange(len(m)) != writer)
        return rank_scores(similarity, writer, (count > 0) & ~same_person)


def split_half_identification(embeddings: np.ndarray, mask, writer_ids=None, seed: int = 0) -> dict:
    """Can each writer's first half of characters be matched to their own second half?

    The halves contain DIFFERENT characters, so this only works if the embeddings capture
    style rather than letter shapes. Used to check the style encoder itself.
    """
    rng = np.random.default_rng(seed)
    mask = np.asarray(mask)
    n = len(mask)
    writer_ids = np.asarray(writer_ids if writer_ids is not None else np.arange(n))
    a = np.zeros((n, embeddings.shape[-1]), np.float32)
    b = np.zeros_like(a)
    for w in range(n):
        chars = rng.permutation(np.flatnonzero(mask[w]))
        half = len(chars) // 2
        a[w], b[w] = embeddings[w, chars[:half]].mean(0), embeddings[w, chars[half:]].mean(0)
    similarity = _normalize(a) @ _normalize(b).T
    rows = [rank_scores(similarity[w], w, (writer_ids != writer_ids[w]) | (np.arange(n) == w)) for w in range(n)]
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
