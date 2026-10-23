"""Training examples for few-shot generation: k reference glyphs + one target glyph.

Every example comes from ONE writer (or font): a random target character, plus 1..max_refs
other characters of the same writer as references. k varies, so the generator learns to
work from one letter as well as from many. Because every training writer wrote all (or
most) characters, we always know the real target glyph - that is what the L1 loss
compares against.

Reading uses the same shuffle-buffer idea as training_data.GlyphBatches: random blocks of
consecutive rows, several examples per row, shuffled inside a pool.
"""
import numpy as np


class FewShotBatches:
    def __init__(self, sources, batch_size: int = 48, max_refs: int = 8, block_rows: int = 16,
                 samples_per_row: int = 4, blocks_per_pool: int = 8, weights=None):
        """sources: list of (images [n, 62, H, W] memory-mapped, mask [n, 62]); weights: how
        often each source is used (default: equally often, so fonts and handwriting are balanced)."""
        self.sources = sources
        self.batch_size, self.max_refs = batch_size, max_refs
        self.block_rows, self.samples_per_row, self.blocks_per_pool = block_rows, samples_per_row, blocks_per_pool
        weights = np.ones(len(sources)) if weights is None else np.asarray(weights, float)
        self.weights = weights / weights.sum()
        # Every row (writer or font) of every source gets one id: source 0 rows first, then source 1, ...
        self.offsets = np.cumsum([0] + [len(images) for images, _ in sources])
        self.n_writers = int(self.offsets[-1])

    def _examples(self, rng, source):
        images, mask = self.sources[source]
        start = int(rng.integers(0, max(1, len(images) - self.block_rows + 1)))
        block = np.asarray(images[start:start + self.block_rows])
        block_mask = np.asarray(mask[start:start + self.block_rows])
        h, w = block.shape[-2:]
        out = []
        for row in range(len(block)):
            available = np.flatnonzero(block_mask[row])
            if len(available) < 2:
                continue
            for _ in range(self.samples_per_row):
                target = int(rng.choice(available))
                others = available[available != target]
                k = int(rng.integers(1, min(self.max_refs, len(others)) + 1))
                refs = rng.choice(others, k, replace=False)
                ref_images = np.zeros((self.max_refs, h, w), np.uint8)
                ref_images[:k] = block[row, refs]
                ref_mask = np.zeros(self.max_refs, bool)
                ref_mask[:k] = True
                out.append((ref_images, ref_mask, target, block[row, target], self.offsets[source] + start + row))
        return out

    def batches(self, seed: int = 0):
        """Endless stream of (refs [B, K, H, W] uint8, ref_mask [B, K], target_char [B], target [B, H, W] uint8,
        writer [B]) - writer ids run from 0 to n_writers - 1."""
        rng = np.random.default_rng(seed)
        pool = []
        while True:
            for source in rng.choice(len(self.sources), self.blocks_per_pool, p=self.weights):
                pool += self._examples(rng, int(source))
            order = rng.permutation(len(pool))
            n_full = len(pool) // self.batch_size * self.batch_size
            for b in range(0, n_full, self.batch_size):
                batch = [pool[i] for i in order[b:b + self.batch_size]]
                yield (np.stack([e[0] for e in batch]), np.stack([e[1] for e in batch]),
                       np.array([e[2] for e in batch], np.int64), np.stack([e[3] for e in batch]),
                       np.array([e[4] for e in batch], np.int64))
            pool = [pool[i] for i in order[n_full:]]
