"""Feed (glyph, character) pairs to training without loading everything into RAM.

Reading single random images from a memory-mapped file is slow (every image is a
separate disk read). A "shuffle buffer" keeps most of the randomness at a fraction of
the cost: read a few random *blocks* of consecutive fonts/writers (fast, sequential
reads), pool them, shuffle inside the pool, and cut it into batches. TensorFlow's
tf.data uses the same trick.
"""
import numpy as np


class GlyphBatches:
    """Batches of (images uint8 [B, H, W], labels int64 [B]) from several glyph sources.

    sources: list of (images [n, 62, H, W] (memory-mapped is fine), mask [n, 62] bool)
    """

    def __init__(self, sources, batch_size: int = 256, block_rows: int = 16, blocks_per_pool: int = 8):
        self.sources = sources
        self.batch_size, self.block_rows, self.blocks_per_pool = batch_size, block_rows, blocks_per_pool
        self.blocks = [(s, start) for s, (images, _) in enumerate(sources)
                       for start in range(0, len(images), block_rows)]
        self.n_samples = sum(int(mask.sum()) for _, mask in sources)

    def __len__(self):
        return int(np.ceil(self.n_samples / self.batch_size))

    def _read_block(self, source: int, start: int):
        images, mask = self.sources[source]
        block = np.asarray(images[start:start + self.block_rows])  # one sequential read
        rows, chars = np.nonzero(mask[start:start + self.block_rows])
        return block[rows, chars], chars.astype(np.int64)

    def epoch(self, shuffle: bool = True, seed: int = 0):
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(self.blocks)) if shuffle else np.arange(len(self.blocks))
        carry_x, carry_y = np.zeros((0,) + self.sources[0][0].shape[2:], np.uint8), np.zeros(0, np.int64)
        for p in range(0, len(order), self.blocks_per_pool):
            parts = [self._read_block(*self.blocks[i]) for i in order[p:p + self.blocks_per_pool]]
            x = np.concatenate([carry_x] + [a for a, _ in parts])
            y = np.concatenate([carry_y] + [b for _, b in parts])
            if shuffle:
                perm = rng.permutation(len(y))
                x, y = x[perm], y[perm]
            n_full = len(y) // self.batch_size * self.batch_size
            for b in range(0, n_full, self.batch_size):
                yield x[b:b + self.batch_size], y[b:b + self.batch_size]
            carry_x, carry_y = x[n_full:], y[n_full:]  # leftovers go into the next pool
        if len(carry_y):
            yield carry_x, carry_y
