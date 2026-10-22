"""Load the processed datasets without filling the laptop's memory.

The .npz files are compressed, so loading one unpacks the whole array into RAM
(the NIST images alone are ~0.9 GB). Instead, each array is extracted once to an
uncompressed .npy file in data/processed/cache/ and opened with memory mapping:
NumPy then reads only the rows that are actually used, straight from disk.
Selected subsets (e.g. clean training fonts) are cached the same way.
"""
import hashlib
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
CACHE = PROCESSED / "cache"


def _array(name: str, key: str) -> np.ndarray:
    """Memory-mapped array `key` from <name>.npz, extracted once without loading it into RAM."""
    path = CACHE / f"{name}_{key}.npy"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = CACHE / f"{name}_{key}.partial"
        with zipfile.ZipFile(PROCESSED / f"{name}.npz") as archive, \
                archive.open(f"{key}.npy") as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst, 2**24)  # streams 16 MB at a time
        tmp.replace(path)
    return np.load(path, mmap_mode="r")


def _subset(name: str, key: str, rows: np.ndarray) -> np.ndarray:
    """Memory-mapped copy of selected rows, cached (the file name changes if the rows change)."""
    tag = hashlib.md5(np.ascontiguousarray(rows).tobytes()).hexdigest()[:10]
    path = CACHE / f"{name}_{key}_{tag}.npy"
    if not path.exists():
        full = _array(name, key)
        tmp = CACHE / f"{name}_{key}_{tag}.partial.npy"
        out = np.lib.format.open_memmap(tmp, mode="w+", dtype=full.dtype, shape=(len(rows),) + full.shape[1:])
        for start in range(0, len(rows), 64):  # copy in small chunks
            out[start:start + 64] = full[rows[start:start + 64]]
        out.flush()
        del out
        tmp.replace(path)
    return np.load(path, mmap_mode="r")


def load_fonts(split: str = None, clean: bool = True, size: int = 64):
    """Font glyphs: (images [n, 62, size, size], mask [n, 62], metadata)."""
    meta = pd.read_csv(PROCESSED / f"fonts_{size}_quality.csv", keep_default_na=False)
    keep = (meta["reason"] == "") if clean else pd.Series(True, index=meta.index)
    if split:
        keep &= meta["split"] == split
    meta = meta[keep]
    images = _subset(f"fonts_{size}", "images", meta.index.to_numpy())
    return images, np.ones(images.shape[:2], bool), meta.reset_index(drop=True)


def load_handwriting(name: str, split: str = None, size: int = 64):
    """Real handwriting, name = "nist" or "drawn": (images, mask, metadata)."""
    meta = pd.read_csv(PROCESSED / f"{name}_{size}.csv", dtype={"session": str, "writer": str})
    if split:
        meta = meta[meta["split"] == split]
    rows = meta.index.to_numpy()
    images = _subset(f"{name}_{size}", "images", rows)
    mask = np.asarray(_array(f"{name}_{size}", "mask")[rows])
    return images, mask, meta.reset_index(drop=True)
