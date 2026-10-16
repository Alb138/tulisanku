"""Download a large file over several connections at once.

Some servers (e.g. the NIST S3 bucket, seen from Indonesia) limit the speed of
each connection. HTTP "Range" requests let us ask for different pieces of the
file in parallel and glue them together. Finished pieces are kept on disk, so
an interrupted download resumes where it stopped (with the same chunk size).

Pieces are small (1 MB) on purpose: with big pieces, the last few stragglers
each crawl along on a single slow connection while the others sit idle
(the "long tail" problem).
"""
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

from tqdm import tqdm


def _fetch_range(url, start, end, path: Path, retries: int = 6):
    expected = end - start + 1
    if path.exists() and path.stat().st_size == expected:
        return  # already downloaded in an earlier run
    for attempt in range(retries):
        try:
            request = Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urlopen(request, timeout=120) as response:
                data = response.read()
            if len(data) == expected:
                path.write_bytes(data)
                return
        except OSError:
            pass
        time.sleep(2 ** attempt)
    raise RuntimeError(f"could not download bytes {start}-{end} of {url}")


def parallel_download(url: str, dest: Path, chunk_mb: float = 1, workers: int = 24):
    dest = Path(dest)
    if dest.exists():
        return dest
    with urlopen(Request(url, method="HEAD"), timeout=60) as response:
        size = int(response.headers["Content-Length"])

    parts = dest.parent / (dest.name + ".parts")
    parts.mkdir(parents=True, exist_ok=True)
    chunk = int(chunk_mb * 2**20)
    ranges = [(i, start, min(start + chunk, size) - 1) for i, start in enumerate(range(0, size, chunk))]

    with ThreadPoolExecutor(workers) as pool, tqdm(total=size, unit="B", unit_scale=True,
                                                   desc=dest.name) as progress:
        futures = [pool.submit(_fetch_range, url, a, b, parts / f"{i:05d}") for i, a, b in ranges]
        for future, (_, a, b) in zip(futures, ranges):
            future.result()
            progress.update(b - a + 1)

    with open(dest, "wb") as out:  # glue the pieces together in order
        for i, _, _ in ranges:
            out.write((parts / f"{i:05d}").read_bytes())
    shutil.rmtree(parts)
    return dest
