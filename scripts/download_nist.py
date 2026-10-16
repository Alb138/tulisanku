"""Download NIST Special Database 19 (handprinted characters, organized by writer).

  python scripts/download_nist.py            -> data/raw/nist/by_write.zip (~540 MB)

The archive holds ~800,000 tiny PNG files; we do NOT extract it - the next
script reads the images straight from the zip.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tulisanku.download import parallel_download  # noqa: E402

URL = "https://s3.amazonaws.com/nist-srd/SD19/by_write.zip"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=Path("data/raw/nist/by_write.zip"))
    parser.add_argument("--workers", type=int, default=24, help="parallel connections")
    parser.add_argument("--chunk-mb", type=float, default=1, help="piece size (keep it the same when resuming)")
    args = parser.parse_args()
    path = parallel_download(URL, args.dest, chunk_mb=args.chunk_mb, workers=args.workers)
    print(f"Saved {path} ({path.stat().st_size / 2**20:.0f} MB)")


if __name__ == "__main__":
    main()
