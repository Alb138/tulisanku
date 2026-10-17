"""Download handwriting collected by the hosted page into data/raw/handwriting/drawn/.

  python scripts/fetch_collected.py --url https://<username>.pythonanywhere.com

The export key is read from the TULISANKU_EXPORT_KEY environment variable, or asked
for (hidden typing) if it is not set. Sessions already on this laptop are kept;
new ones are added.
"""
import argparse
import getpass
import io
import os
import re
import zipfile
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True, help="address of the hosted collection page")
    parser.add_argument("--dest", type=Path, default=Path("data/raw/handwriting/drawn"))
    args = parser.parse_args()

    key = os.environ.get("TULISANKU_EXPORT_KEY") or getpass.getpass("Export key: ")
    try:
        with urlopen(f"{args.url.rstrip('/')}/api/export?{urlencode({'key': key})}", timeout=300) as response:
            data = response.read()
    except HTTPError as e:
        raise SystemExit({403: "Wrong export key.", 404: "Export is not enabled on the server."}
                         .get(e.code, f"Server error {e.code}"))

    # Only accept "<8 hex chars>/<file name>" entries: never trust paths inside a downloaded
    # zip blindly ("../.." names could write outside the destination - the "zip slip" bug).
    archive = zipfile.ZipFile(io.BytesIO(data))
    names = [n for n in archive.namelist() if re.fullmatch(r"[0-9a-f]{8}/[\w.-]+", n)]
    sessions = sorted({name.split("/")[0] for name in names})
    new = [s for s in sessions if not (args.dest / s).exists()]
    for name in names:
        if name.split("/")[0] in new:
            archive.extract(name, args.dest)
    print(f"{len(sessions)} sessions on the server, {len(new)} new ones saved to {args.dest}")


if __name__ == "__main__":
    main()
