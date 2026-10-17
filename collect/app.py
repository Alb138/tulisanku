"""TULISANKU handwriting collection page.

Friends open the page on a phone, tablet or laptop, agree to the consent text, and
draw each of the 62 characters on a canvas with a baseline guide. Each character is
saved the moment it is drawn, so nothing is lost if the browser closes:

  data/raw/handwriting/drawn/<session>/<hex code>.png   the drawing, black on white
  data/raw/handwriting/drawn/<session>/<hex code>.json  the pen strokes (x, y, time)
  data/raw/handwriting/drawn/<session>/session.json     consent time and device

Files are named by character code (61 = "a", 41 = "A") because Windows treats
"a.png" and "A.png" as the same file name.

Run:   python collect/app.py
Then open the printed address on a phone that is on the same Wi-Fi as this laptop.
"""
import argparse
import base64
import json
import re
import socket
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tulisanku.charset import CHARSET  # noqa: E402

STATIC = Path(__file__).parent / "static"
OUT = ROOT / "data" / "raw" / "handwriting" / "drawn"
DEVICES = {"finger", "stylus", "mouse", "other"}
PNG_PREFIX = "data:image/png;base64,"

app = Flask(__name__, static_folder=None)


def session_dir(session: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{8}", session or ""):
        abort(400, "bad session id")
    path = OUT / session
    if not (path / "session.json").exists():
        abort(404, "unknown session")
    return path


@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.get("/static/<path:name>")
def static_files(name):
    return send_from_directory(STATIC, name)


@app.get("/api/charset")
def charset():
    return jsonify(chars=CHARSET)


@app.post("/api/session")
def new_session():
    body = request.get_json(force=True)
    if body.get("consent") is not True:
        abort(400, "consent is required")
    session = uuid.uuid4().hex[:8]
    path = OUT / session
    path.mkdir(parents=True)
    info = {
        "session": session,
        "consent_at": datetime.now(timezone.utc).isoformat(),
        "device": body.get("device") if body.get("device") in DEVICES else "other",
        "user_agent": request.headers.get("User-Agent", ""),
    }
    (path / "session.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return jsonify(session=session)


@app.post("/api/glyph")
def save_glyph():
    body = request.get_json(force=True)
    path = session_dir(body.get("session"))
    char, png = body.get("char", ""), body.get("png", "")
    if len(char) != 1 or char not in CHARSET or not png.startswith(PNG_PREFIX):
        abort(400, "bad glyph")
    code = f"{ord(char):02x}"
    (path / f"{code}.png").write_bytes(base64.b64decode(png[len(PNG_PREFIX):]))
    (path / f"{code}.json").write_text(json.dumps({"char": char, "strokes": body.get("strokes", [])}),
                                       encoding="utf-8")
    return jsonify(ok=True)


def lan_address() -> str:
    """This laptop's address on the local network (what phones should open)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("8.8.8.8", 80))  # no data is sent; this only picks the right network card
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"\n  On this laptop:          http://localhost:{args.port}")
    print(f"  On phones (same Wi-Fi):  http://{lan_address()}:{args.port}\n")
    app.run(host="0.0.0.0", port=args.port)
