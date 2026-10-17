"""TULISANKU handwriting collection page.

Friends open the page on a phone, tablet or laptop, agree to the consent text, and
draw each of the 62 characters on a canvas with a baseline guide. Each character is
saved the moment it is drawn, so nothing is lost if the browser closes:

  data/raw/handwriting/drawn/<session>/<hex code>.png   the drawing, black on white
  data/raw/handwriting/drawn/<session>/<hex code>.json  the pen strokes (x, y, time)
  data/raw/handwriting/drawn/<session>/session.json     consent time and device

Files are named by character code (61 = "a", 41 = "A") because Windows treats
"a.png" and "A.png" as the same file name.

Ways to run it:
  python collect/app.py           -> address for phones on the same Wi-Fi
  python collect/app.py --share   -> temporary public link through a Cloudflare tunnel
  hosted (e.g. PythonAnywhere)    -> permanent public link, see collect/DEPLOY.md

Hosting services start the app themselves through WSGI (the standard way a Python web
app talks to a web server) by importing `app` from this file.

Exporting: if the environment variable TULISANKU_EXPORT_KEY is set, GET
/api/export?key=<that key> returns every session as one zip file (used by
scripts/fetch_collected.py). The key is a secret - it never goes into git.
"""
import argparse
import base64
import hmac
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file, send_from_directory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tulisanku.charset import CHARSET  # noqa: E402

STATIC = Path(__file__).parent / "static"
OUT = Path(os.environ.get("TULISANKU_DATA_DIR", ROOT / "data" / "raw" / "handwriting" / "drawn"))
DEVICES = {"finger", "stylus", "mouse", "other"}
PNG_PREFIX = "data:image/png;base64,"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 2 * 2**20  # a drawing is ~10-50 KB; refuse anything huge


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
    image = base64.b64decode(png[len(PNG_PREFIX):], validate=True)
    if not image.startswith(PNG_SIGNATURE):
        abort(400, "not a PNG image")
    code = f"{ord(char):02x}"
    (path / f"{code}.png").write_bytes(image)
    (path / f"{code}.json").write_text(json.dumps({"char": char, "strokes": body.get("strokes", [])}),
                                       encoding="utf-8")
    return jsonify(ok=True)


@app.get("/api/export")
def export():
    """All sessions as one zip - only for whoever knows the secret export key."""
    key = os.environ.get("TULISANKU_EXPORT_KEY", "")
    if not key:
        abort(404)  # exporting is switched off unless a key is configured
    if not hmac.compare_digest(request.args.get("key", ""), key):  # constant-time comparison
        abort(403)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        if OUT.exists():
            for path in sorted(OUT.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(OUT).as_posix())
    buffer.seek(0)
    return send_file(buffer, mimetype="application/zip", as_attachment=True,
                     download_name="tulisanku_handwriting.zip")


def lan_address() -> str:
    """This laptop's address on the local network (what phones should open)."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("8.8.8.8", 80))  # no data is sent; this only picks the right network card
            return s.getsockname()[0]
        except OSError:
            return "127.0.0.1"


def find_cloudflared():
    return shutil.which("cloudflared") or next(
        (str(p) for p in [ROOT / "tools" / "cloudflared.exe", ROOT / "tools" / "cloudflared"] if p.exists()), None)


def start_tunnel(port: int):
    """Start a Cloudflare quick tunnel and print its public link when it is ready."""
    exe = find_cloudflared()
    if exe is None:
        sys.exit("cloudflared not found - install it first (see 'Collecting handwriting' in README.md)")
    tunnel = subprocess.Popen([exe, "tunnel", "--no-autoupdate", "--url", f"http://localhost:{port}"],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def watch():
        for line in tunnel.stdout:
            match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
            if match:
                print(f"\n  Share this link:  {match.group(0)}")
                print("  (works while this window stays open; it changes on every restart)\n", flush=True)

    threading.Thread(target=watch, daemon=True).start()
    return tunnel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--share", action="store_true", help="also create a public link via Cloudflare")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"\n  On this laptop:          http://localhost:{args.port}")
    print(f"  On phones (same Wi-Fi):  http://{lan_address()}:{args.port}\n", flush=True)
    tunnel = start_tunnel(args.port) if args.share else None
    from waitress import serve  # only needed when running it yourself, not on a hosting service
    try:
        serve(app, host="0.0.0.0", port=args.port)  # waitress: a production server, unlike app.run()
    finally:
        if tunnel:
            tunnel.terminate()
