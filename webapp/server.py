"""TULISANKU web app: draw a few letters, get your handwriting as a font.

The page sends the drawn characters (PNG images from the drawing canvas). The server:
  1. converts them exactly like the collected handwriting (tulisanku.nist.place_writer:
     clean, scale so capitals are 36 px tall, rebuild the baseline) -> 64x64 glyphs
  2. generates every character with the GAN from the drawn letters and removes specks and
     haze (tulisanku.adapt.AdaptedGan; settings chosen by cross-validation over the people
     who drew for us - per-person fine-tuning did not help reliably once it was kept clean)
  3. keeps the drawn characters exactly as drawn and uses generated ones only for the rest
  4. builds a TrueType font (tulisanku.font.build_font) and returns a preview of every glyph

Drawings are processed in memory and not stored. Finished fonts are kept in memory for an
hour so the page can preview and download them, then forgotten.

  python webapp/server.py           -> http://localhost:8001 (+ address for phones on the same Wi-Fi)
  python webapp/server.py --share   -> temporary public link through a Cloudflare tunnel
"""
import argparse
import base64
import io
import re
import secrets
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from flask import Flask, abort, jsonify, request, send_file, send_from_directory
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tulisanku.adapt import AdaptedGan  # noqa: E402
from tulisanku.charset import CHAR_TO_INDEX, CHARSET  # noqa: E402
from tulisanku.font import build_font  # noqa: E402
from tulisanku.nist import UPPER, place_writer  # noqa: E402

CHECKPOINT = ROOT / "checkpoints" / "gan_style" / "ema_015000.pt"
ADAPTATION = dict(clean=True)  # chosen by 2-fold cross-validation (scripts/evaluate_adaptation.py --stage cv)
MIN_LETTERS = 5
SUGGESTED = "AgkR7"  # the five letters used in every experiment: 2 capitals, 2 lowercase, 1 digit
STATIC = Path(__file__).parent / "static"
PNG_PREFIX = "data:image/png;base64,"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
KEEP_SECONDS = 3600

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 8 * 2**20  # 62 drawings of ~10-50 KB each
fonts = {}  # id -> {"ttf": bytes, "family": str, "created": time}
model_lock = threading.Lock()  # one generation at a time: fine-tuning changes the shared model
_model = None


def model() -> AdaptedGan:
    global _model
    if _model is None:
        _model = AdaptedGan(CHECKPOINT, **ADAPTATION)
    return _model


def decode_drawing(data_url: str) -> np.ndarray:
    """PNG data URL from the canvas -> ink image (ink = 255), like the collected drawings."""
    if not isinstance(data_url, str) or not data_url.startswith(PNG_PREFIX):
        abort(400, "each drawing must be a PNG image")
    raw = base64.b64decode(data_url[len(PNG_PREFIX):], validate=True)
    if not raw.startswith(PNG_SIGNATURE):
        abort(400, "each drawing must be a PNG image")
    image = Image.open(io.BytesIO(raw))
    if max(image.size) > 1024:
        abort(400, "drawing too large")
    return 255 - np.asarray(image.convert("L"))


def png_url(glyph: np.ndarray) -> str:
    """64x64 glyph (ink 0..1) -> black-on-white PNG data URL for the preview grid."""
    buffer = io.BytesIO()
    Image.fromarray((255 - np.clip(glyph, 0, 1) * 255).astype(np.uint8)).save(buffer, "PNG")
    return PNG_PREFIX + base64.b64encode(buffer.getvalue()).decode()


def family_name(text) -> str:
    name = re.sub(r"[^A-Za-z0-9 ]", "", str(text or "")).strip()[:30]
    return name or "My Handwriting"


def forget_old_fonts():
    for key in [k for k, v in fonts.items() if time.time() - v["created"] > KEEP_SECONDS]:
        del fonts[key]


@app.get("/")
def index():
    return send_from_directory(STATIC, "index.html")


@app.get("/static/<path:name>")
def static_files(name):
    return send_from_directory(STATIC, name)


@app.get("/api/config")
def config():
    return jsonify(chars=CHARSET, suggested=SUGGESTED, min_letters=MIN_LETTERS)


@app.post("/api/generate")
def generate():
    body = request.get_json(force=True, silent=True) or {}
    drawings = body.get("glyphs")
    if not isinstance(drawings, dict) or any(c not in CHAR_TO_INDEX for c in drawings):
        abort(400, "glyphs must map characters (0-9, A-Z, a-z) to drawings")
    if len(drawings) < MIN_LETTERS:
        abort(400, f"draw at least {MIN_LETTERS} characters")
    if not any(c in UPPER for c in drawings):
        abort(400, "draw at least one capital letter - it sets the size of the whole font")
    ink = {c: decode_drawing(url) for c, url in drawings.items()}
    placed = place_writer(ink)
    if placed is None or placed[1].sum() < MIN_LETTERS:
        abort(400, "some drawings were empty - draw each character clearly")
    images, drawn = placed
    refs = np.flatnonzero(drawn)

    t0 = time.time()
    with model_lock:
        generated = model().predict(images[refs], refs, np.arange(len(CHARSET)))
    glyphs = np.where(drawn[:, None, None], images / 255, generated)  # drawn letters stay exactly as drawn
    family = family_name(body.get("name"))
    with tempfile.TemporaryDirectory() as tmp:
        ttf = build_font(glyphs, Path(tmp) / "font.ttf", family).read_bytes()

    forget_old_fonts()
    font_id = secrets.token_hex(8)
    fonts[font_id] = {"ttf": ttf, "family": family, "created": time.time()}
    return jsonify(id=font_id, family=family, seconds=round(time.time() - t0, 1),
                   glyphs={c: png_url(g) for c, g in zip(CHARSET, glyphs)},
                   drawn={c: bool(d) for c, d in zip(CHARSET, drawn)})


@app.get("/api/font/<font_id>.ttf")
def download(font_id):
    if not re.fullmatch(r"[0-9a-f]{16}", font_id) or font_id not in fonts:
        abort(404, "font not found - it may have expired, please generate it again")
    font = fonts[font_id]
    return send_file(io.BytesIO(font["ttf"]), mimetype="font/ttf", as_attachment=request.args.get("download") == "1",
                     download_name=font["family"].replace(" ", "_") + ".ttf")


@app.errorhandler(400)
@app.errorhandler(404)
@app.errorhandler(413)
def error(e):
    return jsonify(error=getattr(e, "description", str(e))), e.code


@app.errorhandler(500)
def server_error(e):
    return jsonify(error="something went wrong while making the font - please try again"), 500


if __name__ == "__main__":
    from collect.app import lan_address, start_tunnel  # same helpers as the collection page
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--share", action="store_true", help="also create a public link via Cloudflare")
    args = parser.parse_args()
    print("Loading the model...", flush=True)
    model()
    print(f"\n  On this laptop:          http://localhost:{args.port}")
    print(f"  On phones (same Wi-Fi):  http://{lan_address()}:{args.port}\n", flush=True)
    tunnel = start_tunnel(args.port) if args.share else None
    from waitress import serve
    try:
        serve(app, host="0.0.0.0", port=args.port)
    finally:
        if tunnel:
            tunnel.terminate()
