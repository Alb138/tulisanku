// TULISANKU collection page: one character at a time, each saved as soon as it is drawn.
const SIZE = 512;                    // drawing resolution, independent of screen size
const BASELINE = SIZE * 43 / 64;     // same relative height as row 43 of the 64x64 dataset images
const PEN = SIZE * 0.03;             // pen width, similar to the stroke width in the NIST scans

const $ = (id) => document.getElementById(id);
const pad = $("pad"), ctx = pad.getContext("2d");
let chars = "", index = 0, session = null, strokes = [], current = null;
const drawn = {};                    // char -> {png, strokes}, kept so "Back" can restore a drawing

// Neutral wording on purpose: an earlier "small letter a" made people write lowercase letters tiny.
function describe(c) {
  if (/[0-9]/.test(c)) return "digit";
  return c === c.toUpperCase() ? "uppercase" : "lowercase";
}

function paint(target, withGuide) {
  const g = target.getContext("2d");
  g.fillStyle = "#fff"; g.fillRect(0, 0, SIZE, SIZE);
  if (withGuide) {                    // light guide line, only on screen - never in the saved image
    g.strokeStyle = "#9ec5f0"; g.lineWidth = 3;
    g.beginPath(); g.moveTo(SIZE * 0.05, BASELINE); g.lineTo(SIZE * 0.95, BASELINE); g.stroke();
  }
  g.strokeStyle = "#000"; g.lineWidth = PEN; g.lineCap = "round"; g.lineJoin = "round";
  for (const s of strokes) {
    g.beginPath(); g.moveTo(s[0].x, s[0].y);
    for (const p of s.slice(1)) g.lineTo(p.x, p.y);
    if (s.length === 1) g.lineTo(s[0].x + 0.1, s[0].y);   // a single tap makes a dot
    g.stroke();
  }
}

function toPng() {                    // the saved image: black strokes on white, no guide line
  const off = document.createElement("canvas");
  off.width = off.height = SIZE;
  paint(off, false);
  return off.toDataURL("image/png");
}

function point(e) {                   // screen position -> 512x512 drawing coordinates
  const r = pad.getBoundingClientRect();
  return { x: (e.clientX - r.left) * SIZE / r.width, y: (e.clientY - r.top) * SIZE / r.height,
           t: Math.round(e.timeStamp) };
}

pad.addEventListener("pointerdown", (e) => {
  pad.setPointerCapture(e.pointerId);
  current = [point(e)]; strokes.push(current); paint(pad, true);
});
pad.addEventListener("pointermove", (e) => {
  if (!current) return;
  current.push(point(e)); paint(pad, true);
});
for (const ev of ["pointerup", "pointercancel"]) pad.addEventListener(ev, () => { current = null; });

function show(i) {
  index = i;
  const c = chars[i];
  $("char").textContent = c;
  $("hint").textContent = `${describe(c)} ${c}`;
  $("progress").textContent = `${i + 1} / ${chars.length}`;
  $("back").disabled = i === 0;
  strokes = drawn[c] ? drawn[c].strokes : [];
  paint(pad, true);
  $("status").textContent = "";
  try { localStorage.setItem("tulisanku", JSON.stringify({ session, index })); } catch (e) {}
}

async function post(url, body) {
  const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" },
                                 body: JSON.stringify(body) });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

$("next").onclick = async () => {
  if (!strokes.length) { $("status").textContent = "Please draw the character first."; return; }
  const c = chars[index], png = toPng();
  $("next").disabled = true; $("status").textContent = "Saving…";
  try {
    await post("/api/glyph", { session, char: c, png, strokes });
    drawn[c] = { png, strokes };
    if (index + 1 < chars.length) show(index + 1); else finish();
  } catch (e) {
    $("status").textContent = "Could not save - check the connection and press Next again.";
  } finally { $("next").disabled = false; }
};
$("back").onclick = () => show(Math.max(0, index - 1));
$("undo").onclick = () => { strokes.pop(); paint(pad, true); };
$("clear").onclick = () => { strokes = []; paint(pad, true); };

function finish() {
  $("draw").hidden = true; $("done").hidden = false;
  $("code").textContent = session;
  for (const c of chars) {
    if (!drawn[c]) continue;
    const img = document.createElement("img"); img.src = drawn[c].png; img.alt = c;
    $("gallery").appendChild(img);
  }
  try { localStorage.removeItem("tulisanku"); } catch (e) {}
}

const updateStart = () => {
  $("start").disabled = !($("consent").checked && document.querySelector("input[name=device]:checked"));
};
document.querySelectorAll("#intro input").forEach((el) => el.addEventListener("change", updateStart));

$("start").onclick = async () => {
  const device = document.querySelector("input[name=device]:checked").value;
  ({ session } = await post("/api/session", { consent: true, device }));
  $("intro").hidden = true; $("draw").hidden = false;
  show(0);
};

(async () => {
  chars = (await (await fetch("/api/charset")).json()).chars;
  try {                                // resume an unfinished session after a reload
    const saved = JSON.parse(localStorage.getItem("tulisanku") || "null");
    if (saved && saved.session) {
      session = saved.session;
      $("intro").hidden = true; $("draw").hidden = false;
      show(saved.index || 0);
    }
  } catch (e) {}
})();
