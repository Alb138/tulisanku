// TULISANKU web app: draw a few characters -> the server generates the rest -> font.
// The drawing pad is the same as on the collection page (collect/static/app.js): same size,
// guide line and pen width, so drawings look exactly like the handwriting we tested on.
const SIZE = 512;
const BASELINE = SIZE * 43 / 64;
const PEN = SIZE * 0.03;
// After the suggested letters, common letters first, so "Draw more" helps the most early on.
const ORDER_AFTER = "eaonsrtlhdcumbpfwyvjqxzBCDEFGHIJKLMNOPQSTUVWXYZ012345689i";

const $ = (id) => document.getElementById(id);
const pad = $("pad");
let config, order = [], current = "", strokes = [], stroke = null;
const drawn = {};                     // char -> PNG data URL

function describe(c) {
  if (/[0-9]/.test(c)) return "digit";
  return c === c.toUpperCase() ? "uppercase" : "lowercase";
}

function paint(target, withGuide) {
  const g = target.getContext("2d");
  g.fillStyle = "#fff"; g.fillRect(0, 0, SIZE, SIZE);
  if (withGuide) {
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

function toPng() {                    // what is sent: black strokes on white, no guide line
  const off = document.createElement("canvas");
  off.width = off.height = SIZE;
  paint(off, false);
  return off.toDataURL("image/png");
}

function point(e) {
  const r = pad.getBoundingClientRect();
  return { x: (e.clientX - r.left) * SIZE / r.width, y: (e.clientY - r.top) * SIZE / r.height };
}
pad.addEventListener("pointerdown", (e) => {
  pad.setPointerCapture(e.pointerId);
  stroke = [point(e)]; strokes.push(stroke); paint(pad, true);
});
pad.addEventListener("pointermove", (e) => { if (stroke) { stroke.push(point(e)); paint(pad, true); } });
for (const ev of ["pointerup", "pointercancel"]) pad.addEventListener(ev, () => { stroke = null; });

function showOnly(id) {
  for (const s of ["intro", "draw", "working", "result"]) $(s).hidden = s !== id;
  $("error").hidden = true;
}

function fail(message) { $("error").textContent = message; $("error").hidden = false; }

function ask(c) {                     // put character c on the pad
  current = c; strokes = [];
  $("char").textContent = c;
  $("hint").textContent = `${describe(c)} ${c}`;
  $("status").textContent = drawn[c] ? "Drawing it again replaces your earlier version." : "";
  paint(pad, true);
  updateCount();
}

function nextUndrawn() {
  const start = order.indexOf(current);
  for (let i = 1; i <= order.length; i++) {
    const c = order[(start + i) % order.length];
    if (!drawn[c]) return c;
  }
  return null;                        // everything drawn
}

function updateCount() {
  const n = Object.keys(drawn).length;
  const hasCapital = Object.keys(drawn).some((c) => /[A-Z]/.test(c));
  $("count").textContent = `${n} of ${config.chars.length} drawn`;
  const ready = n >= config.min_letters && hasCapital;
  $("generate").disabled = !ready;
  $("generate-hint").textContent = ready
    ? "Ready! Drawing more letters makes the font closer to your handwriting."
    : n < config.min_letters ? `Draw ${config.min_letters - n} more to generate your font.`
      : "Draw at least one capital letter - it sets the size of the whole font.";
  $("strip").replaceChildren(...Object.entries(drawn).map(([c, png]) => {
    const img = document.createElement("img");
    img.src = png; img.alt = c; img.title = `Redraw ${c}`;
    img.onclick = () => ask(c);
    return img;
  }));
}

$("next").onclick = () => {
  if (!strokes.length) { $("status").textContent = "Please draw the character first."; return; }
  drawn[current] = toPng();
  const c = nextUndrawn();
  if (c) ask(c); else { updateCount(); $("status").textContent = "You drew every character!"; }
};
$("skip").onclick = () => { const c = nextUndrawn(); if (c) ask(c); };
$("undo").onclick = () => { strokes.pop(); paint(pad, true); };
$("clear").onclick = () => { strokes = []; paint(pad, true); };

$("start").onclick = () => { showOnly("draw"); ask(order[0]); };

$("generate").onclick = async () => {
  showOnly("working");
  try {
    const res = await fetch("/api/generate", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: $("name").value, glyphs: drawn }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "the server could not make the font");
    await showResult(data);
  } catch (e) {
    showOnly("draw");
    fail(`Could not make your font: ${e.message}`);
  }
};

async function showResult(data) {
  const url = `/api/font/${data.id}.ttf`;
  const face = new FontFace(`Tulisanku-${data.id}`, `url(${url})`);
  await face.load();                  // the browser uses the generated .ttf directly
  document.fonts.add(face);
  $("sample").style.fontFamily = `"Tulisanku-${data.id}", sans-serif`;
  $("family").textContent = data.family;
  $("family-again").textContent = data.family;
  $("download").href = `${url}?download=1`;
  $("grid").replaceChildren(...config.chars.split("").map((c) => {
    const tile = document.createElement("button");
    tile.className = data.drawn[c] ? "tile" : "tile ai";
    tile.title = data.drawn[c] ? `${c} (drawn by you)` : `${c} (written by AI) - tap to draw it yourself`;
    const img = document.createElement("img");
    img.src = data.glyphs[c]; img.alt = c;
    tile.append(img);
    tile.onclick = () => { showOnly("draw"); ask(c); };
    return tile;
  }));
  showOnly("result");
}

$("size").oninput = () => { $("sample").style.fontSize = `${$("size").value}px`; };
$("more").onclick = () => { showOnly("draw"); ask(nextUndrawn() || order[0]); };
$("restart").onclick = () => {
  for (const c of Object.keys(drawn)) delete drawn[c];
  showOnly("intro");
};

(async () => {
  config = await (await fetch("/api/config")).json();
  const rest = ORDER_AFTER.split("").filter((c) => !config.suggested.includes(c));
  order = [...config.suggested, ...rest, ...config.chars.split("").filter((c) => !config.suggested.includes(c) && !rest.includes(c))];
  $("sample").style.fontSize = `${$("size").value}px`;
})();
