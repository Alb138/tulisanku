"""The printable handwriting collection sheet (one A4 page, one box per character).

Layout (all in millimetres, origin top-left of the page):
  - four black corner markers, so a phone photo can be straightened later
  - an 8 x 8 grid of square boxes; the first 62 hold 0-9, A-Z, a-z
  - a light gray baseline in every box at the same relative height as the font
    images (row 43 of 64), so we know exactly where each writer's baseline is
The printed hints and guide lines are light gray: thresholding the photo for
dark pen ink removes them automatically.
"""
from PIL import Image, ImageDraw, ImageFont

from .charset import CHARSET

PAGE_W, PAGE_H = 210.0, 297.0           # A4
MARKER = 10.0                           # corner marker size
MARKER_MARGIN = 8.0                     # distance from page edge to marker
COLS, ROWS = 8, 8
BOX = 20.0                              # box size
GAP_X, GAP_Y = 2.5, 8.0                 # space between boxes (GAP_Y holds the hint)
BASELINE_FRACTION = 43 / 64             # baseline position inside a box, from the top
GRID_TOP = 52.0
GRID_LEFT = (PAGE_W - (COLS * BOX + (COLS - 1) * GAP_X)) / 2

LIGHT = 175                             # gray level for hints and guides (0 = black)


def marker_boxes():
    """(left, top, right, bottom) of the four corner markers, in mm."""
    m, s = MARKER_MARGIN, MARKER
    return [(m, m, m + s, m + s), (PAGE_W - m - s, m, PAGE_W - m, m + s),
            (m, PAGE_H - m - s, m + s, PAGE_H - m), (PAGE_W - m - s, PAGE_H - m - s, PAGE_W - m, PAGE_H - m)]


def box_positions():
    """{char: (left, top)} of each writing box, in mm."""
    positions = {}
    for i, char in enumerate(CHARSET):
        row, col = divmod(i, COLS)
        positions[char] = (GRID_LEFT + col * (BOX + GAP_X), GRID_TOP + row * (BOX + GAP_Y))
    return positions


def render_sheet(dpi: int = 300) -> Image.Image:
    px = dpi / 25.4  # pixels per mm
    page = Image.new("L", (round(PAGE_W * px), round(PAGE_H * px)), 255)
    draw = ImageDraw.Draw(page)

    def font(mm):
        return ImageFont.load_default(size=round(mm * px))

    def rect(l, t, r, b, **kw):
        draw.rectangle([round(l * px), round(t * px), round(r * px), round(b * px)], **kw)

    for box in marker_boxes():
        rect(*box, fill=0)

    draw.text((GRID_LEFT * px, 22 * px), "TULISANKU handwriting sheet", fill=0, font=font(6))
    draw.text((GRID_LEFT * px, 31 * px),
              "Write each character once in its box, resting on the gray line. Use a dark pen.",
              fill=60, font=font(3.2))
    draw.text((GRID_LEFT * px, 37 * px), "Name / ID: ______________________     Consent signed:  [  ]",
              fill=60, font=font(3.2))

    for char, (left, top) in box_positions().items():
        draw.text(((left + 0.5) * px, (top - 5.5) * px), char, fill=LIGHT, font=font(4))
        rect(left, top, left + BOX, top + BOX, outline=LIGHT, width=max(1, round(0.25 * px)))
        y = top + BOX * BASELINE_FRACTION
        draw.line([round((left + 1.5) * px), round(y * px), round((left + BOX - 1.5) * px), round(y * px)],
                  fill=LIGHT, width=max(1, round(0.3 * px)))
    return page
