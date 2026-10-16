"""Create the printable handwriting collection sheet.

  python scripts/make_template_sheet.py   -> templates/tulisanku_sheet.pdf (+ .png preview)

Print at 100% scale ("actual size", not "fit to page") so the boxes keep their size.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tulisanku.template import render_sheet  # noqa: E402


def main():
    out = Path("templates")
    out.mkdir(exist_ok=True)
    page = render_sheet(dpi=300)
    page.save(out / "tulisanku_sheet.pdf", resolution=300)
    page.resize((page.width // 3, page.height // 3)).save(out / "tulisanku_sheet_preview.png")
    print(f"Saved {out / 'tulisanku_sheet.pdf'} and a preview PNG")


if __name__ == "__main__":
    main()
