"""Find and download fonts from the Google Fonts GitHub repository.

The repo has one folder per font family (e.g. ofl/caveat/) with a
METADATA.pb file describing it: its category (HANDWRITING, SERIF, ...),
which scripts it supports, and its font files. We first download only these
metadata files (a few MB), decide which families we want, and then download
just one font file per family. This keeps the download small.
"""
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO_URL = "https://github.com/google/fonts.git"
METADATA_PATTERN = "/*/*/METADATA.pb"


@dataclass
class FontFamily:
    name: str
    categories: list
    license: str          # folder name: "ofl", "apache" or "ufl"
    folder: str           # e.g. "ofl/caveat"
    filename: str         # the single font file we use for this family
    subsets: list
    primary_script: str   # set for fonts whose main script is not Latin

    @property
    def relpath(self) -> str:
        return f"{self.folder}/{self.filename}"

    @property
    def category(self) -> str:
        return self.categories[0] if self.categories else "UNKNOWN"


def _quoted(field: str, text: str) -> list:
    return re.findall(rf'^\s*{field}:\s*"([^"]*)"', text, flags=re.MULTILINE)


def _pick_regular(font_blocks: list) -> str:
    """Pick the 'regular' file: normal style, weight closest to 400.

    Bold/italic files of the same family are near-duplicates of the same style,
    so we keep only one file per family.
    """
    candidates = []
    for block in font_blocks:
        style = (_quoted("style", block) or ["normal"])[0]
        weight = re.search(r"weight:\s*(\d+)", block)
        filename = _quoted("filename", block)
        if filename:
            weight = int(weight.group(1)) if weight else 400
            candidates.append((style != "normal", abs(weight - 400), filename[0]))
    return min(candidates)[2] if candidates else ""


def parse_metadata(path: Path, repo_dir: Path):
    text = path.read_text(encoding="utf-8")
    font_blocks = re.findall(r"^fonts \{(.*?)^\}", text, flags=re.MULTILINE | re.DOTALL)
    filename = _pick_regular(font_blocks)
    if not filename:
        return None
    folder = path.parent.relative_to(repo_dir).as_posix()
    return FontFamily(
        name=(_quoted("name", text) or [path.parent.name])[0],
        categories=_quoted("category", text),
        license=folder.split("/")[0],
        folder=folder,
        filename=filename,
        subsets=_quoted("subsets", text),
        primary_script=(_quoted("primary_script", text) or [""])[0],
    )


def list_families(repo_dir: Path) -> list:
    families = []
    for meta in sorted(repo_dir.glob("*/*/METADATA.pb")):
        family = parse_metadata(meta, repo_dir)
        if family is not None:
            families.append(family)
    return families


# Chinese/Japanese/Korean fonts are very large files and their Latin letters are
# usually borrowed from generic designs, so they are not worth downloading.
CJK_SCRIPTS = {"Jpan", "Kore", "Hans", "Hant", "Hani", "Hira", "Kana"}


def is_latin(family: FontFamily) -> bool:
    """Keep families that include Latin letters (skip fonts without them, and CJK fonts).

    Fonts whose main script is e.g. Devanagari or Thai often also contain Latin
    letters, so we keep them; near-duplicates among them are removed in cleaning.
    """
    return "latin" in family.subsets and family.primary_script not in CJK_SCRIPTS


def _git(repo_dir: Path, *args, stdin: str = None):
    subprocess.run(["git", "-C", str(repo_dir), *args], input=stdin, text=True, check=True)


def clone_metadata(repo_dir: Path):
    """Shallow, 'blobless' clone that only checks out the METADATA.pb files."""
    if (repo_dir / ".git").exists():
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                    REPO_URL, str(repo_dir)], check=True)
    _git(repo_dir, "sparse-checkout", "set", "--no-cone", METADATA_PATTERN)


def _escape_pattern(relpath: str) -> str:
    # Sparse-checkout patterns are globs; filenames like "Caveat[wght].ttf"
    # contain brackets that must be escaped to match literally.
    return "/" + re.sub(r"([\[\]*?\\])", r"\\\1", relpath)


def download_font_files(repo_dir: Path, families: list):
    """Add the chosen font files to the sparse checkout (git downloads only these)."""
    patterns = [METADATA_PATTERN] + [_escape_pattern(f.relpath) for f in families]
    _git(repo_dir, "sparse-checkout", "set", "--no-cone", "--stdin", stdin="\n".join(patterns))
