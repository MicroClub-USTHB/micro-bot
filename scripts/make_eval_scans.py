"""Make the scanned documents of the eval corpus (Req 12.2) in data/sources/scans/.

    uv run python scripts/make_eval_scans.py

Two kinds, both image-only so the OCR path is the only way in:

- page images of real PDFs (Code of Conduct, a candidacy), rasterized with ``pdftoppm``: the
  same text is also in the corpus as a text PDF, so they test near-duplicate handling;
- printed copies of real faculty notices that are NOT otherwise in the corpus (Arabic and
  French), so a question on them can only be answered through OCR. Their source text is kept
  next to them (``<name>.txt``) as the ground truth for evidence checks.

Degradation (skew, blur, grain, JPEG) is seeded, so reruns give the same files.
"""

from __future__ import annotations

import asyncio
import random
import re
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from mcclub_rag.ingest.preprocess import preprocess_url

ROOT = Path(__file__).resolve().parent.parent
SOURCES = ROOT / "data" / "sources"
OUT = SOURCES / "scans"
FONTS = {
    "ar": "/usr/share/fonts/TTF/DejaVuSans.ttf",  # Arabic and Latin (URLs, names)
    "fr": "/usr/share/fonts/TTF/DejaVuSerif.ttf",
}
PAGE_IMAGES = [  # (output, source PDF, pages, angle, as photo)
    ("coc-p3-p8-scan.pdf", "MicroClub Code Of Conduct 2025-2026.pdf", (3, 8), 0.8, False),
    ("sg-candidacy-p2-photo.jpg", "KENPA SG DOC.pdf", (2,), -2.2, True),
]
NOTICES = [  # (output, URL, language, angle)
    (
        "ar-nailekhanim-notice-scan.pdf",
        "https://finfo.usthb.dz/news/  -منحة-“Nailekhanim-Foundation”-بجامعة-خزر",
        "ar",
        -0.6,
    ),
    (
        "fr-m2-library-notice-scan.png",
        "https://finfo.usthb.dz/news/Avis-aux-étudiants-de-2ᵉ-année-Master",
        "fr",
        0.5,
    ),
    (
        "fr-unesco-prize-notice-scan.jpg",
        "https://finfo.usthb.dz/news/UNESCO-Roi-Hamad-Bin-Isa-Al-Khalifa",
        "fr",
        1.4,
    ),
]
_FOOTER = re.compile(r"^#{2,} (Don't miss out|Ne manquez pas)", re.MULTILINE)
_SHARE = re.compile(r"^- (Facebook|Twitter|Messenger|Linkedin|Telegram)\b")


def degrade(image: Image.Image, angle: float, seed: str, *, photo: bool = False) -> Image.Image:
    """Grey, skewed, slightly blurred, with grain; a photo also gets uneven light."""
    rng = random.Random(seed)
    page = image.convert("L").rotate(angle, expand=True, fillcolor=255, resample=Image.BICUBIC)
    page = page.filter(ImageFilter.GaussianBlur(0.7 if photo else 0.4))
    grain = Image.effect_noise(page.size, 18 if photo else 10).filter(ImageFilter.BoxBlur(0.5))
    page = Image.blend(page, grain, 0.12)
    if photo:
        shade = Image.linear_gradient("L").resize(page.size).rotate(rng.uniform(20, 70))
        page = Image.blend(page, shade, 0.10)
    return page


def notice_text(text: str) -> str:
    """The notice body: share links and markdown emphasis dropped."""
    lines = [line.replace("**", "") for line in text.splitlines() if not _SHARE.match(line)]
    body = "\n".join(lines)
    return _FOOTER.split(body, 1)[0].strip()


def wrap(draw: ImageDraw.ImageDraw, text: str, font, width: int, rtl: bool) -> list[str]:
    kwargs = {"direction": "rtl"} if rtl else {}
    lines = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split():
            candidate = f"{line} {word}".strip()
            if draw.textlength(candidate, font=font, **kwargs) > width and line:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
    return lines


def render_notice(text: str, lang: str) -> Image.Image:
    """A4 at 150 dpi, 13 pt type, right-aligned for Arabic."""
    page = Image.new("L", (1240, 1754), 255)
    draw = ImageDraw.Draw(page)
    font = ImageFont.truetype(FONTS[lang], 30)
    rtl = lang == "ar"
    kwargs = {"direction": "rtl"} if rtl else {}
    y = 120
    for line in wrap(draw, text, font, 1040, rtl):
        if rtl:
            x = 1140 - draw.textlength(line, font=font, **kwargs)
        else:
            x = 100
        draw.text((x, y), line, font=font, fill=0, **kwargs)
        y += 52
    return page


def save(pages: list[Image.Image], path: Path) -> None:
    if path.suffix == ".pdf":
        pages[0].save(path, save_all=True, append_images=pages[1:], resolution=150)
    elif path.suffix == ".jpg":
        pages[0].save(path, quality=62)
    else:
        pages[0].save(path, optimize=True)


def rasterize(pdf: Path, page: int) -> Image.Image:
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / "p"
        cmd = ["pdftoppm", "-r", "150", "-gray", "-png", "-f", str(page), "-l", str(page)]
        subprocess.run([*cmd, str(pdf), str(prefix)], check=True)
        (image,) = Path(tmp).glob("p*.png")
        return Image.open(image).copy()


async def main() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("b", ROOT / "scripts" / "build_eval_corpus.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, pdf, pages, angle, photo in PAGE_IMAGES:
        images = [degrade(rasterize(SOURCES / pdf, p), angle, name, photo=photo) for p in pages]
        save(images, OUT / name)
        print(f"wrote {name} ({len(images)} page(s) of {pdf})")
    for name, url, lang, angle in NOTICES:
        doc = await preprocess_url(build.quote_url(url), visibility="public")
        text = notice_text(doc.text)
        (OUT / name).with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
        save([degrade(render_notice(text, lang), angle, name)], OUT / name)
        print(f"wrote {name} ({len(text.split())} words, {lang})")


if __name__ == "__main__":
    asyncio.run(main())
