"""Deskew of photographed pages (eval-fixes Req 12)."""

import io

import pytest
from PIL import Image, ImageDraw

from micro_bot.ingest.deskew import deskew_image, skew_angle


def _page(angle: float) -> Image.Image:
    page = Image.new("L", (1200, 900), 255)
    draw = ImageDraw.Draw(page)
    for row in range(14):
        draw.rectangle((100, 80 + row * 55, 1100, 100 + row * 55), fill=0)  # text lines
    return page.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=255)


def _png(image: Image.Image) -> bytes:
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


@pytest.mark.parametrize("angle", [-3.0, -1.6, 2.2, 4.0])
def test_skew_angle_undoes_the_rotation(angle):
    assert skew_angle(_page(angle)) == pytest.approx(-angle, abs=0.3)


def test_level_page_is_left_alone():
    assert deskew_image(_png(_page(0.0))) is None
    assert deskew_image(_png(_page(0.4))) is None


def test_skewed_page_comes_back_level():
    fixed = deskew_image(_png(_page(2.2)))
    assert fixed is not None
    assert abs(skew_angle(Image.open(io.BytesIO(fixed)))) < 0.5


def test_unreadable_bytes_are_left_to_the_parser():
    assert deskew_image(b"not an image") is None
