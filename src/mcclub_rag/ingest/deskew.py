"""Straighten photographed pages before OCR (eval-fixes Req 12, ISSUES.md I13).

A phone photo skewed by 2.2° lost 59% of its words in Tesseract (41% word recall); straightened,
the same photo reads at 99%. Flatbed scans keep their recall (96%, 97%). Light correction
(dividing by a blurred background) was tried and dropped: it cost the clean scans 2-4 points.

The angle is the one whose horizontal ink profile is the most peaked (text lines line up), on a
downscaled copy, between -5° and 5°. Images only: PDF pages are rasterized inside xberg.
"""

import io

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_ANGLE = 5.0
STEP = 0.2
MIN_ANGLE = 1.0  # below this, rotating (and resampling) is not worth it
_PROBE_SIZE = 800


def skew_angle(image: Image.Image) -> float:
    """Degrees to rotate ``image`` counter-clockwise (PIL convention) to level its text lines."""
    gray = ImageOps.grayscale(image)
    gray.thumbnail((_PROBE_SIZE, _PROBE_SIZE))
    pixels = np.asarray(gray, dtype=np.float32)
    ink = Image.fromarray(((pixels < pixels.mean() - 0.5 * pixels.std()) * 255).astype(np.uint8))
    best, best_score = 0.0, -1.0
    for angle in np.arange(-MAX_ANGLE, MAX_ANGLE + STEP / 2, STEP):
        rotated = ink.rotate(float(angle), resample=Image.Resampling.BILINEAR, fillcolor=0)
        score = float(np.var(np.asarray(rotated, dtype=np.float32).sum(axis=1)))
        if score > best_score:
            best, best_score = float(angle), score
    return round(best, 1)


def deskew_image(data: bytes) -> bytes | None:
    """PNG bytes of the straightened page, or None when it is level enough (or unreadable)."""
    try:
        with Image.open(io.BytesIO(data)) as opened:
            image = ImageOps.exif_transpose(opened).convert("L")
    except (UnidentifiedImageError, OSError, ValueError):
        return None  # let xberg report its own parse error
    angle = skew_angle(image)
    if abs(angle) < MIN_ANGLE:
        return None
    straight = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=255)
    out = io.BytesIO()
    straight.save(out, format="PNG")
    return out.getvalue()
