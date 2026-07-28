"""Image preparation.

Shared by the spike CLI and the server so that quality measured on real pages in
Phase 1 is quality the server actually gets. If these diverged, every number in
spike/README.md would quietly stop describing production.

Deliberately minimal: downscale and re-encode, nothing else. Binarizing,
thresholding, deskewing, and dewarping are classic OCR preprocessing steps and
they *hurt* vision models, which read the natural image and handle perspective
distortion natively.
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageOps

MAX_EDGE = 2000
"""Below roughly 1600px, small print and vulgar fractions start failing."""

QUALITY = 80


def prepare_image(
    source: Path | bytes,
    max_edge: int = MAX_EDGE,
    quality: int = QUALITY,
) -> tuple[bytes, tuple[int, int], tuple[int, int]]:
    """Return (jpeg_bytes, original_size, sent_size).

    Downscaling only — a smaller image is passed through untouched in dimension,
    since upscaling invents detail the model would then read as real.

    The capture client already does this before upload, so for that path this is
    close to a no-op. It still runs server-side because the `<input capture>`
    fallback uploads the raw camera file, EXIF rotation and all.
    """
    handle = Image.open(io.BytesIO(source)) if isinstance(source, bytes) else Image.open(source)
    with handle as img:
        # Phone photos carry rotation in EXIF. Without this the model is handed a
        # sideways page and quietly does much worse.
        img = ImageOps.exif_transpose(img)
        original = img.size
        img = img.convert("RGB")

        scale = max_edge / max(img.size)
        if scale < 1:
            img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
        sent = img.size

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)

    return buf.getvalue(), original, sent
