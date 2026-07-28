"""Image preparation.

These matter because the spike measured extraction quality against exactly this
transformation. If it drifts, every accuracy and cost number recorded in
spike/README.md quietly stops describing what the server sends.
"""

from __future__ import annotations

import io


from PIL import Image

from app.extraction import prepare_image


def _jpeg(width: int, height: int, mode: str = "RGB", **save_kwargs) -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, (width, height), "white" if mode != "L" else 255).save(
        buffer, format="JPEG", **save_kwargs
    )
    return buffer.getvalue()


def _size(data: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(data)) as img:
        return img.size


def test_downscales_the_long_edge_and_preserves_aspect_ratio():
    data, original, sent = prepare_image(_jpeg(3024, 4032), max_edge=2000)

    assert original == (3024, 4032)
    assert sent == (1500, 2000)
    assert _size(data) == (1500, 2000)


def test_downscales_landscape_by_its_long_edge_too():
    _, _, sent = prepare_image(_jpeg(4032, 3024), max_edge=2000)
    assert sent == (2000, 1500)


def test_a_small_image_is_never_upscaled():
    """Upscaling would invent detail the model then reads as real print."""
    _, original, sent = prepare_image(_jpeg(900, 1200), max_edge=2000)
    assert original == sent == (900, 1200)


def test_greyscale_input_is_converted_rather_than_rejected():
    data, _, _ = prepare_image(_jpeg(1000, 1000, mode="L"))
    with Image.open(io.BytesIO(data)) as img:
        assert img.mode == "RGB"


def test_exif_rotation_is_applied():
    """Without this the model is handed a sideways page and quietly does worse.

    The capture client's canvas frames carry no EXIF, but the `<input capture>`
    fallback uploads the raw camera file, which does.
    """
    image = Image.new("RGB", (1000, 500), "white")
    exif = image.getexif()
    exif[274] = 6  # Orientation: rotate 90°
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)

    _, _, sent = prepare_image(buffer.getvalue())

    assert sent == (500, 1000)


def test_accepts_a_path_as_well_as_bytes(tmp_path):
    path = tmp_path / "page.jpg"
    path.write_bytes(_jpeg(2400, 3200))

    _, original, sent = prepare_image(path)

    assert original == (2400, 3200)
    assert sent == (1500, 2000)
