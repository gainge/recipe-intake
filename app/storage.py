"""Page image lifecycle.

Images are transient by contract: written on upload, deleted the moment
extraction succeeds. This is what makes the no-storage goal true on the server
as well as on the phone, and it keeps disk flat regardless of how many books get
digitized.

A failed job keeps its image for a retry window so a transient API error does not
cost a re-shoot, and a sweeper removes anything past the TTL regardless of state
— including directories orphaned by a crash between writing files and committing
the row.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path


def recipe_dir(image_dir: Path, recipe_id: str) -> Path:
    return image_dir / recipe_id


def write_page(image_dir: Path, recipe_id: str, sequence: int, data: bytes) -> Path:
    target = recipe_dir(image_dir, recipe_id)
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{sequence:02d}.jpg"
    path.write_bytes(data)
    return path


def discard(image_dir: Path, recipe_id: str) -> None:
    """Remove every page image for a recipe. Safe to call when already gone."""
    shutil.rmtree(recipe_dir(image_dir, recipe_id), ignore_errors=True)


def has_images(image_dir: Path, recipe_id: str) -> bool:
    directory = recipe_dir(image_dir, recipe_id)
    return directory.is_dir() and any(directory.iterdir())


def read_pages(image_dir: Path, recipe_id: str) -> list[bytes]:
    """Page images in sequence order. Filenames are zero-padded, so sorting is
    the ordering — the reading order the model is given depends on it."""
    directory = recipe_dir(image_dir, recipe_id)
    if not directory.is_dir():
        return []
    return [p.read_bytes() for p in sorted(directory.glob("*.jpg"))]


def sweep(image_dir: Path, ttl_hours: int) -> int:
    """Delete recipe directories older than the TTL. Returns how many went."""
    if not image_dir.is_dir():
        return 0
    cutoff = time.time() - ttl_hours * 3600
    removed = 0
    for directory in image_dir.iterdir():
        if directory.is_dir() and directory.stat().st_mtime < cutoff:
            shutil.rmtree(directory, ignore_errors=True)
            removed += 1
    return removed
