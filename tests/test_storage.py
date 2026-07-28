"""Page image lifecycle.

The ordering guarantee these protect is the one irreversible thing in the
system: once an image is discarded there is no second copy anywhere, so the read
order and the sweeper's cutoff both have to be right.
"""

from __future__ import annotations

import os
import time

from app import storage


def test_pages_read_back_in_capture_order(settings):
    # Written out of order on purpose: sequence, not insertion, defines reading
    # order, and reading order is what the model is given.
    for sequence, payload in ((2, b"third"), (0, b"first"), (1, b"second")):
        storage.write_page(settings.image_dir, "r1", sequence, payload)

    assert storage.read_pages(settings.image_dir, "r1") == [b"first", b"second", b"third"]


def test_double_digit_sequences_still_sort_correctly(settings):
    for sequence in range(11):
        storage.write_page(settings.image_dir, "r1", sequence, bytes([sequence]))

    assert storage.read_pages(settings.image_dir, "r1") == [bytes([i]) for i in range(11)]


def test_discard_removes_everything_and_is_idempotent(settings):
    storage.write_page(settings.image_dir, "r1", 0, b"x")
    assert storage.has_images(settings.image_dir, "r1")

    storage.discard(settings.image_dir, "r1")
    storage.discard(settings.image_dir, "r1")

    assert not storage.has_images(settings.image_dir, "r1")
    assert storage.read_pages(settings.image_dir, "r1") == []


def test_discard_leaves_other_recipes_alone(settings):
    storage.write_page(settings.image_dir, "keep", 0, b"keep")
    storage.write_page(settings.image_dir, "drop", 0, b"drop")

    storage.discard(settings.image_dir, "drop")

    assert storage.has_images(settings.image_dir, "keep")


def test_sweep_removes_only_directories_past_the_ttl(settings):
    storage.write_page(settings.image_dir, "fresh", 0, b"x")
    storage.write_page(settings.image_dir, "stale", 0, b"x")

    old = time.time() - 48 * 3600
    os.utime(storage.recipe_dir(settings.image_dir, "stale"), (old, old))

    assert storage.sweep(settings.image_dir, ttl_hours=24) == 1
    assert storage.has_images(settings.image_dir, "fresh")
    assert not storage.has_images(settings.image_dir, "stale")


def test_sweep_on_a_missing_directory_is_not_an_error(tmp_path):
    assert storage.sweep(tmp_path / "never-created", ttl_hours=24) == 0


def test_has_images_is_false_for_an_empty_directory(settings):
    storage.recipe_dir(settings.image_dir, "r1").mkdir(parents=True)
    assert not storage.has_images(settings.image_dir, "r1")
