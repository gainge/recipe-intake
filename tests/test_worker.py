"""Worker behaviour, with the model stubbed out.

The two properties worth protecting here are both about what survives:

- the image is deleted only after the extraction is committed, and never after
  a failure, because there is no other copy of the page;
- a re-extraction merges human-curated fields rather than replacing them, or
  retry stops being safe to offer as a button.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app import jobs, storage, worker
from app.db import load_json
from app.extraction import ExtractionError, ExtractionResult

PAGE = {
    "raw_text": "Fried Rice\n2 cups rice\n1 tbsp soy sauce",
    "title": "Fried Rice",
    "yield_text": "Serves 4",
    "times": {"prep": "10 minutes", "cook": "15 minutes", "total": ""},
    "ingredients": [
        {
            "raw": "2 cups cooked rice",
            "group": "",
            "name": "rice",
            "key": "rice",
            "amount_text": "2",
            "units": "cups",
            "units_canonical": "cup",
            "processing": "cooked",
            "notes": "",
            "required": True,
            "alternatives": [],
        }
    ],
    "instructions": ["Heat the oil.", "Add the rice."],
    "notes": [],
    "extraction": {"confidence": "high", "issues": []},
}


def _queued(conn, settings, recipe_id="r1") -> str:
    conn.execute(
        "INSERT INTO recipes (id, created_at, updated_at) VALUES (?, '2026-01-01', '2026-01-01')",
        (recipe_id,),
    )
    storage.write_page(settings.image_dir, recipe_id, 0, b"fake-jpeg")
    conn.execute(
        "INSERT INTO pages (id, recipe_id, sequence, image_path, bytes) VALUES ('p1', ?, 0, 'x', 9)",
        (recipe_id,),
    )
    return jobs.enqueue(conn, recipe_id)


def _run(conn, settings, monkeypatch, *, result=None, error=None):
    monkeypatch.setattr(worker, "prepare_image", lambda data: (data, (1, 1), (1, 1)))

    async def fake_extract(*args, **kwargs):
        if error is not None:
            raise error
        return result

    monkeypatch.setattr(worker, "extract", fake_extract)
    job = jobs.claim(conn, 1)[0]
    asyncio.run(worker.run_job(conn, object(), settings, job))
    return job


def _ok(recipe=None) -> ExtractionResult:
    return ExtractionResult(
        recipe=recipe or PAGE, cost_usd=0.04, elapsed_seconds=12.0, usage={}, truncated=False
    )


def test_success_persists_and_then_deletes_the_image(conn, settings, monkeypatch):
    _queued(conn, settings)

    _run(conn, settings, monkeypatch, result=_ok())

    row = conn.execute("SELECT * FROM recipes WHERE id = 'r1'").fetchone()
    assert row["confidence"] == "high"
    assert row["review_status"] == "unreviewed"
    assert row["raw_text"].startswith("Fried Rice")
    assert load_json(row["structured"])["title"] == "Fried Rice"
    assert load_json(row["normalized"])["ingredients"][0]["amount-low"] == 2.0

    assert not storage.has_images(settings.image_dir, "r1")
    assert conn.execute("SELECT image_path FROM pages").fetchone()["image_path"] is None
    assert conn.execute("SELECT status, cost_usd FROM jobs").fetchone()["status"] == "done"


def test_failure_keeps_the_image_so_the_page_need_not_be_re_shot(conn, settings, monkeypatch):
    _queued(conn, settings)

    _run(conn, settings, monkeypatch, error=ExtractionError("API overloaded"))

    assert storage.has_images(settings.image_dir, "r1")
    row = conn.execute("SELECT status, attempts, error FROM jobs").fetchone()
    assert row["status"] == "pending"
    assert row["attempts"] == 1
    assert "overloaded" in row["error"]
    assert conn.execute("SELECT structured FROM recipes").fetchone()["structured"] is None


def test_an_unexpected_exception_fails_the_job_rather_than_the_worker(conn, settings, monkeypatch):
    _queued(conn, settings)

    _run(conn, settings, monkeypatch, error=ValueError("something odd"))

    row = conn.execute("SELECT status, error FROM jobs").fetchone()
    assert row["status"] == "pending"
    assert "something odd" in row["error"]
    assert storage.has_images(settings.image_dir, "r1")


def test_missing_images_fail_the_job_without_calling_the_model(conn, settings, monkeypatch):
    _queued(conn, settings)
    storage.discard(settings.image_dir, "r1")

    called = False

    async def must_not_run(*args, **kwargs):
        nonlocal called
        called = True
        return _ok()

    monkeypatch.setattr(worker, "extract", must_not_run)
    monkeypatch.setattr(worker, "prepare_image", lambda data: (data, (1, 1), (1, 1)))
    job = jobs.claim(conn, 1)[0]
    asyncio.run(worker.run_job(conn, object(), settings, job))

    assert called is False
    assert "re-shot" in conn.execute("SELECT error FROM jobs").fetchone()["error"]


def test_spend_ceiling_fails_closed_and_does_not_retry(conn, settings, monkeypatch):
    settings = settings.model_copy(update={"monthly_spend_ceiling_usd": 0.01})
    conn.execute(
        "INSERT INTO recipes (id, created_at, updated_at) VALUES ('old', '2026-01-01', '2026-01-01')"
    )
    prior = jobs.enqueue(conn, "old")
    jobs.complete(conn, prior, 5.00)

    _queued(conn, settings)
    _run(conn, settings, monkeypatch, result=_ok())

    row = conn.execute("SELECT status, attempts, error FROM jobs WHERE recipe_id = 'r1'").fetchone()
    assert row["status"] == "failed"
    assert row["attempts"] == 0  # not a retryable condition
    assert "ceiling" in row["error"]
    # The page is still on disk, so raising the ceiling and retrying works.
    assert storage.has_images(settings.image_dir, "r1")


def test_re_extraction_preserves_curated_fields(conn, settings, monkeypatch):
    """A hand-set `required` flag or product link must survive a retry, or
    re-extraction is not safe to offer as a button."""
    _queued(conn, settings)
    _run(conn, settings, monkeypatch, result=_ok())

    curated = load_json(conn.execute("SELECT normalized FROM recipes").fetchone()["normalized"])
    curated["source"] = "https://example.com/fried-rice"
    curated["ingredients"][0]["reference"] = "https://example.com/rice"
    curated["ingredients"][0]["required"] = False
    conn.execute("UPDATE recipes SET normalized = ? WHERE id = 'r1'", (json.dumps(curated),))

    storage.write_page(settings.image_dir, "r1", 0, b"fake-jpeg")
    jobs.reset_for_retry(conn, "r1")
    _run(conn, settings, monkeypatch, result=_ok())

    after = load_json(conn.execute("SELECT normalized FROM recipes").fetchone()["normalized"])
    assert after["source"] == "https://example.com/fried-rice"
    assert after["ingredients"][0]["reference"] == "https://example.com/rice"
    assert after["ingredients"][0]["required"] is False
    # Extractor-owned fields still refresh.
    assert after["ingredients"][0]["amount-low"] == 2.0


@pytest.mark.parametrize("confidence", ["high", "medium", "low"])
def test_confidence_is_recorded_verbatim(conn, settings, monkeypatch, confidence):
    _queued(conn, settings)
    page = {**PAGE, "extraction": {"confidence": confidence, "issues": []}}

    _run(conn, settings, monkeypatch, result=_ok(page))

    assert conn.execute("SELECT confidence FROM recipes").fetchone()["confidence"] == confidence
