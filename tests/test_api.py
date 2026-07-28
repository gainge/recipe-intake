"""HTTP surface.

Weighted towards the boundaries rather than the happy path: auth, upload
validation, and the two backstops (rate limit, retry-after-deletion) are what
stand between a leaked token and an unbounded Anthropic bill.
"""

from __future__ import annotations

import json

from app import jobs, storage
from app.db import connect

UPLOAD = "/api/recipes"


def _files(jpeg: bytes, count: int = 1):
    return [("images", (f"{i}.jpg", jpeg, "image/jpeg")) for i in range(count)]


def test_healthz_needs_no_token(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_every_api_route_requires_a_token(client):
    for method, path in (
        ("post", "/api/sessions"),
        ("get", "/api/recipes"),
        ("get", "/api/recipes/anything"),
        ("post", "/api/recipes/anything/retry"),
    ):
        assert getattr(client, method)(path).status_code == 401, path


def test_a_wrong_token_is_rejected(client):
    assert client.get("/api/recipes", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert client.get("/api/recipes", headers={"Authorization": "test-token"}).status_code == 401


def test_a_blank_server_token_refuses_service_rather_than_allowing_it(settings, app_with):
    """An unset token is a misconfiguration. Treating it as "no auth required"
    would put the Anthropic key's spending power on the open internet."""
    with app_with(settings.model_copy(update={"api_token": ""})) as client:
        assert client.get("/api/recipes").status_code == 503


def test_upload_creates_recipe_pages_and_a_job(client, auth, settings, jpeg_bytes):
    session_id = client.post("/api/sessions", data={"book_title": "Ottolenghi"}, headers=auth).json()[
        "session_id"
    ]

    response = client.post(
        UPLOAD, files=_files(jpeg_bytes, 2), data={"session_id": session_id, "hint": "Biryani"}, headers=auth
    )

    assert response.status_code == 202
    body = response.json()
    assert body["pages"] == 2
    recipe_id = body["recipe_id"]

    assert len(storage.read_pages(settings.image_dir, recipe_id)) == 2

    conn = connect(settings.db_path)
    try:
        assert conn.execute("SELECT COUNT(*) c FROM pages WHERE recipe_id = ?", (recipe_id,)).fetchone()["c"] == 2
        job = conn.execute("SELECT status, hint FROM jobs WHERE recipe_id = ?", (recipe_id,)).fetchone()
        assert (job["status"], job["hint"]) == ("pending", "Biryani")
        # Session metadata is denormalized onto the recipe at upload time.
        recipe = conn.execute("SELECT book_title FROM recipes WHERE id = ?", (recipe_id,)).fetchone()
        assert recipe["book_title"] == "Ottolenghi"
    finally:
        conn.close()


def test_upload_works_without_a_session(client, auth, jpeg_bytes):
    assert client.post(UPLOAD, files=_files(jpeg_bytes), headers=auth).status_code == 202


def test_upload_rejects_an_unknown_session(client, auth, jpeg_bytes):
    response = client.post(UPLOAD, files=_files(jpeg_bytes), data={"session_id": "nope"}, headers=auth)
    assert response.status_code == 404


def test_upload_rejects_bytes_that_are_not_an_image(client, auth, settings):
    """Validated on content, not filename — the worker will hand these to the
    model either way."""
    response = client.post(
        UPLOAD, files=[("images", ("page.jpg", b"definitely not a jpeg", "image/jpeg"))], headers=auth
    )

    assert response.status_code == 400
    assert not any(settings.image_dir.rglob("*.jpg"))


def test_upload_rejects_an_oversized_image(client, auth, settings, jpeg_bytes):
    huge = jpeg_bytes + b"\x00" * settings.max_image_bytes
    assert client.post(UPLOAD, files=[("images", ("p.jpg", huge, "image/jpeg"))], headers=auth).status_code == 413


def test_upload_rejects_too_many_pages_for_one_recipe(client, auth, settings, jpeg_bytes):
    over = settings.max_pages_per_recipe + 1
    assert client.post(UPLOAD, files=_files(jpeg_bytes, over), headers=auth).status_code == 400


def test_a_rejected_upload_leaves_no_images_behind(client, auth, settings, jpeg_bytes):
    """A partially-written upload that fails validation must not accumulate on
    disk waiting for the sweeper."""
    client.post(
        UPLOAD,
        files=[*_files(jpeg_bytes), ("images", ("bad.jpg", b"junk", "image/jpeg"))],
        headers=auth,
    )
    assert not any(settings.image_dir.rglob("*.jpg"))


def test_rate_limit_blocks_further_uploads(app_with, auth, settings, jpeg_bytes):
    with app_with(settings.model_copy(update={"uploads_per_hour": 2})) as client:
        assert client.post(UPLOAD, files=_files(jpeg_bytes), headers=auth).status_code == 202
        assert client.post(UPLOAD, files=_files(jpeg_bytes), headers=auth).status_code == 202
        response = client.post(UPLOAD, files=_files(jpeg_bytes), headers=auth)

    assert response.status_code == 429
    assert "rate limit" in response.json()["detail"]


def test_get_recipe_reports_queue_state_before_extraction(client, auth, jpeg_bytes):
    recipe_id = client.post(UPLOAD, files=_files(jpeg_bytes), headers=auth).json()["recipe_id"]

    body = client.get(f"/api/recipes/{recipe_id}", headers=auth).json()

    assert body["status"] == "pending"
    assert body["structured"] is None
    assert body["review"] is None
    assert body["review_status"] == "unreviewed"


def test_get_unknown_recipe_is_404(client, auth):
    assert client.get("/api/recipes/nope", headers=auth).status_code == 404


def test_review_is_derived_on_read_not_stored(client, auth, settings, jpeg_bytes):
    """Tightening a check in validate.py must improve recipes already extracted,
    which only works if nothing persisted the old answer."""
    recipe_id = client.post(UPLOAD, files=_files(jpeg_bytes), headers=auth).json()["recipe_id"]

    structured = {
        "raw_text": "Soup",
        "title": "",
        "yield_text": "",
        "times": {"prep": "", "cook": "", "total": ""},
        "ingredients": [],
        "instructions": [],
        "notes": [],
        "extraction": {"confidence": "low", "issues": [{"path": "title", "detail": "unreadable"}]},
    }
    conn = connect(settings.db_path)
    try:
        conn.execute("UPDATE recipes SET structured = ? WHERE id = ?", (json.dumps(structured), recipe_id))
        assert conn.execute("PRAGMA table_info(recipes)").fetchall()
        assert not any(row[1] == "review" for row in conn.execute("PRAGMA table_info(recipes)"))
    finally:
        conn.close()

    review = client.get(f"/api/recipes/{recipe_id}", headers=auth).json()["review"]

    origins = {issue["origin"] for issue in review["issues"]}
    assert origins == {"model", "validator"}
    paths = {issue["path"] for issue in review["issues"]}
    assert {"title", "ingredients", "instructions"} <= paths


def test_list_filters_by_status_and_since(client, auth, settings, jpeg_bytes):
    first = client.post(UPLOAD, files=_files(jpeg_bytes), headers=auth).json()["recipe_id"]
    second = client.post(UPLOAD, files=_files(jpeg_bytes), headers=auth).json()["recipe_id"]

    conn = connect(settings.db_path)
    try:
        conn.execute("UPDATE jobs SET status = 'done' WHERE recipe_id = ?", (first,))
        conn.execute("UPDATE recipes SET updated_at = '2020-01-01T00:00:00+00:00' WHERE id = ?", (second,))
    finally:
        conn.close()

    done = client.get("/api/recipes?status=done", headers=auth).json()["recipes"]
    assert [r["id"] for r in done] == [first]

    recent = client.get("/api/recipes?since=2025-01-01T00:00:00+00:00", headers=auth).json()["recipes"]
    assert [r["id"] for r in recent] == [first]


def test_retry_requeues_while_the_image_survives(client, auth, settings, jpeg_bytes):
    recipe_id = client.post(UPLOAD, files=_files(jpeg_bytes), headers=auth).json()["recipe_id"]
    conn = connect(settings.db_path)
    try:
        job_id = conn.execute("SELECT id FROM jobs WHERE recipe_id = ?", (recipe_id,)).fetchone()["id"]
        jobs.fail(conn, job_id, "boom", attempts=2, max_attempts=3)

        assert client.post(f"/api/recipes/{recipe_id}/retry", headers=auth).status_code == 202

        row = conn.execute("SELECT status, attempts FROM jobs WHERE id = ?", (job_id,)).fetchone()
        assert (row["status"], row["attempts"]) == ("pending", 0)
    finally:
        conn.close()


def test_retry_is_a_conflict_once_the_image_is_gone(client, auth, settings, jpeg_bytes):
    """The honest answer is that the page must be re-shot — not a queued job
    that will fail again."""
    recipe_id = client.post(UPLOAD, files=_files(jpeg_bytes), headers=auth).json()["recipe_id"]
    storage.discard(settings.image_dir, recipe_id)

    response = client.post(f"/api/recipes/{recipe_id}/retry", headers=auth)

    assert response.status_code == 409
    assert "re-shot" in response.json()["detail"]
