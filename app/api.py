"""HTTP API.

A subset of DESIGN.md §8 — review edits and export formats belong to later
phases. What is here is everything the capture flow needs plus the polling
endpoint downstream consumers will use.

The review queue is *not* stored. `validate.review()` is a pure function over the
verbatim model output, so it is derived on every read. Tightening a check
therefore improves every recipe already extracted, with no migration and no
re-extraction.
"""

from __future__ import annotations

import io
import sqlite3
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from . import jobs, storage
from .auth import require_token
from .config import Settings, get_settings
from .db import connect, load_json, now, transaction
from .extraction import review as derive_review

router = APIRouter(prefix="/api", dependencies=[Depends(require_token)])


def get_conn(settings: Settings = Depends(get_settings)):
    conn = connect(settings.db_path)
    try:
        yield conn
    finally:
        conn.close()


Conn = Annotated[sqlite3.Connection, Depends(get_conn)]
Config = Annotated[Settings, Depends(get_settings)]


class SessionCreated(BaseModel):
    session_id: str


class RecipeCreated(BaseModel):
    recipe_id: str
    job_id: str
    pages: int


def _validate_image(data: bytes, settings: Settings, index: int) -> None:
    """Check the bytes, not the filename.

    A client can call anything `.jpg`; the only honest test is whether Pillow can
    decode it, and the worker will hand it to the model either way.
    """
    if len(data) > settings.max_image_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"page {index + 1} is {len(data) // 1024}KB, over the {settings.max_image_bytes // 1024}KB limit",
        )
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"page {index + 1} is empty")
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"page {index + 1} is not a readable image"
        ) from None


def _recipe_payload(row: sqlite3.Row, job: sqlite3.Row | None) -> dict:
    structured = load_json(row["structured"])
    return {
        "id": row["id"],
        "session_id": row["session_id"],
        "status": job["status"] if job else "unknown",
        "attempts": job["attempts"] if job else 0,
        "error": job["error"] if job else "",
        "confidence": row["confidence"],
        "review_status": row["review_status"],
        "book_title": row["book_title"],
        "page_ref": row["page_ref"],
        "title": (structured or {}).get("title", ""),
        "raw_text": row["raw_text"],
        "structured": structured,
        "normalized": load_json(row["normalized"]),
        # Derived on read, never stored — see the module docstring.
        "review": derive_review(structured) if structured else None,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


@router.post("/sessions", response_model=SessionCreated, status_code=status.HTTP_201_CREATED)
def create_session(conn: Conn, book_title: Annotated[str, Form()] = "") -> SessionCreated:
    session_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO sessions (id, book_title, created_at) VALUES (?, ?, ?)",
        (session_id, book_title, now()),
    )
    return SessionCreated(session_id=session_id)


@router.post("/recipes", response_model=RecipeCreated, status_code=status.HTTP_202_ACCEPTED)
async def create_recipe(
    conn: Conn,
    settings: Config,
    images: Annotated[list[UploadFile], File()],
    session_id: Annotated[str, Form()] = "",
    hint: Annotated[str, Form()] = "",
    page_ref: Annotated[str, Form()] = "",
) -> RecipeCreated:
    """Upload 1..n page images as ONE recipe and enqueue extraction.

    Grouping is the client's job (DESIGN.md §4.1): the person holding the book
    knows a spread is one recipe, and inferring it server-side from the images
    would be guessing at something already known.
    """
    if not images:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "no images supplied")
    if len(images) > settings.max_pages_per_recipe:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"{len(images)} pages for one recipe; the limit is {settings.max_pages_per_recipe}",
        )
    if jobs.uploads_in_last_hour(conn) >= settings.uploads_per_hour:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"rate limit of {settings.uploads_per_hour} uploads/hour reached",
        )

    payloads = [await image.read() for image in images]
    for index, data in enumerate(payloads):
        _validate_image(data, settings, index)

    session = session_id or None
    if session and conn.execute("SELECT 1 FROM sessions WHERE id = ?", (session,)).fetchone() is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown session_id")

    book_title = ""
    if session:
        row = conn.execute("SELECT book_title FROM sessions WHERE id = ?", (session,)).fetchone()
        book_title = row["book_title"] if row else ""

    recipe_id = str(uuid.uuid4())
    written: list[tuple[int, int]] = []
    for sequence, data in enumerate(payloads):
        storage.write_page(settings.image_dir, recipe_id, sequence, data)
        written.append((sequence, len(data)))

    try:
        stamp = now()
        with transaction(conn):
            conn.execute(
                "INSERT INTO recipes (id, session_id, book_title, page_ref, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (recipe_id, session, book_title, page_ref, stamp, stamp),
            )
            conn.executemany(
                "INSERT INTO pages (id, recipe_id, sequence, image_path, bytes) VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        str(uuid.uuid4()),
                        recipe_id,
                        sequence,
                        str(storage.recipe_dir(settings.image_dir, recipe_id) / f"{sequence:02d}.jpg"),
                        size,
                    )
                    for sequence, size in written
                ],
            )
            job_id = jobs.enqueue(conn, recipe_id, hint)
    except Exception:
        # The rows are gone, so the images are unreachable — remove them rather
        # than leaving the sweeper to find them in a day's time.
        storage.discard(settings.image_dir, recipe_id)
        raise

    return RecipeCreated(recipe_id=recipe_id, job_id=job_id, pages=len(payloads))


@router.get("/recipes")
def list_recipes(
    conn: Conn,
    status_filter: Annotated[str, Query(alias="status")] = "",
    review_status: str = "",
    since: str = "",
    limit: Annotated[int, Query(le=200)] = 50,
) -> dict:
    """List and poll. `since` against `updated_at` is the downstream integration
    point — a consumer can pull new work without coupling to this codebase."""
    clauses, params = [], []
    if since:
        clauses.append("r.updated_at > ?")
        params.append(since)
    if review_status:
        clauses.append("r.review_status = ?")
        params.append(review_status)
    if status_filter:
        clauses.append("j.status = ?")
        params.append(status_filter)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = conn.execute(
        "SELECT r.*, j.status, j.attempts, j.error FROM recipes r"
        " LEFT JOIN jobs j ON j.recipe_id = r.id"
        f" {where} ORDER BY r.updated_at DESC LIMIT ?",
        (*params, limit),
    ).fetchall()

    return {
        "recipes": [
            {
                "id": row["id"],
                "title": (load_json(row["structured"]) or {}).get("title", ""),
                "status": row["status"],
                "confidence": row["confidence"],
                "review_status": row["review_status"],
                "book_title": row["book_title"],
                "error": row["error"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]
    }


@router.get("/recipes/{recipe_id}")
def get_recipe(conn: Conn, recipe_id: str) -> dict:
    row = conn.execute("SELECT * FROM recipes WHERE id = ?", (recipe_id,)).fetchone()
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown recipe")
    job = conn.execute(
        "SELECT status, attempts, error FROM jobs WHERE recipe_id = ? ORDER BY created_at DESC LIMIT 1",
        (recipe_id,),
    ).fetchone()
    return _recipe_payload(row, job)


@router.post("/recipes/{recipe_id}/retry", status_code=status.HTTP_202_ACCEPTED)
def retry_recipe(conn: Conn, settings: Config, recipe_id: str) -> dict:
    if conn.execute("SELECT 1 FROM recipes WHERE id = ?", (recipe_id,)).fetchone() is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown recipe")
    if not storage.has_images(settings.image_dir, recipe_id):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "the page images have been deleted; this recipe must be re-shot",
        )
    if not jobs.reset_for_retry(conn, recipe_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no job exists for this recipe")
    return {"recipe_id": recipe_id, "status": "pending"}
