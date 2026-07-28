"""SQLite access and schema.

WAL mode plus a busy timeout is what makes two processes — the API and the
worker — share one file safely. Single user means no write contention worth
worrying about beyond that.

There is no migration framework. The schema is one idempotent block applied at
startup by both processes; for a single-user tool with one deployment, a
migration table would be more machinery than the thing it manages.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    book_title  TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS recipes (
    id            TEXT PRIMARY KEY,
    session_id    TEXT REFERENCES sessions(id),
    raw_text      TEXT NOT NULL DEFAULT '',
    -- Verbatim model output. Written once on success and never rewritten: the
    -- photograph is gone by then, so this is the only surviving source.
    structured    TEXT,
    -- Derived, and safe to regenerate from `structured` — except that it also
    -- carries human-curated fields, which is why it is stored rather than
    -- recomputed on every read.
    normalized    TEXT,
    confidence    TEXT NOT NULL DEFAULT '',
    review_status TEXT NOT NULL DEFAULT 'unreviewed',
    book_title    TEXT NOT NULL DEFAULT '',
    page_ref      TEXT NOT NULL DEFAULT '',
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pages (
    id         TEXT PRIMARY KEY,
    recipe_id  TEXT NOT NULL REFERENCES recipes(id),
    sequence   INTEGER NOT NULL,
    image_path TEXT,          -- NULL once deleted
    bytes      INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS jobs (
    id         TEXT PRIMARY KEY,
    recipe_id  TEXT NOT NULL REFERENCES recipes(id),
    status     TEXT NOT NULL DEFAULT 'pending',  -- pending|running|done|failed
    attempts   INTEGER NOT NULL DEFAULT 0,
    hint       TEXT NOT NULL DEFAULT '',
    error      TEXT NOT NULL DEFAULT '',
    cost_usd   REAL,
    -- Backoff gate. A model overload should not be retried three times in as
    -- many seconds and then given up on.
    next_attempt_at TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_jobs_status  ON jobs(status, next_attempt_at);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_pages_recipe ON pages(recipe_id, sequence);
CREATE INDEX IF NOT EXISTS idx_recipes_updated ON recipes(updated_at);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False because a FastAPI request can touch its connection
    # from both the event loop and the threadpool: sync dependencies run in a
    # worker thread while `async def` endpoints run on the loop. The access is
    # sequential, not concurrent — each request gets its own connection and
    # never shares it — so the check is guarding against something that cannot
    # happen here.
    conn = sqlite3.connect(db_path, timeout=10.0, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db(db_path: Path) -> None:
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA)
    finally:
        conn.close()


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """IMMEDIATE takes the write lock up front.

    With two processes polling the same file, deferring it means discovering the
    conflict halfway through and rolling back work already done.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except Exception:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def load_json(value: str | None) -> dict | None:
    return json.loads(value) if value else None


def dump_json(value: dict | None) -> str | None:
    return json.dumps(value, ensure_ascii=False) if value is not None else None
