"""Queue operations against the jobs table.

Shared by the API (which enqueues and counts) and the worker (which claims and
settles), so the two cannot disagree about what a status means.

The queue is a table, not a broker. One user on a 1GB box does not justify Redis,
and a status column is something you can debug with `sqlite3`.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timedelta, timezone

from .db import now, transaction

TERMINAL = ("done", "failed")


def enqueue(conn: sqlite3.Connection, recipe_id: str, hint: str = "") -> str:
    job_id = str(uuid.uuid4())
    stamp = now()
    conn.execute(
        "INSERT INTO jobs (id, recipe_id, status, hint, created_at, updated_at)"
        " VALUES (?, ?, 'pending', ?, ?, ?)",
        (job_id, recipe_id, hint, stamp, stamp),
    )
    return job_id


def claim(conn: sqlite3.Connection, limit: int) -> list[sqlite3.Row]:
    """Atomically move up to `limit` due jobs to `running` and return them.

    The select and the update share one IMMEDIATE transaction, so two workers —
    or a worker racing its own next poll — can never take the same job.
    """
    stamp = now()
    with transaction(conn):
        rows = conn.execute(
            "SELECT id, recipe_id, hint, attempts FROM jobs"
            " WHERE status = 'pending' AND next_attempt_at <= ?"
            " ORDER BY created_at LIMIT ?",
            (stamp, limit),
        ).fetchall()
        if rows:
            conn.executemany(
                "UPDATE jobs SET status = 'running', updated_at = ? WHERE id = ?",
                [(stamp, row["id"]) for row in rows],
            )
    return rows


def complete(conn: sqlite3.Connection, job_id: str, cost_usd: float | None) -> None:
    conn.execute(
        "UPDATE jobs SET status = 'done', error = '', cost_usd = ?, updated_at = ? WHERE id = ?",
        (cost_usd, now(), job_id),
    )


def fail(conn: sqlite3.Connection, job_id: str, error: str, attempts: int, max_attempts: int) -> str:
    """Record a failure, scheduling a retry unless the budget is spent.

    Returns the resulting status. The page image is deliberately left in place
    either way — it is the retry window's whole purpose, and deleting it here
    would turn a transient API error into a re-shoot.
    """
    attempts += 1
    if attempts >= max_attempts:
        status, retry_at = "failed", ""
    else:
        status = "pending"
        # 30s, then 120s. Long enough for an overloaded API to recover, short
        # enough that a person watching the queue sees it move.
        delay = 30 * (4 ** (attempts - 1))
        retry_at = (datetime.now(timezone.utc) + timedelta(seconds=delay)).isoformat(timespec="seconds")

    conn.execute(
        "UPDATE jobs SET status = ?, attempts = ?, error = ?, next_attempt_at = ?, updated_at = ?"
        " WHERE id = ?",
        (status, attempts, error[:2000], retry_at, now(), job_id),
    )
    return status


def reset_for_retry(conn: sqlite3.Connection, recipe_id: str) -> bool:
    """Put a recipe's job back on the queue. False when there is nothing to do."""
    stamp = now()
    with transaction(conn):
        row = conn.execute(
            "SELECT id FROM jobs WHERE recipe_id = ? ORDER BY created_at DESC LIMIT 1",
            (recipe_id,),
        ).fetchone()
        if row is None:
            return False
        conn.execute(
            "UPDATE jobs SET status = 'pending', attempts = 0, error = '',"
            " next_attempt_at = '', updated_at = ? WHERE id = ?",
            (stamp, row["id"]),
        )
    return True


def month_to_date_spend(conn: sqlite3.Connection) -> float:
    """Actual recorded cost this calendar month.

    Sums real cost rather than counting pages: measured per-page cost varies
    about fivefold with page density, so a page count would be a poor proxy.
    """
    start = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    row = conn.execute(
        "SELECT COALESCE(SUM(cost_usd), 0) AS total FROM jobs WHERE updated_at >= ?",
        (start.isoformat(timespec="seconds"),),
    ).fetchone()
    return float(row["total"])


def uploads_in_last_hour(conn: sqlite3.Connection) -> int:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(timespec="seconds")
    row = conn.execute("SELECT COUNT(*) AS n FROM jobs WHERE created_at >= ?", (cutoff,)).fetchone()
    return int(row["n"])


def requeue_stale_running(conn: sqlite3.Connection) -> int:
    """Reclaim jobs left `running` by a worker that died mid-extraction.

    Called once at worker startup. Without it those jobs sit forever: nothing
    else ever moves a row out of `running`.
    """
    stamp = now()
    with transaction(conn):
        cursor = conn.execute(
            "UPDATE jobs SET status = 'pending', next_attempt_at = '', updated_at = ?"
            " WHERE status = 'running'",
            (stamp,),
        )
        return cursor.rowcount
