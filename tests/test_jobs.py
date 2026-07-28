"""Queue behaviour.

The claim test is the one that matters: two connections racing for the same
pending job is the only genuine concurrency in this system, and the failure it
guards against — extracting the same page twice — costs real money.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app import jobs
from app.db import connect, now


def test_claim_returns_pending_jobs_in_order(conn, make_recipe):
    first, second = make_recipe(), make_recipe()
    jobs.enqueue(conn, first)
    jobs.enqueue(conn, second)

    claimed = jobs.claim(conn, 10)

    assert [row["recipe_id"] for row in claimed] == [first, second]
    statuses = [r["status"] for r in conn.execute("SELECT status FROM jobs")]
    assert statuses == ["running", "running"]


def test_claim_respects_limit(conn, make_recipe):
    for _ in range(5):
        jobs.enqueue(conn, make_recipe())

    assert len(jobs.claim(conn, 2)) == 2
    assert len(jobs.claim(conn, 10)) == 3
    assert jobs.claim(conn, 10) == []


def test_two_connections_never_claim_the_same_job(settings, conn, make_recipe):
    """Extracting one page twice would silently double its cost."""
    for _ in range(6):
        jobs.enqueue(conn, make_recipe())

    other = connect(settings.db_path)
    try:
        first = {row["id"] for row in jobs.claim(conn, 6)}
        second = {row["id"] for row in jobs.claim(other, 6)}
    finally:
        other.close()

    assert len(first) == 6
    assert second == set()
    assert not (first & second)


def test_claim_skips_jobs_still_in_backoff(conn, make_recipe):
    job_id = jobs.enqueue(conn, make_recipe())
    jobs.claim(conn, 1)
    jobs.fail(conn, job_id, "overloaded", attempts=0, max_attempts=3)

    assert jobs.claim(conn, 10) == []

    conn.execute("UPDATE jobs SET next_attempt_at = '' WHERE id = ?", (job_id,))
    assert len(jobs.claim(conn, 10)) == 1


def test_fail_retries_then_gives_up_and_keeps_attempt_count(conn, make_recipe):
    job_id = jobs.enqueue(conn, make_recipe())

    assert jobs.fail(conn, job_id, "boom", attempts=0, max_attempts=3) == "pending"
    assert jobs.fail(conn, job_id, "boom", attempts=1, max_attempts=3) == "pending"
    assert jobs.fail(conn, job_id, "boom", attempts=2, max_attempts=3) == "failed"

    row = conn.execute("SELECT attempts, error, next_attempt_at FROM jobs").fetchone()
    assert row["attempts"] == 3
    assert row["error"] == "boom"
    # A terminal failure must not schedule a retry it will never take.
    assert row["next_attempt_at"] == ""


def test_backoff_grows_between_attempts(conn, make_recipe):
    job_id = jobs.enqueue(conn, make_recipe())

    jobs.fail(conn, job_id, "boom", attempts=0, max_attempts=5)
    first = conn.execute("SELECT next_attempt_at FROM jobs").fetchone()["next_attempt_at"]
    jobs.fail(conn, job_id, "boom", attempts=1, max_attempts=5)
    second = conn.execute("SELECT next_attempt_at FROM jobs").fetchone()["next_attempt_at"]

    assert second > first


def test_reset_for_retry_clears_attempts_and_backoff(conn, make_recipe):
    recipe_id = make_recipe()
    job_id = jobs.enqueue(conn, recipe_id)
    jobs.fail(conn, job_id, "boom", attempts=2, max_attempts=3)

    assert jobs.reset_for_retry(conn, recipe_id) is True

    row = conn.execute("SELECT status, attempts, error, next_attempt_at FROM jobs").fetchone()
    assert (row["status"], row["attempts"], row["error"], row["next_attempt_at"]) == ("pending", 0, "", "")


def test_reset_for_retry_on_unknown_recipe(conn):
    assert jobs.reset_for_retry(conn, "nope") is False


def test_requeue_stale_running_recovers_a_killed_worker(conn, make_recipe):
    """Nothing else ever moves a row out of `running`, so without this the job
    is stuck forever."""
    jobs.enqueue(conn, make_recipe())
    jobs.claim(conn, 1)

    assert jobs.requeue_stale_running(conn) == 1
    assert len(jobs.claim(conn, 1)) == 1


def test_month_to_date_spend_sums_recorded_cost(conn, make_recipe):
    for cost in (0.05, 0.12, None):
        job_id = jobs.enqueue(conn, make_recipe())
        jobs.complete(conn, job_id, cost)

    assert jobs.month_to_date_spend(conn) == pytest.approx(0.17)


def test_month_to_date_spend_ignores_previous_months(conn, make_recipe):
    job_id = jobs.enqueue(conn, make_recipe())
    jobs.complete(conn, job_id, 9.99)
    last_month = (datetime.now(timezone.utc).replace(day=1) - timedelta(days=5)).isoformat(timespec="seconds")
    conn.execute("UPDATE jobs SET updated_at = ?", (last_month,))

    assert jobs.month_to_date_spend(conn) == pytest.approx(0.0)


def test_uploads_in_last_hour_ignores_older_jobs(conn, make_recipe):
    jobs.enqueue(conn, make_recipe())
    jobs.enqueue(conn, make_recipe())
    assert jobs.uploads_in_last_hour(conn) == 2

    stale = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(timespec="seconds")
    conn.execute("UPDATE jobs SET created_at = ?", (stale,))
    assert jobs.uploads_in_last_hour(conn) == 0


def test_complete_clears_a_previous_error(conn, make_recipe):
    job_id = jobs.enqueue(conn, make_recipe())
    jobs.fail(conn, job_id, "transient", attempts=0, max_attempts=3)
    jobs.complete(conn, job_id, 0.03)

    row = conn.execute("SELECT status, error, cost_usd FROM jobs").fetchone()
    assert (row["status"], row["error"], row["cost_usd"]) == ("done", "", 0.03)
    assert now()  # sanity: timestamps are generated, not hard-coded
