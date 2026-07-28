"""Extraction worker.

A separate process from the API, polling the jobs table. Runs several jobs
concurrently because extraction is API-bound — measured pages take 15–92s of
almost pure waiting, so serializing them would make a 25-page sitting a 25-minute
one for no reason. Concurrency costs a socket, not memory.

Ordering that matters: the page image is deleted only after the extraction has
been committed. Reversing those two loses the page, and there is no second copy.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time

import anthropic

from . import jobs, storage
from .config import Settings, get_settings
from .db import connect, dump_json, init_db, load_json, now
from .extraction import ExtractionError, extract, merge_curated, normalize, prepare_image

log = logging.getLogger("worker")


class SpendCeilingReached(ExtractionError):
    """Not a transient failure — retrying makes it worse."""


def _persist(conn: sqlite3.Connection, recipe_id: str, recipe: dict) -> None:
    """Write the extraction, preserving anything a human already curated."""
    row = conn.execute("SELECT normalized FROM recipes WHERE id = ?", (recipe_id,)).fetchone()
    previous = load_json(row["normalized"]) if row else None

    conn.execute(
        "UPDATE recipes SET raw_text = ?, structured = ?, normalized = ?, confidence = ?,"
        " review_status = 'unreviewed', updated_at = ? WHERE id = ?",
        (
            recipe.get("raw_text", ""),
            dump_json(recipe),
            dump_json(merge_curated(previous, normalize(recipe))),
            (recipe.get("extraction") or {}).get("confidence", ""),
            now(),
            recipe_id,
        ),
    )


async def run_job(
    conn: sqlite3.Connection,
    client: anthropic.AsyncAnthropic,
    settings: Settings,
    job: sqlite3.Row,
) -> None:
    job_id, recipe_id = job["id"], job["recipe_id"]
    try:
        if jobs.month_to_date_spend(conn) >= settings.monthly_spend_ceiling_usd:
            raise SpendCeilingReached(
                f"monthly spend ceiling of ${settings.monthly_spend_ceiling_usd:.2f} reached"
            )

        raw_pages = storage.read_pages(settings.image_dir, recipe_id)
        if not raw_pages:
            raise ExtractionError("page images are gone; the recipe must be re-shot")

        # Idempotent for the capture client, which already downscales. Matters
        # for the <input capture> fallback, which uploads the raw camera file.
        images = [prepare_image(data)[0] for data in raw_pages]

        result = await extract(
            client,
            images,
            model=settings.model,
            max_tokens=settings.max_tokens,
            hint=job["hint"] or "",
            thinking=settings.thinking_config,
        )
    except SpendCeilingReached as exc:
        # Fail closed and do not retry — the next attempt would cost money to
        # discover the same thing.
        conn.execute(
            "UPDATE jobs SET status = 'failed', error = ?, updated_at = ? WHERE id = ?",
            (str(exc), now(), job_id),
        )
        log.error("job %s: %s", job_id, exc)
        return
    except ExtractionError as exc:
        status = jobs.fail(conn, job_id, str(exc), job["attempts"], settings.max_attempts)
        log.warning("job %s failed (%s), image retained: %s", job_id, status, exc)
        return
    except Exception as exc:  # noqa: BLE001 - an unexpected bug must not kill the loop
        status = jobs.fail(conn, job_id, f"unexpected error: {exc}", job["attempts"], settings.max_attempts)
        log.exception("job %s crashed (%s)", job_id, status)
        return

    if result.truncated:
        log.warning("job %s hit max_tokens; output may be truncated", job_id)

    _persist(conn, recipe_id, result.recipe)
    jobs.complete(conn, job_id, result.cost_usd)

    # Only now. The row is committed, so the transcription survives the image.
    storage.discard(settings.image_dir, recipe_id)
    conn.execute("UPDATE pages SET image_path = NULL WHERE recipe_id = ?", (recipe_id,))

    log.info(
        "job %s done in %.1fs, $%.4f, confidence %s",
        job_id,
        result.elapsed_seconds,
        result.cost_usd or 0.0,
        (result.recipe.get("extraction") or {}).get("confidence", "?"),
    )


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    init_db(settings.db_path)
    settings.image_dir.mkdir(parents=True, exist_ok=True)

    conn = connect(settings.db_path)
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key or None)
    semaphore = asyncio.Semaphore(settings.worker_concurrency)
    running: set[asyncio.Task] = set()
    last_sweep = 0.0

    reclaimed = jobs.requeue_stale_running(conn)
    if reclaimed:
        log.info("requeued %d job(s) left running by a previous process", reclaimed)
    log.info(
        "worker up: model=%s thinking=%s concurrency=%d",
        settings.model,
        settings.thinking,
        settings.worker_concurrency,
    )

    async def guarded(job: sqlite3.Row) -> None:
        async with semaphore:
            await run_job(conn, client, settings, job)

    try:
        while True:
            if time.monotonic() - last_sweep > settings.sweep_interval_minutes * 60:
                removed = storage.sweep(settings.image_dir, settings.failed_image_ttl_hours)
                if removed:
                    log.info("swept %d image director%s past TTL", removed, "y" if removed == 1 else "ies")
                last_sweep = time.monotonic()

            free = settings.worker_concurrency - len(running)
            for job in jobs.claim(conn, free) if free > 0 else []:
                task = asyncio.create_task(guarded(job))
                running.add(task)
                task.add_done_callback(running.discard)

            await asyncio.sleep(settings.worker_poll_seconds)
    except asyncio.CancelledError:
        pass
    finally:
        for task in running:
            task.cancel()
        await asyncio.gather(*running, return_exceptions=True)
        await client.close()
        conn.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
