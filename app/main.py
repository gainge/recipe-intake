"""ASGI application.

Serves the capture client from the same origin as the API, which keeps CORS out
of the picture entirely and lets the bearer token stay a same-origin secret.

    uvicorn app.main:app --host 127.0.0.1 --port 8000

The worker runs as its own process (`python -m app.worker`), not a background
task here. An extraction that outlives a reload, or a web process restarted by
the reverse proxy, should not take the queue down with it.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .api import router
from .config import get_settings
from .db import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    init_db(settings.db_path)
    settings.image_dir.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="recipe-intake", lifespan=lifespan)
app.include_router(router)


@app.get("/healthz")
def healthz() -> dict:
    """Unauthenticated on purpose — it reports nothing a caller could not learn
    by observing that the port is open."""
    return {"status": "ok"}


_web = get_settings().web_dir
if _web.is_dir():
    app.mount("/", StaticFiles(directory=_web, html=True), name="web")
