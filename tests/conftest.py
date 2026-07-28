"""Shared fixtures.

Every test runs against a real SQLite file in a tmp directory rather than
`:memory:`. The queue's correctness depends on WAL, `BEGIN IMMEDIATE`, and two
connections contending for the same file — none of which an in-memory database
reproduces.
"""

from __future__ import annotations

import io
import uuid

import pytest
from PIL import Image

from app.config import Settings
from app.db import connect, init_db, now


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        api_token="test-token",
        anthropic_api_key="sk-ant-test",
        db_path=tmp_path / "test.db",
        image_dir=tmp_path / "images",
        web_dir=tmp_path / "no-web",
    )


@pytest.fixture
def conn(settings):
    init_db(settings.db_path)
    connection = connect(settings.db_path)
    yield connection
    connection.close()


@pytest.fixture
def make_recipe(conn):
    def _make(recipe_id: str | None = None) -> str:
        recipe_id = recipe_id or str(uuid.uuid4())
        stamp = now()
        conn.execute(
            "INSERT INTO recipes (id, created_at, updated_at) VALUES (?, ?, ?)",
            (recipe_id, stamp, stamp),
        )
        return recipe_id

    return _make


@pytest.fixture
def jpeg_bytes() -> bytes:
    """A real, decodable JPEG — the upload path verifies bytes, not filenames."""
    buffer = io.BytesIO()
    Image.new("RGB", (1200, 1600), (240, 235, 225)).save(buffer, format="JPEG", quality=70)
    return buffer.getvalue()


@pytest.fixture
def app_with(settings):
    """Build a TestClient bound to the given Settings.

    The override key must be the original `get_settings` object, because that is
    what every `Depends()` captured at import time — reassigning the module
    attribute would leave the real (cached, production-path) settings in place.

    The database is initialized here rather than left to the lifespan hook,
    which calls `get_settings()` directly and so would create the file at the
    real configured path instead of the tmp one.
    """
    from contextlib import contextmanager

    from fastapi.testclient import TestClient

    from app import main
    from app.config import get_settings

    @contextmanager
    def _build(overrides: Settings | None = None):
        active = overrides or settings
        init_db(active.db_path)
        active.image_dir.mkdir(parents=True, exist_ok=True)
        main.app.dependency_overrides[get_settings] = lambda: active
        try:
            with TestClient(main.app) as test_client:
                yield test_client
        finally:
            main.app.dependency_overrides.clear()

    return _build


@pytest.fixture
def client(app_with):
    with app_with() as test_client:
        yield test_client


@pytest.fixture
def auth() -> dict[str, str]:
    return {"Authorization": "Bearer test-token"}
