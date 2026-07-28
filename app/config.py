"""Runtime settings, all environment-driven.

The API process and the worker process both import this, so a value read here is
guaranteed identical on both sides — which matters for `IMAGE_DIR`, where a
mismatch would mean the worker silently never finds the images the API wrote.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    """Server-side only. The capture client never sees this and never calls the
    Anthropic API directly."""

    api_token: str = ""
    """Bearer token for every /api route. Empty means the app refuses to start —
    an unauthenticated endpoint in front of a paid API is the real cost risk."""

    model: str = "claude-sonnet-5"

    thinking: str = "adaptive"
    """`adaptive` or `disabled`.

    Measured on seven real pages: disabling thinking halves cost and latency, but
    one page in seven came back with an empty structured body while reporting no
    issues at all, compound ingredient lines stopped being split, and the model's
    own confidence became less honest. See spike/README.md. Worth re-measuring
    against any future model, which is why this is a setting and not a constant.
    """

    max_tokens: int = 16_000
    """Caps thinking *plus* output together, so this needs headroom well above
    the size of the JSON alone."""

    db_path: Path = Path("data/recipe-intake.db")
    image_dir: Path = Path("data/images")

    worker_concurrency: int = 3
    """Extraction is API-bound, not CPU-bound — this buys wall-clock on a
    multi-page sitting without costing meaningful RAM on a 1GB box."""

    worker_poll_seconds: float = 2.0
    max_attempts: int = 3

    failed_image_ttl_hours: int = 24
    """How long a failed job's image survives so a transient API error doesn't
    force re-shooting the page. Successful extractions delete immediately."""

    sweep_interval_minutes: int = 15

    monthly_spend_ceiling_usd: float = 25.0
    """Checked before dispatch, and jobs fail closed when it is reached. Per-page
    cost varies about 5x, so this sums actual recorded cost rather than counting
    pages."""

    uploads_per_hour: int = 60
    max_image_bytes: int = 8 * 1024 * 1024
    max_pages_per_recipe: int = 6

    web_dir: Path = Field(default=Path("web"))

    @property
    def thinking_config(self) -> dict | None:
        """None means send no `thinking` parameter, leaving the model default."""
        return {"type": "disabled"} if self.thinking == "disabled" else None


@lru_cache
def get_settings() -> Settings:
    return Settings()
