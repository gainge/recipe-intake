"""The Anthropic call.

Streaming rather than a single blocking request: measured pages take 15–92s and
`max_tokens` is 16000, which is far enough into timeout territory that the SDK
warns about it. `get_final_message()` still yields the fully parsed result, so
nothing downstream has to deal with stream events.

Structured outputs guarantee the response validates against `Recipe`, so there is
no defensive JSON repair anywhere in this pipeline.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass

import anthropic

from .prompt import SYSTEM_PROMPT, user_instruction
from .schema import Recipe

# USD per million tokens: (input, output).
PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),  # introductory rate through 2026-08-31; list is 3.00 / 15.00
    "claude-haiku-4-5": (1.00, 5.00),
}


class ExtractionError(RuntimeError):
    """Extraction failed in a way worth retrying or reporting to a human.

    Raised rather than returned so the worker cannot mistake a failure for a
    result and delete the page image on the strength of it.
    """


@dataclass(frozen=True)
class ExtractionResult:
    recipe: dict
    cost_usd: float | None
    elapsed_seconds: float
    usage: dict
    truncated: bool


def estimate_cost(model: str, usage) -> float | None:
    rates = PRICING.get(model)
    if rates is None:
        return None
    in_rate, out_rate = rates

    fresh = getattr(usage, "input_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    output = getattr(usage, "output_tokens", 0) or 0

    # Cache reads bill at ~0.1x input, writes at ~1.25x.
    return (
        fresh * in_rate + cache_read * in_rate * 0.1 + cache_write * in_rate * 1.25 + output * out_rate
    ) / 1_000_000


def build_content(images: list[bytes], hint: str = "") -> list[dict]:
    blocks: list[dict] = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.standard_b64encode(data).decode(),
            },
        }
        for data in images
    ]
    blocks.append({"type": "text", "text": user_instruction(hint)})
    return blocks


async def extract(
    client: anthropic.AsyncAnthropic,
    images: list[bytes],
    *,
    model: str,
    max_tokens: int,
    hint: str = "",
    thinking: dict | None = None,
) -> ExtractionResult:
    """Page images -> verbatim structured recipe.

    All images belong to ONE recipe, in reading order — a two-page spread goes in
    a single call so the model can read the ingredient column on the left against
    the method on the right.
    """
    if not images:
        raise ExtractionError("no images supplied")

    extra = {"thinking": thinking} if thinking else {}
    started = time.monotonic()
    try:
        async with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    # Stable across every request, so it is the cache prefix.
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": build_content(images, hint)}],
            output_format=Recipe,
            **extra,
        ) as stream:
            message = await stream.get_final_message()
    except anthropic.APIStatusError as exc:
        raise ExtractionError(f"API error {exc.status_code}: {exc.message}") from exc
    except anthropic.APIConnectionError as exc:
        raise ExtractionError(f"connection error: {exc}") from exc
    elapsed = time.monotonic() - started

    stop_reason = getattr(message, "stop_reason", None)
    if stop_reason == "refusal":
        raise ExtractionError("the request was declined by safety classifiers")

    parsed = message.parsed_output
    if parsed is None:
        raise ExtractionError("response did not parse against the schema")

    usage = message.usage
    return ExtractionResult(
        recipe=parsed.model_dump(),
        cost_usd=estimate_cost(model, usage),
        elapsed_seconds=round(elapsed, 2),
        usage=usage.model_dump() if hasattr(usage, "model_dump") else {},
        # Not fatal on its own — the schema still validated — but it means the
        # model ran out of room, so a reviewer should be told.
        truncated=stop_reason == "max_tokens",
    )
