"""Validation — derived review items.

Pure functions over an extraction payload and its normalized form. Like
`normalize.py`, this runs with no API call, so the checks can be tightened and
re-run over the whole stored corpus.

Every check here answers one question: *would a human, given only this output
and no photograph, be able to tell that something went wrong?* Anything that
fails that test belongs in the review queue rather than in a log line, because
after extraction succeeds the photograph is gone and the review queue is the
only remaining chance to catch it.

Checks are deliberately quiet. A validator that flags legitimate output trains
the reviewer to skim past the queue, which costs more than the errors it catches.
"""

from __future__ import annotations

import re

from .normalize import parse_amount, parse_duration, parse_servings

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {"and", "or", "of", "the", "a", "an", "to", "for", "in", "on", "with", "plus", "into", "at", "from"}
)
_OPTIONAL_SIGNALS = re.compile(r"\b(?:optional|if\s+desired|if\s+you\s+like|to\s+taste|garnish|to\s+serve|for\s+serving)\b")
_IMPRECISE_UNITS = {"pinch", "dash", "to_taste"}


def _stems(text: str) -> set[str]:
    """Crude singularization is enough here — we compare against the same line."""
    stems = set()
    for token in _TOKEN.findall((text or "").lower()):
        if token in _STOPWORDS or len(token) < 2:
            continue
        stems.add(token[:-1] if len(token) > 3 and token.endswith("s") else token)
    return stems


def _issue(path: str, detail: str) -> dict:
    return {"path": path, "detail": detail, "origin": "validator"}


def _check_ingredient(index: int, ing: dict) -> list[dict]:
    at = f"ingredients[{index}]"
    issues: list[dict] = []
    raw = ing.get("raw", "")

    amount_text = ing.get("amount_text", "")
    if not parse_amount(amount_text).ok:
        issues.append(_issue(f"{at}.amount_text", f'amount "{amount_text}" could not be read as a number'))

    canonical = ing.get("units_canonical", "")
    if ing.get("units") and not canonical:
        issues.append(_issue(f"{at}.units_canonical", f'unit "{ing["units"]}" was printed but not mapped to the canonical vocabulary'))
    if canonical in _IMPRECISE_UNITS and amount_text:
        issues.append(_issue(f"{at}.amount_text", f'"{canonical}" carries an amount ("{amount_text}"); one of the two is wrong'))

    # The split of raw into name/processing must not introduce words that were
    # never on the page. This is the only automatic check on the semantic fields,
    # and it is why `raw` is worth carrying.
    if raw:
        source = _stems(raw)
        for field in ("name", "processing"):
            invented = _stems(ing.get(field, "")) - source
            if invented:
                issues.append(
                    _issue(f"{at}.{field}", f'"{", ".join(sorted(invented))}" does not appear in the printed line: "{raw}"')
                )
        for alt_index, alt in enumerate(ing.get("alternatives", [])):
            invented = _stems(alt.get("name", "")) - source
            if invented:
                issues.append(
                    _issue(
                        f"{at}.alternatives[{alt_index}].name",
                        f'"{", ".join(sorted(invented))}" does not appear in the printed line: "{raw}"',
                    )
                )
    else:
        issues.append(_issue(f"{at}.raw", "no verbatim line was captured, so this parse cannot be checked against the page"))

    # required=False is a claim about the page. Unsupported, it silently
    # overwrites a decision that belongs to the cook.
    if not ing.get("required", True) and not _OPTIONAL_SIGNALS.search(raw.lower()):
        issues.append(_issue(f"{at}.required", f'marked optional, but the line does not say so: "{raw}"'))

    return issues


def check(recipe: dict) -> list[dict]:
    """Return derived review items for an extraction payload."""
    issues: list[dict] = []

    if not recipe.get("title"):
        issues.append(_issue("title", "no title was extracted"))
    if not recipe.get("ingredients"):
        issues.append(_issue("ingredients", "no ingredients were extracted"))
    if not recipe.get("instructions"):
        issues.append(_issue("instructions", "no method steps were extracted"))
    if not recipe.get("raw_text"):
        issues.append(_issue("raw_text", "no transcription was captured — nothing survives the photograph being deleted"))

    times = recipe.get("times") or {}
    for field in ("prep", "cook", "total"):
        text = times.get(field, "")
        if text and parse_duration(text) is None:
            issues.append(_issue(f"times.{field}", f'"{text}" could not be read as a duration'))

    yield_text = recipe.get("yield_text", "")
    if yield_text and any(c.isdigit() for c in yield_text) and parse_servings(yield_text) == (None, None):
        issues.append(_issue("yield_text", f'"{yield_text}" has a number but no serving count could be derived'))

    for index, ing in enumerate(recipe.get("ingredients", [])):
        issues.extend(_check_ingredient(index, ing))

    return issues


def review(recipe: dict) -> dict:
    """Model-reported and derived issues in one queue, tagged by origin."""
    extraction = recipe.get("extraction") or {}
    model_issues = [
        {"path": i.get("path", ""), "detail": i.get("detail", ""), "origin": "model"}
        for i in extraction.get("issues", [])
    ]
    return {
        "confidence": extraction.get("confidence", ""),
        "issues": model_issues + check(recipe),
    }
