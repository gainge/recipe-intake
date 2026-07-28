"""Normalization — the derived layer.

Pure functions over an extraction payload. No API calls, no I/O, no model.

This is where the verbatim strings the model produced become the numbers the
display site consumes: "1 1/2" -> 1.5, "2-3" -> (2, 3), "About 30 minutes" -> 30.

Keeping it here rather than in the prompt is the central structural decision of
the pipeline. The page photograph is deleted once extraction succeeds, so a
model mistake is only recoverable by re-shooting the page — but anything in this
file can be fixed and re-run across the whole stored corpus for free. Every
transformation that can be deterministic therefore lives here.

Nothing in this module ever fails. Unparseable input yields None and a flag;
`validate.py` turns that flag into a review item. A dropped number must surface
to a human, never to an exception or a silent zero.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from fractions import Fraction

# Wire keys are kebab-case to match the existing recipes.json corpus. Python-side
# field names stay snake_case; the translation happens only at this boundary.

_VULGAR_FRACTIONS = {
    "¼": " 1/4", "½": " 1/2", "¾": " 3/4",
    "⅓": " 1/3", "⅔": " 2/3",
    "⅕": " 1/5", "⅖": " 2/5", "⅗": " 3/5", "⅘": " 4/5",
    "⅙": " 1/6", "⅚": " 5/6",
    "⅐": " 1/7", "⅛": " 1/8", "⅜": " 3/8", "⅝": " 5/8", "⅞": " 7/8",
    "⅑": " 1/9", "⅒": " 1/10",
}

# Cookbooks and OCR both produce a zoo of dashes. Fold them all to ASCII before
# any range matching, or "2–3" silently fails to parse while "2-3" succeeds.
_DASHES = dict.fromkeys("‐‑‒–—―−⁃", "-")

# Hedges that carry no numeric information. Stripped before parsing so
# "about 2" and "2" reach the same code path.
_APPROXIMATORS = r"(?:about|approx\.?|approximately|around|roughly|scant|generous|heaping|heaped|rounded|big|large|small)"

# Amounts that are real instructions but have no number. These are not parse
# failures — flagging them would train the reviewer to ignore the queue.
_IMPRECISE = re.compile(
    r"^(?:a|an|some|several|few|a\s+few|a\s+couple(?:\s+of)?|couple(?:\s+of)?)?\s*"
    r"(?:pinch|dash|splash|handful|drizzle|sprinkle|knob|squeeze|taste|few|several|some)"
    r"(?:es|s)?(?:\s+of)?$"
)
_TO_TASTE = re.compile(r"^(?:to\s+taste|as\s+needed|as\s+required|to\s+serve|for\s+serving|for\s+garnish)$")

_RANGE = re.compile(r"^(.+?)\s*(?:-|\bto\b|\bor\b)\s*(.+)$")
_MIXED = re.compile(r"^(\d+)\s+(\d+)\s*/\s*(\d+)$")
_FRACTION = re.compile(r"^(\d+)\s*/\s*(\d+)$")
_DECIMAL = re.compile(r"^\d+(?:\.\d+)?$")

_HOURS = re.compile(r"(\d+(?:\s+\d+/\d+)?|\d+/\d+|\d+(?:\.\d+)?)\s*(?:hours?|hrs?)\b")
_MINUTES = re.compile(r"(\d+(?:\s+\d+/\d+)?|\d+/\d+|\d+(?:\.\d+)?)\s*(?:minutes?|mins?)\b")
_BARE_NUMBER = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*$")

_SERVING_WORDS = re.compile(r"\b(?:serv(?:es|ing|ings)|portions?|people|persons?)\b")
_SERVING_RANGE = re.compile(r"(\d+)\s*(?:-|\bto\b|\bor\b)\s*(\d+)")
_SERVING_SINGLE = re.compile(r"(\d+)")

DIMENSIONS: dict[str, str] = {
    "tsp": "volume", "tbsp": "volume", "cup": "volume", "fl_oz": "volume",
    "ml": "volume", "l": "volume", "pt": "volume", "qt": "volume", "gal": "volume",
    "oz": "mass", "lb": "mass", "g": "mass", "kg": "mass",
    "count": "count",
    "pinch": "imprecise", "dash": "imprecise", "to_taste": "imprecise",
    "": "none", "other": "other",
}

# Fields a human owns, not the extractor. Re-extraction must not overwrite them.
CURATED_RECIPE_FIELDS = ("id", "source", "image")
CURATED_INGREDIENT_FIELDS = ("reference", "required")


@dataclass(frozen=True)
class AmountParse:
    low: float | None
    high: float | None
    ok: bool
    """False only when text was present and could not be understood.

    An empty amount and "a pinch" both parse to (None, None) with ok=True —
    they are legitimately numberless, not failures.
    """


def _fold(text: str) -> str:
    """Normalize unicode, dashes, and vulgar fractions down to ASCII arithmetic."""
    text = unicodedata.normalize("NFC", text or "")
    for glyph, ascii_form in _VULGAR_FRACTIONS.items():
        text = text.replace(glyph, ascii_form)
    text = text.translate(str.maketrans(_DASHES))
    return re.sub(r"\s+", " ", text).strip()


def _to_float(frac: Fraction) -> float:
    # Parse through exact rationals so 1/3 does not become 0.33 and then 0.99
    # when a reviewer scales the recipe by three.
    return round(float(frac), 4)


def _scalar(text: str) -> float | None:
    s = re.sub(rf"^{_APPROXIMATORS}\s+", "", text.strip().lower()).strip()
    if not s:
        return None
    if m := _MIXED.match(s):
        whole, num, den = (int(g) for g in m.groups())
        return None if den == 0 else _to_float(whole + Fraction(num, den))
    if m := _FRACTION.match(s):
        num, den = (int(g) for g in m.groups())
        return None if den == 0 else _to_float(Fraction(num, den))
    if _DECIMAL.match(s):
        return _to_float(Fraction(s))
    return None


def parse_amount(text: str) -> AmountParse:
    """"1 1/2" -> (1.5, 1.5); "2-3" -> (2.0, 3.0); "a pinch" -> (None, None)."""
    s = _fold(text).lower()
    if not s:
        return AmountParse(None, None, True)
    if _IMPRECISE.match(s) or _TO_TASTE.match(s):
        return AmountParse(None, None, True)

    if m := _RANGE.match(s):
        low, high = _scalar(m.group(1)), _scalar(m.group(2))
        if low is not None and high is not None:
            # "1-1/2 cups" is one and a half, not a range from 1 to 0.5. A
            # descending "range" is always this typographic convention.
            if high < low:
                combined = _to_float(Fraction(str(low)) + Fraction(str(high)))
                return AmountParse(combined, combined, True)
            return AmountParse(low, high, True)

    if (value := _scalar(s)) is not None:
        return AmountParse(value, value, True)
    return AmountParse(None, None, False)


def parse_duration(text: str) -> int | None:
    """"About 1 hour, mostly unattended" -> 60. None when there is no duration."""
    s = _fold(text).lower()
    if not s:
        return None

    total = 0.0
    found = False
    if m := _HOURS.search(s):
        if (hours := _scalar(m.group(1))) is not None:
            total += hours * 60
            found = True
    if m := _MINUTES.search(s):
        if (minutes := _scalar(m.group(1))) is not None:
            total += minutes
            found = True
    if found:
        return round(total)

    # A bare number in a time field means minutes — that is the only reading.
    if m := _BARE_NUMBER.match(s):
        return round(float(m.group(1)))
    return None


def parse_servings(text: str) -> tuple[int | None, int | None]:
    """"MAKES: 4 to 6 servings" -> (4, 6).

    Requires an explicit serving word. "Makes 12 muffins" yields nothing rather
    than claiming twelve servings — a yield is not a serving count, and the
    verbatim string is preserved either way.
    """
    s = _fold(text).lower()
    if not s or not _SERVING_WORDS.search(s):
        return None, None
    if m := _SERVING_RANGE.search(s):
        return int(m.group(1)), int(m.group(2))
    if m := _SERVING_SINGLE.search(s):
        value = int(m.group(1))
        return value, value
    return None, None


def dimension_of(units_canonical: str) -> str:
    return DIMENSIONS.get(units_canonical, "other")


def _normalize_alternative(alt: dict) -> dict:
    amount = parse_amount(alt.get("amount_text", ""))
    return {
        "name": alt.get("name", ""),
        "amount-low": amount.low,
        "amount-high": amount.high,
        "amount-text": alt.get("amount_text", ""),
        "units": alt.get("units", ""),
        "processing": alt.get("processing", ""),
        "notes": alt.get("notes", ""),
    }


def _normalize_ingredient(ing: dict) -> dict:
    amount = parse_amount(ing.get("amount_text", ""))
    canonical = ing.get("units_canonical", "")
    return {
        "name": ing.get("name", ""),
        "key": ing.get("key", ""),
        "amount-low": amount.low,
        "amount-high": amount.high,
        "amount-text": ing.get("amount_text", ""),
        "units": ing.get("units", ""),
        "units-canonical": canonical,
        "dimension": dimension_of(canonical),
        "processing": ing.get("processing", ""),
        "notes": ing.get("notes", ""),
        "group": ing.get("group", ""),
        "required": ing.get("required", True),
        "reference": None,  # curation-owned; see merge_curated
        "alternatives": [_normalize_alternative(a) for a in ing.get("alternatives", [])],
        "raw": ing.get("raw", ""),
    }


def normalize(recipe: dict) -> dict:
    """Extraction payload -> the shape the display site consumes.

    `id` is deliberately absent: it belongs to whatever owns the collection, not
    to the extractor.
    """
    times = recipe.get("times") or {}
    total = parse_duration(times.get("total", ""))
    if total is None:
        prep = parse_duration(times.get("prep", "")) or 0
        cook = parse_duration(times.get("cook", "")) or 0
        total = (prep + cook) or None

    servings_low, servings_high = parse_servings(recipe.get("yield_text", ""))

    return {
        "title": recipe.get("title", ""),
        "source": None,  # curation-owned
        "image": None,  # curation-owned; this pipeline never retains a photograph
        "time": total,
        "time-text": {
            "prep": times.get("prep", ""),
            "cook": times.get("cook", ""),
            "total": times.get("total", ""),
        },
        "servings-low": servings_low,
        "servings-high": servings_high,
        "yield-text": recipe.get("yield_text", ""),
        "ingredients": [_normalize_ingredient(i) for i in recipe.get("ingredients", [])],
        "instructions": list(recipe.get("instructions", [])),
        "notes": list(recipe.get("notes", [])),
    }


def merge_curated(previous: dict | None, fresh: dict) -> dict:
    """Carry human-owned fields across a re-extraction.

    Without this, retrying a job or re-running an improved prompt silently
    discards every `required` flag and product link the user set by hand, which
    would make re-extraction unsafe to offer as a button.

    Ingredients are matched on `key`, falling back to `name`. A key appearing
    more than once on either side is skipped rather than guessed at — losing a
    curated flag is recoverable, attaching it to the wrong ingredient is not.
    """
    if not previous:
        return fresh

    merged = dict(fresh)
    for field in CURATED_RECIPE_FIELDS:
        if previous.get(field) is not None:
            merged[field] = previous[field]

    def index(ingredients: list[dict]) -> dict[str, dict]:
        counts: dict[str, int] = {}
        for ing in ingredients:
            identity = (ing.get("key") or ing.get("name", "")).strip().lower()
            counts[identity] = counts.get(identity, 0) + 1
        return {
            (ing.get("key") or ing.get("name", "")).strip().lower(): ing
            for ing in ingredients
            if counts[(ing.get("key") or ing.get("name", "")).strip().lower()] == 1
        }

    old = index(previous.get("ingredients", []))
    new = index(merged.get("ingredients", []))
    for identity, new_ing in new.items():
        old_ing = old.get(identity)
        if old_ing is None:
            continue
        for field in CURATED_INGREDIENT_FIELDS:
            if field in old_ing and old_ing[field] is not None:
                new_ing[field] = old_ing[field]
    return merged
