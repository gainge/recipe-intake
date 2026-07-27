"""Cases drawn from the reference corpus and from real cookbook pages.

The point of these is that they run without an API key and without an image, so
the parser can be tightened in a tight loop and then replayed across everything
already extracted.
"""

import pytest

from normalize import merge_curated, normalize, parse_amount, parse_duration, parse_servings


@pytest.mark.parametrize(
    "text,expected",
    [
        ("2", (2.0, 2.0)),
        ("1 1/2", (1.5, 1.5)),
        ("1/2", (0.5, 0.5)),
        ("0.25", (0.25, 0.25)),
        ("1½", (1.5, 1.5)),
        ("¾", (0.75, 0.75)),
        ("2-3", (2.0, 3.0)),
        ("2–3", (2.0, 3.0)),  # en dash, as it appears in real transcriptions
        ("2 to 3", (2.0, 3.0)),
        ("1 or 2", (1.0, 2.0)),
        ("1/2 to 1", (0.5, 1.0)),
        ("about 2", (2.0, 2.0)),
        ("1-1/2", (1.5, 1.5)),  # typographic mixed number, not a descending range
        ("", (None, None)),
        ("Pinch", (None, None)),
        ("a few", (None, None)),
        ("to taste", (None, None)),
    ],
)
def test_parse_amount(text, expected):
    result = parse_amount(text)
    assert (result.low, result.high) == expected
    assert result.ok


def test_thirds_survive_scaling():
    # A float rounded to 0.33 becomes 0.99 cups when a reviewer triples the
    # recipe. Parsing through exact rationals is the whole reason this is here.
    assert parse_amount("1/3").low == pytest.approx(1 / 3, abs=1e-4)


def test_unparseable_amount_is_flagged_not_swallowed():
    result = parse_amount("a bunch-ish")
    assert (result.low, result.high) == (None, None)
    assert not result.ok


@pytest.mark.parametrize(
    "text,expected",
    [
        ("About 30 minutes", 30),
        ("30 mins", 30),
        ("1 hour", 60),
        ("About 1 hour, mostly unattended", 60),
        ("1 hour 15 minutes", 75),
        ("1 1/2 hours", 90),
        ("60", 60),
        ("", None),
        ("overnight", None),
    ],
)
def test_parse_duration(text, expected):
    assert parse_duration(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("MAKES: 4 to 6 servings", (4, 6)),
        ("Serves 4", (4, 4)),
        ("4-5 servings", (4, 5)),
        ("Makes 12 muffins", (None, None)),  # a yield is not a serving count
        ("", (None, None)),
    ],
)
def test_parse_servings(text, expected):
    assert parse_servings(text) == expected


def _ingredient(**overrides):
    base = {
        "raw": "2 cups chopped onion",
        "group": "",
        "name": "onion",
        "key": "onion",
        "amount_text": "2",
        "units": "cups",
        "units_canonical": "cup",
        "processing": "chopped",
        "notes": "",
        "required": True,
        "alternatives": [],
    }
    return base | overrides


def _recipe(**overrides):
    base = {
        "raw_text": "Biryani...",
        "title": "Biryani",
        "yield_text": "MAKES: 4 to 6 servings",
        "times": {"prep": "", "cook": "", "total": "About 30 minutes"},
        "ingredients": [_ingredient()],
        "instructions": ["Cook it."],
        "notes": [],
        "extraction": {"confidence": "high", "issues": []},
    }
    return base | overrides


def test_normalize_derives_the_display_shape():
    result = normalize(_recipe())
    assert result["time"] == 30
    assert (result["servings-low"], result["servings-high"]) == (4, 6)
    assert result["yield-text"] == "MAKES: 4 to 6 servings"

    ing = result["ingredients"][0]
    assert (ing["amount-low"], ing["amount-high"]) == (2.0, 2.0)
    assert ing["name"] == "onion"
    assert ing["processing"] == "chopped"
    assert ing["dimension"] == "volume"
    assert ing["raw"] == "2 cups chopped onion"


def test_total_time_falls_back_to_prep_plus_cook():
    result = normalize(_recipe(times={"prep": "20 minutes", "cook": "1 hour", "total": ""}))
    assert result["time"] == 80


def test_curated_fields_start_empty():
    result = normalize(_recipe())
    assert result["source"] is None
    assert result["image"] is None
    assert result["ingredients"][0]["reference"] is None


def test_alternatives_keep_their_own_amount():
    ing = _ingredient(
        raw="1 tablespoon minced fresh ginger or 1 teaspoon ground ginger",
        name="fresh ginger",
        alternatives=[
            {"name": "ground ginger", "amount_text": "1", "units": "teaspoon", "processing": "", "notes": ""}
        ],
    )
    alt = normalize(_recipe(ingredients=[ing]))["ingredients"][0]["alternatives"][0]
    assert (alt["amount-low"], alt["units"]) == (1.0, "teaspoon")


def test_merge_preserves_curation_across_re_extraction():
    previous = normalize(_recipe())
    previous["source"] = "How to Cook Everything Vegetarian, p. 375"
    previous["ingredients"][0]["required"] = False
    previous["ingredients"][0]["reference"] = "https://example.com/onion"

    fresh = normalize(_recipe(title="Biryani (corrected)"))
    merged = merge_curated(previous, fresh)

    assert merged["title"] == "Biryani (corrected)"  # extraction wins on extracted fields
    assert merged["source"] == "How to Cook Everything Vegetarian, p. 375"
    assert merged["ingredients"][0]["required"] is False
    assert merged["ingredients"][0]["reference"] == "https://example.com/onion"


def test_merge_skips_ambiguous_matches():
    # Two lines sharing a key cannot be matched safely. Losing a curated flag is
    # recoverable; attaching it to the wrong ingredient is not.
    duplicated = [_ingredient(), _ingredient(raw="1 cup sliced onion", processing="sliced")]
    previous = normalize(_recipe(ingredients=duplicated))
    previous["ingredients"][0]["reference"] = "https://example.com/onion"

    merged = merge_curated(previous, normalize(_recipe(ingredients=duplicated)))
    assert merged["ingredients"][0]["reference"] is None
