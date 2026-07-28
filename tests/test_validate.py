"""The validator has to stay quiet on legitimate output.

A queue full of false positives is worse than no queue, because the reviewer
learns to skim it — and the review step is the only thing standing between a
misread fraction and a wrong dish. Half of these tests assert silence.
"""

from app.extraction import validate
from tests.test_normalize import _ingredient, _recipe


def paths(issues):
    return [i["path"] for i in issues]


def test_clean_extraction_produces_no_issues():
    assert validate.check(_recipe()) == []


def test_numberless_amounts_are_not_flagged():
    ing = _ingredient(
        raw="Salt and pepper", name="salt and pepper", key="salt",
        amount_text="", units="", units_canonical="", processing="",
    )
    assert validate.check(_recipe(ingredients=[ing])) == []


def test_unparseable_amount_is_flagged():
    issues = validate.check(_recipe(ingredients=[_ingredient(amount_text="two-ish")]))
    assert "ingredients[0].amount_text" in paths(issues)


def test_invented_name_is_caught_against_the_raw_line():
    # The failure mode that matters most: the model canonicalizing a name into
    # something that was never printed, where nothing downstream would notice.
    ing = _ingredient(raw="2 cups chopped onion", name="shallot")
    issues = validate.check(_recipe(ingredients=[ing]))
    assert "ingredients[0].name" in paths(issues)


def test_plural_and_case_differences_are_not_invention():
    ing = _ingredient(raw="4-6 Green Onions", name="Green Onion", key="green onion",
                      amount_text="4-6", units="", units_canonical="", processing="")
    assert validate.check(_recipe(ingredients=[ing])) == []


def test_unmapped_unit_is_flagged():
    ing = _ingredient(units="cups", units_canonical="")
    assert "ingredients[0].units_canonical" in paths(validate.check(_recipe(ingredients=[ing])))


def test_pinch_carrying_an_amount_is_contradictory():
    ing = _ingredient(raw="Pinch ground cloves", name="ground cloves", key="cloves",
                      amount_text="1", units="Pinch", units_canonical="pinch", processing="")
    assert "ingredients[0].amount_text" in paths(validate.check(_recipe(ingredients=[ing])))


def test_unsupported_optional_flag_is_flagged():
    ing = _ingredient(required=False)
    assert "ingredients[0].required" in paths(validate.check(_recipe(ingredients=[ing])))


def test_supported_optional_flag_is_not_flagged():
    ing = _ingredient(raw="1 pound firm tofu, cubed (optional)", name="firm tofu", key="tofu",
                      amount_text="1", units="pound", units_canonical="lb",
                      processing="cubed", required=False)
    assert validate.check(_recipe(ingredients=[ing])) == []


def test_missing_transcription_is_flagged():
    assert "raw_text" in paths(validate.check(_recipe(raw_text="")))


def test_review_merges_model_and_derived_issues_with_origins():
    recipe = _recipe(
        ingredients=[_ingredient(amount_text="two-ish")],
        extraction={
            "confidence": "medium",
            "issues": [{"path": "title", "detail": "display face is ambiguous"}],
        },
    )
    result = validate.review(recipe)
    assert result["confidence"] == "medium"
    assert {i["origin"] for i in result["issues"]} == {"model", "validator"}
