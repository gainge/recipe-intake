"""Extraction schema — the transcription layer.

This schema describes what the *model* produces: a faithful, verbatim reading of
the page. It is deliberately not the shape the display site consumes. Numeric
coercion, range splitting, and duration parsing all happen afterwards in
`normalize.py`, as pure functions over this payload.

The split exists because the page image is deleted once extraction succeeds.
Anything the model gets wrong is only fixable by re-photographing the page;
anything code gets wrong is fixable by fixing the code and re-running over the
stored payload. So every transformation that *can* be deterministic is kept out
of the model.

Field shapes that are load-bearing:

- `amount_text` is a string. "1 1/2", "2-3", and "a pinch" are all real cookbook
  inputs. The numeric pair is derived downstream; this stays as printed.
- `name` carries the ingredient alone; `processing` carries what the cook does to
  it. "2 cups chopped onion" splits into name "onion" / processing "chopped", so
  the name is usable as a key for shopping lists and pantry matching.
- `raw` preserves the printed line verbatim alongside the parsed fields, which
  makes a bad split recoverable and makes the token-containment lint in
  `validate.py` possible.
- `units_canonical` is a closed enum, so summing across recipes does not require
  reconciling "Cup" / "cup" / "Cups" after the fact.
- `extraction.issues` carry a field path, so the review UI can highlight the
  field rather than just printing a sentence.

Absent values are the empty string rather than null, so downstream consumers
never branch on None.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Closed vocabulary. Unlike ingredient names, the unit space is small and fully
# enumerable, so it goes in the schema itself and the model cannot emit anything
# outside it. `units` keeps whatever the book actually printed.
CanonicalUnit = Literal[
    "",  # no unit — "2 bay leaves", "1 onion"
    "tsp",
    "tbsp",
    "cup",
    "fl_oz",
    "ml",
    "l",
    "pt",
    "qt",
    "gal",
    "oz",
    "lb",
    "g",
    "kg",
    "count",
    "pinch",
    "dash",
    "to_taste",
    "other",
]


class Alternative(BaseModel):
    """A substitution the recipe itself offers: "butter or vegetable oil".

    Printed cookbooks offer these far more often than web recipes do, and the
    alternative frequently carries its own amount ("1 tablespoon fresh ginger or
    1 teaspoon ground ginger"), so it needs amount fields of its own rather than
    being flattened into a note.

    No `units_canonical` here: alternatives are display-only for now, and every
    extra field is another thing the model can get wrong.
    """

    name: str = Field(description="The substitute ingredient, without amount, unit, or processing.")
    amount_text: str = Field(description='Its own amount as printed, if it has one: "1", "1/2". Empty string if none.')
    units: str = Field(description="Its own unit as printed. Empty string if none.")
    processing: str = Field(description='Preparation applied to the substitute: "ground", "melted". Empty string if none.')
    notes: str = Field(description="Any qualifier attached to the substitute. Empty string if none.")


class Ingredient(BaseModel):
    raw: str = Field(description="The ingredient line exactly as printed, verbatim, including punctuation.")
    group: str = Field(
        description='Subheading this line falls under, as printed: "For the sauce". Empty string when the recipe has no subheadings.'
    )
    name: str = Field(
        description=(
            "The ingredient alone — no amount, no unit, no preparation, no qualifier. "
            'From "2 cups chopped onion" this is "onion", not "chopped onion".'
        )
    )
    key: str = Field(
        description=(
            "Lowercase, singular, canonical form of the name for cross-recipe matching: "
            '"onion", "green onion", "vegetable stock". No brand, no processing, no adjectives '
            "that do not change what the ingredient is."
        )
    )
    amount_text: str = Field(
        description='Amount as written: "1 1/2", "2-3", "a few". Never converted to a decimal. Empty string if none.'
    )
    units: str = Field(description='Unit as written: "cups", "g", "tbsp", "clove". Never converted. Empty string if none.')
    units_canonical: CanonicalUnit = Field(
        description=(
            "The printed unit mapped to the closed vocabulary. `oz` is weight and `fl_oz` is volume — "
            'choose by what the book means. `count` for countable units ("clove", "sprig", "can", "medium"). '
            "Empty string when the line has no unit at all."
        )
    )
    processing: str = Field(
        description=(
            'A physical transformation the cook performs on this ingredient: "chopped", "minced", '
            '"softened", "drained", "peeled and chopped". Empty string if none.'
        )
    )
    notes: str = Field(
        description=(
            "Anything qualifying the line that is not a transformation: substitution advice, "
            'brand or provenance, packaging, timing, commentary. "preferably basmati", '
            '"two 4oz cans", "more is better". Empty string if none.'
        )
    )
    required: bool = Field(
        description=(
            "False ONLY when the page explicitly marks this ingredient as dispensable — "
            '"(optional)", "if desired", "for garnish". True in every other case. '
            "Never infer this from the ingredient itself."
        )
    )
    alternatives: list[Alternative] = Field(
        description='Substitutions the recipe offers for this line ("or good-quality vegetable oil"). Empty list if none.'
    )


class Times(BaseModel):
    prep: str = Field(description='As printed, e.g. "20 minutes". Empty string if not stated.')
    cook: str = Field(description='As printed. Empty string if not stated.')
    total: str = Field(description='As printed, e.g. "About 30 minutes". Empty string if not stated.')


class Issue(BaseModel):
    path: str = Field(
        description=(
            "Dotted/indexed path to the field in question, e.g. `ingredients[3].amount_text` or "
            "`title`. Empty string for an issue about the page as a whole."
        )
    )
    detail: str = Field(description="Specific, checkable statement of what a reviewer should verify and why.")


class ExtractionMeta(BaseModel):
    confidence: Literal["high", "medium", "low"] = Field(
        description="Overall confidence that this extraction is accurate and complete."
    )
    issues: list[Issue] = Field(
        description=(
            "Uncertainties a human reviewer should check, each anchored to the field it concerns. "
            "Empty list if the page was fully legible and unambiguous."
        )
    )


class Recipe(BaseModel):
    raw_text: str = Field(
        description=(
            "Faithful transcription of ALL recipe text visible on the page(s), including any "
            "neighbouring recipe, preserving line breaks, headings, and reading order. "
            "The source of truth — the photograph is discarded after extraction."
        )
    )
    title: str = Field(description="Title of the extracted recipe, as printed.")
    yield_text: str = Field(description='Yield or servings as printed: "MAKES: 4 to 6 servings", "Makes 12 muffins". Empty string if absent.')
    times: Times
    ingredients: list[Ingredient] = Field(
        description="Flat, in printed order. Subheadings are carried on each line's `group` field, not by nesting."
    )
    instructions: list[str] = Field(description="Method steps in order, in the book's original wording. One entry per step.")
    notes: list[str] = Field(
        description="Headnotes, variations, storage or serving advice, and any handwritten margin annotations."
    )
    extraction: ExtractionMeta
