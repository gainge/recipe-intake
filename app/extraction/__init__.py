"""Extraction library — everything that turns a page image into a recipe.

Shared by the server worker and by `spike/extract.py`. The spike CLI is not a
throwaway: it remains the fastest way to iterate on the prompt against real
pages, and its `--replay` mode re-derives the pure stages across the whole
stored corpus without an API call.

Three stages, and only the first costs money or is irreversible:

    1. `client.extract`  images -> verbatim structured payload
    2. `normalize`       payload -> the numbers the display site consumes
    3. `validate.review` payload -> derived review items

Stages 2 and 3 are pure functions. That split is the point: the photograph is
deleted once extraction succeeds, so anything the model does is permanent while
anything code does can be fixed and re-run over every page already extracted.
"""

from .client import ExtractionError, ExtractionResult, estimate_cost, extract
from .images import MAX_EDGE, QUALITY, prepare_image
from .normalize import merge_curated, normalize
from .prompt import SYSTEM_PROMPT, user_instruction
from .schema import Recipe
from .validate import review

__all__ = [
    "MAX_EDGE",
    "QUALITY",
    "SYSTEM_PROMPT",
    "ExtractionError",
    "ExtractionResult",
    "Recipe",
    "estimate_cost",
    "extract",
    "merge_curated",
    "normalize",
    "prepare_image",
    "review",
    "user_instruction",
]
