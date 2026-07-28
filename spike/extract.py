#!/usr/bin/env python3
"""Phase 1 spike — validate the extraction prompt and schema against real pages.

Deliberately has no server, no database, and no UI. The only question it exists
to answer is whether image -> structured recipe is accurate enough to build on.

    python spike/extract.py samples/page.jpg
    python spike/extract.py samples/left.jpg samples/right.jpg   # one spread, one recipe
    python spike/extract.py --hint "Biryani" samples/page.jpg    # disambiguate a busy page
    python spike/extract.py --model claude-sonnet-5 samples/page.jpg
    python spike/extract.py --verbose samples/page.jpg           # include the transcription
    python spike/extract.py --replay out/*.json                  # re-derive, no API call

The pipeline is three stages, and only the first one costs money or is
irreversible:

    1. model      images -> verbatim structured payload   (`recipe`)
    2. normalize  payload -> numbers the display site uses (`normalized`)
    3. validate   payload -> derived review items          (`review`)

Stages 2 and 3 are pure functions over stage 1's output, which is what `--replay`
exploits: iterating on the parser costs milliseconds instead of a minute, and an
improved parser can be re-run across every page already extracted.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# Run as a script, so sys.path[0] is spike/ rather than the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.extraction import (  # noqa: E402
    MAX_EDGE,
    QUALITY,
    SYSTEM_PROMPT,
    Recipe,
    estimate_cost,
    merge_curated,
    normalize,
    prepare_image,
    user_instruction,
    validate,
)
from app.extraction.client import PRICING, build_content  # noqa: E402

load_dotenv()

DEFAULT_MODEL = "claude-sonnet-5"
MAX_TOKENS = 16_000

# Image preparation, pricing, and the request shape all live in app/extraction/,
# shared with the server. If the CLI had its own copies, the quality and cost
# measured here would stop describing what production actually sends.


def derive(recipe: dict, previous: dict | None = None) -> tuple[dict, dict]:
    """Stages 2 and 3. Pure — this is everything `--replay` re-runs."""
    return merge_curated(previous, normalize(recipe)), validate.review(recipe)


def _format_amount(ing: dict) -> str:
    low, high = ing.get("amount-low"), ing.get("amount-high")
    if low is None:
        amount = ing.get("amount-text", "")
    elif high is not None and high != low:
        amount = f"{low:g}-{high:g}"
    else:
        amount = f"{low:g}"
    return " ".join(p for p in (amount, ing.get("units", "")) if p)


def report(recipe: dict, normalized: dict, review: dict, verbose: bool) -> None:
    marker = {"high": "✓", "medium": "~", "low": "!"}.get(review["confidence"], "?")
    print(f"\n{'=' * 68}")
    print(f"  {normalized['title'] or '(no title extracted)'}")
    print(f"{'=' * 68}")
    print(f"  confidence   {marker} {review['confidence']}")

    if review["issues"]:
        model_count = sum(1 for i in review["issues"] if i["origin"] == "model")
        derived_count = len(review["issues"]) - model_count
        print(f"  issues       {model_count} reported, {derived_count} derived -- review these --")
        for issue in review["issues"]:
            tag = "model" if issue["origin"] == "model" else "check"
            location = issue["path"] or "(page)"
            print(f"               [{tag}] {location}")
            print(f"                      {issue['detail']}")
    else:
        print("  issues       none")

    if normalized["yield-text"]:
        low, high = normalized["servings-low"], normalized["servings-high"]
        derived = f"  -> serves {low}-{high}" if low is not None and high != low else (f"  -> serves {low}" if low is not None else "")
        print(f"  yield        {normalized['yield-text']}{derived}")
    printed = [f"{k} {v}" for k, v in normalized["time-text"].items() if v]
    if printed:
        derived = f"  -> {normalized['time']} min" if normalized["time"] is not None else ""
        print(f"  times        {', '.join(printed)}{derived}")

    groups = {ing["group"] for ing in normalized["ingredients"]}
    unkeyed = sum(1 for ing in normalized["ingredients"] if not ing["key"])
    print(f"  ingredients  {len(normalized['ingredients'])} in {len(groups)} group{'s' if len(groups) != 1 else ''}"
          + (f", {unkeyed} without a key" if unkeyed else ""))
    print(f"  steps        {len(normalized['instructions'])}")
    if normalized["notes"]:
        print(f"  notes        {len(normalized['notes'])}")

    if verbose:
        print(f"\n{'-' * 68}\n  INGREDIENTS\n{'-' * 68}")
        current = object()
        for ing in normalized["ingredients"]:
            if ing["group"] != current:
                current = ing["group"]
                if current:
                    print(f"\n  [{current}]")
            trailing = [ing["processing"], ing["notes"]] + [
                f"or {' '.join(p for p in (a['amount-text'], a['units'], a['processing'], a['name']) if p)}"
                for a in ing["alternatives"]
            ]
            suffix = "  (" + "; ".join(t for t in trailing if t) + ")" if any(trailing) else ""
            optional = " [optional]" if not ing["required"] else ""
            headline = f"{_format_amount(ing)} {ing['name']}".strip()
            print(f"    {headline}{suffix}{optional}")
            print(f"      key: {ing['key'] or '(none)'}   raw: {ing['raw']}")

        print(f"\n{'-' * 68}\n  STEPS\n{'-' * 68}")
        for number, text in enumerate(normalized["instructions"], 1):
            print(f"    {number}. {text}")

        if normalized["notes"]:
            print(f"\n{'-' * 68}\n  NOTES\n{'-' * 68}")
            for note in normalized["notes"]:
                print(f"    • {note}")

        print(f"\n{'-' * 68}\n  RAW TRANSCRIPTION\n{'-' * 68}")
        print(recipe["raw_text"])


def replay(paths: list[Path], verbose: bool) -> int:
    """Re-derive stages 2 and 3 from stored payloads. No API call, no cost.

    Curated fields already present in the stored `normalized` block survive, so
    replaying does not discard hand review.
    """
    failures = 0
    for path in paths:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"error: {path}: {exc}", file=sys.stderr)
            failures += 1
            continue

        if "recipe" not in payload:
            print(f"error: {path}: no `recipe` block — not an extraction payload", file=sys.stderr)
            failures += 1
            continue

        normalized, review = derive(payload["recipe"], payload.get("normalized"))
        payload["normalized"] = normalized
        payload["review"] = review
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

        print(f"\n{path}")
        report(payload["recipe"], normalized, review, verbose)

    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract a structured recipe from cookbook page photographs.",
        epilog="Multiple images are treated as consecutive pages of ONE recipe.",
    )
    parser.add_argument("paths", nargs="+", type=Path, help="Page image(s) in reading order, or saved JSON with --replay")
    parser.add_argument("--hint", default="", help="Title of the recipe to extract, when a page holds more than one")
    parser.add_argument("--replay", action="store_true", help="re-run normalization and validation over saved JSON; no API call")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"default: {DEFAULT_MODEL}")
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="disable extended thinking; results save under a .nothink suffix so they can be diffed",
    )
    parser.add_argument("--max-edge", type=int, default=MAX_EDGE, help=f"long-edge px before upload (default: {MAX_EDGE})")
    parser.add_argument("--quality", type=int, default=QUALITY, help=f"JPEG quality (default: {QUALITY})")
    parser.add_argument("--out", type=Path, default=Path("out"), help="directory for JSON results (default: out/)")
    parser.add_argument("--verbose", action="store_true", help="print the full parse and transcription")
    args = parser.parse_args()

    missing = [p for p in args.paths if not p.is_file()]
    if missing:
        print(f"error: not found: {', '.join(str(p) for p in missing)}", file=sys.stderr)
        return 1

    if args.replay:
        return replay(args.paths, args.verbose)

    print(f"model: {args.model}{' (thinking disabled)' if args.no_thinking else ''}")
    images: list[bytes] = []
    for path in args.paths:
        data, original, sent = prepare_image(path, args.max_edge, args.quality)
        images.append(data)
        print(f"  {path.name}: {original[0]}x{original[1]} -> {sent[0]}x{sent[1]}, {len(data) / 1024:.0f} KB")

    try:
        client = anthropic.Anthropic()
    except Exception as exc:
        print(f"\nerror: could not construct client: {exc}", file=sys.stderr)
        print("Set ANTHROPIC_API_KEY, or run `ant auth login`.", file=sys.stderr)
        return 1

    print("\nextracting...", flush=True)
    started = time.monotonic()
    try:
        response = client.messages.parse(
            model=args.model,
            max_tokens=MAX_TOKENS,  # caps thinking + output together, so leave headroom
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    # Stable across every request; caches if it clears the model's minimum.
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": build_content(images, args.hint)}],
            output_format=Recipe,
            **({"thinking": {"type": "disabled"}} if args.no_thinking else {}),
        )
    except anthropic.APIStatusError as exc:
        print(f"\nAPI error {exc.status_code}: {exc.message}", file=sys.stderr)
        return 1
    except anthropic.APIConnectionError as exc:
        print(f"\nconnection error: {exc}", file=sys.stderr)
        return 1
    except TypeError as exc:
        # The SDK resolves credentials lazily, at request time rather than at
        # construction, so a missing key surfaces here as a bare TypeError.
        if "authentication" not in str(exc).lower():
            raise
        print("\nerror: no credentials found.", file=sys.stderr)
        print("Set one of:", file=sys.stderr)
        print("  export ANTHROPIC_API_KEY=sk-ant-...", file=sys.stderr)
        print("  ant auth login", file=sys.stderr)
        return 1
    elapsed = time.monotonic() - started

    stop_reason = getattr(response, "stop_reason", None)
    if stop_reason == "refusal":
        print("\nThe request was declined by safety classifiers.", file=sys.stderr)
        return 1
    if stop_reason == "max_tokens":
        print(f"\nwarning: hit max_tokens ({MAX_TOKENS}); output may be truncated.", file=sys.stderr)

    parsed = response.parsed_output
    if parsed is None:
        print("\nerror: response did not parse against the schema.", file=sys.stderr)
        return 1
    recipe = parsed.model_dump()

    usage = response.usage
    cost = estimate_cost(args.model, usage)
    print(f"done in {elapsed:.1f}s")
    print(
        f"  tokens: {getattr(usage, 'input_tokens', 0)} in"
        f" / {getattr(usage, 'output_tokens', 0)} out"
        f"  (cache read {getattr(usage, 'cache_read_input_tokens', 0)},"
        f" write {getattr(usage, 'cache_creation_input_tokens', 0)})"
    )
    if cost is not None:
        print(f"  cost:   ${cost:.4f} this page  ->  ~${cost * 500:.2f} per 500 pages")

    normalized, review = derive(recipe)
    report(recipe, normalized, review, args.verbose)

    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.paths[0].stem
    variant = ".nothink" if args.no_thinking else ""
    dest = args.out / f"{stem}.{args.model}{variant}.json"
    payload = {
        "source_images": [str(p) for p in args.paths],
        "model": args.model,
        "thinking": "disabled" if args.no_thinking else "adaptive",
        "max_edge": args.max_edge,
        "elapsed_seconds": round(elapsed, 2),
        "estimated_cost_usd": round(cost, 6) if cost is not None else None,
        "usage": usage.model_dump() if hasattr(usage, "model_dump") else None,
        # Verbatim model output. Never rewritten — stages 2 and 3 derive from it,
        # and the photograph it came from no longer exists.
        "recipe": recipe,
        "review": review,
        "normalized": normalized,
    }
    dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nsaved: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
