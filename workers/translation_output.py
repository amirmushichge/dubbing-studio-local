from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any


class TranslationOutputError(ValueError):
    """Raised when a model response contains no usable translation payload."""


def validate_translation_payload(payload: Any, expected_ids: Iterable[int]) -> list[dict[str, Any]]:
    expected = list(expected_ids)
    normalized = normalize_translation_items(payload)

    seen = {item["id"] for item in normalized}
    if seen != set(expected) or len(normalized) != len(expected):
        missing = sorted(set(expected) - seen)
        unexpected = sorted(seen - set(expected))
        detail = []
        if missing:
            detail.append(f"missing ids {missing}")
        if unexpected:
            detail.append(f"unexpected ids {unexpected}")
        if not detail:
            detail.append(f"expected {len(expected)} items, received {len(normalized)}")
        raise TranslationOutputError(", ".join(detail))
    return normalized


def normalize_translation_items(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        raise TranslationOutputError("the response is not a JSON array")

    normalized: list[dict[str, Any]] = []
    seen: set[int] = set()
    for position, item in enumerate(payload):
        if not isinstance(item, dict):
            raise TranslationOutputError(f"item {position + 1} is not an object")
        if "id" not in item or "translation" not in item:
            raise TranslationOutputError(f"item {position + 1} is missing id or translation")
        if isinstance(item["id"], bool):
            raise TranslationOutputError(f"item {position + 1} has an invalid id")
        try:
            item_id = int(item["id"])
        except (TypeError, ValueError) as exc:
            raise TranslationOutputError(f"item {position + 1} has an invalid id") from exc
        translation = str(item["translation"]).strip()
        if not translation:
            raise TranslationOutputError(f"item {position + 1} has an empty translation")
        if item_id in seen:
            raise TranslationOutputError(f"translation id {item_id} is duplicated")
        seen.add(item_id)
        normalized.append({"id": item_id, "translation": translation})

    return normalized


def extract_translation_payload(text: str, expected_ids: Iterable[int]) -> list[dict[str, Any]]:
    """Return the first complete, valid JSON array embedded in a model response.

    Models occasionally wrap JSON in Markdown, add an explanation, or emit a
    second array. JSONDecoder.raw_decode deliberately consumes just one value,
    so trailing material cannot corrupt an otherwise valid translation.
    """

    decoder = json.JSONDecoder()
    validation_errors: list[str] = []
    fragments: list[dict[str, Any]] = []
    for start, character in enumerate(text):
        if character != "[":
            continue
        try:
            payload, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, list):
            continue
        try:
            return validate_translation_payload(payload, expected_ids)
        except TranslationOutputError as exc:
            validation_errors.append(str(exc))

        # Hy-MT can legally emit one JSON array per segment. Collect only
        # translation-shaped fragments and validate their complete union below.
        try:
            fragments.extend(normalize_translation_items(payload))
        except TranslationOutputError:
            continue

    if fragments:
        merged: dict[int, dict[str, Any]] = {}
        for item in fragments:
            existing = merged.get(item["id"])
            if existing and existing["translation"] != item["translation"]:
                raise TranslationOutputError(f"translation id {item['id']} has conflicting values")
            merged[item["id"]] = item
        try:
            return validate_translation_payload(list(merged.values()), expected_ids)
        except TranslationOutputError as exc:
            validation_errors.append(str(exc))

    detail = validation_errors[-1] if validation_errors else "no complete JSON array was found"
    raise TranslationOutputError(detail)
