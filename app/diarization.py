from __future__ import annotations

import math
from collections.abc import Sequence


def split_segments_on_word_gaps(segments: list[dict], gap_threshold: float = 1.2) -> list[dict]:
    """Split ASR segments that hide a long silent interval between words."""
    result: list[dict] = []
    for segment in segments:
        words = [word for word in segment.get("words", []) if "start" in word and "end" in word]
        groups: list[list[dict]] = []
        for word in words:
            if groups and float(word["start"]) - float(groups[-1][-1]["end"]) >= gap_threshold:
                groups.append([])
            if not groups:
                groups.append([])
            groups[-1].append(word)
        if len(groups) <= 1:
            item = dict(segment)
            item["id"] = len(result)
            result.append(item)
            continue
        for group_index, group in enumerate(groups):
            text = " ".join(str(word.get("word", "")).strip() for word in group).strip()
            if group_index < len(groups) - 1 and text and text[-1] not in ".,!?;:…":
                text += "…"
            item = dict(segment)
            item.update({
                "id": len(result), "start": float(group[0]["start"]), "end": float(group[-1]["end"]),
                "text": text or segment["text"], "words": group,
            })
            result.append(item)
    return result


def _centroid(rows: Sequence[Sequence[float]]) -> list[float]:
    values = [sum(column) / len(rows) for column in zip(*rows)]
    norm = max(math.sqrt(sum(value * value for value in values)), 1e-8)
    return [value / norm for value in values]


def _normalized_labels(labels: Sequence[int]) -> list[int]:
    mapping: dict[int, int] = {}
    output = []
    for value in labels:
        mapping.setdefault(value, len(mapping))
        output.append(mapping[value])
    return output


def merge_tiny_clusters(
    labels: list[int],
    embeddings: Sequence[Sequence[float]],
    segments: list[dict],
    max_seconds: float = 2.5,
    max_segments: int = 2,
    similarity_threshold: float = 0.82,
) -> list[int]:
    """Fold a tiny, strongly matching diarization fragment into its real voice."""
    if len(set(labels)) < 2 or not len(embeddings):
        return labels
    merged = list(labels)
    for label in sorted(set(labels)):
        positions = [index for index, value in enumerate(merged) if value == label]
        seconds = sum(float(segments[index]["end"]) - float(segments[index]["start"]) for index in positions)
        if len(positions) > max_segments or seconds > max_seconds:
            continue
        centroid = _centroid([embeddings[index] for index in positions])
        best_label, best_similarity = None, -1.0
        for candidate in sorted(set(merged) - {label}):
            candidate_positions = [index for index, value in enumerate(merged) if value == candidate]
            candidate_centroid = _centroid([embeddings[index] for index in candidate_positions])
            similarity = sum(left * right for left, right in zip(centroid, candidate_centroid))
            if similarity > best_similarity:
                best_label, best_similarity = candidate, similarity
        if best_label is not None and best_similarity >= similarity_threshold:
            merged = [best_label if value == label else value for value in merged]
    return _normalized_labels(merged)
