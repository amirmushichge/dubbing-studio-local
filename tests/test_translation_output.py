import pytest

from workers.translation_output import (
    TranslationOutputError,
    extract_translation_payload,
    validate_translation_payload,
)


def test_extracts_fenced_json_with_trailing_commentary() -> None:
    response = '```json\n[{"id": 0, "translation": "你好。"}]\n```\nTranslation complete.'
    assert extract_translation_payload(response, [0]) == [{"id": 0, "translation": "你好。"}]


def test_ignores_second_array_instead_of_merging_both() -> None:
    response = '[{"id": 0, "translation": "你好。"}]\n[{"status": "done"}]'
    assert extract_translation_payload(response, [0]) == [{"id": 0, "translation": "你好。"}]


def test_combines_one_array_per_translation_segment() -> None:
    response = (
        '[{"id": 0, "translation": "Buongiorno."}]\n'
        '[{"id": 1, "translation": "Come stai?"}]\n'
        '[{"id": 2, "translation": "Bene."}]'
    )
    assert extract_translation_payload(response, [0, 1, 2]) == [
        {"id": 0, "translation": "Buongiorno."},
        {"id": 1, "translation": "Come stai?"},
        {"id": 2, "translation": "Bene."},
    ]


def test_rejects_conflicting_fragment_values() -> None:
    response = '[{"id": 0, "translation": "Uno"}]\n[{"id": 0, "translation": "Due"}]'
    with pytest.raises(TranslationOutputError, match="conflicting values"):
        extract_translation_payload(response, [0, 1])


def test_skips_invalid_array_before_valid_translation() -> None:
    response = '["draft"]\n[{"id": "4", "translation": "再见。"}]'
    assert extract_translation_payload(response, [4]) == [{"id": 4, "translation": "再见。"}]


@pytest.mark.parametrize(
    "payload, message",
    [
        ([{"id": 0, "translation": ""}], "empty translation"),
        ([{"id": 0, "translation": "A"}, {"id": 0, "translation": "B"}], "duplicated"),
        ([{"id": 1, "translation": "A"}], "missing ids"),
    ],
)
def test_validation_rejects_incomplete_or_ambiguous_payloads(payload, message: str) -> None:
    with pytest.raises(TranslationOutputError, match=message):
        validate_translation_payload(payload, [0])
