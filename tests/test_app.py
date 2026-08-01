from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.pipeline import normalized_words, write_srt


def test_catalog_has_languages_voices_and_subtitles() -> None:
    response = TestClient(app).get("/api/catalog")
    assert response.status_code == 200
    payload = response.json()
    assert {item["id"] for item in payload["languages"]} >= {"ru", "en", "zh"}
    assert payload["voices"]
    assert payload["subtitle_styles"]


def test_health_explains_runtime_state() -> None:
    payload = TestClient(app).get("/api/health").json()
    assert "runtime_ready" in payload
    assert "checks" in payload
    assert "runtime" in payload


def test_srt_uses_timeline_and_utf8_bom(tmp_path: Path) -> None:
    target = tmp_path / "test.srt"
    write_srt([{"start": 1.25, "end": 3.5, "translation": "Привет, мир!"}], target)
    raw = target.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    assert "00:00:01,250 --> 00:00:03,500" in raw.decode("utf-8-sig")


def test_word_normalization_is_case_insensitive() -> None:
    assert normalized_words("Hello, МИР!") == {"hello", "мир"}
