import re
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
    assert all(not re.search(r"[А-Яа-яЁё]", item["label"]) for item in payload["languages"])
    assert all(not re.search(r"[А-Яа-яЁё]", item["label"]) for item in payload["voices"])
    assert all(not re.search(r"[А-Яа-яЁё]", item["label"]) for item in payload["subtitle_styles"])


def test_health_explains_runtime_state() -> None:
    payload = TestClient(app).get("/api/health").json()
    assert "runtime_ready" in payload
    assert "checks" in payload
    assert "runtime" in payload


def test_srt_uses_timeline_and_utf8_bom(tmp_path: Path) -> None:
    target = tmp_path / "test.srt"
    write_srt([{"start": 1.25, "end": 3.5, "translation": "Hello, world!"}], target)
    raw = target.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    assert "00:00:01,250 --> 00:00:03,500" in raw.decode("utf-8-sig")


def test_word_normalization_is_case_insensitive() -> None:
    assert normalized_words("Hello, WORLD!") == {"hello", "world"}


def test_static_interface_is_english_first() -> None:
    html = (Path(__file__).parents[1] / "static" / "index.html").read_text(encoding="utf-8")
    assert '<html lang="en">' in html
    assert not re.search(r"[А-Яа-яЁё]", html)


def test_static_interface_uses_manrope_only() -> None:
    root = Path(__file__).parents[1]
    styles = (root / "static" / "styles.css").read_text(encoding="utf-8")
    font_files = {path.name.lower() for path in (root / "static" / "fonts").iterdir()}
    assert "font-family:Manrope" in styles
    assert "fraunces" not in styles.lower()
    assert not any("fraunces" in name for name in font_files)


def test_display_headings_do_not_end_with_full_stops() -> None:
    html = (Path(__file__).parents[1] / "static" / "index.html").read_text(encoding="utf-8")
    headings = re.findall(r"<h[1-6][^>]*>(.*?)</h[1-6]>", html, flags=re.DOTALL)
    visible_headings = [re.sub(r"<[^>]+>", "", heading).strip() for heading in headings]
    assert all("." not in heading for heading in visible_headings)
