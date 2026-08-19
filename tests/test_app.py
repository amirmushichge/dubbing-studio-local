import re
from pathlib import Path

from fastapi.testclient import TestClient

from app import main as main_module
from app.config import subtitle_force_style
from app.main import app
from app.pipeline import normalized_words, recovery_action, safe_output_name, select_background, write_srt
from app import store as store_module
from app.store import delete_project, invalidate_current_export


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
    assert payload["app_id"] == "dubbing-studio-local"
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


def test_stale_export_is_archived_and_invalidated() -> None:
    project = {
        "status": "complete", "stage": "Complete", "progress": 100,
        "updated_at": "2026-08-01T00:00:00+00:00",
        "output": {"video": "old.mp4", "subtitles": "old.srt"},
        "render": {"target_language": "en"}, "quality": {"warnings": []}, "exports": [],
    }
    invalidate_current_export(project, "Transcript updated — render required")
    assert project["status"] == "review"
    assert project["output"] == {}
    assert project["exports"][0]["video"] == "old.mp4"


def test_queued_jobs_have_recoverable_payloads() -> None:
    analysis = recovery_action({
        "status": "queued", "stage": "Queued for analysis",
        "analysis": {"source_language": "ru", "speaker_count": 2},
    })
    rendering = recovery_action({
        "status": "queued", "stage": "Queued for rendering",
        "render": {"target_language": "en", "run_id": "abc"},
    })
    assert analysis and analysis[0].__name__ == "analyze" and analysis[1]["speaker_count"] == 2
    assert rendering and rendering[0].__name__ == "render" and rendering[1]["run_id"] == "abc"


def test_output_names_are_version_safe() -> None:
    assert safe_output_name('bad:name/with*chars.') == "bad_name_with_chars"


def test_missing_background_never_falls_back_to_original_speech(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.touch()
    work = tmp_path / "work"
    work.mkdir()
    assert select_background(work, source, mock_mode=True) == source
    try:
        select_background(work, source, mock_mode=False)
    except RuntimeError as exc:
        assert "refusing to mix the original speech" in str(exc)
    else:
        raise AssertionError("Production mode accepted a missing separated background")


def test_quality_review_blocks_download(monkeypatch, tmp_path: Path) -> None:
    video = tmp_path / "result.mp4"
    video.write_bytes(b"video")
    project = {"status": "quality_review", "output": {"video": str(video), "subtitles": None}}
    monkeypatch.setattr(main_module, "get_project", lambda project_id: project)
    response = TestClient(app).get("/api/projects/example/download/output")
    assert response.status_code == 409

    project["status"] = "complete"
    response = TestClient(app).get("/api/projects/example/download/output")
    assert response.status_code == 200


def test_delete_project_removes_only_selected_folder(monkeypatch, tmp_path: Path) -> None:
    projects = tmp_path / "projects"
    selected = projects / "abc123"
    preserved = projects / "keep456"
    selected.mkdir(parents=True)
    preserved.mkdir()
    (selected / "project.json").write_text("{}", encoding="utf-8")
    (selected / "output.mp4").write_bytes(b"video")
    (preserved / "project.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(store_module, "PROJECTS_ROOT", projects)

    delete_project("abc123")

    assert not selected.exists()
    assert preserved.exists()


def test_busy_project_cannot_be_deleted(monkeypatch) -> None:
    monkeypatch.setattr(main_module, "get_project", lambda project_id: {"id": project_id, "status": "rendering"})
    response = TestClient(app).delete("/api/projects/example")
    assert response.status_code == 409


def test_subtitle_controls_change_the_real_ass_style() -> None:
    yellow = subtitle_force_style("social", "large", "yellow")
    black_box = subtitle_force_style("boxed", "small", "black")
    assert "FontName=Manrope" in yellow
    assert "FontSize=24" in yellow
    assert "PrimaryColour=&H0000D4FF" in yellow
    assert "FontSize=16" in black_box
    assert "PrimaryColour=&H00000000" in black_box
    assert "OutlineColour=&H00FFFFFF" in black_box
    assert "BackColour=&H90FFFFFF" in black_box
