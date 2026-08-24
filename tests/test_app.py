import re
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from app import main as main_module
from app import store as store_module
from app.config import subtitle_force_style, venv_python, video_encoder_args
from app.diarization import merge_tiny_clusters
from app.main import app
from app.pipeline import (
    execute,
    normalized_words,
    recovery_action,
    redistribute_line_timing,
    safe_output_name,
    select_background,
    set_progress,
    split_segments_on_word_gaps,
    write_ass,
    write_srt,
)
from app.schemas import RenderRequest
from app.store import delete_project, invalidate_current_export


def test_catalog_has_languages_voices_and_subtitles() -> None:
    response = TestClient(app).get("/api/catalog")
    assert response.status_code == 200
    payload = response.json()
    assert {item["id"] for item in payload["languages"]} >= {"ru", "en", "zh"}
    assert payload["voices"]
    assert payload["subtitle_styles"]
    assert isinstance(payload["capabilities"]["lip_sync"], bool)
    assert all(not re.search(r"[А-Яа-яЁё]", item["label"]) for item in payload["languages"])
    assert all(not re.search(r"[А-Яа-яЁё]", item["label"]) for item in payload["voices"])
    assert all(not re.search(r"[А-Яа-яЁё]", item["label"]) for item in payload["subtitle_styles"])


def test_health_explains_runtime_state() -> None:
    payload = TestClient(app).get("/api/health").json()
    assert payload["app_id"] == "dubbing-studio-local"
    assert "runtime_ready" in payload
    assert "checks" in payload
    assert "runtime" in payload
    assert payload["backend"] in {"cuda", "apple_silicon", "cpu"}
    assert payload["video_encoder"] in {"h264_nvenc", "h264_videotoolbox", "libx264"}


def test_srt_uses_timeline_and_utf8_bom(tmp_path: Path) -> None:
    target = tmp_path / "test.srt"
    write_srt([{"start": 1.25, "end": 3.5, "translation": "Hello, world!"}], target)
    raw = target.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    assert "00:00:01,250 --> 00:00:03,500" in raw.decode("utf-8-sig")


def test_ass_uses_real_scale_and_position_controls(tmp_path: Path) -> None:
    target = tmp_path / "captions.ass"
    write_ass(
        [{"start": 1.25, "end": 3.5, "translation": "Positioned caption"}],
        target, 1920, 1080,
        {"subtitle_style": "clean", "subtitle_size": "medium", "subtitle_color": "yellow", "subtitle_scale": 1.5, "subtitle_x": 25, "subtitle_y": 75},
    )
    text = target.read_text(encoding="utf-8-sig")
    assert r"\pos(480,810)" in text
    assert "Style: Default,Manrope,90.0,&H0000D4FF" in text


def test_ass_escapes_user_control_sequences(tmp_path: Path) -> None:
    target = tmp_path / "captions.ass"
    write_ass(
        [{"start": 0.0, "end": 1.0, "translation": r"Folder\name {not-a-style}"}],
        target, 1920, 1080,
        {"subtitle_style": "clean", "subtitle_size": "medium", "subtitle_color": "white"},
    )
    text = target.read_text(encoding="utf-8-sig")
    assert r"Folder\\name \{not-a-style\}" in text


def test_cross_platform_runtime_and_encoder_profiles(tmp_path: Path) -> None:
    assert venv_python(tmp_path, "nt") == tmp_path / ".venv" / "Scripts" / "python.exe"
    assert venv_python(tmp_path, "posix") == tmp_path / ".venv" / "bin" / "python"
    assert "h264_videotoolbox" in video_encoder_args("high", "apple_silicon")
    assert "h264_nvenc" in video_encoder_args("high", "cuda")
    assert "libx264" in video_encoder_args("high", "cpu")
    apple = video_encoder_args("high", "apple_silicon", source_bitrate=8_000_000)
    assert apple[apple.index("-b:v") + 1] == "10000000"
    cuda = video_encoder_args("high", "cuda", source_bitrate=80_000_000)
    assert cuda[cuda.index("-b:v") + 1] == "0"


def test_macos_launchers_are_real_apple_silicon_profiles() -> None:
    root = Path(__file__).parents[1]
    setup = (root / "setup.command").read_text(encoding="utf-8")
    start = (root / "start.command").read_text(encoding="utf-8")
    assert "$(uname -m)" in setup and '!= "arm64"' in setup
    assert "--profile apple_silicon" in setup
    assert "h264_nvenc" not in setup + start
    assert "DUBBING_STUDIO_BACKEND" in start


def test_word_normalization_is_case_insensitive() -> None:
    assert normalized_words("Hello, WORLD!") == {"hello", "world"}


def test_static_interface_is_english_first() -> None:
    html = (Path(__file__).parents[1] / "static" / "index.html").read_text(encoding="utf-8")
    assert '<html lang="en">' in html
    assert not re.search(r"[А-Яа-яЁё]", html)


def test_delivery_has_no_manual_audio_expression_or_quality_controls() -> None:
    html = (Path(__file__).parents[1] / "static" / "index.html").read_text(encoding="utf-8")
    assert 'id="backgroundVolume"' not in html
    assert 'id="expression"' not in html
    assert 'id="quality"' not in html


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


def test_worker_progress_is_monotonic_and_quiet(monkeypatch) -> None:
    project = {"id": "example", "progress": 44, "stage": "Generating native speech"}
    saved = []
    monkeypatch.setattr("app.pipeline.jobs.raise_if_cancelled", lambda project_id: None)
    monkeypatch.setattr("app.pipeline.load_project", lambda project_id: dict(project))
    monkeypatch.setattr("app.pipeline.save_project", lambda value: saved.append(value) or value)

    set_progress("example", 41, "Generating speech · 3 of 9 lines")

    assert saved[0]["progress"] == 44
    assert saved[0]["stage"] == "Generating speech · 3 of 9 lines"


def test_streamed_worker_percentage_updates_project_progress(monkeypatch, tmp_path: Path) -> None:
    project = {"id": "example", "progress": 10, "stage": "Matching voice profile"}
    monkeypatch.setattr("app.pipeline.project_dir", lambda project_id: tmp_path)
    monkeypatch.setattr("app.pipeline.load_project", lambda project_id: dict(project))
    monkeypatch.setattr("app.pipeline.save_project", lambda value: project.update(value) or value)

    execute(
        "example", "progress_worker",
        [sys.executable, "-c", "print('50%', flush=True); print('100%', flush=True)"],
        progress_range=(10, 20),
    )

    assert project["progress"] == 20
    assert "100%" in (tmp_path / "logs" / "progress_worker.log").read_text(encoding="utf-8")


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


def test_quality_review_allows_download_when_export_exists(monkeypatch, tmp_path: Path) -> None:
    video = tmp_path / "result.mp4"
    video.write_bytes(b"video")
    project = {"status": "quality_review", "output": {"video": str(video), "subtitles": None}}
    monkeypatch.setattr(main_module, "get_project", lambda project_id: project)
    monkeypatch.setattr(main_module, "project_dir", lambda project_id: tmp_path)
    response = TestClient(app).get("/api/projects/example/download/output")
    assert response.status_code == 200


def test_render_uses_automatic_source_preserving_defaults(monkeypatch) -> None:
    project = {"id": "example", "status": "review", "analysis": {"segments": [{"text": "Line"}]}}
    submitted = []
    monkeypatch.setattr(main_module, "get_project", lambda project_id: project)
    monkeypatch.setattr(main_module, "save_project", lambda value: value)
    monkeypatch.setattr(main_module.jobs, "submit", lambda project_id, worker, payload: submitted.append(payload) or True)
    request = RenderRequest(
        target_language="en", background_volume=0.1, expression=1.0, quality="draft",
    )

    response = main_module.start_render("example", request)

    assert response["render"]["background_volume"] == 1.0
    assert response["render"]["expression"] == 0.5
    assert response["render"]["quality"] == "high"
    assert submitted[0]["quality"] == "high"


def test_lip_sync_is_opt_in_and_persisted(monkeypatch) -> None:
    project = {"id": "example", "status": "review", "analysis": {"segments": [{"text": "Line"}]}}
    submitted = []
    monkeypatch.setattr(main_module, "get_project", lambda project_id: project)
    monkeypatch.setattr(main_module, "save_project", lambda value: value)
    monkeypatch.setattr(main_module.config, "MUSETALK_AVAILABLE", True)
    monkeypatch.setattr(main_module.jobs, "submit", lambda project_id, worker, payload: submitted.append(payload) or True)

    assert RenderRequest(target_language="en").lip_sync_enabled is False
    response = main_module.start_render("example", RenderRequest(target_language="en", lip_sync_enabled=True))

    assert response["render"]["lip_sync_enabled"] is True
    assert submitted[0]["lip_sync_enabled"] is True


def test_download_rejects_media_path_outside_project(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    outside = tmp_path / "private.mp4"
    outside.write_bytes(b"video")
    project = {"output": {"video": str(outside), "subtitles": None}}
    monkeypatch.setattr(main_module, "get_project", lambda project_id: project)
    monkeypatch.setattr(main_module, "project_dir", lambda project_id: project_root)
    response = TestClient(app).get("/api/projects/example/download/output")
    assert response.status_code == 404


def test_oversized_upload_removes_the_incomplete_project(monkeypatch, tmp_path: Path) -> None:
    folder = tmp_path / "project"
    for name in ("input", "preview"):
        (folder / name).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(main_module, "create_project", lambda filename: {"id": "example"})
    monkeypatch.setattr(main_module, "project_dir", lambda project_id: folder)
    monkeypatch.setattr(main_module.config, "MAX_UPLOAD_BYTES", 8)
    response = TestClient(app).post(
        "/api/projects",
        files={"video": ("too-large.mp4", b"more than eight bytes", "video/mp4")},
    )
    assert response.status_code == 413
    assert not folder.exists()


def test_invalid_transcript_is_rejected_before_render(monkeypatch) -> None:
    project = {
        "status": "review", "media": {"duration": 5},
        "analysis": {"speakers": [{"id": "SPEAKER_00"}], "segments": []},
    }
    monkeypatch.setattr(main_module, "get_project", lambda project_id: project)
    response = TestClient(app).patch(
        "/api/projects/example/transcript",
        json={"segments": [{"start": 4, "end": 2, "text": "Broken", "speaker": "SPEAKER_00"}]},
    )
    assert response.status_code == 400
    assert "invalid timing" in response.json()["detail"]


def test_transcript_cannot_delete_every_speaker(monkeypatch) -> None:
    project = {
        "status": "review", "media": {"duration": 5},
        "analysis": {"speakers": [{"id": "SPEAKER_00"}], "segments": []},
    }
    monkeypatch.setattr(main_module, "get_project", lambda project_id: project)
    response = TestClient(app).patch(
        "/api/projects/example/transcript",
        json={
            "segments": [{"start": 0, "end": 2, "text": "Line", "speaker": "SPEAKER_00"}],
            "speakers": [],
        },
    )
    assert response.status_code == 400
    assert "at least one speaker" in response.json()["detail"]


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


def test_busy_project_is_cancelled_before_delete(monkeypatch) -> None:
    cancelled = []
    deleted = []
    monkeypatch.setattr(main_module, "get_project", lambda project_id: {"id": project_id, "status": "rendering"})
    monkeypatch.setattr(main_module.jobs, "cancel", lambda project_id, wait=False: cancelled.append((project_id, wait)) or True)
    monkeypatch.setattr(main_module, "delete_project", lambda project_id: deleted.append(project_id))
    response = TestClient(app).delete("/api/projects/example")
    assert response.status_code == 200
    assert cancelled == [("example", True)]
    assert deleted == ["example"]


def test_cancel_marks_a_busy_project_as_stopped(monkeypatch) -> None:
    project = {"id": "example", "status": "queued", "stage": "Queued", "events": []}
    monkeypatch.setattr(main_module, "get_project", lambda project_id: project)
    monkeypatch.setattr(main_module.jobs, "cancel", lambda project_id, wait=False: True)
    monkeypatch.setattr(main_module, "save_project", lambda value: project.update(value) or project)
    monkeypatch.setattr(main_module, "add_event", lambda *args: project["events"].append(args))
    response = TestClient(app).post("/api/projects/example/cancel")
    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"


def test_subtitle_controls_change_the_real_ass_style() -> None:
    yellow = subtitle_force_style("social", "large", "yellow")
    black_box = subtitle_force_style("boxed", "small", "black")
    assert "FontName=Manrope" in yellow
    assert "FontSize=20" in yellow
    assert "PrimaryColour=&H0000D4FF" in yellow
    assert "FontSize=12" in black_box
    assert "PrimaryColour=&H00000000" in black_box
    assert "OutlineColour=&H00FFFFFF" in black_box
    assert "BorderStyle=1" in black_box
    assert "BackColour=" not in black_box
    assert "MarginV=43" in black_box


def test_short_line_borrows_silence_instead_of_becoming_robotically_fast() -> None:
    lines = [
        {"start": 0.0, "end": 1.0},
        {"start": 2.0, "end": 2.44},
    ]
    change = redistribute_line_timing(lines, 1, required_speed=1.914, media_duration=2.8)
    assert change is not None
    assert lines[1]["start"] < 2.0
    assert lines[1]["end"] > 2.44
    assert lines[1]["start"] >= 1.12
    assert lines[1]["end"] <= 2.68
    new_slot = lines[1]["end"] - lines[1]["start"]
    assert 0.44 * 1.914 / new_slot <= 1.26


def test_tiny_matching_speaker_fragment_is_merged() -> None:
    embeddings = [
        [1.0, 0.0], [0.98, 0.02], [0.96, 0.04], [0.0, 1.0],
    ]
    segments = [
        {"start": 0.0, "end": 4.0}, {"start": 5.0, "end": 9.0},
        {"start": 10.0, "end": 10.8}, {"start": 11.0, "end": 15.0},
    ]

    assert merge_tiny_clusters([0, 0, 2, 1], embeddings, segments) == [0, 0, 0, 1]


def test_long_internal_word_gap_becomes_separate_speech_windows() -> None:
    segments = [{
        "id": 0, "start": 12.9, "end": 19.28, "text": "You look terrible.", "speaker": "SPEAKER_01",
        "words": [
            {"start": 12.9, "end": 13.28, "word": " You"},
            {"start": 18.52, "end": 18.9, "word": " look"},
            {"start": 18.9, "end": 19.28, "word": " terrible."},
        ],
    }]

    split = split_segments_on_word_gaps(segments)

    assert [(item["start"], item["end"], item["text"]) for item in split] == [
        (12.9, 13.28, "You…"),
        (18.52, 19.28, "look terrible."),
    ]
    assert [item["id"] for item in split] == [0, 1]
