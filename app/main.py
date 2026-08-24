from __future__ import annotations

import asyncio
import json
import math
import os
import shutil
import subprocess
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .media import VIDEO_SUFFIXES, make_thumbnail, probe
from .pipeline import analyze, jobs, offline_environment, recover_interrupted_jobs, reexport_captions, render
from .schemas import AnalyzeRequest, CaptionRequest, PreviewRequest, RenderRequest, SegmentPatch
from .store import (
    add_event,
    create_project,
    delete_project,
    invalidate_current_export,
    list_projects,
    load_project,
    project_dir,
    save_project,
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    recover_interrupted_jobs()
    yield


app = FastAPI(title="Dubbing Studio", version="0.1.0-alpha.1", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=config.ROOT / "static"), name="static")


def safe_project_asset(project_id: str, selected: str | None) -> Path:
    """Resolve a stored media path without allowing it to escape its project."""
    if not selected:
        raise HTTPException(404, "Media not found.")
    root = project_dir(project_id).resolve()
    path = Path(selected).resolve()
    if not path.is_file() or root not in path.parents:
        raise HTTPException(404, "Media not found.")
    return path


def validate_render_catalog(request: RenderRequest | CaptionRequest) -> None:
    language_ids = {item["id"] for item in config.LANGUAGES}
    style_ids = {item["id"] for item in config.SUBTITLE_STYLES}
    if isinstance(request, RenderRequest) and request.target_language not in language_ids:
        raise HTTPException(400, "Choose a supported translation language.")
    if request.subtitle_style not in style_ids:
        raise HTTPException(400, "Choose a supported subtitle style.")
    if isinstance(request, RenderRequest) and request.voice_mode == "catalog":
        voice_ids = {item["id"] for item in config.VOICE_PERSONAS}
        if request.voice_id not in voice_ids:
            raise HTTPException(400, "Choose a voice for the selected language.")
    if isinstance(request, RenderRequest) and request.lip_sync_enabled and not (config.MUSETALK_AVAILABLE or config.MOCK_MODE):
        raise HTTPException(409, "Local lip sync is not installed on this computer. Run setup again or turn Lip sync off.")


def validate_transcript(project: dict, patch: SegmentPatch) -> None:
    if not patch.segments:
        raise HTTPException(400, "The transcript must contain at least one spoken line.")
    duration = float(project.get("media", {}).get("duration") or 0)
    previous_start = -1.0
    effective_speakers = patch.speakers if patch.speakers is not None else project.get("analysis", {}).get("speakers", [])
    speaker_ids = {item.get("id") for item in effective_speakers if item.get("id")}
    if not speaker_ids:
        raise HTTPException(400, "The transcript must keep at least one speaker.")
    if len(speaker_ids) != len(effective_speakers):
        raise HTTPException(400, "Speaker identifiers must be present and unique.")
    for index, segment in enumerate(patch.segments, start=1):
        try:
            start = float(segment["start"])
            end = float(segment["end"])
            text = str(segment["text"]).strip()
            speaker = segment["speaker"]
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(400, f"Transcript line {index} is incomplete.") from exc
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise HTTPException(400, f"Transcript line {index} has invalid timing.")
        if duration and end > duration + 0.25:
            raise HTTPException(400, f"Transcript line {index} ends outside the video.")
        if start < previous_start:
            raise HTTPException(400, "Transcript lines must remain in chronological order.")
        if not text:
            raise HTTPException(400, f"Transcript line {index} is empty.")
        if speaker_ids and speaker not in speaker_ids:
            raise HTTPException(400, f"Transcript line {index} references an unknown speaker.")
        previous_start = start


@app.get("/")
def index() -> FileResponse:
    return FileResponse(config.ROOT / "static" / "index.html")


@app.get("/api/health")
def health() -> dict:
    checks = {name: path.exists() for name, path in config.RUNTIME_CHECKS.items()}
    ready = all(checks.values())
    return {
        "app_id": "dubbing-studio-local",
        "ok": ready or config.MOCK_MODE,
        "runtime_ready": ready,
        "mock": config.MOCK_MODE,
        "backend": config.BACKEND,
        "device": config.TORCH_DEVICE,
        "video_encoder": config.VIDEO_ENCODER,
        "runtime": str(config.RUNTIME_ROOT),
        "checks": checks,
        "queue": {"active": jobs.active, "waiting": jobs.waiting},
    }


@app.get("/api/catalog")
def catalog() -> dict:
    return {
        "languages": config.LANGUAGES,
        "voices": config.VOICE_PERSONAS,
        "subtitle_styles": config.SUBTITLE_STYLES,
        "capabilities": {"lip_sync": config.MUSETALK_AVAILABLE or config.MOCK_MODE},
    }


@app.get("/api/projects")
def projects() -> list[dict]:
    return list_projects()


@app.post("/api/projects")
async def upload_project(video: UploadFile = File(...)) -> dict:
    suffix = Path(video.filename or "video.mp4").suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise HTTPException(400, "Supported formats: MP4, MOV, MKV, WebM and M4V.")
    project = create_project(video.filename or "video.mp4")
    folder = project_dir(project["id"])
    target = folder / "input" / f"source{suffix}"
    size = 0
    with target.open("wb") as output:
        while chunk := await video.read(1024 * 1024):
            size += len(chunk)
            if size > config.MAX_UPLOAD_BYTES:
                output.close()
                shutil.rmtree(folder, ignore_errors=True)
                raise HTTPException(413, "The file exceeds the local upload limit.")
            output.write(chunk)
    try:
        media = probe(target)
        if not media.get("video_codec") or not media.get("width") or not media.get("height"):
            raise RuntimeError("The file has no readable video stream.")
        if not media.get("audio_codec"):
            raise RuntimeError("The video has no audio track. Add an audible dialogue track and upload it again.")
        if media.get("duration", 0) <= 0:
            raise RuntimeError("The video duration could not be determined.")
        thumbnail = folder / "preview" / "thumbnail.jpg"
        make_thumbnail(target, thumbnail, media["duration"])
    except Exception as exc:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(400, f"Could not read the video: {exc}") from exc
    project = load_project(project["id"])
    project["media"] = media
    project["input"] = str(target)
    project["thumbnail"] = str(thumbnail)
    save_project(project)
    return project


@app.get("/api/projects/{project_id}")
def get_project(project_id: str) -> dict:
    try:
        return load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, "Project not found.") from exc


@app.delete("/api/projects/{project_id}")
def remove_project(project_id: str) -> dict:
    project = get_project(project_id)
    busy = project.get("status") in {"queued", "analyzing", "rendering"} or jobs.active == project_id or project_id in jobs.pending
    stopped = jobs.cancel(project_id, wait=True)
    if busy and not stopped:
        raise HTTPException(409, "Processing is still stopping. Wait a few seconds and delete the project again.")
    try:
        delete_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, "Project not found.") from exc
    return {"deleted": True, "id": project_id}


@app.post("/api/projects/{project_id}/cancel")
def cancel_project(project_id: str) -> dict:
    project = get_project(project_id)
    busy = project.get("status") in {"queued", "analyzing", "rendering"}
    if not busy and jobs.active != project_id and project_id not in jobs.pending:
        raise HTTPException(409, "This project is not processing.")
    jobs.cancel(project_id)
    project = get_project(project_id)
    project.update({"status": "cancelled", "stage": "Stopped by user", "error": None})
    save_project(project)
    add_event(project_id, "warning", "Processing stopped by user")
    return get_project(project_id)


@app.post("/api/projects/{project_id}/analyze")
def start_analysis(project_id: str, request: AnalyzeRequest) -> dict:
    project = get_project(project_id)
    language_ids = {item["id"] for item in config.LANGUAGES}
    if request.source_language != "auto" and request.source_language not in language_ids:
        raise HTTPException(400, "Choose a supported source language or automatic detection.")
    if request.speaker_count != "auto" and not 1 <= request.speaker_count <= 20:
        raise HTTPException(400, "Speaker count must be between 1 and 20.")
    if project["status"] in {"queued", "analyzing", "rendering"}:
        raise HTTPException(409, "This project is already processing.")
    if project.get("output"):
        invalidate_current_export(project, "New analysis requires a new render")
    previous_state = {key: project.get(key) for key in ("status", "stage", "progress")}
    project["status"] = "queued"
    project["stage"] = "Queued for analysis"
    project["progress"] = 0
    project["analysis"].update(request.model_dump())
    save_project(project)
    if not jobs.submit(project_id, analyze, request.model_dump()):
        project.update(previous_state)
        save_project(project)
        raise HTTPException(409, "This project still has a task shutting down.")
    return project


@app.patch("/api/projects/{project_id}/transcript")
def patch_transcript(project_id: str, patch: SegmentPatch) -> dict:
    project = get_project(project_id)
    if project["status"] not in {"review", "failed", "complete", "quality_review"}:
        raise HTTPException(409, "The transcript cannot be changed while the project is processing.")
    validate_transcript(project, patch)
    project["analysis"]["segments"] = patch.segments
    if patch.speakers is not None:
        project["analysis"]["speakers"] = patch.speakers
    if project.get("output"):
        invalidate_current_export(project, "Transcript updated — render required")
    save_project(project)
    if project["stage"] == "Transcript updated — render required":
        add_event(project_id, "warning", "Previous export archived because the transcript changed")
    return project


@app.post("/api/projects/{project_id}/render")
def start_render(project_id: str, request: RenderRequest) -> dict:
    project = get_project(project_id)
    validate_render_catalog(request)
    if project["status"] in {"analyzing", "rendering", "queued"}:
        raise HTTPException(409, "This project is already processing.")
    if not project["analysis"].get("segments"):
        raise HTTPException(400, "Run the analysis first.")
    previous_state = {key: project.get(key) for key in ("status", "stage", "progress")}
    render_request = request.model_dump()
    render_request.update({"background_volume": 1.0, "expression": 0.5, "quality": "high"})
    render_request["run_id"] = uuid.uuid4().hex[:10]
    project["status"] = "queued"
    project["stage"] = "Queued for rendering"
    project["progress"] = 0
    project["render"] = render_request
    save_project(project)
    if not jobs.submit(project_id, render, render_request):
        project.update(previous_state)
        save_project(project)
        raise HTTPException(409, "This project still has a task shutting down.")
    return project


@app.post("/api/projects/{project_id}/captions")
def update_captions(project_id: str, request: CaptionRequest) -> dict:
    project = get_project(project_id)
    validate_render_catalog(request)
    if project["status"] in {"analyzing", "rendering", "queued"}:
        raise HTTPException(409, "This project is already processing.")
    if not project.get("output", {}).get("video"):
        raise HTTPException(409, "Create the dub before adjusting its export captions.")
    render_request = dict(project.get("render") or {})
    if not render_request.get("target_language"):
        raise HTTPException(409, "The previous dub settings are missing.")
    render_request.update(request.model_dump())
    render_request.update({
        "background_volume": 1.0, "expression": 0.5, "quality": "high",
        "burn_subtitles": True, "caption_only": True, "run_id": uuid.uuid4().hex[:10],
    })
    previous_state = {key: project.get(key) for key in ("status", "stage", "progress")}
    project.update({"status": "queued", "stage": "Queued for caption export", "render": render_request})
    save_project(project)
    if not jobs.submit(project_id, reexport_captions, render_request):
        project.update(previous_state)
        save_project(project)
        raise HTTPException(409, "This project still has a task shutting down.")
    return project


@app.post("/api/voice-preview")
def voice_preview(request: PreviewRequest) -> dict:
    try:
        language = config.language(request.language)
        persona = config.voice(request.voice_id)
    except StopIteration as exc:
        raise HTTPException(400, "Choose a supported language and voice.") from exc
    target = config.PREVIEWS_ROOT / request.language / f"{request.voice_id}.wav"
    if not target.exists():
        if not jobs.acquire_external():
            raise HTTPException(409, "The GPU is busy with a dub. Generate the preview after the current task finishes.")
        try:
            payload = {
                "output": str(target), "qwen_root": str(config.QWEN_ROOT),
                "sample_text": language["sample"], "tts_language": language["tts"],
                "voice_description": persona["description"], "device": config.TORCH_DEVICE,
                "dtype": config.TORCH_DTYPE,
            }
            request_path = target.with_suffix(".json")
            request_path.parent.mkdir(parents=True, exist_ok=True)
            request_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            command = (
                ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1.2", str(target)]
                if config.MOCK_MODE
                else [str(config.QWEN_PYTHON), str(config.ROOT / "workers" / "qwen_voice.py"), "preview", str(request_path)]
            )
            result = subprocess.run(
                command, cwd=config.QWEN_ROOT if not config.MOCK_MODE else config.ROOT,
                env=offline_environment(), capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            request_path.with_suffix(".log").write_text(
                (result.stdout or "") + os.linesep + (result.stderr or ""), encoding="utf-8",
            )
            if result.returncode or not target.is_file():
                target.unlink(missing_ok=True)
                raise RuntimeError((result.stderr or result.stdout or "Voice preview failed")[-1200:])
        except Exception as exc:
            raise HTTPException(500, f"Could not generate the voice preview: {exc}") from exc
        finally:
            jobs.release_external()
    return {"url": f"/api/previews/{request.language}/{request.voice_id}.wav"}


@app.get("/api/previews/{language}/{filename}")
def preview_file(language: str, filename: str) -> FileResponse:
    allowed_languages = {item["id"] for item in config.LANGUAGES}
    allowed_files = {f"{item['id']}.wav" for item in config.VOICE_PERSONAS}
    path = (config.PREVIEWS_ROOT / language / filename).resolve()
    expected_parent = (config.PREVIEWS_ROOT / language).resolve()
    if language not in allowed_languages or filename not in allowed_files or not path.is_file() or path.parent != expected_parent:
        raise HTTPException(404)
    return FileResponse(path, media_type="audio/wav")


@app.get("/api/projects/{project_id}/media/{kind}")
def project_media(project_id: str, kind: str) -> FileResponse:
    project = get_project(project_id)
    paths = {
        "input": project.get("input"), "thumbnail": project.get("thumbnail"),
        "output": project.get("output", {}).get("video"), "subtitles": project.get("output", {}).get("subtitles"),
    }
    selected = safe_project_asset(project_id, paths.get(kind))
    media_type = "video/mp4" if kind in {"input", "output"} else "image/jpeg" if kind == "thumbnail" else "application/x-subrip"
    return FileResponse(selected, media_type=media_type, filename=selected.name)


@app.get("/api/projects/{project_id}/download/{kind}")
def project_download(project_id: str, kind: str) -> FileResponse:
    project = get_project(project_id)
    paths = {
        "output": project.get("output", {}).get("video"),
        "subtitles": project.get("output", {}).get("subtitles"),
    }
    selected = safe_project_asset(project_id, paths.get(kind))
    media_type = "video/mp4" if kind == "output" else "application/x-subrip"
    return FileResponse(selected, media_type=media_type, filename=selected.name)


@app.websocket("/ws/projects/{project_id}")
async def project_socket(websocket: WebSocket, project_id: str) -> None:
    await websocket.accept()
    previous = None
    try:
        while True:
            project = load_project(project_id)
            serialized = json.dumps(project, ensure_ascii=False, sort_keys=True)
            if serialized != previous:
                await websocket.send_json(project)
                previous = serialized
            await asyncio.sleep(.8)
    except (WebSocketDisconnect, FileNotFoundError):
        return
