from __future__ import annotations

import asyncio
import json
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .media import VIDEO_SUFFIXES, make_thumbnail, probe
from .pipeline import analyze, jobs, recover_interrupted_jobs, render
from .schemas import AnalyzeRequest, PreviewRequest, RenderRequest, SegmentPatch
from .store import add_event, create_project, invalidate_current_export, list_projects, load_project, project_dir, save_project


@asynccontextmanager
async def lifespan(_: FastAPI):
    recover_interrupted_jobs()
    yield


app = FastAPI(title="Dubbing Studio", version="0.1.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=config.ROOT / "static"), name="static")


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
        "runtime": str(config.RUNTIME_ROOT),
        "checks": checks,
        "queue": {"active": jobs.active, "waiting": jobs.queue.qsize()},
    }


@app.get("/api/catalog")
def catalog() -> dict:
    return {"languages": config.LANGUAGES, "voices": config.VOICE_PERSONAS, "subtitle_styles": config.SUBTITLE_STYLES}


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
                target.unlink(missing_ok=True)
                raise HTTPException(413, "The file exceeds the local upload limit.")
            output.write(chunk)
    try:
        media = probe(target)
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


@app.post("/api/projects/{project_id}/analyze")
def start_analysis(project_id: str, request: AnalyzeRequest) -> dict:
    project = get_project(project_id)
    if project["status"] in {"queued", "analyzing", "rendering"}:
        raise HTTPException(409, "This project is already processing.")
    if project.get("output"):
        invalidate_current_export(project, "New analysis requires a new render")
    project["status"] = "queued"
    project["stage"] = "Queued for analysis"
    project["analysis"].update(request.model_dump())
    save_project(project)
    jobs.submit(project_id, analyze, request.model_dump())
    return project


@app.patch("/api/projects/{project_id}/transcript")
def patch_transcript(project_id: str, patch: SegmentPatch) -> dict:
    project = get_project(project_id)
    if project["status"] not in {"review", "failed", "complete", "quality_review"}:
        raise HTTPException(409, "The transcript cannot be changed while the project is processing.")
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
    if project["status"] in {"analyzing", "rendering", "queued"}:
        raise HTTPException(409, "This project is already processing.")
    if not project["analysis"].get("segments"):
        raise HTTPException(400, "Run the analysis first.")
    render_request = request.model_dump()
    render_request["run_id"] = uuid.uuid4().hex[:10]
    project["status"] = "queued"
    project["stage"] = "Queued for rendering"
    project["render"] = render_request
    save_project(project)
    jobs.submit(project_id, render, render_request)
    return project


@app.post("/api/voice-preview")
def voice_preview(request: PreviewRequest) -> dict:
    language = config.language(request.language)
    persona = config.voice(request.voice_id)
    target = config.PREVIEWS_ROOT / request.language / f"{request.voice_id}.wav"
    if not target.exists():
        if jobs.active or jobs.queue.qsize():
            raise HTTPException(409, "The GPU is busy with a dub. Generate the preview after the current task finishes.")
        payload = {"output": str(target), "qwen_root": str(config.QWEN_ROOT), "sample_text": language["sample"], "tts_language": language["tts"], "voice_description": persona["description"]}
        request_path = target.with_suffix(".json")
        request_path.parent.mkdir(parents=True, exist_ok=True)
        request_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if config.MOCK_MODE:
            subprocess = __import__("subprocess")
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1.2", str(target)], check=True)
        else:
            from .pipeline import execute
            execute("preview", "voice_preview", [str(config.QWEN_PYTHON), str(config.ROOT / "workers" / "qwen_voice.py"), "preview", str(request_path)], cwd=config.QWEN_ROOT)
    return {"url": f"/api/previews/{request.language}/{request.voice_id}.wav"}


@app.get("/api/previews/{language}/{filename}")
def preview_file(language: str, filename: str) -> FileResponse:
    path = config.PREVIEWS_ROOT / language / filename
    if not path.exists() or path.parent != (config.PREVIEWS_ROOT / language):
        raise HTTPException(404)
    return FileResponse(path, media_type="audio/wav")


@app.get("/api/projects/{project_id}/media/{kind}")
def project_media(project_id: str, kind: str) -> FileResponse:
    project = get_project(project_id)
    paths = {
        "input": project.get("input"), "thumbnail": project.get("thumbnail"),
        "output": project.get("output", {}).get("video"), "subtitles": project.get("output", {}).get("subtitles"),
    }
    selected = paths.get(kind)
    if not selected or not Path(selected).exists():
        raise HTTPException(404)
    media_type = "video/mp4" if kind in {"input", "output"} else "image/jpeg" if kind == "thumbnail" else "application/x-subrip"
    return FileResponse(selected, media_type=media_type, filename=Path(selected).name)


@app.get("/api/projects/{project_id}/download/{kind}")
def project_download(project_id: str, kind: str) -> FileResponse:
    project = get_project(project_id)
    if project.get("status") != "complete":
        raise HTTPException(409, "Download is blocked until all quality gates pass.")
    paths = {
        "output": project.get("output", {}).get("video"),
        "subtitles": project.get("output", {}).get("subtitles"),
    }
    selected = paths.get(kind)
    if not selected or not Path(selected).exists():
        raise HTTPException(404)
    media_type = "video/mp4" if kind == "output" else "application/x-subrip"
    return FileResponse(selected, media_type=media_type, filename=Path(selected).name)


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
