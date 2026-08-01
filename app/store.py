from __future__ import annotations

import json
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import PROJECTS_ROOT


_lock = threading.RLock()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def project_dir(project_id: str) -> Path:
    return PROJECTS_ROOT / project_id


def project_file(project_id: str) -> Path:
    return project_dir(project_id) / "project.json"


def delete_project(project_id: str) -> None:
    """Delete one project directory after proving it is inside the project root."""
    with _lock:
        root = PROJECTS_ROOT.resolve()
        folder = project_dir(project_id).resolve()
        if folder.parent != root or folder.name != project_id or not project_file(project_id).is_file():
            raise FileNotFoundError(project_id)
        shutil.rmtree(folder)


def create_project(filename: str) -> dict[str, Any]:
    project_id = uuid.uuid4().hex[:12]
    folder = project_dir(project_id)
    for name in ("input", "work", "output", "logs", "preview"):
        (folder / name).mkdir(parents=True, exist_ok=True)
    stamp = now()
    project = {
        "id": project_id,
        "name": Path(filename).stem,
        "filename": filename,
        "status": "uploaded",
        "stage": "Uploaded",
        "progress": 0,
        "created_at": stamp,
        "updated_at": stamp,
        "media": {},
        "analysis": {"source_language": "auto", "speaker_count": "auto", "speakers": [], "segments": []},
        "render": {},
        "quality": {},
        "error": None,
        "exports": [],
        "events": [],
    }
    save_project(project)
    return project


def archive_current_export(project: dict[str, Any]) -> None:
    """Keep the currently referenced files before the project moves to a newer result."""
    current = project.get("output") or {}
    video = current.get("video")
    if not video:
        return
    exports = project.setdefault("exports", [])
    if any(item.get("video") == video for item in exports):
        return
    exports.append({
        **current,
        "created_at": current.get("created_at") or project.get("updated_at") or now(),
        "render": dict(project.get("render") or {}),
        "quality": dict(project.get("quality") or {}),
    })


def invalidate_current_export(project: dict[str, Any], stage: str) -> None:
    """Archive a stale result and require a new render without deleting any media."""
    archive_current_export(project)
    project.update({
        "status": "review",
        "stage": stage,
        "progress": 100,
        "output": {},
        "quality": {},
        "error": None,
    })


def load_project(project_id: str) -> dict[str, Any]:
    with _lock:
        path = project_file(project_id)
        if not path.exists():
            raise FileNotFoundError(project_id)
        return json.loads(path.read_text(encoding="utf-8"))


def save_project(project: dict[str, Any]) -> dict[str, Any]:
    with _lock:
        project["updated_at"] = now()
        path = project_file(project["id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(project, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
        return project


def update_project(project_id: str, **changes: Any) -> dict[str, Any]:
    project = load_project(project_id)
    project.update(changes)
    return save_project(project)


def add_event(project_id: str, level: str, message: str) -> dict[str, Any]:
    project = load_project(project_id)
    project.setdefault("events", []).append({"at": now(), "level": level, "message": message})
    project["events"] = project["events"][-200:]
    return save_project(project)


def list_projects() -> list[dict[str, Any]]:
    projects = []
    for path in PROJECTS_ROOT.glob("*/project.json"):
        try:
            projects.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(projects, key=lambda item: item.get("updated_at", ""), reverse=True)
