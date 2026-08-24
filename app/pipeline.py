from __future__ import annotations

import json
import os
import queue
import re
import shutil
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Callable

from . import config
from .config import (
    ASR_COMPUTE_TYPE,
    ASR_DEVICE,
    ASR_MODEL,
    DEMUCS_DEVICE,
    HYMT_MODEL,
    HYMT_PYTHON,
    HYMT_ROOT,
    LINLY_PYTHON,
    LINLY_ROOT,
    MOCK_MODE,
    MUSETALK_AVAILABLE,
    MUSETALK_PYTHON,
    MUSETALK_ROOT,
    QWEN_PYTHON,
    QWEN_ROOT,
    SEEDVC_FP16,
    SEEDVC_PYTHON,
    SEEDVC_ROOT,
    TORCH_DEVICE,
    TORCH_DTYPE,
    TORCH_HOME,
)
from .diarization import split_segments_on_word_gaps
from .media import probe
from .store import add_event, archive_current_export, list_projects, load_project, now, project_dir, save_project

WORKERS = config.ROOT / "workers"


class JobCancelled(RuntimeError):
    """Raised when a user stops an active or queued project task."""


def offline_environment(extra: dict | None = None) -> dict[str, str]:
    """Build the deterministic local-only environment shared by all model workers."""
    merged = os.environ.copy()
    merged.update({
        "PYTHONUTF8": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
    })
    if extra:
        merged.update(extra)
    return merged


def input_video(project_id: str) -> Path:
    files = list((project_dir(project_id) / "input").iterdir())
    if not files:
        raise RuntimeError("Input video is missing")
    return files[0]


def set_stage(project_id: str, stage: str, progress: int) -> None:
    jobs.raise_if_cancelled(project_id)
    project = load_project(project_id)
    project.update({"stage": stage, "progress": progress, "error": None})
    save_project(project)
    add_event(project_id, "info", stage)


def set_progress(project_id: str, progress: int, stage: str | None = None) -> None:
    """Persist quiet, monotonic worker progress without flooding project activity."""
    jobs.raise_if_cancelled(project_id)
    project = load_project(project_id)
    progress = max(int(project.get("progress", 0)), min(99, int(progress)))
    changes = {"progress": progress, "error": None}
    if stage:
        changes["stage"] = stage
    project.update(changes)
    save_project(project)


def execute(
    project_id: str,
    name: str,
    command: list[str],
    cwd: Path | None = None,
    env: dict | None = None,
    progress_range: tuple[int, int] | None = None,
    progress_path: Path | None = None,
) -> None:
    jobs.raise_if_cancelled(project_id)
    log = project_dir(project_id) / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    merged = offline_environment(env)
    options = {"cwd": cwd, "env": merged, "text": True, "encoding": "utf-8", "errors": "replace"}
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options["start_new_session"] = True
    if progress_path:
        progress_path.unlink(missing_ok=True)
    with log.open("w", encoding="utf-8", errors="replace") as log_handle:
        options.update({"stdout": log_handle, "stderr": subprocess.STDOUT})
        process = subprocess.Popen(command, **options)
        jobs.register_process(project_id, process)
        try:
            last_progress = None
            while process.poll() is None:
                jobs.raise_if_cancelled(project_id)
                marker = None
                try:
                    if progress_range and progress_path and progress_path.is_file():
                        payload = json.loads(progress_path.read_text(encoding="utf-8"))
                        current = max(0, int(payload.get("current", 0)))
                        total = max(1, int(payload.get("total", 1)))
                        mapped = progress_range[0] + round((progress_range[1] - progress_range[0]) * min(1, current / total))
                        marker = (mapped, str(payload.get("label") or ""))
                    elif progress_range and log.is_file():
                        matches = re.findall(r"(?<!\d)(\d{1,3})%(?!\d)", log.read_text(encoding="utf-8", errors="replace")[-65_536:])
                        if matches:
                            fraction = min(100, int(matches[-1])) / 100
                            mapped = progress_range[0] + round((progress_range[1] - progress_range[0]) * fraction)
                            marker = (mapped, "")
                    if marker is not None and marker != last_progress:
                        label = marker[1] or load_project(project_id).get("stage")
                        set_progress(project_id, marker[0], label)
                        last_progress = marker
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
                time.sleep(.5)
        finally:
            jobs.unregister_process(project_id, process)
    if jobs.is_cancelled(project_id):
        raise JobCancelled(project_id)
    if progress_range:
        try:
            if progress_path and progress_path.is_file():
                payload = json.loads(progress_path.read_text(encoding="utf-8"))
                current = max(0, int(payload.get("current", 0)))
                total = max(1, int(payload.get("total", 1)))
                mapped = progress_range[0] + round((progress_range[1] - progress_range[0]) * min(1, current / total))
                set_progress(project_id, mapped, str(payload.get("label") or load_project(project_id).get("stage")))
            else:
                matches = re.findall(r"(?<!\d)(\d{1,3})%(?!\d)", log.read_text(encoding="utf-8", errors="replace")[-65_536:])
                if matches:
                    fraction = min(100, int(matches[-1])) / 100
                    mapped = progress_range[0] + round((progress_range[1] - progress_range[0]) * fraction)
                    set_progress(project_id, mapped)
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    if process.returncode:
        details = log.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(f"{name}: {details[-2000:]}")


def analyze(project_id: str, request: dict) -> None:
    project = load_project(project_id)
    project["status"] = "analyzing"
    project["analysis"].update(request)
    save_project(project)
    folder = project_dir(project_id)
    source = input_video(project_id)
    work = folder / "work"
    try:
        if MOCK_MODE:
            set_stage(project_id, "Mock transcription", 40)
            time.sleep(.3)
            segments = [
                {"id": 0, "start": 0.0, "end": 2.8, "text": "This is the first transcribed line.", "speaker": "SPEAKER_00", "words": []},
                {"id": 1, "start": 3.1, "end": 5.5, "text": "You can correct it before translation.", "speaker": "SPEAKER_00", "words": []},
            ]
            result = {"language": "en", "language_probability": .99, "speakers": [{"id": "SPEAKER_00", "label": "Speaker 1", "reference": "", "profile": "A natural conversational adult voice."}], "segments": segments}
        else:
            set_stage(project_id, "Extracting audio", 12)
            audio = work / "audio.wav"
            execute(project_id, "extract_audio", ["ffmpeg", "-y", "-v", "error", "-i", str(source), "-vn", "-ar", "44100", "-ac", "2", str(audio)])
            set_stage(project_id, "Separating speech and background", 26)
            demucs_root = work / "demucs"
            execute(project_id, "demucs", [str(LINLY_PYTHON), "-m", "demucs", "--two-stems", "vocals", "-n", "htdemucs", "-d", DEMUCS_DEVICE, "-o", str(demucs_root), str(audio)], cwd=LINLY_ROOT, env={"TORCH_HOME": str(TORCH_HOME)})
            demucs_job = demucs_root / "htdemucs" / "audio"
            shutil.copy2(demucs_job / "vocals.wav", work / "vocals.wav")
            shutil.copy2(demucs_job / "no_vocals.wav", work / "background.wav")
            set_stage(project_id, "Transcribing speech", 48)
            transcript = work / "transcript.json"
            command = [
                str(LINLY_PYTHON), str(WORKERS / "transcribe.py"), str(work / "vocals.wav"), str(transcript),
                "--model", str(ASR_MODEL), "--device", ASR_DEVICE, "--compute-type", ASR_COMPUTE_TYPE,
            ]
            if request.get("source_language", "auto") != "auto":
                command.extend(["--language", request["source_language"]])
            execute(project_id, "transcribe", command, cwd=LINLY_ROOT)
            set_stage(project_id, "Identifying speakers", 72)
            analyzed = work / "analysis.json"
            count = str(request.get("speaker_count", "auto"))
            execute(project_id, "speakers", [str(SEEDVC_PYTHON), str(WORKERS / "cluster_speakers.py"), str(work / "vocals.wav"), str(transcript), str(analyzed), str(work / "references"), "--count", count], cwd=SEEDVC_ROOT)
            result = json.loads(analyzed.read_text(encoding="utf-8"))
        result["segments"] = split_segments_on_word_gaps(result["segments"])
        if not result["segments"]:
            raise RuntimeError("No speech was detected. Check that the video contains an audible dialogue track.")
        project = load_project(project_id)
        project["analysis"].update({
            "detected_language": result["language"], "language_probability": result.get("language_probability"),
            "speakers": result["speakers"], "segments": result["segments"], "references_trimmed": True,
        })
        project.update({"status": "review", "stage": "Review transcript and speakers", "progress": 100})
        save_project(project)
        add_event(project_id, "success", f"Analysis complete: {len(result['segments'])} lines, {len(result['speakers'])} speakers")
    except JobCancelled:
        return
    except Exception as exc:
        fail(project_id, exc)


def write_srt(lines: list[dict], path: Path) -> None:
    def timestamp(value: float) -> str:
        milliseconds = round(value * 1000)
        hours, milliseconds = divmod(milliseconds, 3_600_000)
        minutes, milliseconds = divmod(milliseconds, 60_000)
        seconds, milliseconds = divmod(milliseconds, 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    blocks = []
    for index, item in enumerate(lines, 1):
        text = item["translation"].strip()
        if len(text) > 44 and " " in text:
            words, rows, current = text.split(), [], ""
            for word in words:
                if current and len(current) + len(word) + 1 > 40 and len(rows) < 1:
                    rows.append(current)
                    current = word
                else:
                    current = f"{current} {word}".strip()
            rows.append(current)
            text = "\n".join(rows)
        blocks.append(f"{index}\n{timestamp(item['start'])} --> {timestamp(item['end'])}\n{text}\n")
    path.write_text("\n".join(blocks), encoding="utf-8-sig")


def write_ass(lines: list[dict], path: Path, width: int, height: int, request: dict) -> None:
    """Write positioned ASS captions using the same style controls as the UI."""
    style = config.subtitle_style_values(
        request["subtitle_style"], request.get("subtitle_size", "medium"),
        request.get("subtitle_color", "white"), request.get("subtitle_scale", 1.0),
    )
    x = round(width * float(request.get("subtitle_x", 50)) / 100)
    y = round(height * float(request.get("subtitle_y", 88)) / 100)
    font_size = round(float(style["FontSize"]) * height / 288, 2)

    def stamp(value: float) -> str:
        centiseconds = round(value * 100)
        hours, centiseconds = divmod(centiseconds, 360000)
        minutes, centiseconds = divmod(centiseconds, 6000)
        seconds, centiseconds = divmod(centiseconds, 100)
        return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"

    bold = "-1" if style.get("Bold") == "1" else "0"
    header = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {width}", f"PlayResY: {height}",
        "WrapStyle: 0", "ScaledBorderAndShadow: yes", "", "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,Manrope,{font_size},{style['PrimaryColour']},&H000000FF,{style['OutlineColour']},&H00000000,{bold},0,0,0,100,100,0,0,{style.get('BorderStyle','1')},{style.get('Outline','1.6')},{style.get('Shadow','0')},5,0,0,0,1",
        "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    events = []
    for item in lines:
        text = (
            item["translation"].strip()
            .replace("\\", r"\\")
            .replace("{", r"\{")
            .replace("}", r"\}")
            .replace("\n", r"\N")
        )
        events.append(f"Dialogue: 0,{stamp(item['start'])},{stamp(item['end'])},Default,,0,0,0,,{{\\an5\\pos({x},{y})}}{text}")
    path.write_text("\n".join(header + events) + "\n", encoding="utf-8-sig")


def normalized_words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def safe_output_name(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", value).strip(" ._")
    return cleaned[:120] or "dub"


def apply_lip_sync(project_id: str, mixed_video: Path, work: Path, run_id: str) -> Path:
    """Create a recoverable MuseTalk master without changing the accepted audio mix."""
    if MOCK_MODE:
        return mixed_video
    if not MUSETALK_AVAILABLE:
        raise RuntimeError("Local lip sync is not installed. Run setup again or turn Lip sync off.")
    lip_root = work / "lipsync" / run_id
    results = lip_root / "results"
    lip_root.mkdir(parents=True, exist_ok=True)
    mixed_audio = lip_root / "accepted_mix.wav"
    execute(project_id, "lipsync_audio", [
        "ffmpeg", "-y", "-hide_banner", "-i", str(mixed_video), "-map", "0:a:0",
        "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1", str(mixed_audio),
    ])
    result_name = f"lip-sync-{run_id}.mp4"
    inference_config = lip_root / "inference.json"
    inference_config.write_text(json.dumps({
        "dub": {
            "video_path": str(mixed_video), "audio_path": str(mixed_audio),
            "bbox_shift": 0, "result_name": result_name,
        }
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    command = [
        str(MUSETALK_PYTHON), "-m", "scripts.inference",
        "--inference_config", str(inference_config), "--result_dir", str(results),
        "--unet_model_path", str(MUSETALK_ROOT / "models" / "musetalkV15" / "unet.pth"),
        "--unet_config", str(MUSETALK_ROOT / "models" / "musetalkV15" / "musetalk.json"),
        "--whisper_dir", str(MUSETALK_ROOT / "models" / "whisper"),
        "--version", "v15", "--batch_size", "8", "--parsing_mode", "jaw",
    ]
    if config.BACKEND == "cuda":
        command.append("--use_float16")
    execute(project_id, "lipsync", command, cwd=MUSETALK_ROOT, progress_range=(87, 93))
    generated = results / "v15" / result_name
    if not generated.is_file() or generated.stat().st_size < 1024:
        raise RuntimeError("Lip sync could not find and process a usable face in this video.")
    master = lip_root / "lip_sync_master.mp4"
    execute(project_id, "lipsync_remux", [
        "ffmpeg", "-y", "-hide_banner", "-i", str(generated), "-i", str(mixed_video),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "copy",
        "-shortest", "-movflags", "+faststart", str(master),
    ])
    source_media, master_media = probe(mixed_video), probe(master)
    if (source_media.get("width"), source_media.get("height")) != (master_media.get("width"), master_media.get("height")):
        raise RuntimeError("Lip sync changed the video resolution; the safe pre-lip-sync export was preserved.")
    return master


def select_background(work: Path, source: Path, mock_mode: bool) -> Path:
    background = work / "background.wav"
    if background.exists():
        return background
    if mock_mode:
        return source
    raise RuntimeError("Separated background audio is missing; refusing to mix the original speech under the dub")


def redistribute_line_timing(
    lines: list[dict], index: int, required_speed: float, media_duration: float,
    target_speed: float = 1.25, reserve: float = 0.12,
) -> dict | None:
    """Borrow adjacent silence before shortening a translation beyond natural speech."""
    line = lines[index]
    start, end = float(line["start"]), float(line["end"])
    slot = max(end - start, 0.01)
    needed = max(0.0, slot * required_speed / target_speed - slot)
    if needed <= 0.005:
        return None
    previous_end = float(lines[index - 1]["end"]) if index else 0.0
    next_start = float(lines[index + 1]["start"]) if index + 1 < len(lines) else media_duration
    before_available = max(0.0, start - previous_end - reserve)
    after_available = max(0.0, next_start - end - reserve)
    if before_available + after_available + 0.001 < needed:
        return None
    before = min(before_available, needed / 2)
    after = min(after_available, needed - before)
    before += min(before_available - before, needed - before - after)
    line["start"] = round(start - before, 3)
    line["end"] = round(end + after, 3)
    return {
        "index": index, "old_start": start, "old_end": end,
        "new_start": line["start"], "new_end": line["end"],
    }


def render(project_id: str, request: dict) -> None:
    project = load_project(project_id)
    project.update({"status": "rendering", "render": request, "progress": 0})
    save_project(project)
    folder = project_dir(project_id)
    work = folder / "work"
    source = input_video(project_id)
    try:
        speakers = project["analysis"]["speakers"]
        original_segments = project["analysis"]["segments"]
        segments = split_segments_on_word_gaps(original_segments)
        segments_changed = segments != original_segments
        references_need_rebuild = segments_changed or not project["analysis"].get("references_trimmed")
        if segments_changed:
            project["analysis"]["segments"] = segments
        if references_need_rebuild:
            references = work / "references"
            if references.exists():
                references.replace(work / f"references_pre_gap_split_{uuid.uuid4().hex[:8]}")
            reference_payload = work / "reference_segments.json"
            reference_payload.write_text(json.dumps({"segments": segments, "speakers": speakers}, ensure_ascii=False, indent=2), encoding="utf-8")
            execute(project_id, "rebuild_references", [
                str(SEEDVC_PYTHON), str(WORKERS / "rebuild_references.py"), str(work / "vocals.wav"),
                str(reference_payload), str(references),
            ], cwd=SEEDVC_ROOT)
            project["analysis"]["references_trimmed"] = True
            save_project(project)
        if references_need_rebuild:
            stale_synthesis = work / "synthesis"
            if stale_synthesis.exists():
                stale_synthesis.replace(work / f"synthesis_pre_gap_split_{uuid.uuid4().hex[:8]}")
        if segments_changed:
            add_event(project_id, "warning", "Split transcript lines around long internal pauses before dubbing")
        elif references_need_rebuild:
            add_event(project_id, "warning", "Rebuilt speaker references from active speech windows")
        if not segments:
            raise RuntimeError("No transcript to render")
        if request["voice_mode"] == "catalog" and len(speakers) != 1:
            raise RuntimeError("Catalog voice can only be used for a single-speaker video")
        target = config.language(request["target_language"])
        set_stage(project_id, "Translating and adapting lines", 8)
        translation = work / "translation.json"
        translation_override = work / "translation_override.json"
        translation_review = work / "translation_review.json"
        analysis_payload = {"segments": segments}
        analysis_for_translation = work / "analysis_for_translation.json"
        analysis_for_translation.write_text(json.dumps(analysis_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if translation_review.exists() and translation.exists():
            overridden = json.loads(translation.read_text(encoding="utf-8"))
            reviewed = json.loads(translation_review.read_text(encoding="utf-8"))
            for raw_index, reviewed_text in reviewed.items():
                index = int(raw_index)
                if not 0 <= index < len(overridden) or not str(reviewed_text).strip():
                    raise RuntimeError(f"Invalid reviewed translation line: {raw_index}")
                overridden[index]["translation"] = str(reviewed_text).strip()
            translation.write_text(json.dumps(overridden, ensure_ascii=False, indent=2), encoding="utf-8")
            add_event(project_id, "info", "Using reviewed translations and skipping the full translation pass")
        elif translation_override.exists():
            overridden = json.loads(translation_override.read_text(encoding="utf-8"))
            if len(overridden) != len(segments) or any("translation" not in item for item in overridden):
                raise RuntimeError("Translation override does not match the current transcript")
            translation.write_text(json.dumps(overridden, ensure_ascii=False, indent=2), encoding="utf-8")
            add_event(project_id, "info", "Using the reviewed timing-faithful translation")
        elif MOCK_MODE:
            translated = [dict(item, translation=f"[{target['label']}] {item['text']}") for item in segments]
            translation.write_text(json.dumps(translated, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            detected = project["analysis"].get("detected_language", project["analysis"].get("source_language", "Russian"))
            source_name = next((item["hymt"] for item in config.LANGUAGES if item["id"] == detected), detected)
            execute(project_id, "translate", [
                str(HYMT_PYTHON), str(WORKERS / "translate.py"), str(analysis_for_translation), str(translation),
                "--model", str(HYMT_MODEL), "--source-language", source_name, "--target-language", target["hymt"],
                "--device", TORCH_DEVICE, "--dtype", TORCH_DTYPE,
                "--progress", str(work / "translation_progress.json"),
            ], cwd=HYMT_ROOT, progress_range=(8, 29), progress_path=work / "translation_progress.json")
        set_stage(project_id, "Generating native speech", 30)
        persona = config.voice(request.get("voice_id") or "warm_female")
        qwen_config = {
            "job_dir": str(folder), "translation_path": str(translation), "duration": project["media"]["duration"],
            "qwen_root": str(QWEN_ROOT), "tts_language": target["tts"], "sample_text": target["sample"],
            "voice_mode": request["voice_mode"], "voice_description": persona["description"],
            "expression": request["expression"], "speakers": speakers,
            "device": TORCH_DEVICE, "dtype": TORCH_DTYPE,
            "progress_path": str(work / "synthesis" / "speech_progress.json"),
        }
        qwen_config_path = work / "qwen_config.json"
        qwen_config_path.write_text(json.dumps(qwen_config, ensure_ascii=False, indent=2), encoding="utf-8")
        if MOCK_MODE:
            execute(project_id, "mock_dub", ["ffmpeg", "-y", "-v", "error", "-i", str(source), "-vn", "-ar", "24000", "-ac", "1", str(work / "dub.wav")])
        else:
            qwen_command = [str(QWEN_PYTHON), str(WORKERS / "qwen_voice.py"), "synthesize", str(qwen_config_path)]
            for timing_attempt in range(8):
                try:
                    # Two wording passes are enough. Beyond that, a restrained
                    # 0.78-1.50x timing fit sounds better than adding filler or
                    # repeatedly regenerating the same line.
                    qwen_config["timing_fallback"] = timing_attempt >= 2
                    qwen_config_path.write_text(json.dumps(qwen_config, ensure_ascii=False, indent=2), encoding="utf-8")
                    speech_range = (30, 44) if timing_attempt == 0 else (50, 54)
                    execute(
                        project_id, f"qwen_{timing_attempt + 1}", qwen_command, cwd=QWEN_ROOT,
                        progress_range=speech_range, progress_path=work / "synthesis" / "speech_progress.json",
                    )
                    break
                except RuntimeError as exc:
                    timing_issues_path = work / "synthesis" / "timing_issues.json"
                    if timing_issues_path.exists() and timing_attempt < 7:
                        issues = json.loads(timing_issues_path.read_text(encoding="utf-8"))
                        translated_lines = json.loads(translation.read_text(encoding="utf-8"))
                        adaptations = []
                        adjusted_indexes = set()
                        for issue in issues:
                            line_index = int(issue["index"])
                            error = issue["error"]
                            underfill = re.search(r"Line fills ([\d.]+) of its time slot", error)
                            if underfill:
                                fill_ratio = float(underfill.group(1))
                                adaptations.append({
                                    "index": line_index, "mode": "expand",
                                    "ratio": min(1.95, max(1.15, .9 / max(fill_ratio, .01))),
                                })
                                continue
                            overfill = re.search(r"Line requires ([\d.]+)x speed", error)
                            if not overfill:
                                raise
                            required_speed = float(overfill.group(1))
                            adjustment = redistribute_line_timing(
                                translated_lines, line_index, required_speed, project["media"]["duration"]
                            )
                            if adjustment:
                                adjusted_indexes.add(line_index)
                            else:
                                adaptations.append({
                                    "index": line_index, "mode": "shorten",
                                    "ratio": min(.88, 1.35 / required_speed * .92),
                                })
                        if adjusted_indexes:
                            translation.write_text(json.dumps(translated_lines, ensure_ascii=False, indent=2), encoding="utf-8")
                            add_event(project_id, "warning", f"Adjusted timing with adjacent silence for {len(adjusted_indexes)} lines")
                        if adaptations:
                            adaptation_path = work / "synthesis" / "timing_adaptations.json"
                            adaptation_path.write_text(json.dumps(adaptations, ensure_ascii=False, indent=2), encoding="utf-8")
                            expanded = sum(item["mode"] == "expand" for item in adaptations)
                            shortened = len(adaptations) - expanded
                            add_event(project_id, "warning", f"Adapting {len(adaptations)} lines for visible speech coverage: {expanded} expanded, {shortened} shortened")
                            execute(project_id, f"adapt_timing_{timing_attempt + 1}", [
                                str(HYMT_PYTHON), str(WORKERS / "shorten_translation.py"), str(translation),
                                "--model", str(HYMT_MODEL), "--language", target["hymt"],
                                "--requests", str(adaptation_path), "--device", TORCH_DEVICE, "--dtype", TORCH_DTYPE,
                                "--progress", str(work / "synthesis" / "adapt_progress.json"),
                            ], cwd=HYMT_ROOT, progress_range=(44, 50), progress_path=work / "synthesis" / "adapt_progress.json")
                        synthesis = work / "synthesis"
                        for line_index in adjusted_indexes | {int(item["index"]) for item in adaptations}:
                            for candidate in (
                                synthesis / "raw" / f"{line_index:04d}_{segments[line_index]['speaker']}.wav",
                                synthesis / "raw" / f"{line_index:04d}_{segments[line_index]['speaker']}.txt",
                                synthesis / "fitted" / f"{line_index:04d}_{segments[line_index]['speaker']}.wav",
                                synthesis / "fitted" / f"{line_index:04d}_{segments[line_index]['speaker']}.trim.wav",
                            ):
                                candidate.unlink(missing_ok=True)
                        timing_issues_path.unlink(missing_ok=True)
                        continue
                    underfill = re.search(r"Line fills ([\d.]+) of its time slot; expand its translation: (\d{4})_", str(exc))
                    if underfill and timing_attempt < 7:
                        fill_ratio = float(underfill.group(1))
                        line_index = int(underfill.group(2))
                        ratio = min(1.95, max(1.15, .9 / max(fill_ratio, .01)))
                        add_event(project_id, "warning", f"Line {line_index + 1} leaves visible speech uncovered; expanding and synthesizing again")
                        execute(project_id, f"expand_{line_index}_{timing_attempt + 1}", [
                            str(HYMT_PYTHON), str(WORKERS / "shorten_translation.py"), str(translation),
                            "--index", str(line_index), "--model", str(HYMT_MODEL),
                            "--language", target["hymt"], "--ratio", str(ratio), "--mode", "expand",
                            "--device", TORCH_DEVICE, "--dtype", TORCH_DTYPE,
                            "--progress", str(work / "synthesis" / "adapt_progress.json"),
                        ], cwd=HYMT_ROOT, progress_range=(44, 50), progress_path=work / "synthesis" / "adapt_progress.json")
                        synthesis = work / "synthesis"
                        for candidate in (
                            synthesis / "raw" / f"{line_index:04d}_{segments[line_index]['speaker']}.wav",
                            synthesis / "raw" / f"{line_index:04d}_{segments[line_index]['speaker']}.txt",
                            synthesis / "fitted" / f"{line_index:04d}_{segments[line_index]['speaker']}.wav",
                            synthesis / "fitted" / f"{line_index:04d}_{segments[line_index]['speaker']}.trim.wav",
                        ):
                            candidate.unlink(missing_ok=True)
                        continue
                    match = re.search(r"Line requires ([\d.]+)x speed; shorten its translation: (\d{4})_", str(exc))
                    if not match or timing_attempt == 7:
                        raise
                    required_speed = float(match.group(1))
                    line_index = int(match.group(2))
                    translated_lines = json.loads(translation.read_text(encoding="utf-8"))
                    adjustment = redistribute_line_timing(
                        translated_lines, line_index, required_speed, project["media"]["duration"]
                    )
                    synthesis = work / "synthesis"
                    fitted_candidates = (
                        synthesis / "fitted" / f"{line_index:04d}_{segments[line_index]['speaker']}.wav",
                        synthesis / "fitted" / f"{line_index:04d}_{segments[line_index]['speaker']}.trim.wav",
                    )
                    if adjustment:
                        translation.write_text(json.dumps(translated_lines, ensure_ascii=False, indent=2), encoding="utf-8")
                        add_event(
                            project_id, "warning",
                            f"Line {line_index + 1} used adjacent silence to preserve natural speech speed",
                        )
                        for candidate in fitted_candidates:
                            candidate.unlink(missing_ok=True)
                        continue
                    ratio = min(.88, 1.35 / required_speed * .92)
                    add_event(project_id, "warning", f"Line {line_index + 1} exceeds its time slot; shortening and synthesizing again")
                    execute(project_id, f"shorten_{line_index}_{timing_attempt + 1}", [
                        str(HYMT_PYTHON), str(WORKERS / "shorten_translation.py"), str(translation),
                        "--index", str(line_index), "--model", str(HYMT_MODEL),
                        "--language", target["hymt"], "--ratio", str(ratio),
                        "--device", TORCH_DEVICE, "--dtype", TORCH_DTYPE,
                        "--progress", str(work / "synthesis" / "adapt_progress.json"),
                    ], cwd=HYMT_ROOT, progress_range=(44, 50), progress_path=work / "synthesis" / "adapt_progress.json")
                    for candidate in (
                        synthesis / "raw" / f"{line_index:04d}_{segments[line_index]['speaker']}.wav",
                        synthesis / "raw" / f"{line_index:04d}_{segments[line_index]['speaker']}.txt",
                        *fitted_candidates,
                    ):
                        candidate.unlink(missing_ok=True)
            manifest_path = work / "synthesis" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if request["voice_mode"] == "clone":
                set_stage(project_id, "Matching source voice profiles", 56)
                converted_root = work / "synthesis" / "converted"
                converted_root.mkdir(parents=True, exist_ok=True)
                active_roles = []
                role_count = max(1, len(manifest["roles"]))
                for role_index, role in enumerate(manifest["roles"]):
                    role_start = 56 + round(17 * role_index / role_count)
                    role_end = 56 + round(17 * (role_index + 1) / role_count)
                    set_progress(
                        project_id, role_start,
                        f"Matching voice profiles · {role_index + 1} of {role_count}",
                    )
                    mapping = json.loads(Path(role["mapping"]).read_text(encoding="utf-8"))
                    if not mapping or Path(role["source"]).stat().st_size <= 46:
                        add_event(project_id, "warning", f"Skipped empty detected role {role['speaker']}")
                        continue
                    role_dir = converted_root / role["speaker"]
                    role_dir.mkdir(parents=True, exist_ok=True)
                    execute(project_id, f"seedvc_{role['speaker']}", [
                        str(SEEDVC_PYTHON), str(SEEDVC_ROOT / "inference.py"),
                        "--source", role["source"], "--target", role["reference"], "--output", str(role_dir),
                        "--diffusion-steps", "40", "--length-adjust", "1.0", "--inference-cfg-rate", "0.90",
                        "--f0-condition", "False", "--auto-f0-adjust", "False", "--semi-tone-shift", "0",
                        "--fp16", str(SEEDVC_FP16),
                    ], cwd=SEEDVC_ROOT, progress_range=(role_start, role_end))
                    outputs = sorted(role_dir.glob("*.wav"), key=lambda path: path.stat().st_mtime, reverse=True)
                    if not outputs:
                        raise RuntimeError(f"Seed-VC produced no audio for {role['speaker']}")
                    role["converted"] = str(outputs[0])
                    active_roles.append(role)
                    set_progress(
                        project_id, role_end,
                        f"Matching voice profiles · {role_index + 1} of {role_count}",
                    )
                if not active_roles:
                    raise RuntimeError("No speaker roles contain translated speech")
                manifest["roles"] = active_roles
                manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
                set_progress(project_id, 74, "Assembling translated voices")
                execute(project_id, "assemble_clone", [str(QWEN_PYTHON), str(WORKERS / "assemble_clone.py"), str(manifest_path), str(translation), str(work / "dub.wav"), "--duration", str(project["media"]["duration"])], cwd=QWEN_ROOT)

        set_stage(project_id, "Captions and final mix", 76)
        translated = json.loads(translation.read_text(encoding="utf-8"))
        subtitles = work / "subtitles.srt"
        write_srt(translated, subtitles)
        positioned_subtitles = work / "subtitles.ass"
        write_ass(translated, positioned_subtitles, project["media"]["width"], project["media"]["height"], request)
        run_id = re.sub(r"[^a-zA-Z0-9_-]", "", request.get("run_id", "")) or uuid.uuid4().hex[:10]
        output_stem = f"{safe_output_name(project['name'])}_{request['target_language']}_dub_{run_id}"
        output = folder / "output" / f"{output_stem}.mp4"
        exported_subtitles = folder / "output" / f"{output_stem}.srt"
        background = select_background(work, source, MOCK_MODE)
        if not (work / "dub.wav").exists():
            raise RuntimeError("Synthesized dub audio is missing")
        audio_filter = f"[1:a]volume={request['background_volume']:.3f}[bg];[2:a]volume=1.04,highpass=f=60,lowpass=f=11000,acompressor=threshold=-19dB:ratio=1.8:attack=15:release=140,alimiter=limit=0.94[vo];[bg][vo]amix=inputs=2:duration=longest:normalize=0,loudnorm=I=-15:TP=-2:LRA=7[a]"
        lip_sync_enabled = bool(request.get("lip_sync_enabled"))
        mixed_output = output
        if lip_sync_enabled:
            mixed_output = work / "lipsync" / run_id / "pre_lip_sync_mix.mp4"
            mixed_output.parent.mkdir(parents=True, exist_ok=True)
        command = ["ffmpeg", "-y", "-hide_banner", "-i", str(source), "-i", str(background), "-i", str(work / "dub.wav"), "-filter_complex", audio_filter]
        if not lip_sync_enabled and request["subtitle_enabled"] and request["burn_subtitles"]:
            fonts_dir = (config.ROOT / "static" / "fonts").as_posix().replace(":", r"\:")
            command.extend(["-vf", f"subtitles=work/subtitles.ass:fontsdir='{fonts_dir}'"])
        command.extend(["-map", "0:v:0", "-map", "[a]", *config.video_encoder_args(request["quality"], source_bitrate=project["media"].get("video_bitrate")), "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-ac", "2", "-t", str(project["media"]["duration"]), "-movflags", "+faststart", str(mixed_output)])
        set_progress(project_id, 79, "Mixing translated speech and background")
        execute(project_id, "render", command, cwd=folder)
        if lip_sync_enabled:
            set_stage(project_id, "Synchronizing lip movement", 87)
            lip_sync_master = apply_lip_sync(project_id, mixed_output, work, run_id)
            request["lip_sync_master"] = str(lip_sync_master)
            if request["subtitle_enabled"] and request["burn_subtitles"]:
                fonts_dir = (config.ROOT / "static" / "fonts").as_posix().replace(":", r"\:")
                execute(project_id, "lipsync_captions", [
                    "ffmpeg", "-y", "-hide_banner", "-i", str(lip_sync_master),
                    "-vf", f"subtitles=work/subtitles.ass:fontsdir='{fonts_dir}'",
                    "-map", "0:v:0", "-map", "0:a:0",
                    *config.video_encoder_args(request["quality"], source_bitrate=project["media"].get("video_bitrate")),
                    "-c:a", "copy", "-t", str(project["media"]["duration"]),
                    "-movflags", "+faststart", str(output),
                ], cwd=folder)
            else:
                shutil.copy2(lip_sync_master, output)
            set_progress(project_id, 94, "Lip sync and final video encoded")
        else:
            set_progress(project_id, 88, "Final video encoded")
        shutil.copy2(subtitles, exported_subtitles)

        set_stage(project_id, "Running quality checks", 95 if lip_sync_enabled else 90)
        quality = {"media": probe(output), "max_speed": None, "asr_coverage": None, "warnings": [], "lip_sync_enabled": lip_sync_enabled}
        report_path = work / "synthesis" / ("clone_timing_report.json" if request["voice_mode"] == "clone" else "manifest.json")
        if report_path.exists():
            report_payload = json.loads(report_path.read_text(encoding="utf-8"))
            report = report_payload.get("report", report_payload) if isinstance(report_payload, dict) else report_payload
            speeds = [item.get("speed", 1) for item in report]
            quality["max_speed"] = max(speeds, default=1)
        if not MOCK_MODE:
            qa_asr = work / "qa_asr.json"
            execute(project_id, "qa_asr", [
                str(LINLY_PYTHON), str(WORKERS / "transcribe.py"), str(work / "dub.wav"), str(qa_asr),
                "--language", target["asr"], "--model", str(ASR_MODEL),
                "--device", ASR_DEVICE, "--compute-type", ASR_COMPUTE_TYPE,
            ], cwd=LINLY_ROOT)
            set_progress(project_id, 94, "Checking speech coverage")
            recognized = " ".join(item["text"] for item in json.loads(qa_asr.read_text(encoding="utf-8"))["segments"])
            expected = " ".join(item["translation"] for item in translated)
            expected_words = normalized_words(expected)
            quality["asr_coverage"] = round(len(expected_words & normalized_words(recognized)) / max(1, len(expected_words)), 3)
            if quality["asr_coverage"] < .82:
                quality["warnings"].append("ASR coverage is below 82%; review pronunciation and missing words.")
            qa_audio_path = work / "qa_audio.json"
            qa_command = [str(SEEDVC_PYTHON), str(WORKERS / "qa_audio.py"), str(work / "dub.wav"), str(translation), str(output), str(qa_audio_path)]
            source_vocals = work / "vocals.wav"
            if source_vocals.exists():
                qa_command.extend(["--source-vocals", str(source_vocals)])
            manifest_path = work / "synthesis" / "manifest.json"
            if request["voice_mode"] == "clone" and manifest_path.exists():
                qa_command.extend(["--manifest", str(manifest_path)])
            execute(project_id, "qa_audio", qa_command, cwd=SEEDVC_ROOT)
            set_progress(project_id, 97, "Checking voice and loudness")
            audio_quality = json.loads(qa_audio_path.read_text(encoding="utf-8"))
            quality.update({key: value for key, value in audio_quality.items() if key != "warnings"})
            quality["warnings"].extend(audio_quality["warnings"])
            if request["voice_mode"] == "clone":
                measured = set(quality.get("speaker_similarity") or {})
                # Speaker profiles may remain after the user merges/reassigns
                # diarization labels. QA only requires identities that still
                # own at least one translated segment.
                expected_speakers = {item["speaker"] for item in translated}
                missing_speakers = sorted(expected_speakers - measured)
                if missing_speakers:
                    quality["warnings"].append(f"Voice similarity was not measured for: {', '.join(missing_speakers)}")
            if quality["max_speed"] is None:
                quality["warnings"].append("Timing speed was not measured")
            if quality["asr_coverage"] is None:
                quality["warnings"].append("Target-language ASR coverage was not measured")
        execute(project_id, "decode_check", ["ffmpeg", "-v", "error", "-i", str(output), "-f", "null", os.devnull])
        set_progress(project_id, 99, "Verifying final video")
        project = load_project(project_id)
        archive_current_export(project)
        quality["warnings"] = list(dict.fromkeys(quality["warnings"]))
        accepted = not quality["warnings"]
        project.update({
            "status": "complete",
            "stage": "Complete" if accepted else "Complete with quality notices",
            "progress": 100,
            "quality": quality,
            "render": dict(request),
        })
        project["output"] = {
            "video": str(output), "subtitles": str(exported_subtitles),
            "run_id": run_id, "created_at": now(),
        }
        project.setdefault("exports", []).append({
            **project["output"], "render": dict(request), "quality": dict(quality),
        })
        save_project(project)
        if accepted:
            add_event(project_id, "success", "Dub and quality checks complete")
        else:
            add_event(project_id, "warning", "Export ready with quality notices")
    except JobCancelled:
        return
    except Exception as exc:
        fail(project_id, exc)


def reexport_captions(project_id: str, request: dict) -> None:
    """Rebuild only the final container after caption controls change."""
    project = load_project(project_id)
    project.update({"status": "rendering", "render": request})
    save_project(project)
    folder = project_dir(project_id)
    work = folder / "work"
    source = input_video(project_id)
    try:
        translation = work / "translation.json"
        subtitles = work / "subtitles.srt"
        dub = work / "dub.wav"
        if not translation.exists() or not subtitles.exists() or not dub.exists():
            raise RuntimeError("Caption re-export requires a completed dub")
        translated = json.loads(translation.read_text(encoding="utf-8"))
        positioned_subtitles = work / "subtitles.ass"
        write_ass(translated, positioned_subtitles, project["media"]["width"], project["media"]["height"], request)
        set_stage(project_id, "Rendering caption changes", 82)
        run_id = uuid.uuid4().hex[:10]
        output_stem = f"{safe_output_name(project['name'])}_{request['target_language']}_dub_{run_id}"
        output = folder / "output" / f"{output_stem}.mp4"
        exported_subtitles = folder / "output" / f"{output_stem}.srt"
        lip_sync_master_value = request.get("lip_sync_master") if request.get("lip_sync_enabled") else None
        lip_sync_master = Path(lip_sync_master_value) if lip_sync_master_value else None
        if lip_sync_master and lip_sync_master.is_file():
            if request["subtitle_enabled"]:
                fonts_dir = (config.ROOT / "static" / "fonts").as_posix().replace(":", r"\:")
                command = [
                    "ffmpeg", "-y", "-hide_banner", "-i", str(lip_sync_master),
                    "-vf", f"subtitles=work/subtitles.ass:fontsdir='{fonts_dir}'",
                    "-map", "0:v:0", "-map", "0:a:0",
                    *config.video_encoder_args(request["quality"], source_bitrate=project["media"].get("video_bitrate")),
                    "-c:a", "copy", "-t", str(project["media"]["duration"]),
                    "-movflags", "+faststart", str(output),
                ]
                execute(project_id, "caption_export", command, cwd=folder)
            else:
                shutil.copy2(lip_sync_master, output)
        else:
            background = select_background(work, source, MOCK_MODE)
            audio_filter = f"[1:a]volume={request['background_volume']:.3f}[bg];[2:a]volume=1.04,highpass=f=60,lowpass=f=11000,acompressor=threshold=-19dB:ratio=1.8:attack=15:release=140,alimiter=limit=0.94[vo];[bg][vo]amix=inputs=2:duration=longest:normalize=0,loudnorm=I=-15:TP=-2:LRA=7[a]"
            command = ["ffmpeg", "-y", "-hide_banner", "-i", str(source), "-i", str(background), "-i", str(dub), "-filter_complex", audio_filter]
            if request["subtitle_enabled"]:
                fonts_dir = (config.ROOT / "static" / "fonts").as_posix().replace(":", r"\:")
                command.extend(["-vf", f"subtitles=work/subtitles.ass:fontsdir='{fonts_dir}'"])
            command.extend(["-map", "0:v:0", "-map", "[a]", *config.video_encoder_args(request["quality"], source_bitrate=project["media"].get("video_bitrate")), "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-ac", "2", "-t", str(project["media"]["duration"]), "-movflags", "+faststart", str(output)])
            execute(project_id, "caption_export", command, cwd=folder)
        shutil.copy2(subtitles, exported_subtitles)
        execute(project_id, "caption_decode_check", ["ffmpeg", "-v", "error", "-i", str(output), "-f", "null", os.devnull])
        project = load_project(project_id)
        archive_current_export(project)
        quality = dict(project.get("quality") or {})
        quality["media"] = probe(output)
        project.update({"status": "complete", "stage": "Complete", "progress": 100, "quality": quality, "render": dict(request)})
        project["output"] = {"video": str(output), "subtitles": str(exported_subtitles), "run_id": run_id, "created_at": now()}
        project.setdefault("exports", []).append({**project["output"], "render": dict(request), "quality": dict(quality)})
        save_project(project)
        add_event(project_id, "success", "Caption export complete")
    except JobCancelled:
        return
    except Exception as exc:
        fail(project_id, exc)


def fail(project_id: str, exc: Exception) -> None:
    project = load_project(project_id)
    failed_stage = project.get("stage") or "Processing"
    project.update({
        "status": "failed", "stage": "Failed", "failed_stage": failed_stage,
        "error": str(exc), "progress": project.get("progress", 0),
    })
    save_project(project)
    add_event(project_id, "error", str(exc))


class JobQueue:
    def __init__(self) -> None:
        self.queue: queue.Queue[tuple[str, str, Callable, dict]] = queue.Queue()
        self.active: str | None = None
        self.active_token: str | None = None
        self.pending: dict[str, str] = {}
        self.cancelled: set[str] = set()
        self.processes: dict[str, set[subprocess.Popen]] = {}
        self.external_active = False
        self.lock = threading.Condition()
        self.thread = threading.Thread(target=self._loop, name="dubbing-gpu-worker", daemon=True)
        self.thread.start()

    def submit(self, project_id: str, task: Callable, payload: dict) -> bool:
        with self.lock:
            if project_id in self.pending or self.active == project_id:
                return False
            token = uuid.uuid4().hex
            self.pending[project_id] = token
            position = len(self.pending)
        self.queue.put((project_id, token, task, payload))
        add_event(project_id, "info", f"Task added to the queue: position {position}")
        return True

    @property
    def waiting(self) -> int:
        with self.lock:
            return max(0, len(self.pending) - (1 if self.active in self.pending else 0))

    def acquire_external(self) -> bool:
        """Reserve the accelerator for a synchronous task such as a voice preview."""
        with self.lock:
            if self.active or self.pending or self.external_active:
                return False
            self.external_active = True
            return True

    def release_external(self) -> None:
        with self.lock:
            self.external_active = False
            self.lock.notify_all()

    def is_cancelled(self, project_id: str) -> bool:
        with self.lock:
            token = self.active_token if self.active == project_id else self.pending.get(project_id)
            return token in self.cancelled if token else False

    def raise_if_cancelled(self, project_id: str) -> None:
        if self.is_cancelled(project_id):
            raise JobCancelled(project_id)

    def register_process(self, project_id: str, process: subprocess.Popen) -> None:
        with self.lock:
            self.processes.setdefault(project_id, set()).add(process)
            token = self.active_token if self.active == project_id else self.pending.get(project_id)
            cancelled = token in self.cancelled if token else False
        if cancelled:
            self._terminate_process_tree(process)

    def unregister_process(self, project_id: str, process: subprocess.Popen) -> None:
        with self.lock:
            project_processes = self.processes.get(project_id)
            if project_processes:
                project_processes.discard(process)
                if not project_processes:
                    self.processes.pop(project_id, None)

    @staticmethod
    def _terminate_process_tree(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
            )
        else:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except ProcessLookupError:
                return
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    return

    def cancel(self, project_id: str, wait: bool = False, timeout: float = 15.0) -> bool:
        with self.lock:
            token = self.active_token if self.active == project_id else self.pending.get(project_id)
            if not token:
                return False
            self.cancelled.add(token)
            if self.active != project_id:
                self.pending.pop(project_id, None)
            processes = list(self.processes.get(project_id, ()))
        for process in processes:
            self._terminate_process_tree(process)
        if wait:
            deadline = time.monotonic() + timeout
            with self.lock:
                while self.active == project_id and time.monotonic() < deadline:
                    self.lock.wait(timeout=min(.25, deadline - time.monotonic()))
                return self.active != project_id
        return True

    def _loop(self) -> None:
        while True:
            project_id, token, task, payload = self.queue.get()
            with self.lock:
                while self.external_active:
                    self.lock.wait()
                if token in self.cancelled:
                    self.cancelled.discard(token)
                    self.pending.pop(project_id, None)
                    self.queue.task_done()
                    continue
                self.active = project_id
                self.active_token = token
            try:
                task(project_id, payload)
            except Exception as exc:
                try:
                    fail(project_id, exc)
                except Exception:
                    pass
            finally:
                with self.lock:
                    self.pending.pop(project_id, None)
                    self.cancelled.discard(token)
                    self.active = None
                    self.active_token = None
                    self.processes.pop(project_id, None)
                    self.lock.notify_all()
                self.queue.task_done()


jobs = JobQueue()


def recovery_action(project: dict) -> tuple[Callable, dict] | None:
    if project.get("status") != "queued":
        return None
    stage = str(project.get("stage", "")).lower()
    if "render" in stage and project.get("render", {}).get("target_language"):
        return render, dict(project["render"])
    if "analysis" in stage or "analy" in stage:
        analysis = project.get("analysis") or {}
        return analyze, {
            "source_language": analysis.get("source_language", "auto"),
            "speaker_count": analysis.get("speaker_count", "auto"),
        }
    return None


def recover_interrupted_jobs() -> dict[str, int]:
    recovered = 0
    interrupted = 0
    for project in list_projects():
        status = project.get("status")
        if status == "queued":
            action = recovery_action(project)
            if action and jobs.submit(project["id"], *action):
                recovered += 1
                add_event(project["id"], "info", "Queued task recovered after application restart")
            elif not action:
                project.update({"status": "failed", "stage": "Recovery required", "error": "Queued task payload is incomplete"})
                save_project(project)
                add_event(project["id"], "error", "Queued task could not be recovered; run it again")
                interrupted += 1
        elif status in {"analyzing", "rendering"}:
            project.update({
                "status": "failed", "stage": "Interrupted",
                "error": "Processing was interrupted when the application stopped; run this step again",
            })
            save_project(project)
            add_event(project["id"], "error", "Interrupted processing detected after application restart")
            interrupted += 1
    return {"recovered": recovered, "interrupted": interrupted}
