from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

from . import config
from .config import ASR_MODEL, HYMT_MODEL, HYMT_PYTHON, HYMT_ROOT, LINLY_PYTHON, LINLY_ROOT, MOCK_MODE, QWEN_PYTHON, QWEN_ROOT, SEEDVC_PYTHON, SEEDVC_ROOT, TORCH_HOME
from .media import probe, run
from .store import add_event, load_project, project_dir, save_project


WORKERS = config.ROOT / "workers"


def input_video(project_id: str) -> Path:
    files = list((project_dir(project_id) / "input").iterdir())
    if not files:
        raise RuntimeError("Input video is missing")
    return files[0]


def set_stage(project_id: str, stage: str, progress: int) -> None:
    project = load_project(project_id)
    project.update({"stage": stage, "progress": progress, "error": None})
    save_project(project)
    add_event(project_id, "info", stage)


def execute(project_id: str, name: str, command: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    log = project_dir(project_id) / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    merged = os.environ.copy()
    merged.update({"PYTHONUTF8": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "KMP_DUPLICATE_LIB_OK": "TRUE"})
    if env:
        merged.update(env)
    result = subprocess.run(command, cwd=cwd, env=merged, text=True, encoding="utf-8", errors="replace", capture_output=True)
    log.write_text((result.stdout or "") + "\n" + (result.stderr or ""), encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"{name}: {(result.stderr or result.stdout)[-2000:]}")


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
            set_stage(project_id, "Mock: распознавание", 40)
            time.sleep(.3)
            segments = [
                {"id": 0, "start": 0.0, "end": 2.8, "text": "Пример первой распознанной реплики.", "speaker": "SPEAKER_00", "words": []},
                {"id": 1, "start": 3.1, "end": 5.5, "text": "Её можно исправить перед переводом.", "speaker": "SPEAKER_00", "words": []},
            ]
            result = {"language": "ru", "language_probability": .99, "speakers": [{"id": "SPEAKER_00", "label": "Говорящий 1", "reference": "", "profile": "A natural conversational adult voice."}], "segments": segments}
        else:
            set_stage(project_id, "Извлечение аудио", 12)
            audio = work / "audio.wav"
            execute(project_id, "extract_audio", ["ffmpeg", "-y", "-v", "error", "-i", str(source), "-vn", "-ar", "44100", "-ac", "2", str(audio)])
            set_stage(project_id, "Разделение речи и фона", 26)
            demucs_root = work / "demucs"
            execute(project_id, "demucs", [str(LINLY_PYTHON), "-m", "demucs", "--two-stems", "vocals", "-n", "htdemucs", "-d", "cuda", "-o", str(demucs_root), str(audio)], cwd=LINLY_ROOT, env={"TORCH_HOME": str(TORCH_HOME)})
            demucs_job = demucs_root / "htdemucs" / "audio"
            shutil.copy2(demucs_job / "vocals.wav", work / "vocals.wav")
            shutil.copy2(demucs_job / "no_vocals.wav", work / "background.wav")
            set_stage(project_id, "Распознавание речи", 48)
            transcript = work / "transcript.json"
            command = [str(LINLY_PYTHON), str(WORKERS / "transcribe.py"), str(work / "vocals.wav"), str(transcript), "--model", str(ASR_MODEL)]
            if request.get("source_language", "auto") != "auto":
                command.extend(["--language", request["source_language"]])
            execute(project_id, "transcribe", command, cwd=LINLY_ROOT)
            set_stage(project_id, "Определение говорящих", 72)
            analyzed = work / "analysis.json"
            count = str(request.get("speaker_count", "auto"))
            execute(project_id, "speakers", [str(SEEDVC_PYTHON), str(WORKERS / "cluster_speakers.py"), str(work / "vocals.wav"), str(transcript), str(analyzed), str(work / "references"), "--count", count], cwd=SEEDVC_ROOT)
            result = json.loads(analyzed.read_text(encoding="utf-8"))
        project = load_project(project_id)
        project["analysis"].update({
            "detected_language": result["language"], "language_probability": result.get("language_probability"),
            "speakers": result["speakers"], "segments": result["segments"],
        })
        project.update({"status": "review", "stage": "Проверьте текст и говорящих", "progress": 100})
        save_project(project)
        add_event(project_id, "success", f"Анализ завершён: {len(result['segments'])} реплик, {len(result['speakers'])} говорящих")
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
                    rows.append(current); current = word
                else:
                    current = f"{current} {word}".strip()
            rows.append(current); text = "\n".join(rows)
        blocks.append(f"{index}\n{timestamp(item['start'])} --> {timestamp(item['end'])}\n{text}\n")
    path.write_text("\n".join(blocks), encoding="utf-8-sig")


def normalized_words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def render(project_id: str, request: dict) -> None:
    project = load_project(project_id)
    project.update({"status": "rendering", "render": request, "progress": 0})
    save_project(project)
    folder = project_dir(project_id)
    work = folder / "work"
    source = input_video(project_id)
    try:
        speakers = project["analysis"]["speakers"]
        segments = project["analysis"]["segments"]
        if not segments:
            raise RuntimeError("No transcript to render")
        if request["voice_mode"] == "catalog" and len(speakers) != 1:
            raise RuntimeError("Catalog voice can only be used for a single-speaker video")
        target = config.language(request["target_language"])
        set_stage(project_id, "Перевод и адаптация реплик", 8)
        translation = work / "translation.json"
        analysis_payload = {"segments": segments}
        analysis_for_translation = work / "analysis_for_translation.json"
        analysis_for_translation.write_text(json.dumps(analysis_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if MOCK_MODE:
            translated = [dict(item, translation=f"[{target['label']}] {item['text']}") for item in segments]
            translation.write_text(json.dumps(translated, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            detected = project["analysis"].get("detected_language", project["analysis"].get("source_language", "Russian"))
            source_name = next((item["hymt"] for item in config.LANGUAGES if item["id"] == detected), detected)
            execute(project_id, "translate", [str(HYMT_PYTHON), str(WORKERS / "translate.py"), str(analysis_for_translation), str(translation), "--model", str(HYMT_MODEL), "--source-language", source_name, "--target-language", target["hymt"]], cwd=HYMT_ROOT)
        set_stage(project_id, "Создание нативной речи", 30)
        persona = config.voice(request.get("voice_id") or "warm_female")
        qwen_config = {
            "job_dir": str(folder), "translation_path": str(translation), "duration": project["media"]["duration"],
            "qwen_root": str(QWEN_ROOT), "tts_language": target["tts"], "sample_text": target["sample"],
            "voice_mode": request["voice_mode"], "voice_description": persona["description"],
            "expression": request["expression"], "speakers": speakers,
        }
        qwen_config_path = work / "qwen_config.json"
        qwen_config_path.write_text(json.dumps(qwen_config, ensure_ascii=False, indent=2), encoding="utf-8")
        if MOCK_MODE:
            execute(project_id, "mock_dub", ["ffmpeg", "-y", "-v", "error", "-i", str(source), "-vn", "-ar", "24000", "-ac", "1", str(work / "dub.wav")])
        else:
            qwen_command = [str(QWEN_PYTHON), str(WORKERS / "qwen_voice.py"), "synthesize", str(qwen_config_path)]
            for timing_attempt in range(4):
                try:
                    execute(project_id, f"qwen_{timing_attempt + 1}", qwen_command, cwd=QWEN_ROOT)
                    break
                except RuntimeError as exc:
                    match = re.search(r"Line requires ([\d.]+)x speed; shorten its translation: (\d{4})_", str(exc))
                    if not match or timing_attempt == 3:
                        raise
                    required_speed = float(match.group(1))
                    line_index = int(match.group(2))
                    ratio = min(.88, 1.35 / required_speed * .92)
                    add_event(project_id, "warning", f"Реплика {line_index + 1} длиннее слота; сокращаем перевод и пересинтезируем")
                    execute(project_id, f"shorten_{line_index}_{timing_attempt + 1}", [
                        str(HYMT_PYTHON), str(WORKERS / "shorten_translation.py"), str(translation),
                        "--index", str(line_index), "--model", str(HYMT_MODEL),
                        "--language", target["hymt"], "--ratio", str(ratio),
                    ], cwd=HYMT_ROOT)
                    synthesis = work / "synthesis"
                    for candidate in (
                        synthesis / "raw" / f"{line_index:04d}_{segments[line_index]['speaker']}.wav",
                        synthesis / "fitted" / f"{line_index:04d}_{segments[line_index]['speaker']}.wav",
                        synthesis / "fitted" / f"{line_index:04d}_{segments[line_index]['speaker']}.trim.wav",
                    ):
                        candidate.unlink(missing_ok=True)
            manifest_path = work / "synthesis" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if request["voice_mode"] == "clone":
                set_stage(project_id, "Возвращение оригинальных голосов", 56)
                converted_root = work / "synthesis" / "converted"
                converted_root.mkdir(parents=True, exist_ok=True)
                for role in manifest["roles"]:
                    role_dir = converted_root / role["speaker"]
                    role_dir.mkdir(parents=True, exist_ok=True)
                    execute(project_id, f"seedvc_{role['speaker']}", [
                        str(SEEDVC_PYTHON), str(SEEDVC_ROOT / "inference.py"),
                        "--source", role["source"], "--target", role["reference"], "--output", str(role_dir),
                        "--diffusion-steps", "30", "--length-adjust", "1.0", "--inference-cfg-rate", "0.75",
                        "--f0-condition", "False", "--auto-f0-adjust", "False", "--semi-tone-shift", "0", "--fp16", "True",
                    ], cwd=SEEDVC_ROOT)
                    outputs = sorted(role_dir.glob("*.wav"), key=lambda path: path.stat().st_mtime, reverse=True)
                    if not outputs:
                        raise RuntimeError(f"Seed-VC produced no audio for {role['speaker']}")
                    role["converted"] = str(outputs[0])
                manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
                execute(project_id, "assemble_clone", [str(QWEN_PYTHON), str(WORKERS / "assemble_clone.py"), str(manifest_path), str(translation), str(work / "dub.wav"), "--duration", str(project["media"]["duration"])], cwd=QWEN_ROOT)

        set_stage(project_id, "Субтитры и финальный микс", 76)
        translated = json.loads(translation.read_text(encoding="utf-8"))
        subtitles = work / "subtitles.srt"
        write_srt(translated, subtitles)
        output = folder / "output" / f"{project['name']}_{request['target_language']}_dub.mp4"
        background = work / "background.wav"
        if not background.exists():
            background = source
        audio_filter = f"[1:a]volume={request['background_volume']:.3f}[bg];[2:a]volume=1.04,highpass=f=60,lowpass=f=11000,acompressor=threshold=-19dB:ratio=1.8:attack=15:release=140,alimiter=limit=0.94[vo];[bg][vo]amix=inputs=2:duration=longest:normalize=0,loudnorm=I=-15:TP=-2:LRA=7[a]"
        command = ["ffmpeg", "-y", "-hide_banner", "-i", str(source), "-i", str(background), "-i", str(work / "dub.wav"), "-filter_complex", audio_filter]
        if request["subtitle_enabled"] and request["burn_subtitles"]:
            style = config.subtitle_style(request["subtitle_style"])["force_style"]
            command.extend(["-vf", f"subtitles=work/subtitles.srt:force_style='{style}'"])
        cq = {"draft": "25", "balanced": "21", "high": "19"}[request["quality"]]
        command.extend(["-map", "0:v:0", "-map", "[a]", "-c:v", "h264_nvenc", "-preset", "p6", "-tune", "hq", "-rc", "vbr", "-cq", cq, "-b:v", "4M", "-maxrate", "8M", "-bufsize", "8M", "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-ac", "2", "-t", str(project["media"]["duration"]), "-movflags", "+faststart", str(output)])
        execute(project_id, "render", command, cwd=folder)

        set_stage(project_id, "Контроль качества", 90)
        quality = {"media": probe(output), "max_speed": None, "asr_coverage": None, "warnings": []}
        report_path = work / "synthesis" / ("clone_timing_report.json" if request["voice_mode"] == "clone" else "manifest.json")
        if report_path.exists():
            report_payload = json.loads(report_path.read_text(encoding="utf-8"))
            report = report_payload.get("report", report_payload) if isinstance(report_payload, dict) else report_payload
            speeds = [item.get("speed", 1) for item in report]
            quality["max_speed"] = max(speeds, default=1)
        if not MOCK_MODE:
            qa_asr = work / "qa_asr.json"
            execute(project_id, "qa_asr", [str(LINLY_PYTHON), str(WORKERS / "transcribe.py"), str(work / "dub.wav"), str(qa_asr), "--language", target["asr"], "--model", str(ASR_MODEL)], cwd=LINLY_ROOT)
            recognized = " ".join(item["text"] for item in json.loads(qa_asr.read_text(encoding="utf-8"))["segments"])
            expected = " ".join(item["translation"] for item in translated)
            expected_words = normalized_words(expected)
            quality["asr_coverage"] = round(len(expected_words & normalized_words(recognized)) / max(1, len(expected_words)), 3)
            if quality["asr_coverage"] < .82:
                quality["warnings"].append("ASR coverage is below 82%; review pronunciation and missing words.")
            qa_audio_path = work / "qa_audio.json"
            qa_command = [str(SEEDVC_PYTHON), str(WORKERS / "qa_audio.py"), str(work / "dub.wav"), str(translation), str(output), str(qa_audio_path)]
            manifest_path = work / "synthesis" / "manifest.json"
            if request["voice_mode"] == "clone" and manifest_path.exists():
                qa_command.extend(["--manifest", str(manifest_path)])
            execute(project_id, "qa_audio", qa_command, cwd=SEEDVC_ROOT)
            audio_quality = json.loads(qa_audio_path.read_text(encoding="utf-8"))
            quality.update({key: value for key, value in audio_quality.items() if key != "warnings"})
            quality["warnings"].extend(audio_quality["warnings"])
        execute(project_id, "decode_check", ["ffmpeg", "-v", "error", "-i", str(output), "-f", "null", "NUL"])
        project = load_project(project_id)
        project.update({"status": "complete", "stage": "Готово", "progress": 100, "quality": quality})
        project["output"] = {"video": str(output), "subtitles": str(subtitles)}
        save_project(project)
        add_event(project_id, "success", "Дубляж и проверки завершены")
    except Exception as exc:
        fail(project_id, exc)


def fail(project_id: str, exc: Exception) -> None:
    project = load_project(project_id)
    project.update({"status": "failed", "stage": "Ошибка", "error": str(exc), "progress": project.get("progress", 0)})
    save_project(project)
    add_event(project_id, "error", str(exc))


class JobQueue:
    def __init__(self) -> None:
        self.queue: queue.Queue[tuple[str, Callable, dict]] = queue.Queue()
        self.active: str | None = None
        self.thread = threading.Thread(target=self._loop, name="dubbing-gpu-worker", daemon=True)
        self.thread.start()

    def submit(self, project_id: str, task: Callable, payload: dict) -> None:
        self.queue.put((project_id, task, payload))
        add_event(project_id, "info", f"Задание добавлено в очередь: позиция {self.queue.qsize()}")

    def _loop(self) -> None:
        while True:
            project_id, task, payload = self.queue.get()
            self.active = project_id
            try:
                task(project_id, payload)
            finally:
                self.active = None
                self.queue.task_done()


jobs = JobQueue()
