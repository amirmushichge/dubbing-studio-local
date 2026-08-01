from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path(__import__("os").environ.get("DUBBING_STUDIO_RUNTIME", ROOT / "runtime")).resolve()


def command_version(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, encoding="utf-8", errors="replace")
        return (result.stdout or result.stderr).splitlines()[0] if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


checks = {
    "ffmpeg": (shutil.which("ffmpeg") is not None, command_version(["ffmpeg", "-version"])),
    "nvidia": (shutil.which("nvidia-smi") is not None, command_version(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])),
    "app_python": ((ROOT / ".venv/Scripts/python.exe").is_file(), str(ROOT / ".venv/Scripts/python.exe")),
    "asr_python": ((RUNTIME / "asr/.venv/Scripts/python.exe").is_file(), None),
    "qwen_python": ((RUNTIME / "qwen3-tts/.venv/Scripts/python.exe").is_file(), None),
    "seedvc_python": ((RUNTIME / "seed-vc/.venv/Scripts/python.exe").is_file(), None),
    "hymt_python": ((RUNTIME / "hymt/.venv/Scripts/python.exe").is_file(), None),
    "asr_model": ((RUNTIME / "asr/models/faster-whisper-large-v3/model.bin").is_file(), None),
    "qwen_base": ((RUNTIME / "qwen3-tts/models/Qwen3-TTS-12Hz-1.7B-Base/config.json").is_file(), None),
    "qwen_design": ((RUNTIME / "qwen3-tts/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign/config.json").is_file(), None),
    "hymt_model": ((RUNTIME / "hymt/models/Hy-MT2-7B-FP8/config.json").is_file(), None),
    "seedvc_model": (any((RUNTIME / "seed-vc/checkpoints").glob("models--Plachta--Seed-VC/snapshots/*/DiT_seed*.pth")), None),
}
payload = {"ok": all(value[0] for value in checks.values()), "runtime": str(RUNTIME), "checks": {key: {"ok": value[0], "detail": value[1]} for key, value in checks.items()}}
if "--json" in sys.argv:
    print(json.dumps(payload, ensure_ascii=False, indent=2))
else:
    print("\nДиагностика Dubbing Studio")
    print(f"Runtime: {RUNTIME}\n")
    for name, result in payload["checks"].items():
        print(f"{'OK ' if result['ok'] else 'НЕТ'}  {name}" + (f" — {result['detail']}" if result["detail"] else ""))
    print("\nСистема готова." if payload["ok"] else "\nНе всё установлено. Повторно запустите setup.bat.")
raise SystemExit(0 if payload["ok"] else 1)
