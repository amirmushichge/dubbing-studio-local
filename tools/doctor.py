from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import config


def command_version(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, encoding="utf-8", errors="replace")
        return (result.stdout or result.stderr).splitlines()[0] if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


checks: dict[str, tuple[bool, str | None]] = {
    "ffmpeg": (shutil.which("ffmpeg") is not None, command_version(["ffmpeg", "-version"])),
    "nvidia": (shutil.which("nvidia-smi") is not None, command_version(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])),
    "app_python": ((ROOT / ".venv/Scripts/python.exe").is_file(), str(ROOT / ".venv/Scripts/python.exe")),
}
for name, path in config.RUNTIME_CHECKS.items():
    checks[name] = (path.is_file(), str(path))

payload = {"ok": all(value[0] for value in checks.values()), "runtime": str(config.RUNTIME_ROOT), "checks": {key: {"ok": value[0], "detail": value[1]} for key, value in checks.items()}}
if "--json" in sys.argv:
    print(json.dumps(payload, ensure_ascii=False, indent=2))
else:
    print("\nDubbing Studio diagnostics")
    print(f"Runtime: {config.RUNTIME_ROOT}\n")
    for name, result in payload["checks"].items():
        print(f"{'OK ' if result['ok'] else 'MISS'}  {name}" + (f" — {result['detail']}" if result["detail"] else ""))
    print("\nSystem ready." if payload["ok"] else "\nSetup is incomplete. Run setup.bat again.")
raise SystemExit(0 if payload["ok"] else 1)
