from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import config  # noqa: E402


def command_output(command: list[str], timeout: int = 15) -> tuple[bool, str | None]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
        output = (result.stdout or result.stderr).strip().splitlines()
        return result.returncode == 0, output[0] if output else None
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)


def encoder_available(name: str) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=15,
            encoding="utf-8",
            errors="replace",
        )
        return result.returncode == 0 and name in result.stdout, name
    except (OSError, subprocess.TimeoutExpired):
        return False, name


checks: dict[str, tuple[bool, str | None]] = {}
ffmpeg = shutil.which("ffmpeg")
checks["ffmpeg"] = (ffmpeg is not None, command_output(["ffmpeg", "-version"])[1] if ffmpeg else None)
sox = shutil.which("sox")
checks["sox"] = (sox is not None, command_output(["sox", "--version"])[1] if sox else None)
app_python = config.venv_python(ROOT)
checks["app_python"] = (app_python.is_file(), str(app_python))
checks["platform"] = (
    config.BACKEND != "apple_silicon" or config.IS_APPLE_SILICON,
    f"{platform.system()} {platform.machine()} / backend={config.BACKEND}",
)

if config.BACKEND == "cuda":
    nvidia = shutil.which("nvidia-smi")
    checks["nvidia"] = (
        nvidia is not None,
        command_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])[1] if nvidia else None,
    )
elif config.BACKEND == "apple_silicon":
    checks["mps"] = (
        command_output([
            str(config.QWEN_PYTHON),
            "-c",
            "import torch; assert torch.backends.mps.is_available(); print('PyTorch MPS available')",
        ])
        if config.QWEN_PYTHON.is_file()
        else (False, "Qwen environment is missing")
    )

checks["video_encoder"] = encoder_available(config.VIDEO_ENCODER) if ffmpeg else (False, config.VIDEO_ENCODER)

if not config.MOCK_MODE:
    for name, path in config.RUNTIME_CHECKS.items():
        checks[name] = (path.is_file(), str(path))
    if config.BACKEND == "cuda":
        checks["musetalk_python"] = (config.MUSETALK_PYTHON.is_file(), str(config.MUSETALK_PYTHON))
        checks["musetalk_v15"] = (
            (config.MUSETALK_ROOT / "models" / "musetalkV15" / "unet.pth").is_file(),
            str(config.MUSETALK_ROOT / "models" / "musetalkV15" / "unet.pth"),
        )

payload = {
    "ok": all(value[0] for value in checks.values()),
    "backend": config.BACKEND,
    "runtime": str(config.RUNTIME_ROOT),
    "mock": config.MOCK_MODE,
    "checks": {key: {"ok": value[0], "detail": value[1]} for key, value in checks.items()},
}
if "--json" in sys.argv:
    print(json.dumps(payload, ensure_ascii=False, indent=2))
else:
    print("\nDubbing Studio diagnostics")
    print(f"Backend: {config.BACKEND}")
    print(f"Runtime: {config.RUNTIME_ROOT}\n")
    for name, result in payload["checks"].items():
        print(f"{'OK ' if result['ok'] else 'MISS'}  {name}" + (f" - {result['detail']}" if result["detail"] else ""))
    rerun = "setup.command" if platform.system() == "Darwin" else "setup.bat"
    print("\nSystem ready." if payload["ok"] else f"\nSetup is incomplete. Run {rerun} again.")
raise SystemExit(0 if payload["ok"] else 1)
