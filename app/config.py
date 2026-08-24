from __future__ import annotations

import os
import platform
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
DATA_ROOT = Path(os.environ.get("DUBBING_STUDIO_DATA", ROOT / "data")).resolve()
PROJECTS_ROOT = DATA_ROOT / "projects"
CACHE_ROOT = DATA_ROOT / "cache"
PREVIEWS_ROOT = CACHE_ROOT / "voice-previews"

RUNTIME_ROOT = Path(os.environ.get("DUBBING_STUDIO_RUNTIME", ROOT / "runtime")).resolve()


def venv_python(root: Path, platform_name: str | None = None) -> Path:
    """Return the Python executable used by a virtual environment on this OS."""
    platform_name = platform_name or os.name
    if platform_name == "nt":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


SYSTEM = platform.system()
MACHINE = platform.machine().lower()
IS_APPLE_SILICON = SYSTEM == "Darwin" and MACHINE in {"arm64", "aarch64"}
DEFAULT_BACKEND = "apple_silicon" if IS_APPLE_SILICON else "cuda" if os.name == "nt" else "cpu"
BACKEND = os.environ.get("DUBBING_STUDIO_BACKEND", DEFAULT_BACKEND).strip().lower()
if BACKEND not in {"cuda", "apple_silicon", "cpu"}:
    raise RuntimeError(f"Unsupported DUBBING_STUDIO_BACKEND: {BACKEND}")

TORCH_DEVICE = "mps" if BACKEND == "apple_silicon" else "cuda:0" if BACKEND == "cuda" else "cpu"
DEMUCS_DEVICE = "mps" if BACKEND == "apple_silicon" else "cuda" if BACKEND == "cuda" else "cpu"
ASR_DEVICE = "cpu" if BACKEND == "apple_silicon" else "cuda" if BACKEND == "cuda" else "cpu"
ASR_COMPUTE_TYPE = "int8" if ASR_DEVICE == "cpu" else "float16"
TORCH_DTYPE = "float16" if BACKEND == "apple_silicon" else "bfloat16" if BACKEND == "cuda" else "float32"
VIDEO_ENCODER = "h264_videotoolbox" if BACKEND == "apple_silicon" else "h264_nvenc" if BACKEND == "cuda" else "libx264"
SEEDVC_FP16 = BACKEND == "cuda"


def runtime_path(env_name: str, local_name: str) -> Path:
    """Return an explicitly configured runtime path or the portable local path."""
    if configured := os.environ.get(env_name):
        return Path(configured).expanduser().resolve()
    return RUNTIME_ROOT / local_name


LINLY_ROOT = runtime_path("LINLY_ROOT", "asr")
QWEN_ROOT = runtime_path("QWEN_ROOT", "qwen3-tts")
SEEDVC_ROOT = runtime_path("SEEDVC_ROOT", "seed-vc")
HYMT_ROOT = runtime_path("HYMT_ROOT", "hymt")
MUSETALK_ROOT = runtime_path("MUSETALK_ROOT", "musetalk")

LINLY_PYTHON = venv_python(LINLY_ROOT)
QWEN_PYTHON = venv_python(QWEN_ROOT)
SEEDVC_PYTHON = venv_python(SEEDVC_ROOT)
HYMT_PYTHON = venv_python(HYMT_ROOT)
MUSETALK_PYTHON = venv_python(MUSETALK_ROOT)
def discover_asr_model(root: Path) -> Path:
    """Support both the portable layout and the earlier Hugging Face cache layout."""
    portable = root / "models" / "faster-whisper-large-v3"
    if (portable / "model.bin").is_file():
        return portable
    snapshots = root / "models" / "ASR" / "models--Systran--faster-whisper-large-v3" / "snapshots"
    for candidate in sorted(snapshots.glob("*/model.bin")):
        return candidate.parent
    return portable


ASR_MODEL = discover_asr_model(LINLY_ROOT)
def discover_hymt_model(root: Path) -> Path:
    if configured := os.environ.get("DUBBING_STUDIO_TRANSLATION_MODEL"):
        return Path(configured).expanduser().resolve()
    preferred = (
        ("Hy-MT2-7B", "Hy-MT2-1.8B", "Hy-MT2-7B-FP8")
        if BACKEND == "apple_silicon"
        else ("Hy-MT2-7B-FP8", "Hy-MT2-7B", "Hy-MT2-1.8B")
    )
    for name in preferred:
        candidate = root / "models" / name
        if (candidate / "config.json").is_file():
            return candidate
    return root / "models" / preferred[0]


HYMT_MODEL = discover_hymt_model(HYMT_ROOT)
MUSETALK_AVAILABLE = all(path.is_file() for path in (
    MUSETALK_PYTHON,
    MUSETALK_ROOT / "scripts" / "inference.py",
    MUSETALK_ROOT / "models" / "musetalkV15" / "unet.pth",
    MUSETALK_ROOT / "models" / "musetalkV15" / "musetalk.json",
    MUSETALK_ROOT / "models" / "whisper" / "config.json",
))
TORCH_HOME = LINLY_ROOT / "models" / "torch"

RUNTIME_CHECKS = {
    "asr_python": LINLY_PYTHON,
    "asr_model": ASR_MODEL / "model.bin",
    "qwen_python": QWEN_PYTHON,
    "qwen_base": QWEN_ROOT / "models" / "Qwen3-TTS-12Hz-1.7B-Base" / "config.json",
    "qwen_voice_design": QWEN_ROOT / "models" / "Qwen3-TTS-12Hz-1.7B-VoiceDesign" / "config.json",
    "seedvc_python": SEEDVC_PYTHON,
    "seedvc_entrypoint": SEEDVC_ROOT / "inference.py",
    "hymt_python": HYMT_PYTHON,
    "hymt_model": HYMT_MODEL / "config.json",
}


def video_encoder_args(
    quality: str,
    backend: str | None = None,
    source_bitrate: int | None = None,
) -> list[str]:
    """Return a tested H.264 encoder profile for the active compute backend."""
    backend = backend or BACKEND
    if backend == "cuda":
        cq = {"draft": "25", "balanced": "21", "high": "18"}[quality]
        return [
            "-c:v", "h264_nvenc", "-preset", "p6", "-tune", "hq", "-rc", "vbr",
            "-cq", cq, "-b:v", "0", "-pix_fmt", "yuv420p",
        ]
    if backend == "apple_silicon":
        multiplier = {"draft": 0.8, "balanced": 1.0, "high": 1.25}[quality]
        bitrate = max(2_000_000, min(80_000_000, round((source_bitrate or 8_000_000) * multiplier)))
        return [
            "-c:v", "h264_videotoolbox", "-profile:v", "high", "-pix_fmt", "yuv420p",
            "-b:v", str(bitrate),
        ]
    crf = {"draft": "25", "balanced": "21", "high": "18"}[quality]
    return ["-c:v", "libx264", "-preset", "medium", "-crf", crf, "-pix_fmt", "yuv420p"]

MAX_UPLOAD_BYTES = int(os.environ.get("DUBBING_STUDIO_MAX_UPLOAD", 12 * 1024**3))
MOCK_MODE = os.environ.get("DUBBING_STUDIO_MOCK", "0") == "1"

LANGUAGES = [
    {"id": "zh", "label": "Chinese · Simplified", "tts": "Chinese", "asr": "zh", "hymt": "Chinese", "sample": "你好！这是所选声音的示例。"},
    {"id": "en", "label": "English", "tts": "English", "asr": "en", "hymt": "English", "sample": "Hello! This is a preview of the selected voice."},
    {"id": "ru", "label": "Russian", "tts": "Russian", "asr": "ru", "hymt": "Russian", "sample": "Здравствуйте! Это пример выбранного голоса."},
    {"id": "de", "label": "German", "tts": "German", "asr": "de", "hymt": "German", "sample": "Hallo! Dies ist eine Vorschau der ausgewählten Stimme."},
    {"id": "fr", "label": "French", "tts": "French", "asr": "fr", "hymt": "French", "sample": "Bonjour ! Voici un aperçu de la voix sélectionnée."},
    {"id": "es", "label": "Spanish", "tts": "Spanish", "asr": "es", "hymt": "Spanish", "sample": "¡Hola! Esta es una muestra de la voz seleccionada."},
    {"id": "it", "label": "Italian", "tts": "Italian", "asr": "it", "hymt": "Italian", "sample": "Ciao! Questa è un'anteprima della voce selezionata."},
    {"id": "pt", "label": "Portuguese", "tts": "Portuguese", "asr": "pt", "hymt": "Portuguese", "sample": "Olá! Esta é uma amostra da voz selecionada."},
    {"id": "ja", "label": "Japanese", "tts": "Japanese", "asr": "ja", "hymt": "Japanese", "sample": "こんにちは。選択した音声のサンプルです。"},
    {"id": "ko", "label": "Korean", "tts": "Korean", "asr": "ko", "hymt": "Korean", "sample": "안녕하세요. 선택한 목소리의 예시입니다."},
]

VOICE_PERSONAS = [
    {"id": "warm_female", "label": "Warm female", "icon": "warm", "description": "A warm natural adult female voice, friendly and conversational, restrained emotion, clean studio recording."},
    {"id": "clear_male", "label": "Clear male", "icon": "clear", "description": "A clear natural adult male voice, confident but not commercial, conversational cadence, clean studio recording."},
    {"id": "soft_young", "label": "Soft youthful", "icon": "young", "description": "A soft youthful voice, lively and sincere, natural conversational rhythm, clean studio recording."},
    {"id": "documentary", "label": "Documentary", "icon": "documentary", "description": "A calm mature documentary narrator, articulate and composed, natural pauses, clean studio recording."},
]

SUBTITLE_STYLES = [
    {"id": "clean", "label": "Clean", "description": "Fine outline", "force_style": "FontName=Manrope,FontSize=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=1.6,Shadow=0,Alignment=2,MarginV=55"},
    {"id": "cinema", "label": "Cinema", "description": "Soft cinematic edge", "force_style": "FontName=Manrope,FontSize=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H000B0D12,BorderStyle=1,Outline=2.2,Shadow=0.5,Alignment=2,MarginV=58"},
    {"id": "social", "label": "Social", "description": "Large bold caption", "force_style": "FontName=Manrope,FontSize=20,Bold=1,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=3,Shadow=0,Alignment=2,MarginV=72"},
    {"id": "boxed", "label": "Bold", "description": "Strong readable outline", "force_style": "FontName=Manrope,FontSize=16,Bold=1,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2.6,Shadow=0,Alignment=2,MarginV=55"},
    {"id": "accent", "label": "Editorial", "description": "Strong editorial weight", "force_style": "FontName=Manrope,FontSize=20,Bold=1,PrimaryColour=&H00FFFFFF,OutlineColour=&H00101010,BorderStyle=1,Outline=2.5,Shadow=0,Alignment=2,MarginV=65"},
    {"id": "minimal", "label": "Minimal", "description": "Restrained caption", "force_style": "FontName=Manrope,FontSize=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=1.2,Shadow=0,Alignment=2,MarginV=42"},
]

SUBTITLE_SIZES = {"small": 12, "medium": 16, "large": 20}
SUBTITLE_COLORS = {
    "white": {"primary": "&H00FFFFFF", "outline": "&H00000000"},
    "yellow": {"primary": "&H0000D4FF", "outline": "&H00000000"},
    "black": {"primary": "&H00000000", "outline": "&H00FFFFFF"},
}

for path in (DATA_ROOT, PROJECTS_ROOT, CACHE_ROOT, PREVIEWS_ROOT):
    path.mkdir(parents=True, exist_ok=True)


def language(language_id: str) -> dict:
    return next(item for item in LANGUAGES if item["id"] == language_id)


def voice(voice_id: str) -> dict:
    return next(item for item in VOICE_PERSONAS if item["id"] == voice_id)


def subtitle_style(style_id: str) -> dict:
    return next(item for item in SUBTITLE_STYLES if item["id"] == style_id)


def subtitle_style_values(style_id: str, size: str = "medium", color: str = "white", scale: float = 1.0) -> dict[str, str]:
    """Resolve visible caption controls into normalized libass style values."""
    values = {}
    for item in subtitle_style(style_id)["force_style"].split(","):
        key, value = item.split("=", 1)
        values[key] = value
    palette = SUBTITLE_COLORS[color]
    values.update({
        "FontName": "Manrope",
        "FontSize": str(round(SUBTITLE_SIZES[size] * scale, 2)),
        "PrimaryColour": palette["primary"],
        "OutlineColour": palette["outline"],
    })
    # Shift every future caption down by one selected line height.
    values["MarginV"] = str(max(12, int(values.get("MarginV", "55")) - round(SUBTITLE_SIZES[size] * scale)))
    values.pop("BackColour", None)
    return values


def subtitle_force_style(style_id: str, size: str = "medium", color: str = "white", scale: float = 1.0) -> str:
    """Resolve the visible subtitle controls into one libass force-style string."""
    values = subtitle_style_values(style_id, size, color, scale)
    return ",".join(f"{key}={value}" for key, value in values.items())
