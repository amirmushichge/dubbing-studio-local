from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
DATA_ROOT = Path(os.environ.get("DUBBING_STUDIO_DATA", ROOT / "data")).resolve()
PROJECTS_ROOT = DATA_ROOT / "projects"
CACHE_ROOT = DATA_ROOT / "cache"
PREVIEWS_ROOT = CACHE_ROOT / "voice-previews"

RUNTIME_ROOT = Path(os.environ.get("DUBBING_STUDIO_RUNTIME", ROOT / "runtime")).resolve()


def runtime_path(env_name: str, local_name: str, legacy: str) -> Path:
    """Prefer a portable runtime, while keeping existing installations usable."""
    if configured := os.environ.get(env_name):
        return Path(configured).resolve()
    portable = RUNTIME_ROOT / local_name
    if portable.exists():
        return portable
    return Path(legacy)


LINLY_ROOT = runtime_path("LINLY_ROOT", "asr", r"C:\DubbingStudioRuntime\Linly-Dubbing")
QWEN_ROOT = runtime_path("QWEN_ROOT", "qwen3-tts", r"C:\DubbingStudioRuntime\Qwen3-TTS")
SEEDVC_ROOT = runtime_path("SEEDVC_ROOT", "seed-vc", r"C:\DubbingStudioRuntime\seed-vc")
HYMT_ROOT = runtime_path("HYMT_ROOT", "hymt", r"C:\DubbingStudioRuntime\Hy-MT2")

LINLY_PYTHON = LINLY_ROOT / ".venv" / "Scripts" / "python.exe"
QWEN_PYTHON = QWEN_ROOT / ".venv" / "Scripts" / "python.exe"
SEEDVC_PYTHON = SEEDVC_ROOT / ".venv" / "Scripts" / "python.exe"
HYMT_PYTHON = HYMT_ROOT / ".venv" / "Scripts" / "python.exe"
ASR_MODEL = LINLY_ROOT / "models" / "faster-whisper-large-v3"
HYMT_MODEL = HYMT_ROOT / "models" / "Hy-MT2-7B-FP8"
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

MAX_UPLOAD_BYTES = int(os.environ.get("DUBBING_STUDIO_MAX_UPLOAD", 12 * 1024**3))
MOCK_MODE = os.environ.get("DUBBING_STUDIO_MOCK", "0") == "1"

LANGUAGES = [
    {"id": "zh", "label": "Китайский, упрощённый", "tts": "Chinese", "asr": "zh", "hymt": "Chinese", "sample": "你好！这是所选声音的示例。"},
    {"id": "en", "label": "Английский", "tts": "English", "asr": "en", "hymt": "English", "sample": "Hello! This is a preview of the selected voice."},
    {"id": "ru", "label": "Русский", "tts": "Russian", "asr": "ru", "hymt": "Russian", "sample": "Здравствуйте! Это пример выбранного голоса."},
    {"id": "de", "label": "Немецкий", "tts": "German", "asr": "de", "hymt": "German", "sample": "Hallo! Dies ist eine Vorschau der ausgewählten Stimme."},
    {"id": "fr", "label": "Французский", "tts": "French", "asr": "fr", "hymt": "French", "sample": "Bonjour ! Voici un aperçu de la voix sélectionnée."},
    {"id": "es", "label": "Испанский", "tts": "Spanish", "asr": "es", "hymt": "Spanish", "sample": "¡Hola! Esta es una muestra de la voz seleccionada."},
    {"id": "it", "label": "Итальянский", "tts": "Italian", "asr": "it", "hymt": "Italian", "sample": "Ciao! Questa è un'anteprima della voce selezionata."},
    {"id": "pt", "label": "Португальский", "tts": "Portuguese", "asr": "pt", "hymt": "Portuguese", "sample": "Olá! Esta é uma amostra da voz selecionada."},
    {"id": "ja", "label": "Японский", "tts": "Japanese", "asr": "ja", "hymt": "Japanese", "sample": "こんにちは。選択した音声のサンプルです。"},
    {"id": "ko", "label": "Корейский", "tts": "Korean", "asr": "ko", "hymt": "Korean", "sample": "안녕하세요. 선택한 목소리의 예시입니다."},
]

VOICE_PERSONAS = [
    {"id": "warm_female", "label": "Тёплый женский", "icon": "warm", "description": "A warm natural adult female voice, friendly and conversational, restrained emotion, clean studio recording."},
    {"id": "clear_male", "label": "Чистый мужской", "icon": "clear", "description": "A clear natural adult male voice, confident but not commercial, conversational cadence, clean studio recording."},
    {"id": "soft_young", "label": "Молодой мягкий", "icon": "young", "description": "A soft youthful voice, lively and sincere, natural conversational rhythm, clean studio recording."},
    {"id": "documentary", "label": "Документальный", "icon": "documentary", "description": "A calm mature documentary narrator, articulate and composed, natural pauses, clean studio recording."},
]

SUBTITLE_STYLES = [
    {"id": "clean", "label": "Чистый", "description": "Белый текст с тонкой обводкой", "force_style": "FontName=Arial,FontSize=16,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=1.6,Shadow=0,Alignment=2,MarginV=55"},
    {"id": "cinema", "label": "Кино", "description": "Мягкий жёлто-белый текст", "force_style": "FontName=Arial,FontSize=17,PrimaryColour=&H00F2F5FF,OutlineColour=&H000B0D12,BorderStyle=1,Outline=2.2,Shadow=0.5,Alignment=2,MarginV=58"},
    {"id": "social", "label": "Соцсети", "description": "Крупный жирный текст", "force_style": "FontName=Arial,FontSize=22,Bold=1,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=3,Shadow=0,Alignment=2,MarginV=72"},
    {"id": "boxed", "label": "Плашка", "description": "Текст на полупрозрачном фоне", "force_style": "FontName=Arial,FontSize=17,PrimaryColour=&H00FFFFFF,BackColour=&H90000000,OutlineColour=&H90000000,BorderStyle=3,Outline=6,Shadow=0,Alignment=2,MarginV=55"},
    {"id": "accent", "label": "Акцент", "description": "Жёлтый текст для коротких роликов", "force_style": "FontName=Arial,FontSize=20,Bold=1,PrimaryColour=&H0000E8FF,OutlineColour=&H00101010,BorderStyle=1,Outline=2.5,Shadow=0,Alignment=2,MarginV=65"},
    {"id": "minimal", "label": "Минимал", "description": "Небольшой спокойный текст", "force_style": "FontName=Arial,FontSize=14,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=1.2,Shadow=0,Alignment=2,MarginV=42"},
]

for path in (DATA_ROOT, PROJECTS_ROOT, CACHE_ROOT, PREVIEWS_ROOT):
    path.mkdir(parents=True, exist_ok=True)


def language(language_id: str) -> dict:
    return next(item for item in LANGUAGES if item["id"] == language_id)


def voice(voice_id: str) -> dict:
    return next(item for item in VOICE_PERSONAS if item["id"] == voice_id)


def subtitle_style(style_id: str) -> dict:
    return next(item for item in SUBTITLE_STYLES if item["id"] == style_id)
