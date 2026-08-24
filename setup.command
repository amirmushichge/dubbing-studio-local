#!/bin/bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
RUNTIME_ROOT="${DUBBING_STUDIO_RUNTIME:-$PROJECT_ROOT/runtime}"
LOG_ROOT="$PROJECT_ROOT/logs"
PLAN=0
SKIP_MODELS=0
SKIP_SHORTCUT=0

for argument in "$@"; do
  case "$argument" in
    --plan) PLAN=1 ;;
    --skip-models) SKIP_MODELS=1 ;;
    --skip-shortcut) SKIP_SHORTCUT=1 ;;
    *) echo "Unknown option: $argument"; exit 2 ;;
  esac
done

mkdir -p "$LOG_ROOT"
LOG_FILE="$LOG_ROOT/setup-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

step() { printf '\n==> %s\n' "$1"; }
run() {
  if [[ "$PLAN" == 1 ]]; then printf '[plan]'; printf ' %q' "$@"; printf '\n';
  else "$@"; fi
}

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "This installer requires an Apple Silicon Mac (M1 or newer). Intel Macs are not supported."
  exit 1
fi

MAC_MAJOR="$(sw_vers -productVersion | cut -d. -f1)"
if (( MAC_MAJOR < 14 )); then
  echo "macOS 14 Sonoma or newer is required."
  exit 1
fi

MEMORY_GB="$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))"
if (( MEMORY_GB < 16 )); then
  echo "At least 16 GB of unified memory is required; 32 GB is recommended."
  exit 1
fi
TRANSLATION_MODEL="7b"
REQUIRED_GB=45
if (( MEMORY_GB < 32 )); then
  TRANSLATION_MODEL="1.8b"
  REQUIRED_GB=35
  echo "Using Hy-MT2 1.8B because this Mac has ${MEMORY_GB} GB. Translation quality can be lower than on 32 GB+ Macs."
fi
FREE_GB="$(df -Pk "$PROJECT_ROOT" | awk 'NR==2 {print int($4/1024/1024)}')"
if [[ "$SKIP_MODELS" == 0 ]] && (( FREE_GB < REQUIRED_GB )); then
  echo "Only ${FREE_GB} GB is free. This setup needs at least ${REQUIRED_GB} GB."
  exit 1
fi

step "Checking Apple developer tools"
if ! xcode-select -p >/dev/null 2>&1; then
  if [[ "$PLAN" == 1 ]]; then echo "[plan] request Xcode Command Line Tools";
  else
    xcode-select --install || true
    echo "Finish the Apple Command Line Tools installation, then run setup.command again."
    exit 1
  fi
fi

step "Checking Homebrew"
if ! command -v brew >/dev/null 2>&1; then
  if [[ "$PLAN" == 1 ]]; then echo "[plan] install Homebrew";
  else
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  fi
fi
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

step "Installing Python, Git, FFmpeg and audio libraries"
run brew install python@3.10 git ffmpeg libsndfile sox
BASE_PYTHON="$(brew --prefix python@3.10 2>/dev/null || true)/bin/python3.10"
if [[ "$PLAN" == 0 && ! -x "$BASE_PYTHON" ]]; then
  echo "Python 3.10 was installed but could not be found. Run setup.command again."
  exit 1
fi

ensure_venv() {
  local parent="$1"
  local python="$parent/.venv/bin/python"
  if [[ -x "$python" ]] && "$python" -c 'import sys; assert sys.version_info[:2] == (3, 10)' >/dev/null 2>&1; then return; fi
  if [[ -d "$parent/.venv" ]]; then
    run mv "$parent/.venv" "$parent/.venv.broken-$(date +%Y%m%d-%H%M%S)"
  fi
  run mkdir -p "$parent"
  run "$BASE_PYTHON" -m venv "$parent/.venv"
}

pip_install() {
  local python="$1"; shift
  run "$python" -m pip "$@"
}

ensure_repo() {
  local url="$1" target="$2" commit="$3"
  if [[ ! -d "$target/.git" ]]; then run git clone "$url" "$target"; fi
  if [[ "$PLAN" == 0 ]]; then
    git -C "$target" fetch --quiet origin "$commit"
    git -C "$target" checkout --quiet --detach "$commit"
  else echo "[plan] pin $target to $commit"; fi
}

mkdir -p "$RUNTIME_ROOT"
ensure_venv "$PROJECT_ROOT"
APP_PYTHON="$PROJECT_ROOT/.venv/bin/python"
step "Installing the local web application"
pip_install "$APP_PYTHON" install --upgrade pip wheel
pip_install "$APP_PYTHON" install -r "$PROJECT_ROOT/requirements.txt"

ASR_ROOT="$RUNTIME_ROOT/asr"
ensure_venv "$ASR_ROOT"
ASR_PYTHON="$ASR_ROOT/.venv/bin/python"
step "Installing speech recognition and source separation"
pip_install "$ASR_PYTHON" install --upgrade pip wheel
pip_install "$ASR_PYTHON" install torch==2.7.1 torchaudio==2.7.1
pip_install "$ASR_PYTHON" install -r "$PROJECT_ROOT/requirements/asr.txt"

QWEN_ROOT="$RUNTIME_ROOT/qwen3-tts"
step "Installing Qwen3-TTS"
ensure_repo "https://github.com/QwenLM/Qwen3-TTS.git" "$QWEN_ROOT" "022e286b98fbec7e1e916cb940cdf532cd9f488e"
ensure_venv "$QWEN_ROOT"
QWEN_PYTHON="$QWEN_ROOT/.venv/bin/python"
pip_install "$QWEN_PYTHON" install --upgrade pip wheel
pip_install "$QWEN_PYTHON" install torch==2.7.1 torchaudio==2.7.1
pip_install "$QWEN_PYTHON" install -e "$QWEN_ROOT"

HYMT_ROOT="$RUNTIME_ROOT/hymt"
ensure_venv "$HYMT_ROOT"
HYMT_PYTHON="$HYMT_ROOT/.venv/bin/python"
step "Installing the local Hy-MT2 translator"
pip_install "$HYMT_PYTHON" install --upgrade pip wheel
pip_install "$HYMT_PYTHON" install torch==2.7.1
pip_install "$HYMT_PYTHON" install -r "$PROJECT_ROOT/requirements/macos-hymt.txt"

SEED_ROOT="$RUNTIME_ROOT/seed-vc"
step "Installing Seed-VC and voice quality tools"
ensure_repo "https://github.com/Plachtaa/seed-vc.git" "$SEED_ROOT" "51383efd921027683c89e5348211d93ff12ac2a8"
ensure_venv "$SEED_ROOT"
SEED_PYTHON="$SEED_ROOT/.venv/bin/python"
pip_install "$SEED_PYTHON" install --upgrade pip wheel
pip_install "$SEED_PYTHON" install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
pip_install "$SEED_PYTHON" install -r "$PROJECT_ROOT/requirements/seedvc-extra.txt"
pip_install "$SEED_PYTHON" install --upgrade --no-deps -r "$PROJECT_ROOT/requirements/seedvc-overrides.txt"

if [[ "$SKIP_MODELS" == 0 ]]; then
  step "Downloading local AI models; interrupted downloads resume automatically"
  run "$APP_PYTHON" "$PROJECT_ROOT/tools/download_models.py" --runtime "$RUNTIME_ROOT" --profile apple_silicon --translation-model "$TRANSLATION_MODEL"
  step "Preparing Demucs"
  if [[ "$PLAN" == 0 ]]; then
    TORCH_HOME="$ASR_ROOT/models/torch" PYTORCH_ENABLE_MPS_FALLBACK=1 "$ASR_PYTHON" "$PROJECT_ROOT/tools/prefetch_demucs.py"
  else echo "[plan] prefetch Demucs model"; fi
fi

if [[ "$PLAN" == 0 ]]; then
  export DUBBING_STUDIO_BACKEND=apple_silicon
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  printf '{"installed_at":"%s","runtime":"%s","backend":"apple_silicon","translation_model":"%s","version":"0.1.0-alpha.1"}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$RUNTIME_ROOT" "$TRANSLATION_MODEL" > "$RUNTIME_ROOT/install-state.json"
  chmod +x "$PROJECT_ROOT/setup.command" "$PROJECT_ROOT/start.command" "$PROJECT_ROOT/doctor.command"
  step "Running final diagnostics"
  "$PROJECT_ROOT/doctor.command"
  if [[ "$SKIP_SHORTCUT" == 0 ]]; then
    ln -sfn "$PROJECT_ROOT/start.command" "$HOME/Desktop/Dubbing Studio.command"
    echo "Desktop launcher created: Dubbing Studio.command"
  fi
  echo "Dubbing Studio for Apple Silicon installed successfully."
else
  echo "Setup plan validated successfully; nothing was installed."
fi
echo "Log: $LOG_FILE"
