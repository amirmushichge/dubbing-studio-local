#!/bin/bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
export DUBBING_STUDIO_BACKEND="${DUBBING_STUDIO_BACKEND:-apple_silicon}"
export PYTORCH_ENABLE_MPS_FALLBACK=1

if [[ ! -x "$PYTHON" ]]; then
  echo "The application environment is missing. Run setup.command."
  exit 1
fi
exec "$PYTHON" "$PROJECT_ROOT/tools/doctor.py" "$@"
