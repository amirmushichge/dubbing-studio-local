#!/bin/bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
export DUBBING_STUDIO_BACKEND="${DUBBING_STUDIO_BACKEND:-apple_silicon}"
export PYTHONUTF8=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

if [[ ! -x "$PYTHON" ]]; then
  echo "Dubbing Studio is not installed. Double-click setup.command first."
  read -r -p "Press Return to close..." _
  exit 1
fi

"$PYTHON" "$PROJECT_ROOT/tools/doctor.py"

if lsof -nP -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1; then
  if curl -fsS http://127.0.0.1:8765/api/health | grep -q 'dubbing-studio-local'; then
    open http://127.0.0.1:8765
    echo "Dubbing Studio is already running."
    exit 0
  fi
  echo "Port 8765 is used by another application. Close it and try again."
  read -r -p "Press Return to close..." _
  exit 1
fi

(sleep 2; open http://127.0.0.1:8765) &
echo "Dubbing Studio: http://127.0.0.1:8765"
echo "Close this window or press Control-C to stop the studio."
exec "$PYTHON" -m uvicorn app.main:app --host 127.0.0.1 --port 8765 --app-dir "$PROJECT_ROOT"
