#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

ARCH="$(uname -m)"
case "$ARCH" in
  armv7l|armhf)
    echo "Faster-Whisper needs a supported 64-bit Python/CTranslate2 environment here." >&2
    echo "Use 64-bit Raspberry Pi OS or run bash setup_whisper_cpp.sh instead." >&2
    exit 1
    ;;
esac

if [[ ! -x .venv/bin/python ]]; then
  echo "Creating EMO virtual environment..."
  python3 -m venv .venv
fi

echo "Installing EMO's core dependencies..."
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt

echo "Installing optional Faster-Whisper runtime..."
.venv/bin/python -m pip install -r requirements-faster-whisper.txt

if [[ ! -f .env ]]; then
  cp .env.example .env
fi
chmod 600 .env

MODEL_CACHE="${HOME}/.cache/emo/whisper-en"
mkdir -p "$MODEL_CACHE"
.venv/bin/python scripts/download_whisper_en.py --model tiny.en --cache-dir "$MODEL_CACHE"

# Configure EMO to use the downloaded English-only model without removing other settings.
EMO_WHISPER_CACHE="$MODEL_CACHE" .venv/bin/python - <<'PY'
import os
from pathlib import Path

env_path = Path(".env")
settings = {
    "WHISPER_BACKEND": "faster-whisper",
    "WHISPER_MODEL": "tiny.en",
    "WHISPER_DOWNLOAD_ROOT": os.environ["EMO_WHISPER_CACHE"],
}
lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
seen = set()
updated = []
for line in lines:
    stripped = line.lstrip()
    key = line.split("=", 1)[0].strip() if "=" in line and not stripped.startswith("#") else ""
    if key in settings:
        if key not in seen:
            updated.append(f"{key}={settings[key]}")
            seen.add(key)
    else:
        updated.append(line)
for key, value in settings.items():
    if key not in seen:
        updated.append(f"{key}={value}")
env_path.write_text("\n".join(updated) + "\n", encoding="utf-8")
env_path.chmod(0o600)
PY

echo
echo "English-only Whisper is installed and pre-downloaded."
echo "Configured .env: WHISPER_BACKEND=faster-whisper, WHISPER_MODEL=tiny.en"
echo "Start EMO with: source .venv/bin/activate && python emo.py"
