#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/whisper.cpp"
JOBS="1"
PROJECT_ROOT="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"

echo "[1/5] Installing native build tools..."
sudo apt-get update
sudo apt-get install -y git cmake build-essential

echo "[2/5] Getting Whisper.cpp source..."
if [[ -d "$ROOT/.git" ]]; then
  git -C "$ROOT" pull --ff-only
elif [[ -e "$ROOT" ]]; then
  echo "Path already exists but is not a Git checkout: $ROOT" >&2
  echo "Move that folder aside first if it is not a Whisper.cpp checkout." >&2
  exit 2
else
  git clone https://github.com/ggml-org/whisper.cpp.git "$ROOT"
fi

echo "[3/5] Building on this Raspberry Pi (jobs=$JOBS)..."
cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROOT/build" -j"$JOBS" --config Release

echo "[4/5] Downloading English-only tiny.en weights..."
bash "$ROOT/models/download-ggml-model.sh" tiny.en
MODEL="$ROOT/models/ggml-tiny.en.bin"
BINARY="$ROOT/build/bin/whisper-cli"
if [[ ! -x "$BINARY" || ! -f "$MODEL" ]]; then
  echo "Whisper.cpp binary/model validation failed." >&2
  exit 1
fi

echo "[5/5] Selecting Whisper.cpp in local .env..."
if [[ ! -f "$PROJECT_ROOT/.env" ]]; then cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"; fi
PROJECT_ROOT="$PROJECT_ROOT" WHISPER_BIN="$BINARY" WHISPER_MODEL_PATH="$MODEL" python3 - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["PROJECT_ROOT"]) / ".env"
settings = {
    "WHISPER_BACKEND": "cpp",
    "WHISPER_MODEL": "tiny.en",
    "WHISPER_CPP_BIN": os.environ["WHISPER_BIN"],
    "WHISPER_CPP_MODEL": os.environ["WHISPER_MODEL_PATH"],
}
lines = path.read_text(encoding="utf-8").splitlines()
seen = set()
output = []
for line in lines:
    key = line.split("=", 1)[0].strip() if "=" in line and not line.lstrip().startswith("#") else ""
    if key in settings:
        if key not in seen:
            output.append(f"{key}={settings[key]}")
            seen.add(key)
    else:
        output.append(line)
for key, value in settings.items():
    if key not in seen:
        output.append(f"{key}={value}")
path.write_text("\n".join(output) + "\n", encoding="utf-8")
path.chmod(0o600)
PY

echo "Whisper.cpp is ready for English-only transcription."
echo "Start with: cd '$PROJECT_ROOT' && source .venv/bin/activate && python emo.py"
