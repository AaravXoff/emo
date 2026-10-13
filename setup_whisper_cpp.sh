#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/whisper.cpp"
JOBS="1"

echo "[1/4] Installing native build tools..."
sudo apt-get update
sudo apt-get install -y git cmake build-essential

echo "[2/4] Getting Whisper.cpp source..."
if [[ -d "$ROOT/.git" ]]; then
  git -C "$ROOT" pull --ff-only
elif [[ -e "$ROOT" ]]; then
  echo "Path already exists but is not a Git checkout: $ROOT" >&2
  echo "Move that folder aside first if it is not a Whisper.cpp checkout." >&2
  exit 2
else
  git clone https://github.com/ggml-org/whisper.cpp.git "$ROOT"
fi

echo "[3/4] Building for this Raspberry Pi (jobs=$JOBS)..."
cmake -S "$ROOT" -B "$ROOT/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROOT/build" -j"$JOBS" --config Release

echo "[4/4] Downloading the small multilingual tiny model..."
bash "$ROOT/models/download-ggml-model.sh" tiny

echo
echo "Whisper.cpp setup complete."
echo "Expected binary: $ROOT/build/bin/whisper-cli"
echo "Expected model:  $ROOT/models/ggml-tiny.bin"
echo "Check your .env settings:"
echo "  WHISPER_BACKEND=cpp"
echo "  WHISPER_CPP_BIN=$ROOT/build/bin/whisper-cli"
echo "  WHISPER_CPP_MODEL=$ROOT/models/ggml-tiny.bin"
echo "Then run: cd ~/emo && source .venv/bin/activate && python emo.py"
