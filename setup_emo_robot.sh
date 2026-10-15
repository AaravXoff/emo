#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
cd "$ROOT_DIR"

echo "[1/7] Installing Raspberry Pi OS packages..."
sudo apt-get update
sudo apt-get install -y \
  python3-full python3-venv python3-dev build-essential cmake pkg-config \
  libopenblas-dev libatlas-base-dev libsndfile1 portaudio19-dev libasound2-dev \
  alsa-utils espeak-ng mpg123 curl git unzip

ARCH="$(uname -m)"
case "$ARCH" in
  aarch64|arm64) echo "64-bit ARM detected." ;;
  *) echo "Architecture: $ARCH. Faster-Whisper/CTranslate2 support may be limited." ;;
esac

echo "[2/7] Creating virtual environment and installing core dependencies..."
if [[ ! -x .venv/bin/python ]]; then python3 -m venv .venv; fi
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt

echo "[3/7] Preparing local configuration..."
mkdir -p data assets/songs
if [[ ! -f .env ]]; then cp .env.example .env; fi
chmod 600 .env

echo "[4/7] Installing/starting Ollama and downloading the configured model..."
if ! command -v ollama >/dev/null 2>&1; then
  curl -fsSL https://ollama.com/install.sh | sh
fi
if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama is not available; install it manually and re-run setup." >&2
  exit 1
fi
if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files ollama.service >/dev/null 2>&1; then
  sudo systemctl enable --now ollama || true
fi
MODEL="$(.venv/bin/python -c 'from config import Config; print(Config.from_env().ollama_model)')"
ollama pull "$MODEL"

echo "[5/7] Installing/pre-downloading English-only Whisper tiny.en..."
bash "$ROOT_DIR/setup_english_whisper.sh"

echo "[6/7] Downloading the Vosk wake-word model..."
.venv/bin/python "$ROOT_DIR/scripts/download_vosk_model.py"

echo "[7/7] Creating a private remote-control token if none is set..."
.venv/bin/python - <<'PY'
import secrets
from pathlib import Path

path = Path(".env")
lines = path.read_text(encoding="utf-8").splitlines()
token_lines = [i for i, line in enumerate(lines) if line.startswith("REMOTE_CONTROL_TOKEN=")]
if token_lines and not lines[token_lines[0]].split("=", 1)[1].strip():
    lines[token_lines[0]] = "REMOTE_CONTROL_TOKEN=" + secrets.token_urlsafe(32)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o600)
elif not token_lines:
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\nREMOTE_CONTROL_TOKEN=" + secrets.token_urlsafe(32) + "\n")
    path.chmod(0o600)
PY

echo
echo "Setup finished. Start with:"
echo "  cd '$ROOT_DIR' && source .venv/bin/activate && python emo.py"
echo "Configure the correct SPI LCD kernel driver before expecting Pygame output on that panel."
echo "Optional vision sources: bash scripts/setup_vision_repos.sh"
echo "Optional Chatterbox: bash scripts/setup_chatterbox.sh (not recommended on Pi 3B+ RAM)"
