#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(dirname "$(readlink -f "$0")")"
cd "$ROOT_DIR"

echo "[1/5] Installing system packages..."
sudo apt-get update
sudo apt-get install -y python3-full python3-venv python3-dev build-essential cmake \
  libopenblas-dev libsndfile1 portaudio19-dev alsa-utils espeak-ng curl git

echo "[2/5] Creating virtual environment..."
if [[ ! -d .venv ]]; then python3 -m venv .venv; fi
source .venv/bin/activate

echo "[3/5] Installing core Python dependencies..."
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

echo "[4/5] Creating data directory and local configuration..."
mkdir -p data
if [[ ! -f .env ]]; then cp .env.example .env; fi
chmod 600 .env

echo "[5/5] Core install complete."
echo "Start with: source .venv/bin/activate && python emo.py"
echo "Lowest-memory speech setup: bash setup_whisper_cpp.sh"
echo "Optional English-only Faster-Whisper: bash setup_english_whisper.sh"
echo "Optional offline translations: pip install -r requirements-translation.txt"
