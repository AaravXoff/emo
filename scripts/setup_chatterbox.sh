#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
FORCE=""
if [[ "$#" -gt 0 ]]; then FORCE="$1"; fi
RAM_KB="$(awk '/MemTotal:/ { print $2; exit }' /proc/meminfo)"
RAM_MB="$((RAM_KB / 1024))"
ARCH="$(uname -m)"
if [[ "$RAM_MB" -lt 2200 && "$FORCE" != "--force" ]]; then
  echo "Refusing optional Chatterbox setup on this $RAM_MB MB machine." >&2
  echo "Its neural/Torch runtime is not suitable for a stock Pi 3B+ (about 1 GB RAM)." >&2
  echo "On a capable computer use: bash scripts/setup_chatterbox.sh --force" >&2
  exit 2
fi
case "$ARCH" in
  aarch64|arm64|x86_64) ;;
  *) echo "Untested Chatterbox architecture: $ARCH" >&2; exit 2 ;;
esac
if [[ ! -x .venv/bin/python ]]; then
  echo "Create the environment first with bash setup_emo_robot.sh" >&2
  exit 2
fi
.venv/bin/python -m pip install -r requirements-chatterbox.txt
echo "Set TTS_ENGINE=chatterbox in .env to opt in. Expect much higher RAM use than espeak-ng."
