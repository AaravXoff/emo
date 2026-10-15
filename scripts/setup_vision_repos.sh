#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/vendor"
clone_or_update() {
  local url="$1" dir="$2"
  if [[ -d "$dir/.git" ]]; then
    git -C "$dir" pull --ff-only
  elif [[ -e "$dir" ]]; then
    echo "Existing non-git directory: $dir" >&2
    return 2
  else
    git clone --depth 1 "$url" "$dir"
  fi
}
clone_or_update https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe.git "$ROOT/vendor/hand-gesture-recognition-using-mediapipe"
clone_or_update https://github.com/ageitgey/face_recognition.git "$ROOT/vendor/face_recognition"
clone_or_update https://github.com/ultralytics/ultralytics.git "$ROOT/vendor/ultralytics"
echo "Vision source checkouts are under vendor/."
echo "Install optional heavy dependencies separately: .venv/bin/pip install -r requirements-vision.txt"
echo "On Pi 3B+ enable one vision task at a time; face recognition and YOLO can exceed available memory."
