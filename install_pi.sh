#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(dirname "$(readlink -f "$0")")"
exec bash "$ROOT_DIR/setup_emo_robot.sh" "$@"
