#!/usr/bin/env bash
set -euo pipefail

echo "[Video2Humanoid] Environment check"

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "MISSING: $1" >&2
    return 1
  fi
  echo "OK: $1 -> $(command -v "$1")"
}

require_path() {
  if [[ ! -e "$1" ]]; then
    echo "MISSING: $1" >&2
    return 1
  fi
  echo "OK: $1"
}

require_cmd python
require_cmd conda
require_cmd git

require_path "${GEMX_ROOT:?Set GEMX_ROOT}"
require_path "${SOMA_RETARGETER_ROOT:?Set SOMA_RETARGETER_ROOT}"
require_path "${ISAAC_ROOT:?Set ISAAC_ROOT}"
require_path "${SOMA_ASSETS_ROOT:?Set SOMA_ASSETS_ROOT}/SOMA_neutral.npz"

conda env list | grep -E 'gemx|retarget-soma-ik' || {
  echo "MISSING: expected conda envs gemx and retarget-soma-ik" >&2
  exit 1
}

echo "Environment check complete."
