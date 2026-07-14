#!/usr/bin/env bash
set -euo pipefail

DEPS_ROOT="${DEPS_ROOT:-$HOME/video2humanoid_deps}"
mkdir -p "$DEPS_ROOT"

clone_if_missing() {
  local url="$1"
  local dst="$2"
  if [[ -d "$dst/.git" ]]; then
    echo "Already exists: $dst"
  else
    git clone "$url" "$dst"
  fi
}

clone_if_missing "https://github.com/NVIDIA/soma-retargeter.git" "$DEPS_ROOT/soma-retargeter"
clone_if_missing "https://github.com/NVlabs/SOMA-X.git" "$DEPS_ROOT/SOMA-X"
clone_if_missing "https://github.com/NVlabs/ProtoMotions.git" "$DEPS_ROOT/ProtoMotions"

cat <<'MSG'

Manual steps still required:
1. Install Isaac Sim 5.x from NVIDIA official installer.
2. Install or clone GEM-X according to its official instructions.
3. Download licensed SMPL/SOMA assets where required by upstream projects.
4. Run scripts/check_environment.sh and update configs/config.yaml paths.

MSG
