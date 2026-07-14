#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python -m pip install -e .
python -m pip install -r requirements.txt

echo "Installed Video2Humanoid-Retarget package."
echo "Run scripts/check_environment.sh to verify external robotics dependencies."
