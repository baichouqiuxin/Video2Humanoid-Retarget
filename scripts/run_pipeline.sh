#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_pipeline.sh input/my_video.mp4 [outputs/my_video]" >&2
  exit 1
fi

INPUT_VIDEO="$1"
OUTPUT_DIR="${2:-}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$REPO_ROOT"
if [[ -n "$OUTPUT_DIR" ]]; then
  python run.py --input "$INPUT_VIDEO" --output "$OUTPUT_DIR"
else
  python run.py --input "$INPUT_VIDEO"
fi
