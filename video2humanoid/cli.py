"""Console entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from video2humanoid.config import load_config, resolve_path
from video2humanoid.pipeline import default_output_dir, run_pipeline


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Video-based humanoid retargeting pipeline")
    parser.add_argument("--input", required=True, type=Path, help="Input monocular video")
    parser.add_argument("--output", type=Path, default=None, help="Output directory")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"), help="YAML config")
    return parser.parse_args()


def main() -> None:
    """Run the command-line pipeline."""

    args = parse_args()
    config = load_config(args.config)
    input_video = resolve_path(args.input)
    output_dir = resolve_path(args.output) if args.output else default_output_dir(input_video, config)
    run_pipeline(input_video=input_video, output_dir=output_dir, config=config)
