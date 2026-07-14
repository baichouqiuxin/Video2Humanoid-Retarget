"""GEM-X motion extractor backend."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from video2humanoid.config import ProjectConfig, resolve_path
from video2humanoid.interfaces import ExtractionResult, MotionExtractor
from video2humanoid.utils.subprocess import run_command


class GemXExtractor(MotionExtractor):
    """Run GEM-X or reuse an existing GEM-X `hpe_results.pt`."""

    def extract(self, video_path: Path, output_dir: Path, config: ProjectConfig, logger: logging.Logger) -> ExtractionResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        hpe_results = output_dir / "hpe_results.pt"

        existing_hpe = config.extractor.get("existing_hpe_results")
        if existing_hpe:
            source = resolve_path(existing_hpe)
            logger.info("Reusing existing GEM-X HPE results: %s", source)
            shutil.copy2(source, hpe_results)
            return ExtractionResult(hpe_results=hpe_results, human_dir=output_dir)

        gemx_root = resolve_path(config.extractor["gemx_root"])
        conda_env = config.extractor.get("conda_env", "gemx")
        script = gemx_root / "scripts" / "demo" / "demo_soma.py"
        command = [
            "conda",
            "run",
            "-n",
            conda_env,
            "python",
            str(script),
            "--video",
            str(video_path),
            "--output_root",
            str(output_dir.parent),
        ]
        if config.extractor.get("static_cam", False):
            command.append("--static_cam")
        if config.extractor.get("retarget_in_gemx", False):
            command.append("--retarget")

        run_command(command, logger, cwd=gemx_root)
        gemx_output = output_dir.parent / video_path.stem / "hpe_results.pt"
        if not gemx_output.exists():
            raise FileNotFoundError(f"GEM-X did not produce {gemx_output}")
        shutil.copy2(gemx_output, hpe_results)
        return ExtractionResult(hpe_results=hpe_results, human_dir=output_dir)
