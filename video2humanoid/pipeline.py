"""End-to-end pipeline orchestration."""

from __future__ import annotations

import json
from pathlib import Path

from video2humanoid.config import ProjectConfig, resolve_path
from video2humanoid.logging_utils import setup_run_logger
from video2humanoid.registry import get_extractor, get_retargeter, get_simulator


def run_pipeline(input_video: Path, output_dir: Path, config: ProjectConfig) -> dict[str, str]:
    """Run extraction, retargeting, and simulation for one video."""

    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_run_logger(output_dir / "logs")
    logger.info("Starting pipeline")
    logger.info("Input video: %s", input_video)
    logger.info("Output directory: %s", output_dir)

    extractor_name = config.extractor.get("name", "gemx")
    retargeter_name = config.retargeter.get("name", "soma_g1")
    simulator_name = config.simulator.get("name", "isaac_kinematic")

    extraction = get_extractor(extractor_name).extract(input_video, output_dir / "human", config, logger)
    retarget = get_retargeter(retargeter_name).retarget(extraction, output_dir / "retarget", config, logger)
    simulation = get_simulator(simulator_name).simulate(retarget, output_dir / "simulation", config, logger)

    manifest = {
        "input_video": str(input_video),
        "hpe_results": str(extraction.hpe_results),
        "soma_bvh": str(retarget.soma_bvh),
        "robot_csv": str(retarget.robot_csv),
        "robot_npy": str(retarget.robot_npy),
        "usd_path": str(simulation.usd_path),
    }
    if retarget.raw_robot_csv is not None:
        manifest["raw_robot_csv"] = str(retarget.raw_robot_csv)
    if retarget.ground_alignment_report is not None:
        manifest["ground_alignment_report"] = str(retarget.ground_alignment_report)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Pipeline complete. Manifest: %s", manifest_path)
    return manifest


def default_output_dir(input_video: Path, config: ProjectConfig) -> Path:
    """Build a default output directory from config and video stem."""

    root = resolve_path(config.paths.get("output_root", "outputs"), config.config_path.parent.parent)
    return root / input_video.stem
