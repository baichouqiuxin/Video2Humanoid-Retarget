"""SOMA to Unitree G1 retargeter backend."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from video2humanoid.config import ProjectConfig, resolve_path
from video2humanoid.interfaces import ExtractionResult, MotionRetargeter, RetargetResult
from video2humanoid.utils.subprocess import run_command


class SomaG1Retargeter(MotionRetargeter):
    """Retarget GEM-X SOMA predictions to Unitree G1 via soma-retargeter/Newton."""

    def retarget(
        self,
        extraction: ExtractionResult,
        output_dir: Path,
        config: ProjectConfig,
        logger: logging.Logger,
    ) -> RetargetResult:
        bvh_dir = output_dir / "bvh"
        csv_dir = output_dir / "csv"
        motion_dir = output_dir / "motion"
        logs_dir = output_dir / "logs"
        for directory in (bvh_dir, csv_dir, motion_dir, logs_dir):
            directory.mkdir(parents=True, exist_ok=True)

        sample_name = output_dir.parent.name
        soma_bvh = bvh_dir / f"{sample_name}_soma.bvh"
        raw_robot_csv = csv_dir / f"{sample_name}_soma_raw.csv"
        robot_csv = raw_robot_csv
        robot_npy = output_dir / "robot.npy"
        metadata_json = output_dir / "robot_metadata.json"

        tools_root = resolve_path(config.retargeter["tools_root"])
        gemx_root = resolve_path(config.extractor["gemx_root"])
        reference_bvh = resolve_path(config.retargeter["reference_bvh"])
        gemx_env = config.extractor.get("conda_env", "gemx")
        retarget_env = config.retargeter.get("conda_env", "retarget-soma-ik")
        fps = str(config.retargeter.get("fps", 30))

        run_command(
            [
                "conda",
                "run",
                "-n",
                gemx_env,
                "python",
                str(tools_root / "export_gem_soma_bvh.py"),
                "--hpe",
                str(extraction.hpe_results),
                "--output-bvh",
                str(soma_bvh),
                "--fps",
                fps,
                "--gemx-root",
                str(gemx_root),
                "--reference-bvh",
                str(reference_bvh),
            ],
            logger,
        )

        batch_config = logs_dir / "bvh_to_g1_config.json"
        batch_config.write_text(
            json.dumps(
                {
                    "import_folder": str(bvh_dir),
                    "export_folder": str(csv_dir),
                    "batch_size": 1,
                    "retargeter": "Newton",
                    "retarget_source": "soma",
                    "retarget_target": "unitree_g1",
                    "retarget_source_facing_direction": config.retargeter.get("source_facing_direction", "Mujoco"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        soma_retargeter_root = resolve_path(config.retargeter["soma_retargeter_root"])
        run_command(
            [
                "env",
                "-u",
                "PYTHONPATH",
                "conda",
                "run",
                "-n",
                retarget_env,
                "python",
                str(soma_retargeter_root / "app" / "bvh_to_csv_converter.py"),
                "--config",
                str(batch_config),
            ],
            logger,
            cwd=soma_retargeter_root,
        )

        generated_csv = csv_dir / f"{soma_bvh.stem}.csv"
        if generated_csv != raw_robot_csv:
            generated_csv.rename(raw_robot_csv)

        ground_alignment = config.ground_alignment
        ground_report = None
        if ground_alignment.get("enabled", True):
            robot_csv = csv_dir / f"{sample_name}_soma_ground_aligned.csv"
            ground_report = logs_dir / "ground_alignment.json"
            foot_bodies = ground_alignment.get("foot_bodies", {})
            run_command(
                [
                    "env", "-u", "PYTHONPATH", "conda", "run", "-n", retarget_env, "python",
                    str(tools_root / "ground_align_g1_csv.py"),
                    "--input-csv", str(raw_robot_csv), "--output-csv", str(robot_csv), "--report", str(ground_report),
                    "--fps", fps,
                    "--left-foot-body", foot_bodies.get("left", "left_ankle_roll_link"),
                    "--right-foot-body", foot_bodies.get("right", "right_ankle_roll_link"),
                    "--contact-height-m", str(ground_alignment.get("contact_height_m", 0.06)),
                    "--max-vertical-speed-mps", str(ground_alignment.get("max_vertical_speed_mps", 0.5)),
                    "--max-horizontal-speed-mps", str(ground_alignment.get("max_horizontal_speed_mps", 0.8)),
                    "--contact-gap-frames", str(ground_alignment.get("contact_gap_frames", 2)),
                    "--min-contact-frames", str(ground_alignment.get("min_contact_frames", 2)),
                    "--max-down-step-m", str(ground_alignment.get("max_down_step_m", 0.03)),
                    "--sole-clearance-m", str(ground_alignment.get("sole_clearance_m", 0.0)),
                ],
                logger,
            )

        run_command(
            [
                "env",
                "-u",
                "PYTHONPATH",
                "conda",
                "run",
                "-n",
                retarget_env,
                "python",
                str(tools_root / "export_g1_robot_npy.py"),
                "--csv",
                str(robot_csv),
                "--output-npy",
                str(robot_npy),
                "--output-json",
                str(metadata_json),
                "--fps",
                fps,
            ],
            logger,
        )

        return RetargetResult(
            soma_bvh=soma_bvh,
            robot_csv=robot_csv,
            robot_npy=robot_npy,
            retarget_dir=output_dir,
            raw_robot_csv=raw_robot_csv,
            ground_alignment_report=ground_report,
        )
