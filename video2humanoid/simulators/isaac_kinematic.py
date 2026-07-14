"""Isaac Sim kinematic replay backend."""

from __future__ import annotations

import logging
from pathlib import Path

from video2humanoid.config import ProjectConfig, resolve_path
from video2humanoid.interfaces import RetargetResult, SimulationResult, Simulator
from video2humanoid.utils.subprocess import run_command


class IsaacKinematicReplay(Simulator):
    """Generate a kinematic USD replay for Isaac Sim visualization."""

    def simulate(
        self,
        retarget: RetargetResult,
        output_dir: Path,
        config: ProjectConfig,
        logger: logging.Logger,
    ) -> SimulationResult:
        usd_dir = output_dir / "usd"
        screenshots_dir = output_dir / "screenshots"
        usd_dir.mkdir(parents=True, exist_ok=True)
        screenshots_dir.mkdir(parents=True, exist_ok=True)

        sample_name = output_dir.parent.name
        usd_path = usd_dir / f"{sample_name}_g1_kinematic_replay.usd"
        tools_root = resolve_path(config.simulator["tools_root"])
        conda_env = config.simulator.get("conda_env", config.retargeter.get("conda_env", "retarget-soma-ik"))
        fps = str(config.simulator.get("fps", config.retargeter.get("fps", 30)))

        command = [
                "env",
                "-u",
                "PYTHONPATH",
                "conda",
                "run",
                "-n",
                conda_env,
                "python",
                str(tools_root / "export_g1_kinematic_usd.py"),
                "--csv",
                str(retarget.robot_csv),
                "--output-usd",
                str(usd_path),
                "--fps",
                fps,
                "--human-bvh",
                str(retarget.soma_bvh),
            ]
        ground_alignment = config.ground_alignment
        if ground_alignment.get("enabled", True):
            command.extend(
                [
                    "--ground-aligned",
                    "--ground-scope",
                    str(ground_alignment.get("ground_scope", "feet")),
                    "--sole-clearance",
                    str(ground_alignment.get("sole_clearance_m", 0.0)),
                ]
            )
        run_command(command, logger)

        return SimulationResult(usd_path=usd_path, simulation_dir=output_dir)
