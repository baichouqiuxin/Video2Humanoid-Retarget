"""Plugin interfaces for extraction, retargeting, and simulation."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from video2humanoid.config import ProjectConfig


@dataclass(frozen=True)
class ExtractionResult:
    """Artifacts produced by a motion extractor."""

    hpe_results: Path
    human_dir: Path


@dataclass(frozen=True)
class RetargetResult:
    """Artifacts produced by a retargeter."""

    soma_bvh: Path
    robot_csv: Path
    robot_npy: Path
    retarget_dir: Path
    raw_robot_csv: Path | None = None
    ground_alignment_report: Path | None = None


@dataclass(frozen=True)
class SimulationResult:
    """Artifacts produced by a simulator/replay backend."""

    usd_path: Path
    simulation_dir: Path


class MotionExtractor(ABC):
    """Base class for video-to-human-motion extractors."""

    @abstractmethod
    def extract(self, video_path: Path, output_dir: Path, config: ProjectConfig, logger: logging.Logger) -> ExtractionResult:
        """Extract human motion from a monocular video."""


class MotionRetargeter(ABC):
    """Base class for human-to-robot retargeters."""

    @abstractmethod
    def retarget(
        self,
        extraction: ExtractionResult,
        output_dir: Path,
        config: ProjectConfig,
        logger: logging.Logger,
    ) -> RetargetResult:
        """Retarget extracted human motion to robot motion."""


class Simulator(ABC):
    """Base class for simulation or replay backends."""

    @abstractmethod
    def simulate(
        self,
        retarget: RetargetResult,
        output_dir: Path,
        config: ProjectConfig,
        logger: logging.Logger,
    ) -> SimulationResult:
        """Create simulation/replay outputs."""
