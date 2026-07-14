"""Backend registry."""

from __future__ import annotations

from video2humanoid.extractors.gemx import GemXExtractor
from video2humanoid.interfaces import MotionExtractor, MotionRetargeter, Simulator
from video2humanoid.retargeters.soma_g1 import SomaG1Retargeter
from video2humanoid.simulators.isaac_kinematic import IsaacKinematicReplay


def get_extractor(name: str) -> MotionExtractor:
    """Return a motion extractor by name."""

    if name == "gemx":
        return GemXExtractor()
    raise ValueError(f"Unknown extractor: {name}")


def get_retargeter(name: str) -> MotionRetargeter:
    """Return a retargeter by name."""

    if name == "soma_g1":
        return SomaG1Retargeter()
    raise ValueError(f"Unknown retargeter: {name}")


def get_simulator(name: str) -> Simulator:
    """Return a simulator/replay backend by name."""

    if name == "isaac_kinematic":
        return IsaacKinematicReplay()
    raise ValueError(f"Unknown simulator: {name}")
