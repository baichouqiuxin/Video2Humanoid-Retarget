"""Subprocess utilities with logging."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Sequence


def run_command(command: Sequence[str], logger: logging.Logger, cwd: Path | None = None) -> None:
    """Run a command and stream captured output to logs."""

    logger.info("Running command: %s", " ".join(str(part) for part in command))
    result = subprocess.run(
        [str(part) for part in command],
        cwd=str(cwd) if cwd else None,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.stdout:
        logger.info(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}: {' '.join(command)}")
