"""Configuration loading for Video2Humanoid-Retarget."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised only without PyYAML.
    yaml = None


@dataclass(frozen=True)
class ProjectConfig:
    """Runtime configuration resolved from YAML files."""

    raw: dict[str, Any]
    config_path: Path

    @property
    def paths(self) -> dict[str, Any]:
        return self.raw.get("paths", {})

    @property
    def extractor(self) -> dict[str, Any]:
        return self.raw.get("extractor", {})

    @property
    def retargeter(self) -> dict[str, Any]:
        return self.raw.get("retargeter", {})

    @property
    def ground_alignment(self) -> dict[str, Any]:
        return self.raw.get("ground_alignment", {})

    @property
    def simulator(self) -> dict[str, Any]:
        return self.raw.get("simulator", {})


def load_config(path: str | Path) -> ProjectConfig:
    """Load a YAML project configuration."""

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = _load_yaml(handle.read())
    return ProjectConfig(raw=raw, config_path=config_path)


def resolve_path(value: str | Path, base_dir: Path | None = None) -> Path:
    """Resolve a user path with `~` and optional relative base support."""

    path = Path(os.path.expandvars(str(value))).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve()


def _load_yaml(text: str) -> dict[str, Any]:
    if yaml is not None:
        return yaml.safe_load(text) or {}
    return _load_simple_yaml(text)


def _load_simple_yaml(text: str) -> dict[str, Any]:
    """Small fallback for simple nested mapping configs.

    PyYAML remains the supported parser. This fallback keeps smoke tests and
    environment checks usable before dependencies are installed.
    """

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line_without_comment = raw_line.split("#", 1)[0].rstrip()
        if not line_without_comment.strip():
            continue
        indent = len(line_without_comment) - len(line_without_comment.lstrip(" "))
        key, separator, value = line_without_comment.strip().partition(":")
        if not separator:
            continue
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        value = value.strip()
        if value:
            parent[key] = _parse_scalar(value)
            continue
        child: dict[str, Any] = {}
        parent[key] = child
        stack.append((indent, child))
    return root


def _parse_scalar(value: str) -> Any:
    if value in {"null", "None", "~"}:
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
