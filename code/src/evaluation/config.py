"""Dependency-free experiment configuration loading."""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    datasets: tuple[str, ...]
    systems: tuple[str, ...]
    seeds: tuple[int, ...]
    models: tuple[str, ...]
    output_dir: str


def load_config(path: str | Path) -> ExperimentConfig:
    source = Path(path)
    if source.suffix.lower() == ".json":
        raw = json.loads(source.read_text(encoding="utf-8"))
    elif source.suffix.lower() in {".toml", ".tml"}:
        with source.open("rb") as handle:
            raw = tomllib.load(handle)
    else:
        raise ValueError(f"unsupported config format: {source.suffix}")
    if source.name == "pyproject.toml":
        raw = _project_experiment_mapping(raw)
    return _from_mapping(raw)


def _project_experiment_mapping(raw: dict[str, Any]) -> dict[str, Any]:
    """Read ClaimLedger's experiment profile from ``[tool.claimledger.experiment]``."""
    tool = raw.get("tool")
    if isinstance(tool, dict):
        claimledger = tool.get("claimledger")
        if isinstance(claimledger, dict):
            experiment = claimledger.get("experiment")
            if isinstance(experiment, dict):
                return experiment
    raise ValueError("pyproject.toml is missing [tool.claimledger.experiment]")


def _from_mapping(raw: dict[str, Any]) -> ExperimentConfig:
    required = {"name", "datasets", "systems", "seeds", "models", "output_dir"}
    missing = required.difference(raw)
    if missing:
        raise ValueError(f"missing config keys: {sorted(missing)}")
    return ExperimentConfig(
        name=str(raw["name"]),
        datasets=tuple(str(item) for item in raw["datasets"]),
        systems=tuple(str(item) for item in raw["systems"]),
        seeds=tuple(int(item) for item in raw["seeds"]),
        models=tuple(str(item) for item in raw["models"]),
        output_dir=str(raw["output_dir"]),
    )
