"""Run provenance helpers for reproducible experiments."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    git_commit: str | None
    config_hash: str | None
    dataset_hashes: dict[str, str]
    model: str | None
    seed: int | None
    python_version: str
    platform: str
    created_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def create_run_manifest(
    *,
    run_id: str,
    config_path: str | Path | None = None,
    dataset_paths: Iterable[str | Path] = (),
    model: str | None = None,
    seed: int | None = None,
) -> RunManifest:
    dataset_hashes = {
        str(Path(path)): sha256_file(path)
        for path in dataset_paths
    }
    return RunManifest(
        run_id=run_id,
        git_commit=_git_commit(),
        config_hash=sha256_file(config_path) if config_path else None,
        dataset_hashes=dataset_hashes,
        model=model,
        seed=seed,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        created_at_utc=datetime.now(UTC).isoformat(timespec="seconds"),
    )
