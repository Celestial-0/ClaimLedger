"""Internal I/O and hashing utilities shared by dataset adapters."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
import gzip
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    """Compute sha256 hex digest of a file in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> Any:
    """Load JSON from a file path."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: Any, indent: int = 2) -> Path:
    """Write JSON to a file path, ensuring parent directories exist."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=indent, sort_keys=True) + "\n", encoding="utf-8")
    return target


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read all JSONL rows from a file (transparently supports .gz or uncompressed)."""
    p = Path(path)
    if not p.exists() and p.with_suffix(p.suffix + ".gz").exists():
        p = p.with_suffix(p.suffix + ".gz")

    if p.suffix == ".gz" or str(p).endswith(".jsonl.gz"):
        with gzip.open(p, "rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with p.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> Path:
    """Write records to a JSONL file, ensuring parent directories exist."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return target
