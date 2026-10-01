"""Serialization helpers for result artifacts consumed by plots and LaTeX."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from .schemas import MetricRecord


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_metric_records(path: str | Path, records: Iterable[MetricRecord]) -> None:
    rows = [record.to_dict() for record in records]
    if not rows:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
