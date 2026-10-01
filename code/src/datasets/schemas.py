"""Dataset-level schemas shared by benchmark adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DatasetExample:
    example_id: str
    question: str
    answer: str | None
    task_type: str | None
    evidence: tuple[str, ...]
    raw: dict[str, Any]
