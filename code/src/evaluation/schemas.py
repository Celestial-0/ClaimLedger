"""Versioned machine-readable result contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class MetricRecord:
    """One aggregate metric with its denominator and uncertainty metadata."""

    run_id: str
    system: str
    dataset: str
    condition: str
    metric: str
    value: float
    numerator: int | None = None
    denominator: int | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    ci_method: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
