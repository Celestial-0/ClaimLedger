"""Statistical helpers shared by benchmark runners and report generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

from .metrics import PairedComparison, paired_exact_active_comparison
from .scenarios import ScenarioOutcome


@dataclass(frozen=True)
class PairedTestRecord:
    system: str
    baseline: str
    system_accuracy: float
    baseline_accuracy: float
    system_wins: int
    baseline_wins: int
    ties: int
    exact_p: float
    holm_adjusted_p: float

    @classmethod
    def from_comparison(
        cls,
        *,
        system: str,
        baseline: str,
        comparison: PairedComparison,
        holm_adjusted_p: float,
    ) -> "PairedTestRecord":
        return cls(
            system=system,
            baseline=baseline,
            system_accuracy=comparison.system_accuracy,
            baseline_accuracy=comparison.baseline_accuracy,
            system_wins=comparison.system_wins,
            baseline_wins=comparison.baseline_wins,
            ties=comparison.ties,
            exact_p=comparison.mcnemar_exact_p,
            holm_adjusted_p=holm_adjusted_p,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def paired_comparisons(
    *,
    system: str,
    system_outcomes: list[ScenarioOutcome],
    baseline_outcomes: Mapping[str, list[ScenarioOutcome]],
) -> list[PairedTestRecord]:
    comparisons = {
        baseline: paired_exact_active_comparison(system_outcomes, outcomes)
        for baseline, outcomes in baseline_outcomes.items()
    }
    adjusted = holm_bonferroni({name: result.mcnemar_exact_p for name, result in comparisons.items()})
    return [
        PairedTestRecord.from_comparison(
            system=system,
            baseline=baseline,
            comparison=comparison,
            holm_adjusted_p=adjusted[baseline],
        )
        for baseline, comparison in comparisons.items()
    ]


def holm_bonferroni(p_values: Mapping[str, float]) -> dict[str, float]:
    """Return Holm step-down adjusted p-values in the original key order."""

    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, (name, p_value) in enumerate(ordered):
        corrected = min(1.0, (count - index) * p_value)
        running = max(running, corrected)
        adjusted[name] = running
    return adjusted
