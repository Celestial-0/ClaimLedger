"""Summary-based memory baseline (MemGPT/Letta-style approximation).

Compresses all claim evidence into a single summary blob.
No individual claim tracking, no governance, no conflict detection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..evaluation.scenarios import ScenarioOutcome

if TYPE_CHECKING:
    from ..evaluation.benchmark import BenchmarkCase


def run_summary_memory_cases(cases: list[BenchmarkCase]) -> list[ScenarioOutcome]:
    """Summary memory baseline: absorb everything, lose claim-level tracking."""
    outcomes: list[ScenarioOutcome] = []

    for case in cases:
        # All claims are absorbed into the summary — no governance
        actual_active = len(case.claims)

        outcomes.append(ScenarioOutcome(
            name=case.task_type,
            expected_active=case.expected_active,
            actual_active=actual_active,
            conflict_disclosed=False,
            poisoned_active=any(c.is_poisoned for c in case.claims),
            abstained=False,
        ))

    return outcomes
