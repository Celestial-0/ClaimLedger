"""Provenance-only control baseline.

The control keeps claims with non-empty evidence and source fields, but it does
not apply temporal, conflict, trust, or poison governance. The input contract
already requires these fields, so this baseline is expected to tie with naive
memory on the current benchmark and makes that limitation explicit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..evaluation.scenarios import ScenarioOutcome

if TYPE_CHECKING:
    from ..evaluation.benchmark import BenchmarkCase


def run_provenance_only_cases(cases: list[BenchmarkCase]) -> list[ScenarioOutcome]:
    outcomes: list[ScenarioOutcome] = []
    for case in cases:
        active = [claim for claim in case.claims if claim.evidence.strip() and claim.source.strip()]
        outcomes.append(
            ScenarioOutcome(
                name=case.task_type,
                expected_active=case.expected_active,
                actual_active=len(active),
                conflict_disclosed=False,
                poisoned_active=any(claim.is_poisoned for claim in active),
                abstained=len(active) == 0,
            )
        )
    return outcomes
