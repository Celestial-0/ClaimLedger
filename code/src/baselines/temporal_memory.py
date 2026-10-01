"""Temporal-only control baseline.

The control applies valid-time filtering but does not detect conflicts, validate
provenance, or quarantine poisoned writes. It is intentionally not presented as
an external temporal-memory system.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..evaluation.scenarios import ScenarioOutcome

if TYPE_CHECKING:
    from ..evaluation.benchmark import BenchmarkCase


def run_temporal_only_cases(cases: list[BenchmarkCase]) -> list[ScenarioOutcome]:
    outcomes: list[ScenarioOutcome] = []
    for case in cases:
        if case.query_time is None:
            active = list(case.claims)
        else:
            active = [
                claim
                for claim in case.claims
                if (claim.valid_from is None or claim.valid_from <= case.query_time)
                and (claim.valid_until is None or case.query_time < claim.valid_until)
            ]
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
