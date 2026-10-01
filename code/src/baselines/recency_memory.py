"""Recency-weighted memory baseline.

Keeps only the most recent claim per (subject, predicate) pair.
No poison checking, no conflict disclosure, no lifecycle governance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..evaluation.scenarios import ScenarioOutcome

if TYPE_CHECKING:
    from ..evaluation.benchmark import BenchmarkCase


def run_recency_memory_cases(cases: list[BenchmarkCase]) -> list[ScenarioOutcome]:
    """Recency baseline: always trust the newest claim, no governance."""
    outcomes: list[ScenarioOutcome] = []

    for case in cases:
        claims = list(case.claims)

        # Sort by (valid_from descending, index descending) so newest first
        scored = []
        for i, c in enumerate(claims):
            v_from = c.valid_from if c.valid_from is not None else ""
            scored.append(((v_from, i), c))
        scored.sort(key=lambda x: x[0], reverse=True)

        # Keep only the most recent claim for each (subject, predicate) pair
        surviving = []
        seen: set[tuple[str, str]] = set()
        for _, c in scored:
            pair = (c.subject, c.predicate)
            if pair not in seen:
                seen.add(pair)
                surviving.append(c)

        outcomes.append(ScenarioOutcome(
            name=case.task_type,
            expected_active=case.expected_active,
            actual_active=len(surviving),
            conflict_disclosed=False,
            poisoned_active=any(c.is_poisoned for c in surviving),
            abstained=False,
        ))

    return outcomes
