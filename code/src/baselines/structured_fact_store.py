"""Structured fact-store proxy baseline.

This is a transparent local proxy for a fact/graph memory: it normalizes claims
to ``(subject, predicate)`` keys, applies valid-time filtering, and keeps the
latest surviving fact per key. It is not an implementation of an external
knowledge-graph product and should be labelled accordingly in the paper.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..evaluation.scenarios import ScenarioOutcome

if TYPE_CHECKING:
    from ..evaluation.benchmark import BenchmarkCase


def run_structured_fact_store_cases(cases: list[BenchmarkCase]) -> list[ScenarioOutcome]:
    outcomes: list[ScenarioOutcome] = []
    for case in cases:
        if case.query_time is None:
            eligible = list(case.claims)
        else:
            eligible = [
                claim
                for claim in case.claims
                if (claim.valid_from is None or claim.valid_from <= case.query_time)
                and (claim.valid_until is None or case.query_time < claim.valid_until)
            ]
        eligible.sort(key=lambda claim: claim.valid_from or "", reverse=True)
        facts = []
        seen: set[tuple[str, str]] = set()
        for claim in eligible:
            key = (claim.subject, claim.predicate)
            if key not in seen:
                seen.add(key)
                facts.append(claim)
        outcomes.append(
            ScenarioOutcome(
                name=case.task_type,
                expected_active=case.expected_active,
                actual_active=len(facts),
                conflict_disclosed=False,
                poisoned_active=any(claim.is_poisoned for claim in facts),
                abstained=len(facts) == 0,
            )
        )
    return outcomes
