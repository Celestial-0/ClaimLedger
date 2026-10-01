"""Null baseline: lexical retrieval with oracle relation labels without ledger governance.

Evaluates claim eligibility under oracle labels without applying temporal,
conflict, or trust policy gates. This isolates the contribution of ledger
governance policies from the presence of ground-truth relation annotations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..evaluation.scenarios import ScenarioOutcome

if TYPE_CHECKING:
    from ..evaluation.benchmark import BenchmarkCase


def _valid_at(claim, query_time: str | None) -> bool:
    if query_time is None:
        return True
    if claim.valid_from is not None and claim.valid_from > query_time:
        return False
    if claim.valid_until is not None and claim.valid_until <= query_time:
        return False
    return True


def run_no_ledger_oracle_labels_cases(
    cases: list[BenchmarkCase],
) -> list[ScenarioOutcome]:
    """Lexical retrieval + gold labels, no ledger governance gate."""
    outcomes: list[ScenarioOutcome] = []

    for case in cases:
        active = [claim for claim in case.claims if _valid_at(claim, case.query_time)]
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
