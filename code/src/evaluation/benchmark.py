"""JSONL benchmark runner for ClaimLedger experiments.

The schema is intentionally close to what a LoCoMo/LongMemEval adapter can emit:
one case per line, with observed memory claims and expected retrieval behavior.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.ledger import ClaimLedger
from ..core.models import ClaimStatus, TrustTier
from .scenarios import ScenarioOutcome
from ..baselines.no_ledger_oracle_labels import run_no_ledger_oracle_labels_cases


@dataclass(frozen=True)
class BenchmarkClaim:
    subject: str
    predicate: str
    object: str
    evidence: str
    source: str
    valid_from: str | None = None
    valid_until: str | None = None
    trust_tier: TrustTier = TrustTier.MEDIUM
    confidence: float = 0.75
    is_poisoned: bool = False


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    task_type: str
    query_time: str | None
    claims: list[BenchmarkClaim]
    expected_active: int
    expect_conflict: bool = False
    expect_abstain: bool = False


def load_jsonl(path: str | Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            cases.append(_case_from_json(raw, line_number))
    return cases


def run_claimledger_cases(cases: list[BenchmarkCase]) -> list[ScenarioOutcome]:
    outcomes: list[ScenarioOutcome] = []
    for case in cases:
        with ClaimLedger() as ledger:
            claim_ids = [_insert_benchmark_claim(ledger, claim) for claim in case.claims]
            _apply_case_relations(ledger, case, claim_ids)
            result = ledger.eligible_claims(at_time=case.query_time)
            outcomes.append(
                ScenarioOutcome(
                    name=case.task_type,
                    expected_active=case.expected_active,
                    actual_active=len(result.eligible),
                    conflict_disclosed=bool(result.conflicted),
                    poisoned_active=any(claim.is_poisoned for claim in result.eligible),
                    abstained=len(result.eligible) == 0,
                )
            )
    return outcomes


def run_no_conflict_cases(cases: list[BenchmarkCase]) -> list[ScenarioOutcome]:
    """Ablation: keep temporal/provenance gates but disable conflict handling."""

    outcomes: list[ScenarioOutcome] = []
    for case in cases:
        with ClaimLedger() as ledger:
            claim_ids = [_insert_benchmark_claim(ledger, claim) for claim in case.claims]
            _apply_case_relations(ledger, case, claim_ids, apply_conflict=False)
            result = ledger.eligible_claims(at_time=case.query_time)
            outcomes.append(
                ScenarioOutcome(
                    name=case.task_type,
                    expected_active=case.expected_active,
                    actual_active=len(result.eligible),
                    conflict_disclosed=bool(result.conflicted),
                    poisoned_active=any(claim.is_poisoned for claim in result.eligible),
                    abstained=len(result.eligible) == 0,
                )
            )
    return outcomes


def run_no_temporal_cases(cases: list[BenchmarkCase]) -> list[ScenarioOutcome]:
    """Ablation: keep ledger relations but ignore query-time validity checks."""

    outcomes: list[ScenarioOutcome] = []
    for case in cases:
        with ClaimLedger() as ledger:
            claim_ids = [_insert_benchmark_claim(ledger, claim) for claim in case.claims]
            _apply_case_relations(ledger, case, claim_ids)
            rows = ledger.all_claims()
            active = [claim for claim in rows if claim.status == ClaimStatus.ACTIVE]
            conflicted = [claim for claim in rows if claim.status == ClaimStatus.CONFLICTED]
            outcomes.append(
                ScenarioOutcome(
                    name=case.task_type,
                    expected_active=case.expected_active,
                    actual_active=len(active),
                    conflict_disclosed=bool(conflicted),
                    poisoned_active=any(claim.is_poisoned for claim in active),
                    abstained=len(active) == 0,
                )
            )
    return outcomes


def run_trust_only_cases(cases: list[BenchmarkCase]) -> list[ScenarioOutcome]:
    """Ablation: use initial trust gating and poison quarantine, no ledger relations."""

    outcomes: list[ScenarioOutcome] = []
    for case in cases:
        with ClaimLedger() as ledger:
            claim_ids = [_insert_benchmark_claim(ledger, claim) for claim in case.claims]
            _apply_case_relations(
                ledger,
                case,
                claim_ids,
                apply_stale=False,
                apply_conflict=False,
            )
            result = ledger.eligible_claims(at_time=case.query_time)
            outcomes.append(
                ScenarioOutcome(
                    name=case.task_type,
                    expected_active=case.expected_active,
                    actual_active=len(result.eligible),
                    conflict_disclosed=bool(result.conflicted),
                    poisoned_active=any(claim.is_poisoned for claim in result.eligible),
                    abstained=len(result.eligible) == 0,
                )
            )
    return outcomes


def run_naive_cases(cases: list[BenchmarkCase]) -> list[ScenarioOutcome]:
    """Naive baseline: every observed claim is treated as active memory."""

    outcomes: list[ScenarioOutcome] = []
    for case in cases:
        active = _naive_active_claims(case)
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


def _build_baseline_runners() -> dict[str, Any]:
    """Build the full baseline runners dict with lazy imports to avoid circular deps."""
    from ..baselines.vector_rag import run_vector_rag_cases
    from ..baselines.summary_memory import run_summary_memory_cases
    from ..baselines.recency_memory import run_recency_memory_cases
    from ..baselines.temporal_memory import run_temporal_only_cases
    from ..baselines.provenance_memory import run_provenance_only_cases
    from ..baselines.structured_fact_store import run_structured_fact_store_cases
    from ..baselines.no_ledger_oracle_labels import run_no_ledger_oracle_labels_cases

    return {
        "claimledger": run_claimledger_cases,
        "naive": run_naive_cases,
        "no_conflict": run_no_conflict_cases,
        "no_temporal": run_no_temporal_cases,
        "trust_only": run_trust_only_cases,
        "no_ledger_oracle_labels": run_no_ledger_oracle_labels_cases,
        "vector_rag": run_vector_rag_cases,
        "summary_memory": run_summary_memory_cases,
        "recency_memory": run_recency_memory_cases,
        "temporal_only": run_temporal_only_cases,
        "provenance_only": run_provenance_only_cases,
        "structured_fact_store": run_structured_fact_store_cases,
    }


# Canonical baseline and ablation runners registry (single source of truth).
BASELINE_RUNNERS: dict[str, Any] = _build_baseline_runners()


def get_all_runners() -> dict[str, Any]:
    """Return all lightweight baselines and ablations."""
    return BASELINE_RUNNERS


def _case_from_json(raw: dict[str, Any], line_number: int) -> BenchmarkCase:
    required = {"case_id", "task_type", "claims", "expected_active"}
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(f"Line {line_number} missing required keys: {', '.join(missing)}")
    return BenchmarkCase(
        case_id=str(raw["case_id"]),
        task_type=str(raw["task_type"]),
        query_time=raw.get("query_time"),
        claims=[_claim_from_json(item, line_number) for item in raw["claims"]],
        expected_active=int(raw["expected_active"]),
        expect_conflict=bool(raw.get("expect_conflict", False)),
        expect_abstain=bool(raw.get("expect_abstain", False)),
    )


def _claim_from_json(raw: dict[str, Any], line_number: int) -> BenchmarkClaim:
    required = {"subject", "predicate", "object", "evidence", "source"}
    missing = sorted(required - raw.keys())
    if missing:
        raise ValueError(f"Line {line_number} has claim missing keys: {', '.join(missing)}")
    return BenchmarkClaim(
        subject=str(raw["subject"]),
        predicate=str(raw["predicate"]),
        object=str(raw["object"]),
        evidence=str(raw["evidence"]),
        source=str(raw["source"]),
        valid_from=raw.get("valid_from"),
        valid_until=raw.get("valid_until"),
        trust_tier=TrustTier(str(raw.get("trust_tier", TrustTier.MEDIUM))),
        confidence=float(raw.get("confidence", 0.75)),
        is_poisoned=bool(raw.get("is_poisoned", False)),
    )


def _insert_benchmark_claim(ledger: ClaimLedger, claim: BenchmarkClaim) -> str:
    return ledger.add_claim(
        subject=claim.subject,
        predicate=claim.predicate,
        object=claim.object,
        evidence=claim.evidence,
        source=claim.source,
        valid_from=claim.valid_from,
        valid_until=claim.valid_until,
        trust_tier=claim.trust_tier,
        confidence=claim.confidence,
        is_poisoned=claim.is_poisoned,
    )


def _apply_case_relations(
    ledger: ClaimLedger,
    case: BenchmarkCase,
    claim_ids: list[str],
    *,
    apply_stale: bool = True,
    apply_conflict: bool = True,
    apply_poison: bool = True,
) -> None:
    if apply_stale and case.task_type == "stale_update" and len(claim_ids) >= 2:
        ledger.supersede(claim_ids[0], claim_ids[-1], "benchmark stale-update relation")
    if apply_conflict and case.task_type in {"direct_conflict", "conflict_flood"} and len(claim_ids) >= 2:
        ledger.mark_conflict(claim_ids[0], claim_ids[1], "benchmark conflict relation")
    if apply_poison:
        for claim_id, claim in zip(claim_ids, case.claims, strict=False):
            if claim.is_poisoned:
                ledger.quarantine(claim_id, "benchmark poisoned write")


def _naive_active_claims(case: BenchmarkCase) -> list[BenchmarkClaim]:
    if case.query_time is None:
        return case.claims
    return [
        claim
        for claim in case.claims
        if (claim.valid_from is None or claim.valid_from <= case.query_time)
        and (claim.valid_until is None or claim.valid_until > case.query_time)
    ]







