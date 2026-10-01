"""Non-oracle extraction-error model for ClaimLedger mechanism cases.

The headline mechanism runner (``run_claimledger_cases``) operates in *oracle*
mode: conflict relations, poison quarantines, and temporal-validity fields are
supplied to the ledger as gold labels. That isolates gate correctness but says
nothing about how the system degrades when those labels come from a real
extractor that makes mistakes.

This module converts oracle cases into *non-oracle* cases by simulating an
extractor with a per-label detection probability. Three independent recall-type
error channels are modelled:

* conflict detection -- with probability ``1 - p_detect`` a ``contradicts``
  relation is missed, so the ledger never calls ``mark_conflict``;
* poison detection -- with probability ``1 - p_detect`` a poisoned write's
  ``is_poisoned`` flag is missed, so the write is never quarantined;
* temporal-field extraction -- with probability ``1 - p_detect`` a claim's
  ``valid_until`` bound is dropped, so a stale claim stays eligible past its
  true expiry.

The result is a degradation curve: as ``p_detect`` sweeps from 1.0 (oracle)
toward 0.0 (extractor misses every relation), the governance delta erodes
toward the naive-memory baseline.

Deterministic for a fixed ``(case, p_detect, seed)`` triple so the sweep is
reproducible and resumable.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

from ..core.ledger import ClaimLedger
from .benchmark import BenchmarkCase, _insert_benchmark_claim
from .scenarios import ScenarioOutcome


@dataclass(frozen=True)
class ExtractionErrorConfig:
    """Per-channel detection probability and the seed for the error draw."""

    p_detect: float
    seed: int


def run_claimledger_extraction_error_cases(
    cases: list[BenchmarkCase],
    *,
    p_detect: float,
    seed: int,
) -> list[ScenarioOutcome]:
    """Run ClaimLedger with simulated extractor label errors.

    ``p_detect`` is the probability the extractor correctly surfaces each
    relation/field; 1.0 reproduces oracle mode and lower values inject
    recall-type errors. ``seed`` is independent of the runner seed so the
    extraction-error sweep is a separate reproducibility axis.
    """
    if not 0.0 <= p_detect <= 1.0:
        raise ValueError(f"p_detect must be in [0, 1]; got {p_detect}")

    outcomes: list[ScenarioOutcome] = []
    for case in cases:
        # One RNG per (case, p_detect, seed) so a single case's error draw is
        # stable regardless of sweep ordering or other cases.
        rng = random.Random(f"extraction-error|{case.case_id}|{p_detect}|{seed}")
        with ClaimLedger() as ledger:
            claim_ids = [
                _insert_benchmark_claim(ledger, _corrupt_claim(claim, rng, p_detect))
                for claim in case.claims
            ]
            _apply_corrupted_relations(ledger, case, claim_ids, rng, p_detect)
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



def _corrupt_claim(claim, rng: random.Random, p_detect: float):
    """Return the claim with extractor-induced field errors applied.

    Only the poison flag and the closing temporal bound are modelled as
    extractor-dependent; identity fields (subject/predicate/object, evidence)
    are the extractor's *output* and are treated as given here. ``valid_from``
    errors shift the activation boundary and are a separate, smaller effect; we
    model the dominant ``valid_until`` (stale-retention) channel only.
    """
    # Missed poison detection: the extractor fails to flag the write.
    is_poisoned = claim.is_poisoned and (rng.random() < p_detect)
    # Missed closing-bound extraction: a stale claim loses its expiry.
    valid_until = claim.valid_until
    if claim.valid_until is not None and rng.random() >= p_detect:
        valid_until = None

    if is_poisoned == claim.is_poisoned and valid_until == claim.valid_until:
        return claim  # unchanged: avoid rebuilding the frozen dataclass
    return replace(claim, is_poisoned=is_poisoned, valid_until=valid_until)


def _apply_corrupted_relations(
    ledger: ClaimLedger,
    case: BenchmarkCase,
    claim_ids: list[str],
    rng: random.Random,
    p_detect: float,
) -> None:
    """Apply relations, dropping each with probability ``1 - p_detect``.

    Mirrors ``benchmark._apply_case_relations`` but each explicit relation
    (supersession, conflict) is gated by an independent detection draw. Poison
    quarantine additionally requires the extractor to have kept the poison flag
    (corrupted in ``_corrupt_claim``); a missed flag means there is nothing to
    quarantine, matching a real extractor that never raised the alarm.
    """
    if case.task_type == "stale_update" and len(claim_ids) >= 2:
        if rng.random() < p_detect:
            ledger.supersede(claim_ids[0], claim_ids[-1], "extractor supersession relation")
    if case.task_type in {"direct_conflict", "conflict_flood"} and len(claim_ids) >= 2:
        if rng.random() < p_detect:
            ledger.mark_conflict(claim_ids[0], claim_ids[1], "extractor conflict relation")
    # Quarantine only poisoned writes whose flag survived extraction, and only
    # when the extractor's detection draw succeeds for that relation.
    for claim_id, claim in zip(claim_ids, case.claims, strict=False):
        if claim.is_poisoned and rng.random() < p_detect:
            ledger.quarantine(claim_id, "extractor poisoned write")
