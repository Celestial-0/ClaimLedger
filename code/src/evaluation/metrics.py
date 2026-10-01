"""Evaluation metrics and confidence interval computation for ClaimLedger runs."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .scenarios import ScenarioOutcome


@dataclass(frozen=True)
class MetricReport:
    temporal_validity_accuracy: float
    conflict_disclosure_rate: float
    poison_activation_rate: float
    abstention_correctness: float
    exact_active_count_accuracy: float


@dataclass(frozen=True)
class ConfidenceInterval:
    mean: float
    lower: float
    upper: float


@dataclass(frozen=True)
class PairedComparison:
    system_accuracy: float
    baseline_accuracy: float
    system_wins: int
    baseline_wins: int
    ties: int
    mcnemar_exact_p: float


TEMPORAL_TASKS = {
    "stale_update",
    "expired_claim",
    "future_effective",
    "valid_window",
    "temporal_boundary",
    "temporal_overlap",
    "retroactive_correction",
}


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def compute_metrics(outcomes: list[ScenarioOutcome]) -> MetricReport:
    temporal = [outcome for outcome in outcomes if outcome.name in TEMPORAL_TASKS]
    conflicts = [
        outcome
        for outcome in outcomes
        if outcome.name in {"direct_conflict", "conflict_flood"}
    ]
    poison = [
        outcome
        for outcome in outcomes
        if outcome.name in {"poisoned_write", "trusted_source_poison"}
    ]
    abstention_cases = [outcome for outcome in outcomes if outcome.expected_active == 0]

    return MetricReport(
        temporal_validity_accuracy=_rate(
            sum(outcome.actual_active == outcome.expected_active for outcome in temporal),
            len(temporal),
        ),
        conflict_disclosure_rate=_rate(
            sum(outcome.conflict_disclosed for outcome in conflicts),
            len(conflicts),
        ),
        poison_activation_rate=_rate(
            sum(outcome.poisoned_active for outcome in poison),
            len(poison),
        ),
        abstention_correctness=_rate(
            sum(outcome.abstained and outcome.actual_active == 0 for outcome in abstention_cases),
            len(abstention_cases),
        ),
        exact_active_count_accuracy=_rate(
            sum(outcome.actual_active == outcome.expected_active for outcome in outcomes),
            len(outcomes),
        ),
    )


def exact_active_correctness(outcomes: list[ScenarioOutcome]) -> list[bool]:
    return [outcome.actual_active == outcome.expected_active for outcome in outcomes]


def bootstrap_mean_ci(
    values: list[bool],
    *,
    iterations: int = 2000,
    seed: int = 13,
    alpha: float = 0.05,
) -> ConfidenceInterval:
    if not values:
        return ConfidenceInterval(mean=0.0, lower=0.0, upper=0.0)
    rng = random.Random(seed)
    n = len(values)
    numeric = [1.0 if value else 0.0 for value in values]
    samples = []
    for _ in range(iterations):
        samples.append(sum(rng.choice(numeric) for _ in range(n)) / n)
    samples.sort()
    lower_index = max(0, int((alpha / 2) * iterations))
    upper_index = min(iterations - 1, int((1 - alpha / 2) * iterations))
    return ConfidenceInterval(
        mean=sum(numeric) / n,
        lower=samples[lower_index],
        upper=samples[upper_index],
    )


def bootstrap_cluster_mean_ci(
    clusters: Mapping[str, Sequence[bool]],
    *,
    iterations: int = 2000,
    seed: int = 13,
    alpha: float = 0.05,
) -> ConfidenceInterval:
    """Bootstrap a case-level mean by resampling whole clusters.

    REALTALK cases share a conversation and question across clean, conflict,
    poison, and temporal variants. Resampling individual cases therefore
    understates dependence. This helper resamples conversations with
    replacement and retains all cases belonging to each sampled conversation.
    """
    nonempty = {name: list(values) for name, values in clusters.items() if values}
    if not nonempty:
        return ConfidenceInterval(mean=0.0, lower=0.0, upper=0.0)
    cluster_names = sorted(nonempty)
    observed = [value for name in cluster_names for value in nonempty[name]]
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(iterations):
        sampled_values = [
            value
            for _ in cluster_names
            for value in nonempty[rng.choice(cluster_names)]
        ]
        samples.append(sum(sampled_values) / len(sampled_values))
    samples.sort()
    lower_index = max(0, int((alpha / 2) * iterations))
    upper_index = min(iterations - 1, int((1 - alpha / 2) * iterations))
    return ConfidenceInterval(
        mean=sum(observed) / len(observed),
        lower=samples[lower_index],
        upper=samples[upper_index],
    )


def paired_exact_active_comparison(
    system_outcomes: list[ScenarioOutcome],
    baseline_outcomes: list[ScenarioOutcome],
) -> PairedComparison:
    if len(system_outcomes) != len(baseline_outcomes):
        raise ValueError("paired comparisons require equal outcome counts")
    system_correct = exact_active_correctness(system_outcomes)
    baseline_correct = exact_active_correctness(baseline_outcomes)
    system_wins = sum(sys and not base for sys, base in zip(system_correct, baseline_correct, strict=True))
    baseline_wins = sum(base and not sys for sys, base in zip(system_correct, baseline_correct, strict=True))
    ties = len(system_correct) - system_wins - baseline_wins
    return PairedComparison(
        system_accuracy=_rate(sum(system_correct), len(system_correct)),
        baseline_accuracy=_rate(sum(baseline_correct), len(baseline_correct)),
        system_wins=system_wins,
        baseline_wins=baseline_wins,
        ties=ties,
        mcnemar_exact_p=_exact_two_sided_binomial_p(system_wins, baseline_wins),
    )


def _exact_two_sided_binomial_p(first_wins: int, second_wins: int) -> float:
    """Return a stable two-sided exact binomial p-value.

    The direct ``0.5 ** discordant`` calculation underflows to zero for
    large discordant counts.  Results below the smallest reporting floor are
    retained as ``1e-300`` so downstream JSON and Holm correction never emit
    the misleading value ``p=0``; manuscript reporting should render this
    floor as ``p < 1e-300``.
    """
    discordant = first_wins + second_wins
    if discordant == 0:
        return 1.0
    smaller = min(first_wins, second_wins)
    log_terms = [
        math.lgamma(discordant + 1)
        - math.lgamma(k + 1)
        - math.lgamma(discordant - k + 1)
        - discordant * math.log(2.0)
        for k in range(smaller + 1)
    ]
    pivot = max(log_terms)
    log_tail = pivot + math.log(sum(math.exp(term - pivot) for term in log_terms))
    log_p = math.log(2.0) + log_tail
    return max(1e-300, min(1.0, math.exp(log_p)))
