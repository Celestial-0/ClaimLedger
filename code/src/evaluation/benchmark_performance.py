"""Repeated latency and storage measurements for the SQLite ClaimLedger backend."""

from __future__ import annotations

import math
import os
import random
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..core.ledger import ClaimLedger
from ..core.models import ClaimStatus, TrustTier


@dataclass(frozen=True)
class PerformanceReport:
    """Aggregated performance metrics for one claim-count scale."""

    scale: int
    repetitions: int
    write_latency_ms_mean: float
    write_latency_ms_p50: float
    write_latency_ms_p95: float
    write_latency_ms_p99: float
    query_latency_ms_mean: float
    query_latency_ms_p50: float
    query_latency_ms_p95: float
    query_latency_ms_p99: float
    storage_bytes: int
    storage_bytes_per_claim: float
    total_write_time_s: float
    total_query_time_s: float


def _calculate_percentile(latencies_ns: list[int], percentile: float) -> float:
    """Return a nearest-rank latency percentile in milliseconds."""
    if not latencies_ns:
        return 0.0
    ordered = sorted(latencies_ns)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1] / 1_000_000.0


def _calculate_mean(latencies_ns: list[int]) -> float:
    """Calculate the mean latency in milliseconds."""
    if not latencies_ns:
        return 0.0
    return (sum(latencies_ns) / len(latencies_ns)) / 1_000_000.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _run_trial(scale: int, rng: random.Random) -> tuple[list[int], list[int], int, float, float]:
    fd, db_path_str = tempfile.mkstemp(suffix=".db", prefix="claimledger_bench_")
    os.close(fd)
    db_path = Path(db_path_str)
    predicates = [
        "is_affiliated_with",
        "has_degree_from",
        "employed_by",
        "certified_by",
        "owns_asset",
    ]
    sources = ["source_a", "source_b", "source_c", "source_d"]
    trust_tiers = list(TrustTier)
    base_time = datetime(2025, 1, 1, tzinfo=timezone.utc)

    try:
        write_latencies_ns: list[int] = []
        start_write_phase = time.perf_counter()
        with ClaimLedger(db_path) as ledger:
            for i in range(scale):
                subject = f"user_{rng.randint(1, 100)}"
                predicate = rng.choice(predicates)
                obj = f"entity_{rng.randint(1, 50)}"
                valid_from = base_time + timedelta(days=rng.randint(0, 365))
                valid_until = valid_from + timedelta(days=rng.randint(30, 365))
                start_write = time.perf_counter_ns()
                ledger.add_claim(
                    claim_id=f"claim_{scale}_{i}",
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    evidence=f"Evidence for {subject} {predicate} {obj}",
                    source=rng.choice(sources),
                    valid_from=valid_from.isoformat(),
                    valid_until=valid_until.isoformat(),
                    trust_tier=rng.choice(trust_tiers),
                    confidence=rng.random(),
                    status=ClaimStatus.ACTIVE,
                    is_poisoned=False,
                )
                write_latencies_ns.append(time.perf_counter_ns() - start_write)
            total_write_time_s = time.perf_counter() - start_write_phase

            query_latencies_ns: list[int] = []
            start_query_phase = time.perf_counter()
            for _ in range(100):
                query_time = base_time + timedelta(days=rng.randint(0, 700))
                start_query = time.perf_counter_ns()
                ledger.eligible_claims(at_time=query_time.isoformat())
                query_latencies_ns.append(time.perf_counter_ns() - start_query)
            total_query_time_s = time.perf_counter() - start_query_phase

        storage_bytes = db_path.stat().st_size
        return (
            write_latencies_ns,
            query_latencies_ns,
            storage_bytes,
            total_write_time_s,
            total_query_time_s,
        )
    finally:
        if db_path.exists():
            try:
                db_path.unlink()
            except OSError:
                pass


def run_performance_benchmark(
    scales: list[int] | None = None,
    *,
    repetitions: int = 1,
    seed: int = 42,
) -> list[PerformanceReport]:
    """Run fresh-database trials at each scale and aggregate latency percentiles."""
    if scales is None:
        scales = [100, 1000, 10000, 100000]
    if repetitions < 1:
        raise ValueError("repetitions must be at least 1")
    if any(scale < 0 for scale in scales):
        raise ValueError("scales must be non-negative")

    reports: list[PerformanceReport] = []
    for scale in scales:
        all_write_latencies: list[int] = []
        all_query_latencies: list[int] = []
        storage_values: list[float] = []
        write_totals: list[float] = []
        query_totals: list[float] = []
        for repetition in range(repetitions):
            trial_rng = random.Random(seed + (scale * 1_000_003) + repetition)
            write_latencies, query_latencies, storage_bytes, write_total, query_total = _run_trial(
                scale,
                trial_rng,
            )
            all_write_latencies.extend(write_latencies)
            all_query_latencies.extend(query_latencies)
            storage_values.append(float(storage_bytes))
            write_totals.append(write_total)
            query_totals.append(query_total)

        mean_storage = _mean(storage_values)
        reports.append(
            PerformanceReport(
                scale=scale,
                repetitions=repetitions,
                write_latency_ms_mean=_calculate_mean(all_write_latencies),
                write_latency_ms_p50=_calculate_percentile(all_write_latencies, 0.50),
                write_latency_ms_p95=_calculate_percentile(all_write_latencies, 0.95),
                write_latency_ms_p99=_calculate_percentile(all_write_latencies, 0.99),
                query_latency_ms_mean=_calculate_mean(all_query_latencies),
                query_latency_ms_p50=_calculate_percentile(all_query_latencies, 0.50),
                query_latency_ms_p95=_calculate_percentile(all_query_latencies, 0.95),
                query_latency_ms_p99=_calculate_percentile(all_query_latencies, 0.99),
                storage_bytes=round(mean_storage),
                storage_bytes_per_claim=mean_storage / scale if scale else 0.0,
                total_write_time_s=_mean(write_totals),
                total_query_time_s=_mean(query_totals),
            )
        )
    return reports


def format_performance_table(reports: list[PerformanceReport]) -> str:
    """Format repeated performance reports into a compact Markdown table."""
    lines = [
        "| Scale | Repeats | Write mean/p50/p95/p99 (ms) | Query mean/p50/p95/p99 (ms) | Storage (MB) | Bytes/claim |",
        "|------:|--------:|----------------------------:|----------------------------:|-------------:|------------:|",
    ]
    for report in reports:
        storage_mb = report.storage_bytes / (1024 * 1024)
        write = "/".join(
            f"{value:.3f}"
            for value in (
                report.write_latency_ms_mean,
                report.write_latency_ms_p50,
                report.write_latency_ms_p95,
                report.write_latency_ms_p99,
            )
        )
        query = "/".join(
            f"{value:.3f}"
            for value in (
                report.query_latency_ms_mean,
                report.query_latency_ms_p50,
                report.query_latency_ms_p95,
                report.query_latency_ms_p99,
            )
        )
        lines.append(
            f"| {report.scale} | {report.repetitions} | {write} | {query} | "
            f"{storage_mb:.2f} | {report.storage_bytes_per_claim:.1f} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(format_performance_table(run_performance_benchmark([100, 1000, 10000], repetitions=3)))
