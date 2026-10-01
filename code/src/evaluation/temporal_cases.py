"""Temporal-boundary case generation for ClaimLedger experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TemporalBoundaryReport:
    source_claim_count: int
    usable_claim_count: int
    generated_case_count: int


def generate_temporal_boundaries(
    claims_path: str | Path,
    output_path: str | Path,
    *,
    limit: int | None = None,
) -> TemporalBoundaryReport:
    claims = _load_claims(claims_path)
    usable = [claim for claim in claims if _parse_date(claim.get("valid_from")) is not None]
    if limit is not None:
        usable = usable[:limit]

    cases: list[dict[str, Any]] = []
    for idx, claim in enumerate(usable, start=1):
        cases.extend(_claim_to_temporal_cases(claim, idx))

    _write_jsonl(output_path, cases)
    return TemporalBoundaryReport(
        source_claim_count=len(claims),
        usable_claim_count=len(usable),
        generated_case_count=len(cases),
    )


def _claim_to_temporal_cases(claim: dict[str, Any], idx: int) -> list[dict[str, Any]]:
    base_date = _parse_date(claim["valid_from"])
    if base_date is None:
        raise ValueError(f"claim {idx} has invalid valid_from={claim.get('valid_from')!r}")

    day0 = _iso(base_date)
    day1 = _iso(base_date + timedelta(days=1))
    day2 = _iso(base_date + timedelta(days=2))
    day3 = _iso(base_date + timedelta(days=3))
    case_prefix = f"{claim['sample_id']}_{claim['evidence_id'].replace(':', '_')}_{idx}"
    base_claim = _benchmark_claim(claim)

    expired_claim = {**base_claim, "valid_from": day0, "valid_until": day1}
    future_claim = {**base_claim, "valid_from": day2, "valid_until": None}
    active_claim = {**base_claim, "valid_from": day0, "valid_until": day3}
    old_overlap = {**base_claim, "valid_from": day0, "valid_until": day1}
    new_overlap = {
        **base_claim,
        "object": f"updated({base_claim['object']})",
        "valid_from": day1,
        "valid_until": day3,
        "evidence": f"Temporal update after {claim['evidence_id']}: {claim['evidence']}",
    }

    return [
        {
            "case_id": f"{case_prefix}_expired",
            "task_type": "temporal_boundary",
            "query_time": day2,
            "expected_active": 0,
            "expect_abstain": True,
            "claims": [expired_claim],
        },
        {
            "case_id": f"{case_prefix}_future_effective",
            "task_type": "temporal_boundary",
            "query_time": day1,
            "expected_active": 0,
            "expect_abstain": True,
            "claims": [future_claim],
        },
        {
            "case_id": f"{case_prefix}_active_window",
            "task_type": "temporal_boundary",
            "query_time": day1,
            "expected_active": 1,
            "claims": [active_claim],
        },
        {
            "case_id": f"{case_prefix}_overlap_boundary",
            "task_type": "temporal_boundary",
            "query_time": day2,
            "expected_active": 1,
            "claims": [old_overlap, new_overlap],
        },
    ]


def _benchmark_claim(claim: dict[str, Any]) -> dict[str, Any]:
    return {
        "subject": claim["subject"],
        "predicate": claim["predicate"],
        "object": claim["object"],
        "evidence": claim["evidence"],
        "source": claim["source"],
        "valid_from": claim["valid_from"],
        "valid_until": claim.get("valid_until"),
        "trust_tier": claim["trust_tier"],
        "confidence": claim["confidence"],
    }


def _parse_date(value: object) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _iso(value: date) -> str:
    return value.isoformat()


def _load_claims(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
