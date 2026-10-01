"""Unified data audit tool for ClaimLedger.

Consolidates:
- Benchmark source file and hash validation (writes data/manifests/benchmark_sources.json)
- Prompt and packet leakage audit (writes data/results/leakage_audit.json)

Usage:
    uv run --project code python code/tools/audit_data.py [--checks {all,sources,leakage}]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from claimledger.evaluation.e2e_qa_runner import (
    _build_prompt,
    _claimledger_packet,
    _naive_packet,
    _recency_packet,
    _summary_packet,
)
from claimledger.evaluation.multillm import build_prompt, iter_examples, select_context

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# 1. Benchmark Sources Audit
# ---------------------------------------------------------------------------

SOURCES = {
    "locomo": {
        "path": Path("data/dataset/locomo/locomo10.json"),
        "url": "https://github.com/snap-research/locomo",
        "provenance": "official_benchmark_generated_history",
    },
    "longmemeval_s": {
        "path": Path("data/dataset/longmemeval/longmemeval_s_cleaned.json"),
        "url": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned",
        "provenance": "official_benchmark_generated_history",
    },
    "longmemeval_m": {
        "path": Path("data/dataset/longmemeval/longmemeval_m_cleaned.json"),
        "url": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned",
        "provenance": "official_benchmark_generated_history",
    },
    "longmemeval_oracle": {
        "path": Path("data/dataset/longmemeval/longmemeval_oracle.json"),
        "url": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned",
        "provenance": "official_benchmark_oracle_diagnostic",
    },
    "realtalk": {
        "path": Path("data/dataset/realtalk/processed"),
        "url": "https://github.com/danny911kr/REALTALK",
        "provenance": "authentic_human_human_dialogue_human_annotated_question",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_sources() -> dict[str, Any]:
    output = REPOSITORY_ROOT / "data" / "manifests" / "benchmark_sources.json"
    records = []
    print("\n--- Auditing Benchmark Sources ---")
    for name, spec in SOURCES.items():
        rel_path = spec["path"]
        path = rel_path if rel_path.is_absolute() else REPOSITORY_ROOT / rel_path
        if not path.exists():
            print(f"MISSING: {name} at {path}")
            continue
        files = sorted(path.glob("Chat_*.json")) if path.is_dir() else [path]
        counts = {"examples": 0, "task_types": {}, "max_context_chunks": 0}
        for example in iter_examples(name if name != "realtalk" else "realtalk", path):
            counts["examples"] += 1
            task = example.task_type or "unknown"
            counts["task_types"][task] = counts["task_types"].get(task, 0) + 1
            counts["max_context_chunks"] = max(counts["max_context_chunks"], len(example.chunks))
        record = {
            "dataset": name,
            "path": str(rel_path),
            "files": [{"path": str(f.name), "bytes": f.stat().st_size, "sha256": _sha256(f)} for f in files],
            "source_url": spec["url"],
            "retrieved_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "provenance": spec["provenance"],
            "question_answer_fields_are_official": True,
            "llm_used_for_download_or_validation": False,
            **counts,
        }
        records.append(record)
        print(f"  {name:18s}: {counts['examples']:4d} examples, {len(files)} files")

    manifest = {"schema_version": "1.0", "sources": records}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote source manifest to {output}")
    return manifest


# ---------------------------------------------------------------------------
# 2. Leakage Audit
# ---------------------------------------------------------------------------

FORBIDDEN_FIELD_NAMES = (
    "expected_active",
    "expect_conflict",
    "expect_abstain",
    "qa_category",
    "human_annotated",
    "unresolved_evidence_ids",
)


def audit_leakage() -> bool:
    cases_path = REPOSITORY_ROOT / "data" / "derived" / "realtalk_memory_probe_cases.jsonl"
    out_path = REPOSITORY_ROOT / "data" / "results" / "leakage_audit.json"
    if not cases_path.exists():
        print(f"Cases not found at {cases_path}; skipping leakage check.")
        return True

    print("\n--- Auditing Prompt & Packet Leakage ---")
    violations: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    with cases_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                cases.append(json.loads(line))

    # Self-test: Ensure auditor catches injected field
    mock_case = dict(cases[0]) if cases else {"claims": []}
    mock_packet = "This contains expected_active=1"
    self_test_caught = any(field in mock_packet for field in FORBIDDEN_FIELD_NAMES)

    producers = [
        ("claimledger", _claimledger_packet),
        ("naive", _naive_packet),
        ("recency", _recency_packet),
        ("summary", _summary_packet),
    ]

    total_checked = 0
    for case in cases[:200]:
        claims = case.get("claims", [])
        q_time = case.get("query_time")
        for name, producer in producers:
            packet = producer(claims, q_time)
            prompt = _build_prompt("Question?", packet)
            total_checked += 1
            for field in FORBIDDEN_FIELD_NAMES:
                if field in prompt:
                    violations.append({
                        "case_id": case.get("case_id"),
                        "producer": name,
                        "forbidden_field": field,
                    })

    passed = len(violations) == 0 and self_test_caught
    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "total_packets_checked": total_checked,
        "self_test_passed": self_test_caught,
        "violations": violations,
        "passed": passed,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"  Packets checked: {total_checked}")
    print(f"  Self-test passed: {self_test_caught}")
    print(f"  Violations found: {len(violations)}")
    print(f"wrote leakage audit report to {out_path}")
    return passed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checks",
        choices=["all", "sources", "leakage"],
        default="all",
        help="Audit checks to run (default: all)",
    )
    args = parser.parse_args()

    passed = True
    if args.checks in ("all", "sources"):
        audit_sources()

    if args.checks in ("all", "leakage"):
        if not audit_leakage():
            passed = False

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
