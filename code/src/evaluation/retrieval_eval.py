"""Run deterministic context-retrieval measurements over official QA releases."""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .multillm import iter_examples, retrieval_scores, select_context

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_PATHS = {
    "locomo": Path("data/dataset/locomo/locomo10.json"),
    "realtalk": Path("data/dataset/realtalk/processed"),
    "longmemeval_s": Path("data/dataset/longmemeval/longmemeval_s_cleaned.json"),
    "longmemeval_m": Path("data/dataset/longmemeval/longmemeval_m_cleaned.json"),
    "longmemeval_oracle": Path("data/dataset/longmemeval/longmemeval_oracle.json"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_manifest(path: Path) -> dict[str, object]:
    files = sorted(path.glob("Chat_*.json")) if path.is_dir() else [path]
    return {
        "path": str(path),
        "files": [{"path": str(item), "bytes": item.stat().st_size, "sha256": _sha256(item)} for item in files],
    }


def run_retrieval_evaluation(
    datasets: list[str] | None = None,
    output_root: Path | str = Path("data/results/runs"),
    run_id: str = "retrieval-official-benchmarks-20260818-v2",
    limit: int | None = None,
    top_k: int = 8,
    max_chars: int = 24_000,
) -> Path:
    """Run deterministic context retrieval evaluation across benchmarks."""
    selected_datasets = datasets or list(DEFAULT_PATHS.keys())
    out_root = Path(output_root) if Path(output_root).is_absolute() else REPOSITORY_ROOT / output_root
    strategies = ["claimledger", "lexical", "recency", "oracle"]
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    per_case = run_dir / "retrieval_results.jsonl"
    aggregate_path = run_dir / "retrieval_aggregate.json"
    manifest_path = run_dir / "retrieval_manifest.json"

    resolved_sources = {
        d: (DEFAULT_PATHS[d] if DEFAULT_PATHS[d].is_absolute() else REPOSITORY_ROOT / DEFAULT_PATHS[d])
        for d in selected_datasets
    }

    manifest = {
        "schema_version": "1.0",
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "datasets": selected_datasets,
        "strategies": strategies,
        "top_k": top_k,
        "max_chars": max_chars,
        "limit": limit,
        "llm_used": False,
        "sources": {d: _source_manifest(path) for d, path in resolved_sources.items()},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    groups: dict[tuple[str, str, str | None], list[dict[str, object]]] = defaultdict(list)
    overall: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    started = time.perf_counter()

    with per_case.open("w", encoding="utf-8") as handle:
        for dataset in selected_datasets:
            source = resolved_sources[dataset]
            if not source.exists():
                raise FileNotFoundError(f"missing source dataset at {source}")
            for example in iter_examples(dataset, source, limit=limit):
                for strategy in strategies:
                    t0 = time.perf_counter()
                    selection_max_chars = 10_000_000 if strategy == "oracle" else max_chars
                    context = select_context(example, strategy, top_k=top_k, max_chars=selection_max_chars)
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    record = {
                        "dataset": dataset,
                        "example_id": example.example_id,
                        "task_type": example.task_type,
                        "strategy": strategy,
                        **retrieval_scores(example, context),
                        "context_chars": sum(len(chunk.text) for chunk in context),
                        "source_chunk_count": len(example.chunks),
                        "selection_latency_ms": elapsed_ms,
                    }
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    group_key = (dataset, strategy, example.task_type)
                    groups[group_key].append(record)
                    overall[(dataset, strategy)].append(record)
            handle.flush()

    aggregate = []
    for (dataset, strategy, task_type), records in sorted(groups.items()):
        n = len(records)
        aggregate.append({
            "dataset": dataset,
            "strategy": strategy,
            "task_type": task_type,
            "n": n,
            "evidence_recall_any": sum(bool(r["retrieved_evidence_any"]) for r in records) / n,
            "evidence_recall_all": sum(bool(r["retrieved_evidence_all"]) for r in records) / n,
            "mean_context_chars": sum(int(r["context_chars"]) for r in records) / n,
            "mean_source_chunk_count": sum(int(r["source_chunk_count"]) for r in records) / n,
            "selection_latency_ms_mean": sum(float(r["selection_latency_ms"]) for r in records) / n,
        })
    for (dataset, strategy), records in sorted(overall.items()):
        n = len(records)
        aggregate.append({
            "dataset": dataset,
            "strategy": strategy,
            "task_type": None,
            "n": n,
            "evidence_recall_any": sum(bool(r["retrieved_evidence_any"]) for r in records) / n,
            "evidence_recall_all": sum(bool(r["retrieved_evidence_all"]) for r in records) / n,
            "mean_context_chars": sum(int(r["context_chars"]) for r in records) / n,
            "mean_source_chunk_count": sum(int(r["source_chunk_count"]) for r in records) / n,
            "selection_latency_ms_mean": sum(float(r["selection_latency_ms"]) for r in records) / n,
        })
    aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    overall_counts = [int(item["n"]) for item in aggregate if item["task_type"] is None]
    manifest["completed_examples"] = sum(overall_counts) // len(strategies) if overall_counts else 0
    manifest["elapsed_seconds"] = time.perf_counter() - started
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return run_dir
