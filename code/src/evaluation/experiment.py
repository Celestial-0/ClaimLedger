"""Run configured ClaimLedger mechanism evaluations and emit report artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .benchmark import BenchmarkCase, get_all_runners, load_jsonl
from .config import ExperimentConfig, load_config
from .metrics import TEMPORAL_TASKS, bootstrap_cluster_mean_ci
from .reporting import write_jsonl, write_metric_records
from .run_manifest import create_run_manifest
from .schemas import MetricRecord
from .statistics import paired_comparisons

DATASETS = {
    "realtalk_memory_probes": Path("data/derived/realtalk_memory_probe_cases.jsonl"),
}

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _resolve_config_path(path: Path) -> Path:
    """Resolve a config path from either the current directory or repository root."""
    if path.is_absolute():
        return path
    candidates = (Path.cwd() / path, REPOSITORY_ROOT / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (REPOSITORY_ROOT / path).resolve()


def _resolve_repository_path(path: Path) -> Path:
    """Anchor relative datasets and artifacts to the repository root."""
    return path if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _cluster_id(case_id: str) -> str:
    """Return the REALTALK conversation key for clustered uncertainty."""
    return case_id.split("_q", 1)[0] if "_q" in case_id else case_id


def _metric_cluster_values(
    cases: list[BenchmarkCase], outcomes: list[Any], metric: str
) -> tuple[list[bool], dict[str, list[bool]]]:
    values: list[bool] = []
    clustered: dict[str, list[bool]] = {}
    for case, outcome in zip(cases, outcomes, strict=True):
        if metric == "exact_active_count_accuracy":
            value = outcome.actual_active == case.expected_active
        elif metric == "conflict_disclosure_rate":
            if outcome.name != "direct_conflict":
                continue
            value = outcome.conflict_disclosed
        elif metric == "poison_activation_rate":
            if outcome.name != "poisoned_write":
                continue
            value = outcome.poisoned_active
        elif metric == "abstention_correctness":
            if case.expected_active != 0:
                continue
            value = outcome.abstained and outcome.actual_active == 0
        elif metric == "temporal_validity_accuracy":
            if outcome.name not in TEMPORAL_TASKS:
                continue
            value = outcome.actual_active == case.expected_active
        else:
            raise KeyError(metric)
        values.append(value)
        clustered.setdefault(_cluster_id(case.case_id), []).append(value)
    return values, clustered


def _metric_records(
    run_id: str,
    dataset: str,
    system: str,
    cases: list[BenchmarkCase],
    outcomes: list[Any],
    seed: int,
) -> list[MetricRecord]:
    records: list[MetricRecord] = []
    for metric in (
        "exact_active_count_accuracy",
        "conflict_disclosure_rate",
        "poison_activation_rate",
        "abstention_correctness",
        "temporal_validity_accuracy",
    ):
        values, clustered_values = _metric_cluster_values(cases, outcomes, metric)
        if not values:
            continue
        interval = bootstrap_cluster_mean_ci(clustered_values, seed=seed)
        records.append(
            MetricRecord(
                run_id=run_id,
                system=system,
                dataset=dataset,
                condition="mechanism",
                metric=metric,
                value=interval.mean,
                numerator=sum(values),
                denominator=len(values),
                ci_lower=interval.lower,
                ci_upper=interval.upper,
                ci_method=f"conversation-clustered bootstrap; seed={seed}",
            )
        )
    return records


def run_dataset(config: ExperimentConfig, dataset: str, output_root: Path, config_path: Path) -> Path:
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset '{dataset}'; choose from {sorted(DATASETS)}")
    dataset_path = _resolve_repository_path(DATASETS[dataset])
    cases = load_jsonl(dataset_path)
    runners = get_all_runners()
    selected = [name for name in config.systems if name in runners]
    missing = sorted(set(config.systems) - set(selected))
    if missing:
        raise ValueError(f"unknown systems in config: {missing}")

    for seed in config.seeds:
        run_id = f"{config.name}-{dataset}-seed{seed}"
        run_dir = output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = create_run_manifest(
            run_id=run_id,
            config_path=config_path,
            dataset_paths=[dataset_path],
            seed=seed,
        )
        manifest.write_json(run_dir / "run_manifest.json")

        outcomes = {name: runners[name](cases) for name in selected}
        per_case_rows = []
        for system, system_outcomes in outcomes.items():
            for case, outcome in zip(cases, system_outcomes, strict=True):
                per_case_rows.append(
                    {
                        "run_id": run_id,
                        "system": system,
                        "case_id": case.case_id,
                        "task_type": case.task_type,
                        "expected_active": case.expected_active,
                        "actual_active": outcome.actual_active,
                        "exact_active_correct": outcome.actual_active == case.expected_active,
                        "conflict_disclosed": outcome.conflict_disclosed,
                        "poisoned_active": outcome.poisoned_active,
                        "abstained": outcome.abstained,
                    }
                )
        write_jsonl(run_dir / "per_case.jsonl", per_case_rows)
        records = [
            record
            for name in selected
            for record in _metric_records(run_id, dataset, name, cases, outcomes[name], seed)
        ]
        write_jsonl(run_dir / "metrics.jsonl", (record.to_dict() for record in records))
        write_metric_records(run_dir / "metrics.csv", records)

        baseline_outcomes = {
            name: outcomes[name]
            for name in selected
            if name != "claimledger"
        }
        comparisons = (
            paired_comparisons(
                system="claimledger",
                system_outcomes=outcomes["claimledger"],
                baseline_outcomes=baseline_outcomes,
            )
            if baseline_outcomes
            else []
        )
        (run_dir / "paired_tests.json").write_text(
            json.dumps([record.to_dict() for record in comparisons], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return output_root


def run_experiment(
    config_path: Path | str = Path("pyproject.toml"),
    datasets: list[str] | None = None,
    output_dir: Path | str | None = None,
) -> Path:
    """Run full benchmark experiment from config."""
    cfg_path = _resolve_config_path(Path(config_path))
    config = load_config(cfg_path)
    selected_datasets = datasets or [name for name in config.datasets if name in DATASETS]
    out_root = _resolve_repository_path(Path(output_dir) if output_dir else Path(config.output_dir))
    for dataset in selected_datasets:
        run_dataset(config, dataset, out_root, cfg_path)
    return out_root
