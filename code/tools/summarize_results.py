"""Unified summary generator for ClaimLedger evaluation results.

Consolidates:
- REALTALK mechanism run summaries (JSON and Markdown)
- Extraction-error non-oracle degradation curves
- Tier-2 multi-LLM matrix aggregation and gate reports

Usage:
    uv run --project code python code/tools/summarize_results.py [--target {all,realtalk,extraction,multillm}]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# 1. REALTALK Mechanism Summary
# ---------------------------------------------------------------------------

def summarize_realtalk(
    runs_dir: Path,
    cases_path: Path,
    output_json: Path | None = None,
    output_md: Path | None = None,
) -> dict[str, Any]:
    metric_rows: list[dict[str, Any]] = []
    for path in sorted(runs_dir.glob("*/metrics.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    metric_rows.append(json.loads(line))

    per_case_paths = sorted(runs_dir.glob("*/per_case.jsonl"))
    cases: list[dict[str, Any]] = []
    if cases_path.exists():
        with cases_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    cases.append(json.loads(line))

    task_counts = Counter(str(case.get("task_type")) for case in cases)
    metric_values: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in metric_rows:
        metric_values[str(row["system"])][str(row["metric"])].append(row)

    def _agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
        num = sum(int(r.get("numerator", 0)) for r in rows)
        den = sum(int(r.get("denominator", 0)) for r in rows)
        vals = [float(r["value"]) for r in rows]
        return {
            "numerator": num,
            "denominator": den,
            "pooled_value": num / den if den else None,
            "seed_min": min(vals) if vals else None,
            "seed_max": max(vals) if vals else None,
        }

    summary = {
        "dataset": "REALTALK",
        "case_count": len(cases),
        "task_counts": dict(sorted(task_counts.items())),
        "run_count": len(per_case_paths),
        "systems": sorted(metric_values),
        "metrics": {
            system: {metric: _agg(r_list) for metric, r_list in sorted(metrics.items())}
            for system, metrics in sorted(metric_values.items())
        },
    }

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    md_lines = [
        "# Fresh REALTALK Results Summary",
        "",
        f"Cases: {summary['case_count']}; runs: {summary['run_count']}; source: REALTALK.",
        "",
        "| System | Exact active-count accuracy | Conflict disclosure | Poison activation | Abstention correctness |",
        "|---|---:|---:|---:|---:|",
    ]
    for system, metrics in summary["metrics"].items():
        vals = [
            metrics.get(m, {}).get("pooled_value")
            for m in (
                "exact_active_count_accuracy",
                "conflict_disclosure_rate",
                "poison_activation_rate",
                "abstention_correctness",
            )
        ]
        md_lines.append(
            "| {} | {} | {} | {} | {} |".format(
                system, *("—" if v is None else f"{v:.3f}" for v in vals)
            )
        )
    md_lines.extend(["", "Task counts:"])
    md_lines.extend(f"- `{t}`: {c}" for t, c in summary["task_counts"].items())
    md_lines.append("")
    md_content = "\n".join(md_lines)

    if output_md:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(md_content, encoding="utf-8")

    return summary


# ---------------------------------------------------------------------------
# 2. Extraction-Error Degradation Table
# ---------------------------------------------------------------------------

def summarize_extraction_error(sweep_file: Path) -> dict[float, dict[str, float]]:
    if not sweep_file.exists():
        return {}
    rows = [json.loads(l) for l in sweep_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_p: dict[float, dict[str, float]] = defaultdict(dict)
    for row in rows:
        by_p[row["p_detect"]][row["metric"]] = row["value"]

    print("\n--- Extraction-Error Degradation Curve ---")
    print("p_detect | exact_acc | temporal_acc | conflict_discl | poison_act | abstain_corr")
    print("---------+-----------+--------------+----------------+------------+-------------")
    for p in sorted(by_p, reverse=True):
        m = by_p[p]
        print(
            f"{p:>8.2f} | "
            f"{m.get('exact_active_count_accuracy', 0):>9.3f} | "
            f"{m.get('temporal_validity_accuracy', 0):>12.3f} | "
            f"{m.get('conflict_disclosure_rate', 0):>14.3f} | "
            f"{m.get('poison_activation_rate', 0):>10.3f} | "
            f"{m.get('abstention_correctness', 0):>12.3f}"
        )
    return by_p


# ---------------------------------------------------------------------------
# 3. Tier-2 Multi-LLM Summary & Acceptance Gate
# ---------------------------------------------------------------------------

def summarize_multillm_tier2(runs_dir: Path) -> None:
    expected_n = {
        "realtalk": 728,
        "locomo": 1_986,
        "longmemeval_s": 500,
        "longmemeval_m": 500,
        "longmemeval_oracle": 500,
    }
    legs = {
        "tier2-realtalk-v1": ("realtalk", {"claimledger", "lexical", "recency"}),
        "tier2-locomo-v1": ("locomo", {"claimledger", "lexical", "recency"}),
        "tier2-longmemes-v1": ("longmemeval_s", {"claimledger", "lexical", "recency"}),
        "tier2-longmemem-v1": ("longmemeval_m", {"claimledger", "lexical", "recency"}),
        "tier2-longmeo-v1": ("longmemeval_oracle", {"claimledger", "lexical", "recency"}),
        "tier2-realtalk-fc-v1": ("realtalk", {"full_context"}),
        "tier2-locomo-fc-v1": ("locomo", {"full_context"}),
        "tier2-longmemes-fc-v1": ("longmemeval_s", {"full_context"}),
    }

    records: list[dict[str, Any]] = []
    for leg_id, (dataset, expected_strategies) in legs.items():
        res_file = runs_dir / leg_id / "results.jsonl"
        if not res_file.exists():
            continue
        with res_file.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    r = json.loads(line)
                    if r.get("strategy") in expected_strategies:
                        records.append({**r, "dataset": dataset})

    if not records:
        return

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        grouped[(r["dataset"], r["strategy"], r["model"])].append(r)

    rows: list[dict[str, Any]] = []
    for (dataset, strategy, model), r_list in sorted(grouped.items()):
        n = len(r_list)
        f1_mean = sum(float(x.get("token_f1", 0)) for x in r_list) / n
        em_mean = sum(float(x.get("exact_match", 0)) for x in r_list) / n
        abst_mean = sum(float(x.get("abstention", 0)) for x in r_list) / n
        rows.append({
            "dataset": dataset,
            "strategy": strategy,
            "model": model,
            "n": n,
            "token_f1": f1_mean,
            "exact_match": em_mean,
            "abstention_rate": abst_mean,
        })

    summary_csv = runs_dir / "tier2_summary.csv"
    if rows:
        with summary_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {len(rows)} multi-LLM tier-2 cells to {summary_csv}")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=["all", "realtalk", "extraction", "multillm"],
        default="all",
        help="Summary domain to run (default: all).",
    )
    parser.add_argument(
        "--runs",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "results" / "runs",
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "derived" / "realtalk_memory_probe_cases.jsonl",
    )
    parser.add_argument(
        "--sweep",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "results" / "extraction_error" / "sweep_metrics.jsonl",
    )
    args = parser.parse_args()

    if args.target in ("all", "realtalk") and args.runs.exists():
        out_json = REPOSITORY_ROOT / "data" / "results" / "realtalk_summary.json"
        out_md = REPOSITORY_ROOT / "data" / "results" / "realtalk_summary.md"
        summary = summarize_realtalk(args.runs, args.cases, out_json, out_md)
        print(f"REALTALK summary: {summary['case_count']} cases across {summary['run_count']} runs.")

    if args.target in ("all", "extraction") and args.sweep.exists():
        summarize_extraction_error(args.sweep)

    if args.target in ("all", "multillm") and args.runs.exists():
        summarize_multillm_tier2(args.runs)


if __name__ == "__main__":
    main()
