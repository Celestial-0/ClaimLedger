"""Command-line interface for the ClaimLedger prototype."""

from __future__ import annotations

import argparse
import json
import tomllib
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from .attacks import sweep_trust_escalation
from .attacks.adaptive import generate_policy_aware_attacks
from .attacks.perturbations import generate_qa_perturbations
from .core.ledger import ClaimLedger
from .datasets.locomo_adapter import adapt_locomo
from .datasets.realtalk_adapter import prepare_realtalk_cases
from .evaluation.benchmark import BASELINE_RUNNERS, get_all_runners, load_jsonl
from .evaluation.benchmark_performance import format_performance_table, run_performance_benchmark
from .evaluation.e2e_qa_runner import format_results_table, run_e2e_qa, save_results
from .evaluation.experiment import run_experiment
from .evaluation.extraction_error import run_claimledger_extraction_error_cases
from .evaluation.metrics import (
    bootstrap_mean_ci,
    compute_metrics,
    exact_active_correctness,
    paired_exact_active_comparison,
)
from .evaluation.multillm import (
    SUPPORTED_DATASETS,
    SUPPORTED_STRATEGIES,
    run_multillm,
)
from .evaluation.reporting import write_jsonl as _write_jsonl
from .evaluation.retrieval_eval import DEFAULT_PATHS as RETRIEVAL_DEFAULT_PATHS, run_retrieval_evaluation
from .evaluation.run_manifest import create_run_manifest as _create_run_manifest
from .evaluation.scenarios import run_demo_scenarios
from .evaluation.temporal_cases import generate_temporal_boundaries
from .extraction.llm import compare_extractors, extract_claims_llm
from .extraction.rule_based import extract_locomo_claims
from .integrations.ollama_client import OllamaClient
from .retrieval.qa import evaluate_locomo_qa


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="claimledger")
    sub = parser.add_subparsers(dest="command", required=True)

    init_db = sub.add_parser("init-db", help="Create an empty ClaimLedger SQLite database.")
    init_db.add_argument("path", type=Path)

    demo = sub.add_parser("demo", help="Run deterministic demo scenarios and print metrics.")
    demo.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    bench = sub.add_parser("run-benchmark", aliases=["run-jsonl"], help="Run a JSONL benchmark fixture.")
    bench.add_argument("path", type=Path)
    bench.add_argument("--baseline", choices=sorted(BASELINE_RUNNERS), default="claimledger")
    bench.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    compare = sub.add_parser("compare-jsonl", help="Compare ClaimLedger against JSONL benchmark baselines.")
    compare.add_argument("path", type=Path)
    compare.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    run_exp = sub.add_parser("run-experiment", help="Run full benchmark evaluation across systems and seeds.")
    run_exp.add_argument("--config", type=Path, default=Path("code/pyproject.toml"))
    run_exp.add_argument("--dataset", action="append", help="Dataset name")
    run_exp.add_argument("--output", type=Path, default=None, help="Output directory")

    adapt = sub.add_parser("adapt-locomo", help="Convert a LoCoMo-style JSON export to ClaimLedger JSONL.")
    adapt.add_argument("input", type=Path)
    adapt.add_argument("output", type=Path)
    adapt.add_argument("--limit", type=int, default=None)

    qa = sub.add_parser("eval-locomo-qa", help="Evaluate lexical evidence retrieval on LoCoMo QA annotations.")
    qa.add_argument("input", type=Path)
    qa.add_argument("--method", choices=["lexical", "bm25"], default="lexical")
    qa.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    extract = sub.add_parser("extract-locomo-claims", help="Extract deterministic claims from LoCoMo dialogue turns.")
    extract.add_argument("input", type=Path)
    extract.add_argument("output", type=Path)
    extract.add_argument("--include-fallback", action="store_true")
    extract.add_argument("--json", action="store_true", help="Print machine-readable JSON report.")

    perturb = sub.add_parser("generate-qa-perturbations", help="Generate QA-relevant perturbation cases.")
    perturb.add_argument("locomo", type=Path)
    perturb.add_argument("claims", type=Path)
    perturb.add_argument("output", type=Path)
    perturb.add_argument("--limit", type=int, default=None)
    perturb.add_argument("--json", action="store_true", help="Print machine-readable JSON report.")

    adaptive = sub.add_parser("generate-adaptive-attacks", help="Generate policy-aware adversarial stress cases.")
    adaptive.add_argument("input", type=Path)
    adaptive.add_argument("output", type=Path)
    adaptive.add_argument("--flood-size", type=int, default=3)
    adaptive.add_argument("--json", action="store_true", help="Print machine-readable JSON report.")

    temporal = sub.add_parser("generate-temporal-boundaries", help="Generate temporal-boundary benchmark cases.")
    temporal.add_argument("claims", type=Path)
    temporal.add_argument("output", type=Path)
    temporal.add_argument("--limit", type=int, default=None)
    temporal.add_argument("--json", action="store_true", help="Print machine-readable JSON report.")

    e2e = sub.add_parser("e2e-qa", help="Run end-to-end LLM QA evaluation.")
    e2e.add_argument("locomo", type=Path, help="Path to LoCoMo JSON file.")
    e2e.add_argument("perturbations", type=Path, help="Path to QA perturbation JSONL.")
    e2e.add_argument("--models", nargs="+", default=["phi4-mini", "gemma3:4b", "qwen3:4b"])
    e2e.add_argument("--strategies", nargs="+", default=None)
    e2e.add_argument("--limit", type=int, default=None)
    e2e.add_argument("--output", type=Path, default=None)
    e2e.add_argument("--json", action="store_true")

    llm_ext = sub.add_parser("llm-extract", help="Extract claims using LLM via Ollama.")
    llm_ext.add_argument("input", type=Path, help="Path to LoCoMo JSON.")
    llm_ext.add_argument("output", type=Path, help="Output JSONL path.")
    llm_ext.add_argument("--model", default="phi4-mini")
    llm_ext.add_argument("--sample-size", type=int, default=50)
    llm_ext.add_argument("--json", action="store_true")

    ext_compare = sub.add_parser("compare-extractors", help="Compare rule-based vs LLM claim extraction.")
    ext_compare.add_argument("rule_claims", type=Path)
    ext_compare.add_argument("llm_claims", type=Path)
    ext_compare.add_argument("--json", action="store_true")

    perf = sub.add_parser("perf", aliases=["benchmark-perf"], help="Run latency and storage benchmarks.")
    perf.add_argument("--scales", nargs="+", type=int, default=[100, 1000, 10000, 50000])
    perf.add_argument("--repetitions", type=int, default=3)
    perf.add_argument("--seed", type=int, default=20260818)
    perf.add_argument("--output", type=Path, default=None)
    perf.add_argument("--manifest", type=Path, default=None)
    perf.add_argument("--json", action="store_true")

    sweep = sub.add_parser(
        "extraction-sweep",
        help="Sweep the non-oracle extraction-error model.",
    )
    sweep.add_argument(
        "--p-detect",
        type=float,
        nargs="+",
        default=[1.0, 0.95, 0.9, 0.8, 0.7, 0.5, 0.3, 0.0],
        help="Detection probabilities to sweep (default: oracle-to-failing range).",
    )
    sweep.add_argument("--seed", type=int, default=0, help="Extraction-error draw seed.")
    sweep.add_argument(
        "--output",
        type=Path,
        default=Path("data/results/extraction_error"),
        help="Output directory for sweep_metrics.jsonl, sweep_per_case.jsonl, run_manifest.json.",
    )

    probe = sub.add_parser(
        "trust-probe",
        help="Run the adaptive trust-escalation poisoning probe.",
    )
    probe.add_argument(
        "--n-benign", type=int, nargs="+", default=[0, 1, 3, 5, 10, 25, 50]
    )
    probe.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    probe.add_argument(
        "--output",
        type=Path,
        default=Path("data/results/adaptive_poison"),
        help="Output directory for probe_outcomes.jsonl, summary.json, run_manifest.json.",
    )

    prep_rt = sub.add_parser("prepare-realtalk", help="Prepare REALTALK-derived ClaimLedger cases.")
    prep_rt.add_argument("--input", type=Path, default=Path("data/dataset/realtalk/processed"))
    prep_rt.add_argument("--output", type=Path, default=Path("data/derived/realtalk_memory_probe_cases.jsonl"))
    prep_rt.add_argument("--manifest", type=Path, default=Path("data/manifests/realtalk_preparation.json"))
    prep_rt.add_argument("--source-commit", default="b903e06a9770bf4e5fe9018c3e132889666d3b4a")

    ret = sub.add_parser("retrieval", help="Run deterministic context-retrieval measurements.")
    ret.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(RETRIEVAL_DEFAULT_PATHS),
        default=list(RETRIEVAL_DEFAULT_PATHS.keys()),
    )
    ret.add_argument("--output-root", type=Path, default=Path("data/results/runs"))
    ret.add_argument("--run-id", default="retrieval-official-benchmarks-20260818-v2")
    ret.add_argument("--limit", type=int, default=None)
    ret.add_argument("--top-k", type=int, default=8)
    ret.add_argument("--max-chars", type=int, default=24000)

    mllm = sub.add_parser("multillm", help="Run the resumable local-Ollama benchmark matrix.")
    mllm.add_argument("--dataset", choices=sorted(SUPPORTED_DATASETS), required=True)
    mllm.add_argument("--source-path", type=Path, default=None)
    mllm.add_argument("--output-root", type=Path, default=Path("data/results/runs"))
    mllm.add_argument("--run-id", default=None)
    mllm.add_argument("--models", nargs="+", default=None)
    mllm.add_argument("--strategies", nargs="+", default=["claimledger", "lexical", "recency"])
    mllm.add_argument("--seeds", nargs="+", type=int, default=None)
    mllm.add_argument("--limit", type=int, default=None)
    mllm.add_argument("--full", action="store_true")
    mllm.add_argument("--top-k", type=int, default=8)
    mllm.add_argument("--max-chars", type=int, default=24000)
    mllm.add_argument("--base-url", default="http://127.0.0.1:11434")
    mllm.add_argument("--validate-only", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "init-db":
        with ClaimLedger(args.path):
            pass
        print(f"Initialized {args.path}")
        return

    if args.command == "demo":
        outcomes = run_demo_scenarios()
        metrics = compute_metrics(outcomes)
        payload = {
            "outcomes": [outcome.__dict__ for outcome in outcomes],
            "metrics": metrics.__dict__,
        }
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"Ran {len(outcomes)} scenarios")
            print(f"Active count accuracy:      {metrics.exact_active_count_accuracy:.4f}")
            print(f"Temporal validity accuracy: {metrics.temporal_validity_accuracy:.4f}")
            print(f"Conflict disclosure rate:   {metrics.conflict_disclosure_rate:.4f}")
            print(f"Poison activation rate:     {metrics.poison_activation_rate:.4f}")
            print(f"Abstention correctness:     {metrics.abstention_correctness:.4f}")
        return

    if args.command in ("run-benchmark", "run-jsonl"):
        cases = load_jsonl(args.path)
        outcomes = BASELINE_RUNNERS[args.baseline](cases)
        metrics = compute_metrics(outcomes)
        if args.json:
            print(json.dumps(metrics.__dict__, indent=2, sort_keys=True))
        else:
            print(f"Loaded {len(cases)} cases")
            print(f"Active count accuracy:      {metrics.exact_active_count_accuracy:.4f}")
            print(f"Temporal validity accuracy: {metrics.temporal_validity_accuracy:.4f}")
            print(f"Conflict disclosure rate:   {metrics.conflict_disclosure_rate:.4f}")
            print(f"Poison activation rate:     {metrics.poison_activation_rate:.4f}")
            print(f"Abstention correctness:     {metrics.abstention_correctness:.4f}")
        return

    if args.command == "compare-jsonl":
        cases = load_jsonl(args.path)
        claimledger_outcomes = BASELINE_RUNNERS["claimledger"](cases)
        results = {
            baseline: paired_exact_active_comparison(
                claimledger_outcomes,
                runner(cases),
            )
            for baseline, runner in BASELINE_RUNNERS.items()
            if baseline != "claimledger"
        }
        if args.json:
            print(json.dumps({k: v.__dict__ for k, v in results.items()}, indent=2, sort_keys=True))
        else:
            for baseline, report in results.items():
                print(
                    f"{baseline:25s} diff: {report.mean_difference:+.4f} "
                    f"p: {report.mcnemar_exact_p:.4e} (wins: {report.system_wins}, losses: {report.baseline_wins})"
                )
        return

    if args.command == "run-experiment":
        out = run_experiment(config_path=args.config, datasets=args.dataset, output_dir=args.output)
        print(f"wrote runs under {out}")
        return

    if args.command == "adapt-locomo":
        count = adapt_locomo(args.input, args.output, limit=args.limit)
        print(f"Wrote {count} cases to {args.output}")
        return

    if args.command == "eval-locomo-qa":
        metrics = evaluate_locomo_qa(args.input, method=args.method)
        if args.json:
            print(json.dumps(metrics.__dict__, indent=2, sort_keys=True))
        else:
            print(f"Evaluated {metrics.total_questions} questions")
            print(f"Recall@1: {metrics.recall_at_1:.4f}")
            print(f"Recall@3: {metrics.recall_at_3:.4f}")
            print(f"Recall@5: {metrics.recall_at_5:.4f}")
            print(f"MRR:      {metrics.mean_reciprocal_rank:.4f}")
        return

    if args.command == "extract-locomo-claims":
        report = extract_locomo_claims(
            args.input,
            args.output,
            include_fallback=args.include_fallback,
        )
        if args.json:
            print(json.dumps(report.__dict__, indent=2, sort_keys=True))
        else:
            print(f"Processed {report.total_sessions} sessions, {report.total_turns} turns")
            print(f"Extracted {report.total_claims_extracted} claims")
            print(f"Entity count: {report.entity_count}")
        return

    if args.command == "generate-qa-perturbations":
        report = generate_qa_perturbations(
            args.locomo,
            args.claims,
            args.output,
            limit=args.limit,
        )
        if args.json:
            print(json.dumps(report.__dict__, indent=2, sort_keys=True))
        else:
            print(f"Generated {report.total_perturbations} perturbations")
            print(f"Direct conflict: {report.direct_conflict_count}")
            print(f"Poisoned write:  {report.poisoned_write_count}")
        return

    if args.command == "generate-adaptive-attacks":
        report = generate_policy_aware_attacks(
            args.input,
            args.output,
            flood_size=args.flood_size,
        )
        if args.json:
            print(json.dumps(report.__dict__, indent=2, sort_keys=True))
        else:
            print(f"Generated {report.total_attacks} adaptive attacks")
            print(f"Trust escalation:  {report.trust_escalation_count}")
            print(f"Cascade flood:     {report.cascade_flood_count}")
            print(f"Payload preserved: {report.payload_preserved_count}")
        return

    if args.command == "generate-temporal-boundaries":
        report = generate_temporal_boundaries(
            args.claims,
            args.output,
            limit=args.limit,
        )
        if args.json:
            print(json.dumps(report.__dict__, indent=2, sort_keys=True))
        else:
            print(f"Generated {report.total_cases} temporal boundary cases")
            print(f"Boundary queries: {report.boundary_query_count}")
            print(f"Overlap queries:  {report.overlap_query_count}")
            print(f"Pre-start/post-end: {report.out_of_bounds_count}")
        return

    if args.command == "e2e-qa":
        results = run_e2e_qa(
            args.locomo,
            args.perturbations,
            models=args.models,
            strategies=args.strategies,
            limit=args.limit,
        )
        if args.output:
            save_results(results, args.output)
        if args.json:
            print(json.dumps([r.__dict__ for r in results], indent=2, default=str))
        else:
            print(format_results_table(results))
        return

    if args.command == "llm-extract":
        report = extract_claims_llm(
            args.input,
            args.output,
            model=args.model,
            sample_size=args.sample_size,
        )
        if args.json:
            print(json.dumps(report.__dict__, indent=2, sort_keys=True))
        else:
            print(f"Extracted {report.total_claims_extracted} claims from {report.total_turns} turns")
            print(f"Claims/turn: {report.claims_per_turn:.3f}, Errors: {report.extraction_errors}")
        return

    if args.command == "compare-extractors":
        result = compare_extractors(args.rule_claims, args.llm_claims)
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(f"Rule claims: {result['rule_claim_count']}")
            print(f"LLM claims:  {result['llm_claim_count']}")
            print(f"Overlap:     {result['overlap_count']}")
            print(f"Precision:   {result['precision']:.4f}")
            print(f"Recall:      {result['recall']:.4f}")
            print(f"F1:          {result['f1']:.4f}")
        return

    if args.command in ("perf", "benchmark-perf"):
        reports = run_performance_benchmark(
            scales=args.scales or [100, 1000, 10000, 50000],
            repetitions=args.repetitions,
            seed=args.seed,
        )
        if args.output:
            out_path = args.output if args.output.is_absolute() else Path.cwd() / args.output
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps([report.__dict__ for report in reports], indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            manifest_p = args.manifest or (out_path.parent.parent / "manifests" / out_path.name)
            manifest_p = manifest_p if manifest_p.is_absolute() else Path.cwd() / manifest_p
            manifest_p.parent.mkdir(parents=True, exist_ok=True)
            manifest = _create_run_manifest(
                run_id="claimledger-performance-realtalk-release",
                config_path=Path.cwd() / "code/pyproject.toml",
                seed=args.seed,
            )
            manifest_p.write_text(
                json.dumps(
                    {
                        **manifest.to_dict(),
                        "scales": args.scales,
                        "repetitions": args.repetitions,
                        "result_path": str(out_path),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            print(
                json.dumps(
                    {"output": str(out_path), "manifest": str(manifest_p), "records": len(reports)},
                    indent=2,
                )
            )
        elif args.json:
            print(json.dumps([r.__dict__ for r in reports], indent=2))
        else:
            print(format_performance_table(reports))
        return

    if args.command == "prepare-realtalk":
        input_dir = args.input if args.input.is_absolute() else Path.cwd() / args.input
        output = args.output if args.output.is_absolute() else Path.cwd() / args.output
        manifest = args.manifest if args.manifest.is_absolute() else Path.cwd() / args.manifest
        report = prepare_realtalk_cases(
            input_dir,
            output,
            manifest_path=manifest,
            source_commit=args.source_commit,
        )
        print(json.dumps(report.__dict__, indent=2, sort_keys=True))
        return

    if args.command == "retrieval":
        run_dir = run_retrieval_evaluation(
            datasets=args.datasets,
            output_root=args.output_root,
            run_id=args.run_id,
            limit=args.limit,
            top_k=args.top_k,
            max_chars=args.max_chars,
        )
        print(f"wrote retrieval evaluation under {run_dir}")
        return

    if args.command == "multillm":
        cfg_candidates = (Path("code/pyproject.toml"), Path("pyproject.toml"))
        settings: dict = {}
        for c in cfg_candidates:
            if c.exists():
                with c.open("rb") as handle:
                    settings = tomllib.load(handle).get("tool", {}).get("claimledger", {}).get("multillm", {})
                break
        default_sources = {
            "locomo": Path("data/dataset/locomo/locomo10.json"),
            "realtalk": Path("data/dataset/realtalk/processed"),
            "longmemeval_s": Path("data/dataset/longmemeval/longmemeval_s_cleaned.json"),
            "longmemeval_m": Path("data/dataset/longmemeval/longmemeval_m_cleaned.json"),
            "longmemeval_oracle": Path("data/dataset/longmemeval/longmemeval_oracle.json"),
        }
        source_path = args.source_path or default_sources.get(args.dataset)
        models = args.models or list(settings.get("models", []))
        seeds = args.seeds or [int(s) for s in settings.get("seeds", [0])]
        run_id = args.run_id or f"multillm-{args.dataset}"
        if args.validate_only:
            client = OllamaClient(base_url=args.base_url)
            if not client.is_available():
                raise SystemExit("Ollama is not reachable")
            available = set(client.list_models())
            missing = sorted(set(models) - available)
            if missing:
                raise SystemExit(f"Missing Ollama tags: {missing}")
            print(json.dumps({"ollama_available": True, "available_tags": sorted(available)}, indent=2))
            return
        run_dir = run_multillm(
            dataset=args.dataset,
            source_path=source_path,
            output_root=args.output_root,
            run_id=run_id,
            models=models,
            strategies=args.strategies,
            seeds=seeds,
            limit=args.limit,
            top_k=args.top_k,
            max_chars=args.max_chars,
            max_output_tokens=int(settings.get("max_output_tokens", 256)),
            num_ctx=int(settings.get("num_ctx", 8192)),
            temperature=float(settings.get("temperature", 0.0)),
            timeout_s=int(settings.get("timeout_seconds", 180)),
            base_url=args.base_url,
        )
        print(f"wrote multillm run under {run_dir}")
        return

    if args.command == "extraction-sweep":
        output_dir = args.output if args.output.is_absolute() else (Path.cwd() / args.output).resolve()
        dataset_path = (Path.cwd() / "data/derived/realtalk_memory_probe_cases.jsonl").resolve()
        cases = load_jsonl(dataset_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = _create_run_manifest(
            run_id=f"extraction-error-sweep-seed{args.seed}",
            config_path=None,
            dataset_paths=[dataset_path],
            seed=args.seed,
        )
        manifest.write_json(output_dir / "run_manifest.json")
        metric_fields = [
            "exact_active_count_accuracy",
            "temporal_validity_accuracy",
            "conflict_disclosure_rate",
            "poison_activation_rate",
            "abstention_correctness",
        ]
        metric_rows: list = []
        per_case_rows: list = []
        for p_detect in sorted(set(args.p_detect), reverse=True):
            outcomes = run_claimledger_extraction_error_cases(cases, p_detect=p_detect, seed=args.seed)
            metrics = compute_metrics(outcomes)
            for field in metric_fields:
                metric_rows.append(
                    {
                        "run_id": manifest.run_id,
                        "p_detect": p_detect,
                        "seed": args.seed,
                        "metric": field,
                        "value": getattr(metrics, field),
                        "n_cases": len(cases),
                    }
                )
            for case, outcome in zip(cases, outcomes, strict=True):
                per_case_rows.append(
                    {
                        "run_id": manifest.run_id,
                        "p_detect": p_detect,
                        "seed": args.seed,
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
        _write_jsonl(output_dir / "sweep_metrics.jsonl", metric_rows)
        _write_jsonl(output_dir / "sweep_per_case.jsonl", per_case_rows)
        print(f"wrote extraction-error sweep under {output_dir}")
        return

    if args.command == "trust-probe":
        output_dir = args.output if args.output.is_absolute() else (Path.cwd() / args.output).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = _create_run_manifest(
            run_id="adaptive-trust-escalation-probe",
            config_path=None,
            dataset_paths=[],
            seed=args.seeds[0] if args.seeds else None,
        )
        manifest.write_json(output_dir / "run_manifest.json")
        outcomes = sweep_trust_escalation(
            n_benign_values=sorted(set(args.n_benign)), seeds=sorted(set(args.seeds))
        )
        _write_jsonl(output_dir / "probe_outcomes.jsonl", (asdict(o) for o in outcomes))
        grouped: dict = defaultdict(list)
        for o in outcomes:
            grouped[(o.mode, o.n_benign)].append(o.payload_active)
        summary = [
            {
                "mode": mode,
                "n_benign": n_benign,
                "activation_rate": sum(flags) / len(flags),
                "n_seeds": len(flags),
            }
            for (mode, n_benign), flags in sorted(grouped.items())
        ]
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"wrote adaptive-poison probe under {output_dir}")
        return

    raise AssertionError(f"unhandled command {args.command}")
