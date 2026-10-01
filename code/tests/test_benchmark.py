import unittest
from pathlib import Path

from claimledger.evaluation.benchmark import (
    BASELINE_RUNNERS,
    load_jsonl,
    run_claimledger_cases,
    run_naive_cases,
    run_no_conflict_cases,
    run_no_temporal_cases,
    get_all_runners,
)
from claimledger.evaluation.metrics import (
    _exact_two_sided_binomial_p,
    compute_metrics,
    paired_exact_active_comparison,
)


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mini_memory_cases.jsonl"
TEMPORAL_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "temporal_boundary_cases.jsonl"


class BenchmarkTests(unittest.TestCase):
    def test_fixture_loads(self) -> None:
        cases = load_jsonl(FIXTURE)

        self.assertEqual(len(cases), 3)
        self.assertEqual(cases[0].task_type, "stale_update")

    def test_claimledger_beats_naive_on_fixture(self) -> None:
        cases = load_jsonl(FIXTURE)

        claimledger_metrics = compute_metrics(run_claimledger_cases(cases))
        naive_metrics = compute_metrics(run_naive_cases(cases))

        self.assertEqual(claimledger_metrics.exact_active_count_accuracy, 1.0)
        self.assertLess(naive_metrics.exact_active_count_accuracy, 1.0)
        self.assertEqual(claimledger_metrics.poison_activation_rate, 0.0)
        self.assertGreater(naive_metrics.poison_activation_rate, 0.0)

    def test_no_conflict_ablation_loses_conflict_disclosure(self) -> None:
        cases = load_jsonl(FIXTURE)

        metrics = compute_metrics(run_no_conflict_cases(cases))

        self.assertEqual(metrics.conflict_disclosure_rate, 0.0)
        self.assertLess(metrics.exact_active_count_accuracy, 1.0)

    def test_temporal_boundary_suite_separates_no_temporal_ablation(self) -> None:
        cases = load_jsonl(TEMPORAL_FIXTURE)

        claimledger_metrics = compute_metrics(run_claimledger_cases(cases))
        no_temporal_metrics = compute_metrics(run_no_temporal_cases(cases))

        self.assertEqual(len(cases), 10)
        self.assertEqual(claimledger_metrics.temporal_validity_accuracy, 1.0)
        self.assertEqual(claimledger_metrics.exact_active_count_accuracy, 1.0)
        self.assertLess(no_temporal_metrics.temporal_validity_accuracy, 1.0)
        self.assertLess(no_temporal_metrics.exact_active_count_accuracy, 1.0)

    def test_baseline_registry_contains_expected_runners(self) -> None:
        self.assertIn("claimledger", BASELINE_RUNNERS)
        self.assertIn("naive", BASELINE_RUNNERS)
        self.assertIn("no_conflict", BASELINE_RUNNERS)
        self.assertIn("no_temporal", BASELINE_RUNNERS)
        self.assertIn("trust_only", BASELINE_RUNNERS)

        runners = get_all_runners()
        self.assertIn("temporal_only", runners)
        self.assertIn("provenance_only", runners)
        self.assertIn("structured_fact_store", runners)

    def test_paired_comparison_reports_system_wins(self) -> None:
        cases = load_jsonl(FIXTURE)

        report = paired_exact_active_comparison(run_claimledger_cases(cases), run_naive_cases(cases))

        self.assertGreater(report.system_wins, report.baseline_wins)
        self.assertLessEqual(report.mcnemar_exact_p, 1.0)

    def test_exact_binomial_p_value_has_nonzero_reporting_floor(self) -> None:
        p_value = _exact_two_sided_binomial_p(1452, 0)

        self.assertGreaterEqual(p_value, 1e-300)
        self.assertLess(p_value, 1e-200)


if __name__ == "__main__":
    unittest.main()





