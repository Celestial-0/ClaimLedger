import unittest
from pathlib import Path

from claimledger.evaluation.benchmark import load_jsonl, run_claimledger_cases
from claimledger.evaluation.extraction_error import run_claimledger_extraction_error_cases
from claimledger.evaluation.metrics import compute_metrics


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mini_memory_cases.jsonl"


class ExtractionErrorTests(unittest.TestCase):
    def test_oracle_detection_reproduces_claimledger(self) -> None:
        # p_detect = 1.0 means the extractor never misses a label, so the
        # non-oracle runner must reproduce oracle ClaimLedger exactly.
        cases = load_jsonl(FIXTURE)
        oracle = compute_metrics(run_claimledger_cases(cases))
        non_oracle = compute_metrics(
            run_claimledger_extraction_error_cases(cases, p_detect=1.0, seed=0)
        )
        self.assertEqual(non_oracle, oracle)

    def test_zero_detection_degrades_below_perfect(self) -> None:
        # p_detect = 0.0 means the extractor misses every conflict relation and
        # poison flag, so exact accuracy must drop below the oracle's 1.0.
        cases = load_jsonl(FIXTURE)
        metrics = compute_metrics(
            run_claimledger_extraction_error_cases(cases, p_detect=0.0, seed=0)
        )
        self.assertLess(metrics.exact_active_count_accuracy, 1.0)

    def test_degradation_is_monotone_at_endpoints(self) -> None:
        # The governance delta must be non-increasing as detection falls from
        # oracle (1.0) to failing (0.0): accuracy(1.0) >= accuracy(0.0).
        cases = load_jsonl(FIXTURE)
        high = compute_metrics(
            run_claimledger_extraction_error_cases(cases, p_detect=1.0, seed=0)
        )
        low = compute_metrics(
            run_claimledger_extraction_error_cases(cases, p_detect=0.0, seed=0)
        )
        self.assertGreaterEqual(
            high.exact_active_count_accuracy, low.exact_active_count_accuracy
        )

    def test_deterministic_for_fixed_case_p_seed(self) -> None:
        # Same (case, p_detect, seed) must give identical outcomes (reproducible
        # and resumable sweep).
        cases = load_jsonl(FIXTURE)
        first = run_claimledger_extraction_error_cases(cases, p_detect=0.5, seed=7)
        second = run_claimledger_extraction_error_cases(cases, p_detect=0.5, seed=7)
        self.assertEqual(first, second)

    def test_invalid_p_detect_rejected(self) -> None:
        cases = load_jsonl(FIXTURE)
        with self.assertRaises(ValueError):
            run_claimledger_extraction_error_cases(cases, p_detect=1.5, seed=0)


if __name__ == "__main__":
    unittest.main()
