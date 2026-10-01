import tempfile
import unittest
from pathlib import Path

from claimledger.datasets.locomo_adapter import adapt_locomo
from claimledger.evaluation.benchmark import load_jsonl, run_claimledger_cases
from claimledger.evaluation.metrics import compute_metrics


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "locomo_tiny.json"


class AdapterTests(unittest.TestCase):
    def test_locomo_adapter_outputs_runnable_cases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "cases.jsonl"
            count = adapt_locomo(FIXTURE, output)

            cases = load_jsonl(output)
            metrics = compute_metrics(run_claimledger_cases(cases))

        self.assertEqual(count, 3)
        self.assertEqual(len(cases), 3)
        self.assertEqual(metrics.exact_active_count_accuracy, 1.0)
        self.assertEqual(metrics.poison_activation_rate, 0.0)

    def test_locomo_adapter_normalizes_session_times(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "cases.jsonl"
            adapt_locomo(FIXTURE, output)
            cases = load_jsonl(output)

        stale = cases[0]
        self.assertEqual(stale.query_time, "2000-01-02")
        self.assertEqual(stale.claims[0].valid_from, "2000-01-01")
        self.assertEqual(stale.claims[0].valid_until, "2000-01-02")
        self.assertIn("raw_time=", stale.claims[0].evidence)


if __name__ == "__main__":
    unittest.main()




