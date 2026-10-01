import tempfile
import unittest
from pathlib import Path

from claimledger.evaluation.benchmark import load_jsonl, run_claimledger_cases, run_naive_cases
from claimledger.extraction.rule_based import extract_locomo_claims
from claimledger.evaluation.metrics import compute_metrics
from claimledger.attacks.perturbations import generate_qa_perturbations


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "locomo_tiny.json"


class PerturbationTests(unittest.TestCase):
    def test_generate_qa_perturbations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            claims = Path(tmp) / "claims.jsonl"
            cases_path = Path(tmp) / "cases.jsonl"
            extract_locomo_claims(FIXTURE, claims)
            report = generate_qa_perturbations(FIXTURE, claims, cases_path)
            cases = load_jsonl(cases_path)
            claimledger = compute_metrics(run_claimledger_cases(cases))
            naive = compute_metrics(run_naive_cases(cases))

        self.assertGreaterEqual(report.matched_claim_count, 1)
        self.assertEqual(report.generated_case_count, report.matched_claim_count * 3)
        self.assertGreater(claimledger.exact_active_count_accuracy, naive.exact_active_count_accuracy)


if __name__ == "__main__":
    unittest.main()




