import tempfile
import unittest
from pathlib import Path

from claimledger.extraction.rule_based import extract_locomo_claims


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "locomo_tiny.json"


class ClaimExtractionTests(unittest.TestCase):
    def test_extracts_rule_based_claims(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "claims.jsonl"
            report = extract_locomo_claims(FIXTURE, output)
            lines = output.read_text(encoding="utf-8").splitlines()

        self.assertGreaterEqual(report.claim_count, 2)
        self.assertEqual(report.sample_count, 1)
        self.assertTrue(any("lives_in" in line for line in lines))
        self.assertTrue(any("moved_to" in line for line in lines))


if __name__ == "__main__":
    unittest.main()




