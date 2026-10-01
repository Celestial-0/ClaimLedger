import unittest
from pathlib import Path

from claimledger.retrieval.qa import evaluate_locomo_qa


FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "locomo_tiny.json"


class QARetrievalTests(unittest.TestCase):
    def test_tiny_locomo_qa_retrieval(self) -> None:
        report = evaluate_locomo_qa(FIXTURE)

        self.assertEqual(report.question_count, 1)
        self.assertEqual(report.recall_at_1_any, 1.0)
        self.assertEqual(report.recall_at_5_all, 1.0)
        self.assertEqual(report.mean_reciprocal_rank, 1.0)

    def test_tiny_locomo_bm25_retrieval(self) -> None:
        report = evaluate_locomo_qa(FIXTURE, method="bm25")

        self.assertEqual(report.question_count, 1)
        self.assertEqual(report.recall_at_1_any, 1.0)
        self.assertEqual(report.recall_at_5_all, 1.0)


if __name__ == "__main__":
    unittest.main()




