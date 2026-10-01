import unittest

from claimledger.evaluation.metrics import bootstrap_cluster_mean_ci, compute_metrics
from claimledger.evaluation.scenarios import run_demo_scenarios


class EvaluationTests(unittest.TestCase):
    def test_demo_metrics_are_expected(self) -> None:
        metrics = compute_metrics(run_demo_scenarios())

        self.assertEqual(metrics.temporal_validity_accuracy, 1.0)
        self.assertEqual(metrics.conflict_disclosure_rate, 1.0)
        self.assertEqual(metrics.poison_activation_rate, 0.0)
        self.assertEqual(metrics.abstention_correctness, 1.0)

    def test_cluster_bootstrap_resamples_whole_groups(self) -> None:
        interval = bootstrap_cluster_mean_ci(
            {"conversation-a": [True, True], "conversation-b": [False, False]},
            iterations=200,
            seed=7,
        )

        self.assertEqual(interval.mean, 0.5)
        self.assertGreaterEqual(interval.lower, 0.0)
        self.assertLessEqual(interval.upper, 1.0)


if __name__ == "__main__":
    unittest.main()




