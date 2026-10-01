import tempfile
import unittest
from pathlib import Path

from claimledger.evaluation.config import load_config
from claimledger.evaluation.benchmark_performance import run_performance_benchmark
from claimledger.evaluation.run_manifest import create_run_manifest
from claimledger.evaluation.schemas import MetricRecord
from claimledger.evaluation.statistics import holm_bonferroni


class RunContractTests(unittest.TestCase):
    def test_config_and_metric_contracts_are_machine_readable(self) -> None:
        config = load_config(Path(__file__).resolve().parents[1] / "pyproject.toml")
        metric = MetricRecord(
            run_id="run-1",
            system="claimledger",
            dataset="fixture",
            condition="clean",
            metric="accuracy",
            value=1.0,
            numerator=1,
            denominator=1,
        )

        self.assertEqual(config.name, "claimledger-realtalk")
        self.assertIn("claimledger", config.systems)
        self.assertEqual(metric.to_dict()["denominator"], 1)

    def test_manifest_hashes_inputs_and_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            dataset = directory / "dataset.jsonl"
            output = directory / "manifest.json"
            dataset.write_text('{"case": 1}\n', encoding="utf-8")

            manifest = create_run_manifest(
                run_id="run-1",
                dataset_paths=[dataset],
                seed=7,
            )
            manifest.write_json(output)

            self.assertEqual(manifest.seed, 7)
            self.assertIn(str(dataset), manifest.dataset_hashes)
            self.assertTrue(output.exists())

    def test_holm_adjustment_is_monotonic_and_bounded(self) -> None:
        adjusted = holm_bonferroni({"a": 0.001, "b": 0.02, "c": 0.5})

        self.assertLessEqual(adjusted["a"], adjusted["b"])
        self.assertLessEqual(adjusted["b"], adjusted["c"])
        self.assertTrue(all(0.0 <= value <= 1.0 for value in adjusted.values()))

    def test_performance_contract_reports_repeated_percentiles(self) -> None:
        reports = run_performance_benchmark([10], repetitions=2, seed=7)

        self.assertEqual(len(reports), 1)
        report = reports[0]
        self.assertEqual(report.repetitions, 2)
        self.assertGreaterEqual(report.write_latency_ms_p50, 0.0)
        self.assertGreaterEqual(report.write_latency_ms_p95, report.write_latency_ms_p50)
        self.assertGreaterEqual(report.query_latency_ms_p99, report.query_latency_ms_p95)


if __name__ == "__main__":
    unittest.main()
