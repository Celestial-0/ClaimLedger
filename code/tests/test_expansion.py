import json
import tempfile
import unittest
from pathlib import Path

from claimledger.attacks.adaptive import generate_policy_aware_attacks
from claimledger.datasets.longmemeval import iter_longmemeval, load_longmemeval


class ExpansionTests(unittest.TestCase):
    def test_longmemeval_adapter_normalizes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "longmemeval.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "question_id": "q1",
                        "question": "What changed?",
                        "answer": "The address",
                        "question_type": "knowledge_update",
                        "evidence": "turn-7",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            examples = load_longmemeval(path)

        self.assertEqual(examples[0].example_id, "q1")
        self.assertEqual(examples[0].evidence, ("turn-7",))
        self.assertEqual(examples[0].task_type, "knowledge_update")

    def test_longmemeval_adapter_streams_top_level_json_array(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "longmemeval.json"
            path.write_text(
                json.dumps(
                    [
                        {"question_id": "q1", "question": "one", "answer": "1"},
                        {"question_id": "q2", "question": "two", "answer": "2"},
                    ]
                ),
                encoding="utf-8",
            )
            examples = list(iter_longmemeval(path, limit=1))

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].example_id, "q1")

    def test_longmemeval_adapter_rejects_truncated_array(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "truncated.json"
            path.write_text('[{"question_id":"q1"}', encoding="utf-8")

            with self.assertRaises(ValueError):
                load_longmemeval(path)

    def test_policy_aware_attack_generator_creates_both_stress_types(self) -> None:
        source_rows = [
            {
                "case_id": "c1",
                "task_type": "direct_conflict",
                "expected_active": 0,
                "claims": [
                    {
                        "subject": "u",
                        "predicate": "p",
                        "object": "a",
                        "evidence": "e",
                        "source": "s",
                    },
                    {
                        "subject": "u",
                        "predicate": "p",
                        "object": "b",
                        "evidence": "e",
                        "source": "s",
                    },
                ],
            },
            {
                "case_id": "c2",
                "task_type": "poisoned_write",
                "expected_active": 1,
                "claims": [
                    {
                        "subject": "u",
                        "predicate": "p",
                        "object": "clean",
                        "evidence": "e",
                        "source": "s",
                    },
                    {
                        "subject": "u",
                        "predicate": "p",
                        "object": "poison",
                        "evidence": "e",
                        "source": "s",
                        "is_poisoned": True,
                    },
                ],
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.jsonl"
            output = Path(tmp) / "adaptive.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in source_rows), encoding="utf-8")
            report = generate_policy_aware_attacks(source, output, flood_size=2)
            generated = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(report.generated_case_count, 2)
        self.assertEqual({row["task_type"] for row in generated}, {"conflict_flood", "trusted_source_poison"})
        flood = next(row for row in generated if row["task_type"] == "conflict_flood")
        self.assertEqual(len(flood["claims"]), 4)
        poison = next(row for row in generated if row["task_type"] == "trusted_source_poison")
        poisoned = next(claim for claim in poison["claims"] if claim.get("is_poisoned"))
        self.assertEqual(poisoned["trust_tier"], "high")


if __name__ == "__main__":
    unittest.main()
