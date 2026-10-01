import json
import tempfile
import unittest
from pathlib import Path

from claimledger.evaluation.multillm import (
    QAExample,
    ContextChunk,
    build_prompt,
    expand_evidence_ids,
    run_multillm,
    parse_answer_response,
    read_raw_outputs,
    score_answer,
    select_context,
)
from claimledger.integrations.ollama_client import OllamaResponse


class _FakeClient:
    def __init__(self, model: str, seed: int) -> None:
        self.model = model
        self.seed = seed

    def chat(self, messages, *, max_tokens=256):  # noqa: ANN001
        return OllamaResponse(
            model=self.model,
            text="Paris",
            total_duration_ns=1_000_000,
            prompt_eval_count=8,
            eval_count=1,
        )


class MultiLLMTests(unittest.TestCase):
    def test_evidence_ranges_are_expanded_for_turn_level_recall(self) -> None:
        self.assertEqual(expand_evidence_ids(["D1:2-4"]), ("D1:2", "D1:3", "D1:4"))

    def test_structured_answer_parser_preserves_failures(self) -> None:
        self.assertEqual(parse_answer_response('{"answer":"Paris"}'), ("Paris", True, None))
        parsed, ok, error = parse_answer_response("not-json")
        self.assertEqual((parsed, ok), ("not-json", False))
        self.assertTrue(error)

    def test_prompt_has_no_gold_answer_or_evidence_labels(self) -> None:
        example = QAExample(
            dataset="fixture",
            example_id="q1",
            question="Where is the office?",
            answer="SECRET-GOLD",
            task_type="single",
            chunks=(ContextChunk("turn-1", "s1", "The office is in Paris.", None, "fixture"),),
            evidence_ids=("turn-1",),
            provenance="fixture",
            metadata={},
        )
        prompt = build_prompt(example, example.chunks)
        self.assertNotIn("SECRET-GOLD", prompt)
        self.assertNotIn("turn-1", prompt)

    def test_evaluation_nonce_is_nonsemantic_and_unique(self) -> None:
        example = QAExample(
            dataset="fixture",
            example_id="q1",
            question="Where is the office?",
            answer="Paris",
            task_type="single",
            chunks=(ContextChunk("turn-1", "s1", "The office is in Paris.", None, "fixture"),),
            evidence_ids=("turn-1",),
            provenance="fixture",
            metadata={},
        )
        first = build_prompt(example, example.chunks, evaluation_id="abc123")
        second = build_prompt(example, example.chunks, evaluation_id="def456")
        self.assertNotEqual(first, second)
        self.assertIn("carries no task information", first)

    def test_claimledger_context_is_deterministic_and_scored(self) -> None:
        example = QAExample(
            dataset="fixture",
            example_id="q1",
            question="Where is the office?",
            answer="Paris",
            task_type="single",
            chunks=(
                ContextChunk("a", "s1", "The office is in Paris.", "2024-01-01", "fixture"),
                ContextChunk("b", "s2", "The weather is rainy.", "2024-02-01", "fixture"),
            ),
            evidence_ids=("a",),
            provenance="fixture",
            metadata={},
        )
        first = select_context(example, "claimledger", top_k=1)
        second = select_context(example, "claimledger", top_k=1)
        self.assertEqual(first, second)
        self.assertEqual(first[0].chunk_id, "a")
        self.assertEqual(score_answer("Paris", "Paris")["exact_match"], True)

    def test_full_context_baseline_is_distinct_from_top_k_retrieval(self) -> None:
        example = QAExample(
            dataset="fixture",
            example_id="q1",
            question="Where is the office?",
            answer="Paris",
            task_type="single",
            chunks=(
                ContextChunk("a", "s1", "The office is in Paris.", None, "fixture"),
                ContextChunk("b", "s2", "The weather is rainy.", None, "fixture"),
            ),
            evidence_ids=("a",),
            provenance="fixture",
            metadata={},
        )

        self.assertEqual(len(select_context(example, "full_context", top_k=1)), 2)

    def test_runner_writes_raw_results_and_resumes_without_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "locomo.json"
            source.write_text(json.dumps([{
                "sample_id": "sample-1",
                "conversation": {
                    "session_1": [{"dia_id": "D1:1", "speaker": "A", "text": "The office is in Paris."}],
                    "session_1_date_time": "2024-01-01",
                },
                "qa": [{"question": "Where is the office?", "answer": "Paris", "evidence": ["D1:1"], "category": 1}],
            }]), encoding="utf-8")

            run_multillm(
                dataset="locomo",
                source_path=source,
                output_root=root / "runs",
                run_id="run-1",
                models=["fake:model"],
                strategies=["claimledger"],
                seeds=[0],
                client_factory=lambda model, seed: _FakeClient(model, seed),
            )
            run_multillm(
                dataset="locomo",
                source_path=source,
                output_root=root / "runs",
                run_id="run-1",
                models=["fake:model"],
                strategies=["claimledger"],
                seeds=[0],
                client_factory=lambda model, seed: _FakeClient(model, seed),
            )
            run_dir = root / "runs" / "run-1"
            results = [line for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines() if line]
            raw = [line for line in (run_dir / "raw_outputs.jsonl").read_text(encoding="utf-8").splitlines() if line]
            aggregate = json.loads((run_dir / "aggregate.json").read_text(encoding="utf-8"))

        self.assertEqual(len(results), 1)
        self.assertEqual(len(raw), 1)
        self.assertEqual(aggregate[0]["n"], 1)

    def test_read_raw_outputs_plain_and_gzip(self) -> None:
        import gzip
        from claimledger.datasets._io import read_jsonl

        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            sample_data = [{"step": 1, "text": "hello"}, {"step": 2, "text": "world"}]

            # 1. Plain jsonl
            plain_file = run_dir / "raw_outputs.jsonl"
            with plain_file.open("w", encoding="utf-8") as f:
                for row in sample_data:
                    f.write(json.dumps(row) + "\n")
            loaded = read_raw_outputs(run_dir)
            self.assertEqual(loaded, sample_data)
            self.assertEqual(read_jsonl(plain_file), sample_data)

            # 2. Gzipped jsonl
            plain_file.unlink()
            gz_file = run_dir / "raw_outputs.jsonl.gz"
            with gzip.open(gz_file, "wt", encoding="utf-8") as f:
                for row in sample_data:
                    f.write(json.dumps(row) + "\n")
            loaded_gz = read_raw_outputs(run_dir)
            self.assertEqual(loaded_gz, sample_data)
            # read_jsonl transparently finding .gz when given plain path
            self.assertEqual(read_jsonl(plain_file), sample_data)
            self.assertEqual(read_jsonl(gz_file), sample_data)


if __name__ == "__main__":
    unittest.main()
