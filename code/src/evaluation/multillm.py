"""Resumable, provenance-preserving multi-LLM benchmark evaluation.

The runner in this module is intentionally independent of a particular
benchmark. It streams official examples, selects context with auditable
strategies, sends requests to local Ollama models, and appends one raw record
and one scored record per completed cell. A run can therefore be interrupted
and resumed without silently changing its denominator.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from ..core.ledger import ClaimLedger
from ..core.models import TrustTier
from ..datasets.longmemeval import iter_longmemeval
from ..integrations.ollama_client import OllamaClient, OllamaResponse


SUPPORTED_DATASETS = {"locomo", "realtalk", "longmemeval_s", "longmemeval_m", "longmemeval_oracle"}
SUPPORTED_STRATEGIES = {"claimledger", "lexical", "recency", "full_context", "oracle"}
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


@dataclass(frozen=True)
class ContextChunk:
    chunk_id: str
    session_id: str
    text: str
    timestamp: str | None
    source: str


@dataclass(frozen=True)
class QAExample:
    dataset: str
    example_id: str
    question: str
    answer: str
    task_type: str | None
    chunks: tuple[ContextChunk, ...]
    evidence_ids: tuple[str, ...]
    provenance: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class EvaluationCell:
    dataset: str
    example_id: str
    model: str
    strategy: str
    seed: int

    @property
    def key(self) -> str:
        return "|".join((self.dataset, self.example_id, self.model, self.strategy, str(self.seed)))


def iter_examples(dataset: str, path: str | Path, *, limit: int | None = None) -> Iterator[QAExample]:
    """Yield official QA examples without synthesizing questions or answers."""

    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"unsupported dataset: {dataset}")
    source = Path(path)
    if dataset.startswith("longmemeval_"):
        yield from _iter_longmemeval_examples(dataset, source, limit=limit)
    elif dataset == "locomo":
        yield from _iter_locomo_examples(source, limit=limit)
    else:
        yield from _iter_realtalk_examples(source, limit=limit)


def _iter_longmemeval_examples(dataset: str, path: Path, *, limit: int | None) -> Iterator[QAExample]:
    for item in iter_longmemeval(path, limit=limit):
        raw = item.raw
        yield QAExample(
            dataset=dataset,
            example_id=item.example_id,
            question=item.question,
            answer=item.answer or "",
            task_type=item.task_type,
            chunks=tuple(_longmemeval_chunks(raw)),
            evidence_ids=item.evidence,
            provenance="official_benchmark_generated_history",
            metadata={
                "question_date": raw.get("question_date"),
                "haystack_session_count": len(raw.get("haystack_sessions", [])),
            },
        )


def _longmemeval_chunks(raw: dict[str, Any]) -> list[ContextChunk]:
    sessions = raw.get("haystack_sessions", [])
    session_ids = raw.get("haystack_session_ids", [])
    dates = raw.get("haystack_dates", [])
    if not isinstance(sessions, list):
        return []
    chunks: list[ContextChunk] = []
    for index, session in enumerate(sessions):
        if isinstance(session, dict):
            session_id = str(session.get("session_id", session.get("id", index)))
            turns = session.get("turns", session.get("messages", session.get("content", [])))
            timestamp = session.get("date", session.get("timestamp"))
        else:
            session_id = str(session_ids[index]) if isinstance(session_ids, list) and index < len(session_ids) else str(index)
            turns = session
            timestamp = dates[index] if isinstance(dates, list) and index < len(dates) else None
        if isinstance(turns, list):
            text = "\n".join(_turn_text(turn) for turn in turns if isinstance(turn, dict)).strip()
        else:
            text = str(turns)
        if text:
            chunks.append(ContextChunk(session_id, session_id, text, _optional_str(timestamp), "LongMemEval"))
    return chunks


def _iter_locomo_examples(path: Path, *, limit: int | None) -> Iterator[QAExample]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    samples = raw if isinstance(raw, list) else raw.get("data", raw.get("samples", [raw]))
    if not isinstance(samples, list):
        raise ValueError("LoCoMo input must be a list or contain data/samples")
    emitted = 0
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        sample_id = str(sample.get("sample_id", sample.get("id", "sample")))
        chunks = tuple(_locomo_chunks(sample))
        qa_items = sample.get("qa", [])
        if not isinstance(qa_items, list):
            continue
        for index, qa in enumerate(qa_items):
            if not isinstance(qa, dict) or not str(qa.get("question", "")).strip():
                continue
            evidence = expand_evidence_ids(qa.get("evidence", []))
            yield QAExample(
                dataset="locomo",
                example_id=f"{sample_id}:q{index + 1}",
                question=str(qa.get("question", "")),
                answer=str(qa.get("answer", "")),
                task_type=str(qa.get("category")) if qa.get("category") is not None else None,
                chunks=chunks,
                evidence_ids=evidence,
                provenance="official_benchmark_generated_history",
                metadata={"sample_id": sample_id, "category": qa.get("category")},
            )
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def _locomo_chunks(sample: dict[str, Any]) -> list[ContextChunk]:
    conversation = sample.get("conversation", {})
    if not isinstance(conversation, dict):
        return []
    names = sorted(
        (key for key in conversation if re.fullmatch(r"session_\d+", key)),
        key=lambda key: int(key.split("_", 1)[1]),
    )
    chunks: list[ContextChunk] = []
    for session in names:
        turns = conversation.get(session, [])
        date = conversation.get(f"{session}_date_time")
        if not isinstance(turns, list):
            continue
        for index, turn in enumerate(turns):
            if not isinstance(turn, dict):
                continue
            chunk_id = str(turn.get("dia_id", f"{session}:{index + 1}"))
            text = str(turn.get("text", turn.get("clean_text", ""))).strip()
            if text:
                chunks.append(ContextChunk(chunk_id, session, f"{turn.get('speaker', 'speaker')}: {text}", _optional_str(date), "LoCoMo"))
    return chunks


def _iter_realtalk_examples(path: Path, *, limit: int | None) -> Iterator[QAExample]:
    files = sorted(path.glob("Chat_*.json")) if path.is_dir() else [path]
    emitted = 0
    for source in files:
        conversation = json.loads(source.read_text(encoding="utf-8"))
        chunks = tuple(_realtalk_chunks(conversation))
        for index, qa in enumerate(conversation.get("qa", [])):
            if not isinstance(qa, dict) or not str(qa.get("question", "")).strip():
                continue
            yield QAExample(
                dataset="realtalk",
                example_id=f"{source.stem}:q{index + 1}",
                question=str(qa.get("question", "")),
                answer=str(qa.get("answer", "")),
                task_type=str(qa.get("category")) if qa.get("category") is not None else None,
                chunks=chunks,
                evidence_ids=expand_evidence_ids(qa.get("evidence", [])),
                provenance="authentic_human_human_dialogue_human_annotated_question",
                metadata={"source_file": source.name, "conversation": source.stem},
            )
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def _realtalk_chunks(conversation: dict[str, Any]) -> list[ContextChunk]:
    chunks: list[ContextChunk] = []
    names = sorted(
        (key for key in conversation if re.fullmatch(r"session_\d+", key)),
        key=lambda key: int(key.split("_", 1)[1]),
    )
    for session in names:
        turns = conversation.get(session, [])
        if not isinstance(turns, list):
            continue
        for index, turn in enumerate(turns):
            if not isinstance(turn, dict):
                continue
            chunk_id = str(turn.get("dia_id", f"{session}:{index + 1}"))
            text = str(turn.get("clean_text", turn.get("text", ""))).strip()
            if text:
                chunks.append(ContextChunk(chunk_id, session, f"{turn.get('speaker', 'speaker')}: {text}", _optional_str(turn.get("date_time")), "REALTALK"))
    return chunks


def select_context(example: QAExample, strategy: str, *, top_k: int = 8, max_chars: int = 24_000) -> tuple[ContextChunk, ...]:
    """Select a context packet; gold evidence is used only by Oracle."""

    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(f"unsupported strategy: {strategy}")
    if top_k <= 0 or max_chars <= 0:
        return ()
    if strategy == "oracle":
        gold = set(example.evidence_ids)
        ranked = [chunk for chunk in example.chunks if chunk.chunk_id in gold or chunk.session_id in gold]
    elif strategy == "full_context":
        ranked = list(example.chunks)
    elif strategy == "recency":
        ranked = sorted(example.chunks, key=lambda chunk: (chunk.timestamp or "", chunk.chunk_id), reverse=True)
    else:
        scored = [(lexical_score(example.question, chunk.text), chunk) for chunk in example.chunks]
        ranked = [chunk for _, chunk in sorted(scored, key=lambda pair: (-pair[0], pair[1].chunk_id))]
    selected = _fit_context(ranked[:top_k] if strategy not in {"full_context", "oracle"} else ranked, max_chars)
    if strategy == "claimledger":
        selected = _govern_context(selected)
    return tuple(selected)


def _govern_context(chunks: list[ContextChunk]) -> list[ContextChunk]:
    with ClaimLedger() as ledger:
        identifiers: dict[str, str] = {}
        for index, chunk in enumerate(chunks):
            claim_id = ledger.add_claim(
                subject=f"context:{chunk.chunk_id}",
                predicate="contains",
                object=chunk.text,
                evidence=chunk.text,
                source=chunk.chunk_id,
                valid_from=chunk.timestamp,
                trust_tier=TrustTier.CONFIRMED,
                confidence=1.0,
            )
            identifiers[claim_id] = chunk.chunk_id
        eligible = ledger.eligible_claims(at_time=None)
        allowed = {record.source for record in eligible.eligible}
    return [chunk for chunk in chunks if chunk.chunk_id in allowed]


def _fit_context(chunks: Iterable[ContextChunk], max_chars: int) -> list[ContextChunk]:
    selected: list[ContextChunk] = []
    used = 0
    for chunk in chunks:
        cost = len(chunk.text) + 64
        if selected and used + cost > max_chars:
            break
        selected.append(chunk)
        used += cost
    return selected


def lexical_score(question: str, text: str) -> float:
    query = _tokens(question)
    document = _tokens(text)
    if not query or not document:
        return 0.0
    overlap = set(query) & set(document)
    numerator = sum(query[token] * document[token] for token in overlap)
    denominator = math.sqrt(sum(value * value for value in document.values()))
    return numerator / denominator if denominator else 0.0


def build_prompt(example: QAExample, context: Iterable[ContextChunk], *, evaluation_id: str | None = None) -> str:
    """Build a prompt without inserting answers, evidence labels, or gold IDs.

    ``evaluation_id`` is a deterministic, non-semantic nonce used to prevent
    Ollama prompt-cache warmth from making one context strategy look faster
    merely because it follows an identical prompt.
    """

    lines = [
        "Answer the question using only the memory context below.",
        "If the context is insufficient, say: I don't have enough information to answer this.",
        "Return only a JSON object with one string field named answer. Put the concise final answer in that field; do not include analysis or chain-of-thought.",
        "",
        "MEMORY CONTEXT:",
    ]
    if evaluation_id:
        lines.insert(3, f"Evaluation nonce (ignore; it carries no task information): {evaluation_id}")
    selected = list(context)
    if not selected:
        lines.append("[No retrieved memory]")
    else:
        for index, chunk in enumerate(selected, start=1):
            metadata = []
            if chunk.session_id:
                metadata.append(f"session={chunk.session_id}")
            if chunk.timestamp:
                metadata.append(f"timestamp={chunk.timestamp}")
            prefix = f"[{index}]"
            if metadata:
                prefix += " (" + ", ".join(metadata) + ")"
            lines.append(f"{prefix} {chunk.text}")
    lines.extend(("", f"QUESTION: {example.question}", "ANSWER:"))
    return "\n".join(lines)


def score_answer(prediction: str, answer: str) -> dict[str, Any]:
    normalized_prediction = normalize_answer(prediction)
    normalized_answer = normalize_answer(answer)
    return {
        "exact_match": normalized_prediction == normalized_answer,
        "token_f1": token_f1(prediction, answer),
        "contains_answer": bool(normalized_answer) and normalized_answer in normalized_prediction,
        "abstained": is_abstention(prediction),
    }


def parse_answer_response(text: str) -> tuple[str, bool, str | None]:
    """Parse the structured Ollama answer while preserving parse failures."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return text.strip(), False, f"JSONDecodeError: {exc.msg}"
    if not isinstance(payload, dict) or not isinstance(payload.get("answer"), str):
        return text.strip(), False, "response does not match the answer schema"
    return payload["answer"].strip(), True, None


def normalize_answer(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def token_f1(prediction: str, answer: str) -> float:
    predicted = normalize_answer(prediction).split()
    gold = normalize_answer(answer).split()
    if not gold:
        return 1.0 if not predicted else 0.0
    if not predicted:
        return 0.0
    overlap = sum((Counter(predicted) & Counter(gold)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def is_abstention(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in (
        "i don't have enough information",
        "i do not have enough information",
        "i don't know",
        "cannot determine",
        "insufficient information",
        "no relevant memory",
    ))


def retrieval_scores(example: QAExample, context: Iterable[ContextChunk]) -> dict[str, Any]:
    selected = list(context)
    gold = set(example.evidence_ids)
    retrieved = {chunk.chunk_id for chunk in selected} | {chunk.session_id for chunk in selected}
    return {
        "retrieved_evidence_any": bool(gold & retrieved),
        "retrieved_evidence_all": bool(gold) and gold.issubset(retrieved),
        "retrieved_count": len(selected),
        "gold_evidence_count": len(gold),
    }


def run_multillm(
    *,
    dataset: str,
    source_path: str | Path,
    output_root: str | Path,
    run_id: str,
    models: list[str],
    strategies: list[str],
    seeds: list[int],
    limit: int | None = None,
    top_k: int = 8,
    max_chars: int = 24_000,
    max_output_tokens: int = 256,
    num_ctx: int = 8_192,
    temperature: float = 0.0,
    timeout_s: int = 180,
    base_url: str = "http://127.0.0.1:11434",
    client_factory: Callable[[str, int], OllamaClient] | None = None,
) -> Path:
    """Run/resume a multi-model evaluation and return its run directory."""

    if not models or not strategies or not seeds:
        raise ValueError("models, strategies, and seeds must all be non-empty")
    if any(strategy not in SUPPORTED_STRATEGIES for strategy in strategies):
        raise ValueError(f"strategies must be drawn from {sorted(SUPPORTED_STRATEGIES)}")
    run_dir = Path(output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    results_path = run_dir / "results.jsonl"
    raw_path = run_dir / "raw_outputs.jsonl"
    failures_path = run_dir / "failures.jsonl"
    completed = _completed_keys(results_path)

    manifest = _load_or_create_manifest(
        manifest_path,
        dataset=dataset,
        source_path=source_path,
        run_id=run_id,
        models=models,
        strategies=strategies,
        seeds=seeds,
        config={"top_k": top_k, "max_chars": max_chars, "max_output_tokens": max_output_tokens, "num_ctx": num_ctx, "temperature": temperature, "think": False, "response_format": ANSWER_SCHEMA, "timeout_s": timeout_s, "limit": limit, "prompt_cache_isolation": "deterministic per-cell nonce"},
    )
    clients: dict[tuple[str, int], OllamaClient] = {}
    for model in models:
        for seed in seeds:
            clients[(model, seed)] = client_factory(model, seed) if client_factory else OllamaClient(
                model=model,
                base_url=base_url,
                temperature=temperature,
                system_prompt="You are a careful benchmark QA system. Return only the final answer, with no chain-of-thought, analysis, preamble, or explanation.",
                timeout_s=timeout_s,
                think=False,
                response_format=ANSWER_SCHEMA,
                _options={"seed": seed, "num_ctx": num_ctx},
            )
    if client_factory is None:
        _validate_ollama_models(clients, models)
        manifest["ollama_models"] = {model: clients[(model, seeds[0])].show_model() for model in models}
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    processed = 0
    with raw_path.open("a", encoding="utf-8") as raw_handle, results_path.open("a", encoding="utf-8") as result_handle, failures_path.open("a", encoding="utf-8") as failure_handle:
        for example in iter_examples(dataset, source_path, limit=limit):
            context_by_strategy = {strategy: select_context(example, strategy, top_k=top_k, max_chars=max_chars) for strategy in strategies}
            for model in models:
                for strategy in strategies:
                    context = context_by_strategy[strategy]
                    retrieval = retrieval_scores(example, context)
                    for seed in seeds:
                        cell = EvaluationCell(dataset, example.example_id, model, strategy, seed)
                        if cell.key in completed:
                            continue
                        nonce = hashlib.sha256(cell.key.encode("utf-8")).hexdigest()[:16]
                        prompt = build_prompt(example, context, evaluation_id=nonce)
                        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                        started = time.perf_counter()
                        try:
                            response = clients[(model, seed)].chat([{"role": "user", "content": prompt}], max_tokens=max_output_tokens)
                            generated = response.text
                            parsed_answer, parse_ok, parse_error = parse_answer_response(generated)
                            elapsed_ms = (time.perf_counter() - started) * 1000
                            raw_record = {
                                "schema_version": "1.0",
                                "status": "success",
                                "evaluation_key": cell.key,
                                "dataset": dataset,
                                "example_id": example.example_id,
                                "model": model,
                                "strategy": strategy,
                                "seed": seed,
                                "prompt_hash": prompt_hash,
                                "prompt": prompt,
                                "response": generated,
                                "parsed_answer": parsed_answer,
                                "parse_ok": parse_ok,
                                "parse_error": parse_error,
                                "thinking": response.thinking,
                                "ollama_model": response.model,
                                "total_duration_ns": response.total_duration_ns,
                                "prompt_eval_count": response.prompt_eval_count,
                                "eval_count": response.eval_count,
                                "wall_latency_ms": elapsed_ms,
                            }
                            _write_record(raw_handle, raw_record)
                            scores = score_answer(parsed_answer, example.answer)
                            result_record = {
                                **asdict(cell),
                                "evaluation_key": cell.key,
                                "question": example.question,
                                "answer": example.answer,
                                "generated_answer_raw": generated,
                                "generated_answer": parsed_answer,
                                "output_parse_ok": parse_ok,
                                "output_parse_error": parse_error,
                                "task_type": example.task_type,
                                "provenance": example.provenance,
                                **scores,
                                **retrieval,
                                "latency_ms": elapsed_ms,
                                "prompt_tokens": response.prompt_eval_count,
                                "completion_tokens": response.eval_count,
                                "prompt_hash": prompt_hash,
                            }
                            _write_record(result_handle, result_record)
                            result_handle.flush()
                            completed.add(cell.key)
                        except Exception as exc:  # noqa: BLE001 - errors are persisted for audit and retry
                            elapsed_ms = (time.perf_counter() - started) * 1000
                            _write_record(raw_handle, {"schema_version": "1.0", "status": "error", "evaluation_key": cell.key, "dataset": dataset, "example_id": example.example_id, "model": model, "strategy": strategy, "seed": seed, "prompt_hash": prompt_hash, "prompt": prompt, "error_type": type(exc).__name__, "error": str(exc), "wall_latency_ms": elapsed_ms})
                            _write_record(failure_handle, {"evaluation_key": cell.key, "error_type": type(exc).__name__, "error": str(exc), "prompt_hash": prompt_hash})
                        raw_handle.flush()
                        failure_handle.flush()
                        processed += 1
    _write_aggregate(results_path, run_dir / "aggregate.json")
    manifest["completed_cells"] = len(completed)
    manifest["last_updated_utc"] = _utc_now()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return run_dir


def read_raw_outputs(run_dir: str | Path) -> list[dict[str, Any]]:
    """Load raw Ollama audit traces from a run directory (.jsonl or .jsonl.gz)."""
    target = Path(run_dir)
    gz_path = target / "raw_outputs.jsonl.gz"
    plain_path = target / "raw_outputs.jsonl"
    if gz_path.exists():
        with gzip.open(gz_path, "rt", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if plain_path.exists():
        with plain_path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    raise FileNotFoundError(f"Neither raw_outputs.jsonl nor raw_outputs.jsonl.gz found in {target}")


def _validate_ollama_models(clients: dict[tuple[str, int], OllamaClient], models: list[str]) -> None:
    probe = next(iter(clients.values()))
    if not probe.is_available():
        raise RuntimeError("Ollama is not reachable at the configured base URL")
    available = set(probe.list_models())
    missing = [model for model in models if model not in available]
    if missing:
        raise RuntimeError(f"Ollama model tags are missing: {missing}; available={sorted(available)}")


def _completed_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if record.get("evaluation_key"):
                    completed.add(str(record["evaluation_key"]))
    return completed


def _load_or_create_manifest(path: Path, **values: Any) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    source = Path(values["source_path"])
    if source.is_file():
        source_hash = _sha256_file(source)
    else:
        source_hash = {str(item): _sha256_file(item) for item in sorted(source.glob("Chat_*.json"))}
    manifest = {"schema_version": "1.0", "created_at_utc": _utc_now(), "source_sha256": source_hash, "no_synthetic_qa": True, **values}
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return manifest


def _write_record(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def _write_aggregate(results_path: Path, output_path: Path) -> None:
    groups: dict[tuple[str, str, str | None], list[dict[str, Any]]] = defaultdict(list)
    if results_path.exists():
        with results_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    groups[(str(record["model"]), str(record["strategy"]), record.get("task_type"))].append(record)
    rows = []
    for (model, strategy, task_type), records in sorted(groups.items()):
        n = len(records)
        rows.append({
            "model": model,
            "strategy": strategy,
            "task_type": task_type,
            "n": n,
            "exact_match": sum(bool(r["exact_match"]) for r in records) / n,
            "token_f1": sum(float(r["token_f1"]) for r in records) / n,
            "contains_answer": sum(bool(r["contains_answer"]) for r in records) / n,
            "abstained": sum(bool(r["abstained"]) for r in records) / n,
            "output_parse_rate": sum(bool(r.get("output_parse_ok", False)) for r in records) / n,
            "retrieved_evidence_any": sum(bool(r["retrieved_evidence_any"]) for r in records) / n,
            "retrieved_evidence_all": sum(bool(r["retrieved_evidence_all"]) for r in records) / n,
            "latency_ms_mean": sum(float(r["latency_ms"]) for r in records) / n,
        })
    output_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _turn_text(turn: dict[str, Any]) -> str:
    role = turn.get("role", turn.get("speaker", "speaker"))
    content = turn.get("content", turn.get("text", turn.get("clean_text", "")))
    return f"{role}: {content}".strip()


def _tokens(text: str) -> Counter[str]:
    return Counter(token for token in re.findall(r"[a-z0-9']+", text.lower()) if len(token) > 1)


def _as_string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    return ()


def expand_evidence_ids(value: Any) -> tuple[str, ...]:
    """Expand common REALTALK/LoCoMo turn ranges for retrieval scoring."""

    expanded: list[str] = []
    for raw in _as_string_tuple(value):
        for item in (part.strip() for part in raw.split(";") if part.strip()):
            match = re.fullmatch(r"(D\d+):(\d+)-(D\d+:)?(\d+)", item)
            if match and (match.group(3) is None or match.group(3).rstrip(":") == match.group(1)):
                expanded.extend(f"{match.group(1)}:{index}" for index in range(int(match.group(2)), int(match.group(4)) + 1))
            else:
                expanded.append(item.rstrip(".,;:"))
    return tuple(expanded)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
