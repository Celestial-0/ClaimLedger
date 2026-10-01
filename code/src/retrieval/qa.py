"""LoCoMo QA evidence-retrieval harness.

This module evaluates whether a simple retriever can recover annotated evidence
turns for LoCoMo questions. It is deliberately dependency-free so it can serve as
the first reproducible retrieval baseline before adding embeddings or LLMs.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


@dataclass(frozen=True)
class Turn:
    sample_id: str
    session: str
    dia_id: str
    speaker: str
    text: str


@dataclass(frozen=True)
class QACase:
    sample_id: str
    question: str
    answer: str
    evidence_ids: tuple[str, ...]
    category: int | None


@dataclass(frozen=True)
class RetrievalResult:
    sample_id: str
    category: int | None
    evidence_count: int
    retrieved_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class QAMetricReport:
    question_count: int
    recall_at_1_any: float
    recall_at_3_any: float
    recall_at_5_any: float
    recall_at_5_all: float
    mean_reciprocal_rank: float
    by_category: dict[str, dict[str, float]]


def evaluate_locomo_qa(
    path: str | Path,
    *,
    k_values: tuple[int, ...] = (1, 3, 5),
    method: str = "lexical",
) -> QAMetricReport:
    samples = _load_samples(path)
    all_results: list[RetrievalResult] = []
    for sample in samples:
        turns = _extract_turns(sample)
        cases = _extract_qa_cases(sample)
        index = _build_bm25_index(turns) if method == "bm25" else None
        for case in cases:
            all_results.append(_retrieve_case(case, turns, max(k_values), method=method, bm25_index=index))
    return _metrics(all_results, k_values)


def _load_samples(path: str | Path) -> list[dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("data", raw.get("samples", [raw]))
    if not isinstance(raw, list):
        raise ValueError("LoCoMo QA input must be a JSON list or dict containing data/samples")
    return raw


def _extract_turns(sample: dict[str, Any]) -> list[Turn]:
    sample_id = str(sample.get("sample_id", sample.get("id", "unknown_sample")))
    conversation = sample.get("conversation", {})
    if not isinstance(conversation, dict):
        return []
    session_names = sorted(
        [key for key in conversation if key.startswith("session_") and not key.endswith("_date_time")],
        key=_session_index,
    )
    turns: list[Turn] = []
    for session in session_names:
        raw_turns = conversation.get(session, [])
        if not isinstance(raw_turns, list):
            continue
        for turn in raw_turns:
            if not isinstance(turn, dict):
                continue
            turns.append(
                Turn(
                    sample_id=sample_id,
                    session=session,
                    dia_id=str(turn.get("dia_id", "")),
                    speaker=str(turn.get("speaker", "")),
                    text=str(turn.get("text", "")),
                )
            )
    return turns


def _extract_qa_cases(sample: dict[str, Any]) -> list[QACase]:
    sample_id = str(sample.get("sample_id", sample.get("id", "unknown_sample")))
    qa_items = sample.get("qa", [])
    if not isinstance(qa_items, list):
        return []
    cases: list[QACase] = []
    for item in qa_items:
        if not isinstance(item, dict):
            continue
        evidence = item.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = [evidence]
        cases.append(
            QACase(
                sample_id=sample_id,
                question=str(item.get("question", "")),
                answer=str(item.get("answer", "")),
                evidence_ids=tuple(str(eid) for eid in evidence),
                category=_as_int(item.get("category")),
            )
        )
    return cases


def _retrieve_case(
    case: QACase,
    turns: list[Turn],
    k: int,
    *,
    method: str,
    bm25_index: dict[str, Any] | None,
) -> RetrievalResult:
    query_tokens = _tokens(case.question)
    if method == "bm25":
        if bm25_index is None:
            raise ValueError("bm25_index is required for BM25 retrieval")
        score_fn = lambda turn: _bm25_score(query_tokens, turn.dia_id, bm25_index)
    elif method == "lexical":
        score_fn = lambda turn: _score(query_tokens, _tokens(f"{turn.speaker} {turn.text}"))
    else:
        raise ValueError(f"Unsupported retrieval method: {method}")
    scored = sorted(
        ((turn, score_fn(turn)) for turn in turns),
        key=lambda item: (-item[1], item[0].dia_id),
    )
    retrieved = tuple(turn.dia_id for turn, _ in scored[:k])
    return RetrievalResult(
        sample_id=case.sample_id,
        category=case.category,
        evidence_count=len(case.evidence_ids),
        retrieved_ids=retrieved,
        evidence_ids=case.evidence_ids,
    )


def _score(query: Counter[str], document: Counter[str]) -> float:
    if not query or not document:
        return 0.0
    overlap = set(query) & set(document)
    return sum(query[token] * document[token] for token in overlap) / math.sqrt(sum(v * v for v in document.values()))


def _build_bm25_index(turns: list[Turn]) -> dict[str, Any]:
    docs: dict[str, Counter[str]] = {}
    doc_freq: Counter[str] = Counter()
    lengths: dict[str, int] = {}
    for turn in turns:
        tokens = _tokens(f"{turn.speaker} {turn.text}")
        docs[turn.dia_id] = tokens
        lengths[turn.dia_id] = sum(tokens.values())
        doc_freq.update(tokens.keys())
    avg_len = sum(lengths.values()) / len(lengths) if lengths else 0.0
    return {"docs": docs, "doc_freq": doc_freq, "lengths": lengths, "avg_len": avg_len, "n": len(turns)}


def _bm25_score(query: Counter[str], dia_id: str, index: dict[str, Any]) -> float:
    docs: dict[str, Counter[str]] = index["docs"]
    doc_freq: Counter[str] = index["doc_freq"]
    lengths: dict[str, int] = index["lengths"]
    avg_len: float = index["avg_len"]
    n_docs: int = index["n"]
    doc = docs.get(dia_id, Counter())
    if not query or not doc or avg_len == 0 or n_docs == 0:
        return 0.0
    k1 = 1.5
    b = 0.75
    score = 0.0
    for token, qf in query.items():
        tf = doc.get(token, 0)
        if tf == 0:
            continue
        df = doc_freq.get(token, 0)
        idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        denom = tf + k1 * (1 - b + b * lengths[dia_id] / avg_len)
        score += qf * idf * (tf * (k1 + 1) / denom)
    return score


def _metrics(results: list[RetrievalResult], k_values: tuple[int, ...]) -> QAMetricReport:
    if not results:
        return QAMetricReport(0, 0.0, 0.0, 0.0, 0.0, 0.0, {})
    by_category: dict[str, list[RetrievalResult]] = defaultdict(list)
    for result in results:
        by_category[str(result.category)].append(result)
    category_metrics = {
        category: {
            "count": float(len(items)),
            "recall_at_5_any": _recall_any(items, 5),
            "recall_at_5_all": _recall_all(items, 5),
        }
        for category, items in sorted(by_category.items())
    }
    return QAMetricReport(
        question_count=len(results),
        recall_at_1_any=_recall_any(results, 1),
        recall_at_3_any=_recall_any(results, 3),
        recall_at_5_any=_recall_any(results, 5),
        recall_at_5_all=_recall_all(results, 5),
        mean_reciprocal_rank=_mrr(results),
        by_category=category_metrics,
    )


def _recall_any(results: list[RetrievalResult], k: int) -> float:
    hits = 0
    for result in results:
        gold = set(result.evidence_ids)
        retrieved = set(result.retrieved_ids[:k])
        hits += bool(gold & retrieved)
    return hits / len(results) if results else 0.0


def _recall_all(results: list[RetrievalResult], k: int) -> float:
    hits = 0
    for result in results:
        gold = set(result.evidence_ids)
        retrieved = set(result.retrieved_ids[:k])
        hits += bool(gold) and gold.issubset(retrieved)
    return hits / len(results) if results else 0.0


def _mrr(results: list[RetrievalResult]) -> float:
    total = 0.0
    for result in results:
        gold = set(result.evidence_ids)
        rank = next((idx for idx, dia_id in enumerate(result.retrieved_ids, start=1) if dia_id in gold), None)
        if rank is not None:
            total += 1 / rank
    return total / len(results) if results else 0.0


def _tokens(text: str) -> Counter[str]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return Counter(word for word in words if word not in STOPWORDS and len(word) > 1)


def _session_index(name: str) -> int:
    try:
        return int(name.split("_", 1)[1])
    except (IndexError, ValueError):
        return 0


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None



