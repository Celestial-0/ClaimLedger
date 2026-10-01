"""Vector-space retrieval baseline using TF-IDF cosine similarity.

Treats all claims as retrievable documents without lifecycle governance,
quarantine, conflict disclosure, or temporal filtering.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING

from ..evaluation.scenarios import ScenarioOutcome

if TYPE_CHECKING:
    from ..evaluation.benchmark import BenchmarkCase


def _compute_tf(text: str) -> dict[str, float]:
    words = text.lower().split()
    if not words:
        return {}
    tf: dict[str, float] = defaultdict(float)
    for word in words:
        tf[word] += 1.0
    for word in tf:
        tf[word] = tf[word] / len(words)
    return dict(tf)


def _cosine_sim(tf1: dict[str, float], tf2: dict[str, float], idfs: dict[str, float]) -> float:
    score = 0.0
    for word, val in tf1.items():
        if word in tf2:
            score += val * tf2[word] * (idfs.get(word, 0.0) ** 2)
    norm1 = sum((v * idfs.get(k, 0.0)) ** 2 for k, v in tf1.items())
    norm2 = sum((v * idfs.get(k, 0.0)) ** 2 for k, v in tf2.items())
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return score / (math.sqrt(norm1) * math.sqrt(norm2))


def run_vector_rag_cases(cases: list[BenchmarkCase]) -> list[ScenarioOutcome]:
    """Vector RAG baseline: retrieve by similarity, trust everything."""
    outcomes: list[ScenarioOutcome] = []

    for case in cases:
        docs = case.claims
        N = len(docs)

        doc_tfs: list[dict[str, float]] = []
        df: dict[str, float] = defaultdict(float)

        for claim in docs:
            text = claim.evidence if claim.evidence else claim.object
            tf = _compute_tf(text)
            doc_tfs.append(tf)
            for word in tf:
                df[word] += 1.0

        idfs: dict[str, float] = {}
        for word, count in df.items():
            idfs[word] = math.log((N + 1) / (1.0 + count)) + 1.0

        # Build query from task type and first claim context
        query_text = case.task_type
        if docs:
            query_text += " " + docs[0].subject + " " + docs[0].predicate

        query_tf = _compute_tf(query_text)

        scored = []
        for i, tf in enumerate(doc_tfs):
            sim = _cosine_sim(query_tf, tf, idfs)
            scored.append((sim, docs[i]))
        scored.sort(key=lambda x: x[0], reverse=True)

        k = min(5, N)
        retrieved = [doc for _, doc in scored[:k]]

        outcomes.append(ScenarioOutcome(
            name=case.task_type,
            expected_active=case.expected_active,
            actual_active=len(retrieved),
            conflict_disclosed=False,
            poisoned_active=any(c.is_poisoned for c in retrieved),
            abstained=False,
        ))

    return outcomes
