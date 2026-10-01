"""Adapter for locally obtained LongMemEval JSON or JSONL exports.

The adapter intentionally does not download or redistribute benchmark data.
It normalizes common LongMemEval fields while retaining the raw example for
task-specific evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Iterator
from typing import Any

from .schemas import DatasetExample


def load_longmemeval(path: str | Path, *, limit: int | None = None) -> list[DatasetExample]:
    return list(iter_longmemeval(path, limit=limit))


def iter_longmemeval(path: str | Path, *, limit: int | None = None) -> Iterator[DatasetExample]:
    """Stream LongMemEval JSON/JSONL examples without reading the full file.

    LongMemEval-M is multi-gigabyte at the released history scale. A normal
    ``read_text``/``json.loads`` path creates an avoidable memory spike and
    prevents resumable evaluation. This iterator accepts either a top-level
    JSON array or JSONL and yields one normalized example at a time.
    """
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        first = _first_non_whitespace(handle)
        if first == "[":
            yield from _iter_json_array(handle, limit=limit)
            return
        if first:
            line = first + handle.readline()
            count = 0
            while line:
                stripped = line.strip()
                if stripped:
                    yield _normalize(json.loads(stripped), count)
                    count += 1
                    if limit is not None and count >= limit:
                        return
                line = handle.readline()


def _first_non_whitespace(handle: Any) -> str:
    while True:
        character = handle.read(1)
        if not character or not character.isspace():
            return character


def _iter_json_array(handle: Any, *, limit: int | None) -> Iterator[DatasetExample]:
    decoder = json.JSONDecoder()
    buffer = ""
    count = 0
    end_of_array = False
    while not end_of_array:
        chunk = handle.read(1024 * 1024)
        if chunk:
            buffer += chunk
        elif not buffer.strip():
            break

        while True:
            buffer = buffer.lstrip()
            if buffer.startswith(","):
                buffer = buffer[1:].lstrip()
            if buffer.startswith("]"):
                end_of_array = True
                break
            if not buffer:
                break
            try:
                raw, consumed = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                break
            if not isinstance(raw, dict):
                raise ValueError("LongMemEval JSON array must contain objects")
            yield _normalize(raw, count)
            count += 1
            buffer = buffer[consumed:]
            if limit is not None and count >= limit:
                return
    if not end_of_array:
        raise ValueError("truncated LongMemEval JSON array")


def _normalize(raw: dict[str, Any], index: int) -> DatasetExample:
    question = raw.get("question", raw.get("input", ""))
    answer = raw.get("answer", raw.get("gold_answer"))
    evidence = raw.get("answer_session_ids", raw.get("evidence", raw.get("supporting_facts", [])))
    if not evidence:
        evidence = _evidence_session_ids(raw)
    if isinstance(evidence, str):
        evidence = [evidence]
    if not isinstance(evidence, list):
        evidence = []
    return DatasetExample(
        example_id=str(raw.get("question_id", raw.get("id", index))),
        question=str(question),
        answer=None if answer is None else str(answer),
        task_type=None if raw.get("question_type") is None else str(raw["question_type"]),
        evidence=tuple(str(item) for item in evidence),
        raw=raw,
    )


def _evidence_session_ids(raw: dict[str, Any]) -> list[str]:
    """Recover official answer-bearing session IDs when present."""
    answer_session_ids = raw.get("answer_session_ids")
    if isinstance(answer_session_ids, list):
        return [str(identifier) for identifier in answer_session_ids]

    sessions = raw.get("haystack_sessions", [])
    session_ids = raw.get("haystack_session_ids", [])
    identifiers: list[str] = []
    if not isinstance(sessions, list):
        return identifiers
    for index, session in enumerate(sessions):
        if isinstance(session, dict):
            session_id = session.get("session_id", session.get("id"))
            turns = session.get("turns", session.get("messages", session.get("content", [])))
            has_answer = bool(session.get("has_answer", False))
        else:
            session_id = session_ids[index] if isinstance(session_ids, list) and index < len(session_ids) else index
            turns = session
            has_answer = False
        if isinstance(turns, list):
            has_answer = has_answer or any(
                isinstance(turn, dict) and bool(turn.get("has_answer", False))
                for turn in turns
            )
        if has_answer and session_id is not None:
            identifiers.append(str(session_id))
    return identifiers
