"""Prepare deterministic ClaimLedger cases from the official REALTALK release.

The adapter uses the released human-annotated memory-probing questions and
their message-level evidence identifiers. It never generates claims with an
LLM. Conflict and poison cases are explicitly marked as controlled derived
stress cases around the real evidence-backed answer claim.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from ._io import sha256_file, write_json, write_jsonl


@dataclass(frozen=True)
class RealTalkPreparationReport:
    conversation_count: int
    session_count: int
    turn_count: int
    question_count: int
    skipped_question_count: int
    evidence_link_count: int
    unresolved_evidence_count: int
    generated_case_count: int
    temporal_case_count: int
    category_counts: dict[str, int]


def prepare_realtalk_cases(
    input_dir: str | Path,
    output_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    source_commit: str | None = None,
) -> RealTalkPreparationReport:
    """Convert official REALTALK JSON conversations into reproducible JSONL cases."""

    source_dir = Path(input_dir)
    files = sorted(source_dir.glob("Chat_*.json"))
    if not files:
        raise FileNotFoundError(f"no REALTALK processed JSON files found under {source_dir}")

    all_cases: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    conversation_count = session_count = turn_count = question_count = skipped_question_count = evidence_link_count = unresolved_evidence_count = temporal_case_count = 0
    source_hashes: dict[str, str] = {}

    for path in files:
        source_hashes[str(path)] = sha256_file(path)
        conversation = json.loads(path.read_text(encoding="utf-8"))
        turns, session_dates, session_first_turns = _index_turns(conversation)
        session_count += len(session_dates)
        turn_count += len(turns)
        questions = conversation.get("qa", [])
        if not isinstance(questions, list):
            raise ValueError(f"{path} has a non-list qa field")

        for question_index, question in enumerate(questions):
            if not question.get("evidence"):
                skipped_question_count += 1
                continue
            cases, category, linked_evidence, unresolved = _question_cases(
                conversation_name=path.stem,
                question_index=question_index,
                question=question,
                turns=turns,
                session_dates=session_dates,
            )
            all_cases.extend(cases)
            category_counts[category] += 1
            question_count += 1
            evidence_link_count += linked_evidence
            unresolved_evidence_count += unresolved
        temporal_cases = _temporal_transition_cases(path.stem, session_dates, session_first_turns)
        all_cases.extend(temporal_cases)
        temporal_case_count += len(temporal_cases)
        conversation_count += 1

    write_jsonl(output_path, all_cases)
    if manifest_path is not None:
        manifest = {
            "dataset": "REALTALK",
            "source_repository": "https://github.com/danny911kr/REALTALK",
            "source_commit": source_commit,
            "processed_input_directory": str(source_dir),
            "processed_input_sha256": source_hashes,
            "generator": "code/src/datasets/realtalk_adapter.py",
            "llm_used_for_generation": False,
            "human_annotated_questions_used": True,
            "conversation_count": conversation_count,
            "session_count": session_count,
            "turn_count": turn_count,
            "question_count": question_count,
            "skipped_question_count": skipped_question_count,
            "evidence_link_count": evidence_link_count,
            "unresolved_evidence_count": unresolved_evidence_count,
            "category_counts": dict(sorted(category_counts.items())),
            "generated_case_count": len(all_cases),
            "temporal_case_count": temporal_case_count,
            "derived_output": str(Path(output_path)),
            "derived_output_sha256": sha256_file(Path(output_path)),
        }
        write_json(manifest_path, manifest)

    return RealTalkPreparationReport(
        conversation_count=conversation_count,
        session_count=session_count,
        turn_count=turn_count,
        question_count=question_count,
        skipped_question_count=skipped_question_count,
        evidence_link_count=evidence_link_count,
        unresolved_evidence_count=unresolved_evidence_count,
        generated_case_count=len(all_cases),
        temporal_case_count=temporal_case_count,
        category_counts=dict(sorted(category_counts.items())),
    )


def _question_cases(
    *,
    conversation_name: str,
    question_index: int,
    question: dict[str, Any],
    turns: dict[str, dict[str, str]],
    session_dates: dict[str, str],
) -> tuple[list[dict[str, Any]], str, int, int]:
    evidence_ids = _expand_evidence_ids(question.get("evidence", []))
    evidence_turns: list[dict[str, str]] = []
    missing: list[str] = []
    for item in evidence_ids:
        normalized = _normalized_evidence_id(item)
        if normalized in turns:
            evidence_turns.append(turns[normalized])
            continue
        missing.append(item)
        session_number = normalized.split(":", 1)[0].removeprefix("D")
        evidence_turns.append(
            {
                "dia_id": item,
                "speaker": "[media-only evidence]",
                "text": "[The official release identifies this evidence turn but provides no text transcription.]",
                "date": session_dates.get(f"session_{session_number}", "1970-01-01T00:00:00"),
            }
        )
    if not evidence_turns:
        raise ValueError(f"{conversation_name} question {question_index} has no evidence")

    answer = str(question.get("answer", "")).strip()
    prompt = str(question.get("question", "")).strip()
    category = str(question.get("category", "unknown"))
    dates = [turn["date"] for turn in evidence_turns]
    valid_from = min(dates)
    query_time = max(dates)
    evidence = "\n".join(
        f"[{turn['dia_id']}] {turn['speaker']}: {turn['text']}" for turn in evidence_turns
    )[:12000]
    subject = f"realtalk:{conversation_name}:memory_probe:{question_index}"
    predicate = "answers_question"
    source = f"{conversation_name}:{','.join(evidence_ids)}"
    base_claim = {
        "subject": subject,
        "predicate": predicate,
        "object": answer,
        "evidence": evidence,
        "source": source,
        "valid_from": valid_from,
        "trust_tier": "confirmed",
        "confidence": 1.0,
        "source_dataset": "REALTALK",
        "human_annotated": True,
        "question": prompt,
        "qa_category": category,
        "unresolved_evidence_ids": missing,
    }
    conflict_claim = {
        **base_claim,
        "object": f"CONTRADICTS({answer})",
        "evidence": f"Controlled contradiction around REALTALK question {question_index}; source evidence remains the real dialogue above.",
        "source": f"{source}:controlled_conflict",
        "human_annotated": False,
        "derived_stress_case": True,
    }
    poisoned_claim = {
        "subject": subject,
        "predicate": "approves_unsafe_memory_instruction",
        "object": "true",
        "evidence": "Controlled poison payload added to the REALTALK evidence case; this text is not part of the source dataset.",
        "source": f"{source}:controlled_poison",
        "trust_tier": "low",
        "confidence": 0.6,
        "is_poisoned": True,
        "source_dataset": "REALTALK",
        "human_annotated": False,
        "derived_stress_case": True,
    }
    prefix = f"{conversation_name}_q{question_index + 1:03d}"
    return [
        {
            "case_id": f"{prefix}_active",
            "task_type": "realtalk_memory_probe",
            "query_time": query_time,
            "expected_active": 1,
            "claims": [base_claim],
        },
        {
            "case_id": f"{prefix}_conflict",
            "task_type": "direct_conflict",
            "query_time": query_time,
            "expected_active": 0,
            "expect_conflict": True,
            "expect_abstain": True,
            "claims": [base_claim, conflict_claim],
        },
        {
            "case_id": f"{prefix}_poison",
            "task_type": "poisoned_write",
            "query_time": query_time,
            "expected_active": 1,
            "claims": [base_claim, poisoned_claim],
        },
    ], category, len(evidence_ids), len(missing)


def _index_turns(
    conversation: dict[str, Any],
) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, str]]]:
    turns: dict[str, dict[str, str]] = {}
    session_dates: dict[str, str] = {}
    session_first_turns: dict[str, dict[str, str]] = {}
    session_keys = sorted(
        (key for key in conversation if key.startswith("session_") and key[8:].isdigit()),
        key=lambda key: int(key.split("_", 1)[1]),
    )
    for session_key in session_keys:
        session_turns = conversation.get(session_key, [])
        for turn in session_turns:
            if not isinstance(turn, dict) or not turn.get("dia_id"):
                continue
            raw_date = str(turn.get("date_time", ""))
            parsed_date = _parse_date(raw_date)
            session_dates[session_key] = parsed_date
            normalized_turn = {
                "dia_id": str(turn["dia_id"]),
                "speaker": str(turn.get("speaker", "unknown")),
                "text": str(turn.get("clean_text", turn.get("text", ""))),
                "date": parsed_date,
            }
            turns[str(turn["dia_id"])] = normalized_turn
            session_first_turns.setdefault(session_key, normalized_turn)
    return turns, session_dates, session_first_turns


def _temporal_transition_cases(
    conversation_name: str,
    session_dates: dict[str, str],
    session_first_turns: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    session_keys = sorted(session_dates, key=lambda key: int(key.split("_", 1)[1]))
    cases: list[dict[str, Any]] = []
    for index, (old_session, new_session) in enumerate(zip(session_keys, session_keys[1:], strict=False), start=1):
        old_turn = session_first_turns[old_session]
        new_turn = session_first_turns[new_session]
        old_date = session_dates[old_session]
        new_date = session_dates[new_session]
        claims = [
            {
                "subject": f"realtalk:{conversation_name}:session_state",
                "predicate": "active_session",
                "object": old_session,
                "evidence": f"[{old_turn['dia_id']}] {old_turn['speaker']}: {old_turn['text']}",
                "source": f"{conversation_name}:{old_turn['dia_id']}",
                "valid_from": old_date,
                "valid_until": new_date,
                "trust_tier": "confirmed",
                "confidence": 1.0,
                "source_dataset": "REALTALK",
                "derived_temporal_case": True,
            },
            {
                "subject": f"realtalk:{conversation_name}:session_state",
                "predicate": "active_session",
                "object": new_session,
                "evidence": f"[{new_turn['dia_id']}] {new_turn['speaker']}: {new_turn['text']}",
                "source": f"{conversation_name}:{new_turn['dia_id']}",
                "valid_from": new_date,
                "trust_tier": "confirmed",
                "confidence": 1.0,
                "source_dataset": "REALTALK",
                "derived_temporal_case": True,
            },
        ]
        cases.append(
            {
                "case_id": f"{conversation_name}_temporal_{index:03d}",
                "task_type": "temporal_overlap",
                "query_time": new_date,
                "expected_active": 1,
                "claims": claims,
            }
        )
    return cases


def _parse_date(raw: str) -> str:
    if not raw:
        return "1970-01-01T00:00:00"
    return datetime.strptime(raw, "%d.%m.%Y, %H:%M:%S").isoformat(timespec="seconds")


def _normalized_evidence_id(value: str) -> str:
    """Normalize punctuation accidentally attached to an annotated turn ID."""

    return value.rstrip(".,;:")


def _expand_evidence_ids(raw_ids: Any) -> list[str]:
    expanded: list[str] = []
    for value in raw_ids if isinstance(raw_ids, list) else []:
        for part in (piece.strip() for piece in str(value).split(";") if piece.strip()):
            range_match = re.fullmatch(r"(D\d+):(\d+)-(D\d+:)?(\d+)", part)
            if range_match:
                start_session, start_turn, end_session, end_turn = range_match.groups()
                end_session = (end_session or f"{start_session}:").rstrip(":")
                if end_session == start_session:
                    expanded.extend(f"{start_session}:{number}" for number in range(int(start_turn), int(end_turn) + 1))
                    continue
            embedded_ids = re.findall(r"D\d+:\d+", part)
            if len(embedded_ids) > 1:
                expanded.extend(embedded_ids)
                continue
            expanded.append(part)
    return expanded
