"""Deterministic claim extraction baseline for LoCoMo-style dialogue."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExtractedClaim:
    sample_id: str
    evidence_id: str
    subject: str
    predicate: str
    object: str
    evidence: str
    source: str
    valid_from: str
    valid_until: str | None
    trust_tier: str
    confidence: float
    extractor: str


@dataclass(frozen=True)
class ExtractionReport:
    sample_count: int
    turn_count: int
    claim_count: int
    claims_per_turn: float
    by_predicate: dict[str, int]


def extract_locomo_claims(
    input_path: str | Path,
    output_path: str | Path,
    *,
    include_fallback: bool = False,
) -> ExtractionReport:
    samples = _load_samples(input_path)
    claims: list[ExtractedClaim] = []
    turn_count = 0
    for sample in samples:
        sample_claims, sample_turns = _extract_sample(sample, include_fallback=include_fallback)
        claims.extend(sample_claims)
        turn_count += sample_turns
    _write_jsonl(output_path, (asdict(claim) for claim in claims))
    counts = Counter(claim.predicate for claim in claims)
    return ExtractionReport(
        sample_count=len(samples),
        turn_count=turn_count,
        claim_count=len(claims),
        claims_per_turn=len(claims) / turn_count if turn_count else 0.0,
        by_predicate=dict(sorted(counts.items())),
    )


def _extract_sample(sample: dict[str, Any], *, include_fallback: bool) -> tuple[list[ExtractedClaim], int]:
    sample_id = str(sample.get("sample_id", sample.get("id", "unknown_sample")))
    conversation = sample.get("conversation", {})
    if not isinstance(conversation, dict):
        return [], 0
    session_names = sorted(
        [key for key in conversation if key.startswith("session_") and not key.endswith("_date_time")],
        key=_session_index,
    )
    claims: list[ExtractedClaim] = []
    turn_count = 0
    for session_idx, session in enumerate(session_names):
        valid_from = _session_date(session_idx)
        raw_turns = conversation.get(session, [])
        if not isinstance(raw_turns, list):
            continue
        for turn in raw_turns:
            if not isinstance(turn, dict):
                continue
            turn_count += 1
            claims.extend(_extract_turn(sample_id, session, turn, valid_from, include_fallback=include_fallback))
    return claims, turn_count


def _extract_turn(
    sample_id: str,
    session: str,
    turn: dict[str, Any],
    valid_from: str,
    *,
    include_fallback: bool,
) -> list[ExtractedClaim]:
    speaker = str(turn.get("speaker", "speaker")).strip() or "speaker"
    dia_id = str(turn.get("dia_id", "unknown_turn"))
    text = str(turn.get("text", "")).strip()
    source = f"{sample_id}:{session}:{dia_id}"
    subject = f"locomo:{sample_id}:{speaker}"
    claims: list[ExtractedClaim] = []

    patterns = [
        (r"\bI (?:still )?live in ([A-Z][A-Za-z .'-]+)", "lives_in", 0.86),
        (r"\bI moved (?:the shop )?to ([A-Z][A-Za-z .'-]+)", "moved_to", 0.82),
        (r"\bI (?:am|'m) (?:a |an )?([A-Za-z -]+ woman)\b", "identity", 0.78),
        (r"\bI researched ([A-Za-z -]+)", "researched", 0.76),
        (r"\bI ran (?:a |an )?([A-Za-z -]+)", "participated_in", 0.72),
        (r"\bI painted (?:a |an )?([A-Za-z -]+)", "created", 0.72),
        (r"\bI (?:went|go) to ([^.?!]+)", "visited", 0.70),
        (r"\bI (?:want|wanted|would like) to ([^.?!]+)", "intends", 0.68),
        (r"\bI (?:started|began) ([^.?!]+)", "started", 0.68),
        (r"\bI (?:love|like|enjoy) ([^.?!]+)", "likes", 0.66),
        (r"\bI work(?:ed)? (?:at|in|as) (?:a |an |the )?([A-Za-z][A-Za-z -]{2,80})", "works_in", 0.70),
    ]
    for pattern, predicate, confidence in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            obj = _clean_object(match.group(1))
            if obj:
                claims.append(
                    ExtractedClaim(
                        sample_id=sample_id,
                        evidence_id=dia_id,
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        evidence=f"{speaker}: {text}",
                        source=source,
                        valid_from=valid_from,
                        valid_until=None,
                        trust_tier="confirmed",
                        confidence=confidence,
                        extractor="rules-v1",
                    )
                )

    if include_fallback and not claims and len(text) >= 40:
        claims.append(
            ExtractedClaim(
                sample_id=sample_id,
                evidence_id=dia_id,
                subject=subject,
                predicate="said",
                object=text[:180],
                evidence=f"{speaker}: {text}",
                source=source,
                valid_from=valid_from,
                valid_until=None,
                trust_tier="medium",
                confidence=0.55,
                extractor="fallback-v1",
            )
        )
    return claims


def _clean_object(value: str) -> str:
    value = re.split(r"[.!?,;:]", value.strip())[0]
    return re.sub(r"\s+", " ", value).strip()


def _load_samples(path: str | Path) -> list[dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("data", raw.get("samples", [raw]))
    if not isinstance(raw, list):
        raise ValueError("LoCoMo input must be a JSON list or dict containing data/samples")
    return raw


def _write_jsonl(path: str | Path, rows) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _session_date(session_idx: int) -> str:
    return f"2000-01-{session_idx + 1:02d}"


def _session_index(name: str) -> int:
    try:
        return int(name.split("_", 1)[1])
    except (IndexError, ValueError):
        return 0



