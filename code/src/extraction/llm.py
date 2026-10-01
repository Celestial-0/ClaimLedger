"""LLM-based claim extraction via Ollama.

Replaces the deterministic rule-based extractor with an LLM-driven
zero-shot structured extraction pipeline. Extracts atomic claims in
the form (subject, predicate, object, evidence, valid_from, valid_until)
from raw dialogue turns.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..integrations.ollama_client import OllamaClient


@dataclass(frozen=True)
class LLMExtractedClaim:
    """A single claim extracted by the LLM."""

    sample_id: str
    turn_id: str
    subject: str
    predicate: str
    object: str
    evidence: str
    valid_from: str | None
    valid_until: str | None
    confidence: float


@dataclass(frozen=True)
class LLMExtractionReport:
    """Summary of LLM extraction over a sample."""

    model_name: str
    total_turns: int
    total_claims_extracted: int
    claims_per_turn: float
    extraction_errors: int


_EXTRACTION_SYSTEM_PROMPT = """\
You are a structured claim extractor. Given a dialogue turn, extract all \
factual claims as JSON. Each claim must have:
- "subject": who or what the claim is about
- "predicate": the relationship or attribute (use snake_case like lives_in, works_at, likes, visited, etc.)
- "object": the value or target
- "evidence": the exact quote from the turn supporting the claim
- "valid_from": ISO date if mentioned or inferrable, else null
- "valid_until": ISO date if mentioned or inferrable, else null
- "confidence": float 0-1, how confident you are this is a factual claim

Return a JSON array of claims. If no factual claims exist, return [].
Return ONLY valid JSON, no explanations."""


def extract_claims_llm(
    locomo_path: str | Path,
    output_path: str | Path,
    *,
    model: str = "phi4-mini",
    base_url: str = "http://127.0.0.1:11434",
    sample_size: int = 50,
    seed: int = 42,
) -> LLMExtractionReport:
    """Extract claims from a random sample of dialogue turns using an LLM.

    Parameters
    ----------
    locomo_path : path
        Path to the LoCoMo JSON file.
    output_path : path
        Output path for the extraction results JSONL.
    model : str
        Ollama model to use for extraction.
    sample_size : int
        Number of turns to sample for extraction.
    seed : int
        Random seed for reproducible sampling.

    Returns
    -------
    LLMExtractionReport
        Summary statistics of the extraction run.
    """
    import random

    client = OllamaClient(
        model=model,
        base_url=base_url,
        temperature=0.0,
        system_prompt=_EXTRACTION_SYSTEM_PROMPT,
        timeout_s=120,
    )

    # Load and flatten dialogue turns
    turns = _load_dialogue_turns(locomo_path)

    # Sample turns
    rng = random.Random(seed)
    sampled = rng.sample(turns, min(sample_size, len(turns)))

    results: list[dict[str, Any]] = []
    errors = 0
    total_claims = 0

    for i, turn in enumerate(sampled):
        turn_text = f"{turn['speaker']}: {turn['text']}"
        prompt = f"Dialogue turn:\n{turn_text}\n\nExtract factual claims as JSON array:"

        try:
            response = client.generate(prompt, max_tokens=512)
            claims = _parse_claims_json(response.text, turn)
            total_claims += len(claims)
        except Exception:
            claims = []
            errors += 1

        result_entry = {
            "turn_index": i,
            "sample_id": turn.get("sample_id", ""),
            "turn_id": turn.get("dia_id", ""),
            "session": turn.get("session", ""),
            "speaker": turn.get("speaker", ""),
            "text": turn.get("text", ""),
            "extracted_claims": claims,
            "claim_count": len(claims),
        }
        results.append(result_entry)

    # Write results
    with Path(output_path).open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    claims_per_turn = total_claims / len(sampled) if sampled else 0.0

    return LLMExtractionReport(
        model_name=model,
        total_turns=len(sampled),
        total_claims_extracted=total_claims,
        claims_per_turn=claims_per_turn,
        extraction_errors=errors,
    )


def compare_extractors(
    rule_claims_path: str | Path,
    llm_claims_path: str | Path,
) -> dict[str, Any]:
    """Compare rule-based and LLM-based extraction outputs.

    Returns a dict with precision/recall/F1 estimates based on
    subject-predicate-object overlap between the two extractors.
    """
    rule_claims = _load_jsonl(rule_claims_path)
    llm_results = _load_jsonl(llm_claims_path)

    # Build rule claim fingerprints by turn
    rule_by_turn: dict[str, set[str]] = {}
    for claim in rule_claims:
        turn_key = claim.get("evidence_id", claim.get("source", ""))
        fp = _claim_fingerprint(claim)
        rule_by_turn.setdefault(turn_key, set()).add(fp)

    # Build LLM claim fingerprints by turn
    llm_by_turn: dict[str, set[str]] = {}
    for result in llm_results:
        turn_key = result.get("turn_id", "")
        for claim in result.get("extracted_claims", []):
            fp = _claim_fingerprint(claim)
            llm_by_turn.setdefault(turn_key, set()).add(fp)

    # Calculate overlap metrics
    all_turns = set(rule_by_turn) | set(llm_by_turn)
    total_rule = sum(len(v) for v in rule_by_turn.values())
    total_llm = sum(len(v) for v in llm_by_turn.values())
    total_overlap = 0
    for turn in all_turns:
        r = rule_by_turn.get(turn, set())
        l = llm_by_turn.get(turn, set())
        total_overlap += len(r & l)

    precision = total_overlap / total_llm if total_llm else 0.0
    recall = total_overlap / total_rule if total_rule else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "rule_claim_count": total_rule,
        "llm_claim_count": total_llm,
        "overlap_count": total_overlap,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _load_dialogue_turns(path: str | Path) -> list[dict[str, Any]]:
    """Flatten LoCoMo JSON into individual dialogue turns."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    samples = raw if isinstance(raw, list) else raw.get("data", raw.get("samples", [raw]))
    turns: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = str(sample.get("sample_id", sample.get("id", "")))
        conversation = sample.get("conversation", {})
        if not isinstance(conversation, dict):
            continue
        sessions = sorted(
            [k for k in conversation if k.startswith("session_") and not k.endswith("_date_time")],
            key=lambda s: int(s.split("_")[1]) if s.split("_")[1].isdigit() else 0,
        )
        for session in sessions:
            session_date = conversation.get(f"{session}_date_time", "")
            for turn in conversation.get(session, []):
                if not isinstance(turn, dict):
                    continue
                turns.append({
                    "sample_id": sample_id,
                    "session": session,
                    "session_date": session_date,
                    "dia_id": str(turn.get("dia_id", "")),
                    "speaker": str(turn.get("speaker", "")),
                    "text": str(turn.get("text", "")),
                })
    return turns


def _parse_claims_json(
    raw_text: str,
    turn: dict[str, Any],
) -> list[dict[str, Any]]:
    """Parse LLM output into structured claim dicts."""
    text = raw_text.strip()
    
    # Strip markdown code fence if present
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Look for JSON array or object in text
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        text = match.group(0)
    else:
        match_obj = re.search(r"\{.*\}", text, re.DOTALL)
        if match_obj:
            text = match_obj.group(0)

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []

    if isinstance(parsed, dict):
        parsed = [parsed]
    elif not isinstance(parsed, list):
        return []

    claims = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        subj = str(item.get("subject", "")).strip()
        pred = str(item.get("predicate", "")).strip()
        obj = str(item.get("object", "")).strip()
        if not subj or not obj:
            continue
        claims.append({
            "subject": subj,
            "predicate": pred or "stated",
            "object": obj,
            "evidence": str(item.get("evidence", turn.get("text", ""))),
            "valid_from": item.get("valid_from"),
            "valid_until": item.get("valid_until"),
            "confidence": float(item.get("confidence", 0.75)),
        })
    return claims


def _claim_fingerprint(claim: dict[str, Any]) -> str:
    """Create a normalized fingerprint for deduplication/matching."""
    s = str(claim.get("subject", "")).lower().strip()
    p = str(claim.get("predicate", "")).lower().strip()
    o = str(claim.get("object", "")).lower().strip()
    return f"{s}|{p}|{o}"


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows
