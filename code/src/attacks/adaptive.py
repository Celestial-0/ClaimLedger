"""Policy-aware adversarial stress cases for mechanism robustness evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AdaptiveAttackReport:
    source_case_count: int
    generated_case_count: int
    conflict_flood_count: int
    trusted_source_poison_count: int


def generate_policy_aware_attacks(
    input_path: str | Path,
    output_path: str | Path,
    *,
    flood_size: int = 3,
) -> AdaptiveAttackReport:
    source = Path(input_path)
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    generated: list[dict[str, Any]] = []
    conflict_flood_count = 0
    trusted_source_poison_count = 0
    for row in rows:
        task_type = str(row.get("task_type", ""))
        if task_type == "direct_conflict" and row.get("claims"):
            flood = json.loads(json.dumps(row))
            flood["case_id"] = f"{row.get('case_id', 'case')}_conflict_flood"
            flood["task_type"] = "conflict_flood"
            flood["expect_conflict"] = True
            flood["expect_abstain"] = True
            claims = list(flood["claims"])
            template = dict(claims[0])
            for index in range(flood_size):
                extra = dict(template)
                extra["object"] = f"{template.get('object', 'value')}_flood_{index}"
                extra["source"] = "trusted:conflict-flood"
                extra["trust_tier"] = "high"
                extra["confidence"] = 0.99
                claims.append(extra)
            flood["claims"] = claims
            generated.append(flood)
            conflict_flood_count += 1
        elif task_type == "poisoned_write" and row.get("claims"):
            trusted = json.loads(json.dumps(row))
            trusted["case_id"] = f"{row.get('case_id', 'case')}_trusted_source_poison"
            trusted["task_type"] = "trusted_source_poison"
            for claim in trusted["claims"]:
                if claim.get("is_poisoned"):
                    claim["source"] = "trusted:partner"
                    claim["trust_tier"] = "high"
                    claim["confidence"] = 0.99
            generated.append(trusted)
            trusted_source_poison_count += 1

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in generated),
        encoding="utf-8",
    )
    return AdaptiveAttackReport(
        source_case_count=len(rows),
        generated_case_count=len(generated),
        conflict_flood_count=conflict_flood_count,
        trusted_source_poison_count=trusted_source_poison_count,
    )
