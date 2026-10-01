"""Core value types for ClaimLedger."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ClaimStatus(StrEnum):
    CANDIDATE = "candidate"
    QUARANTINED = "quarantined"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"
    CONFLICTED = "conflicted"
    REJECTED = "rejected"


class TrustTier(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CONFIRMED = "confirmed"


class RelationKind(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    SUPERSEDES = "supersedes"
    DERIVED_FROM = "derived_from"
    DUPLICATES = "duplicates"


@dataclass(frozen=True)
class ValidTime:
    """Inclusive-start, exclusive-end validity interval."""

    valid_from: str | None = None
    valid_until: str | None = None

    def contains(self, at_time: str | None) -> bool:
        if at_time is None:
            return self.valid_until is None
        if self.valid_from is not None and self.valid_from > at_time:
            return False
        if self.valid_until is not None and self.valid_until <= at_time:
            return False
        return True


@dataclass(frozen=True)
class TransactionTime:
    """Transaction-time interval for append-only history."""

    tx_from: str
    tx_until: str | None = None


@dataclass(frozen=True)
class EvidenceRef:
    """Stable reference to the evidence supporting a claim."""

    source: str
    locator: str
    content_hash: str | None = None


@dataclass(frozen=True)
class ClaimRelation:
    """Typed relation between two claim records."""

    from_claim_id: str
    to_claim_id: str
    relation: RelationKind


@dataclass(frozen=True)
class ClaimRecord:
    claim_id: str
    subject: str
    predicate: str
    object: str
    evidence: str
    source: str
    valid_from: str | None
    valid_until: str | None
    tx_from: str
    tx_until: str | None
    trust_tier: TrustTier
    confidence: float
    status: ClaimStatus
    is_poisoned: bool


@dataclass(frozen=True)
class EligibilityResult:
    eligible: list[ClaimRecord]
    conflicted: list[ClaimRecord]
    quarantined: list[ClaimRecord]
    rejected: list[ClaimRecord]



