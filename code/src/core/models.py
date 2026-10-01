"""Backward-compatible exports for the core schema types."""

from .schema import (
    ClaimRecord,
    ClaimRelation,
    ClaimStatus,
    EligibilityResult,
    EvidenceRef,
    RelationKind,
    TransactionTime,
    TrustTier,
    ValidTime,
)

__all__ = [
    "ClaimRecord",
    "ClaimRelation",
    "ClaimStatus",
    "EligibilityResult",
    "EvidenceRef",
    "RelationKind",
    "TransactionTime",
    "TrustTier",
    "ValidTime",
]
