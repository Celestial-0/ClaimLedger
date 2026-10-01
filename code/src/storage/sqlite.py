"""SQLite-backed ClaimLedger prototype."""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterable
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path

from ..core.schema import ClaimRecord, ClaimStatus, EligibilityResult, RelationKind, TrustTier


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class ClaimLedger(AbstractContextManager["ClaimLedger"]):
    """A small SQLite-backed claim ledger with auditable state transitions."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.init_schema()

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()

    def close(self) -> None:
        self.conn.close()

    def init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS claims (
                claim_id TEXT PRIMARY KEY,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                evidence TEXT NOT NULL,
                source TEXT NOT NULL,
                valid_from TEXT,
                valid_until TEXT,
                tx_from TEXT NOT NULL,
                tx_until TEXT,
                trust_tier TEXT NOT NULL,
                confidence REAL NOT NULL,
                status TEXT NOT NULL,
                is_poisoned INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS claim_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_time TEXT NOT NULL,
                details TEXT NOT NULL DEFAULT '',
                FOREIGN KEY (claim_id) REFERENCES claims (claim_id)
            );

            CREATE TABLE IF NOT EXISTS claim_relations (
                from_claim_id TEXT NOT NULL,
                to_claim_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                event_time TEXT NOT NULL,
                PRIMARY KEY (from_claim_id, to_claim_id, relation),
                FOREIGN KEY (from_claim_id) REFERENCES claims (claim_id),
                FOREIGN KEY (to_claim_id) REFERENCES claims (claim_id)
            );
            """
        )
        self.conn.commit()

    def add_claim(
        self,
        *,
        subject: str,
        predicate: str,
        object: str,
        evidence: str,
        source: str,
        valid_from: str | None = None,
        valid_until: str | None = None,
        trust_tier: TrustTier = TrustTier.MEDIUM,
        confidence: float = 0.75,
        status: ClaimStatus | None = None,
        is_poisoned: bool = False,
        claim_id: str | None = None,
    ) -> str:
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if not evidence.strip():
            raise ValueError("evidence must not be empty")
        if not source.strip():
            raise ValueError("source must not be empty")
        if valid_from is not None and valid_until is not None and valid_until <= valid_from:
            raise ValueError("valid_until must be later than valid_from")
        claim_id = claim_id or str(uuid.uuid4())
        chosen_status = status or self._initial_status(trust_tier, confidence)
        now = _now()
        self.conn.execute(
            """
            INSERT INTO claims (
                claim_id, subject, predicate, object, evidence, source,
                valid_from, valid_until, tx_from, tx_until, trust_tier,
                confidence, status, is_poisoned
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?)
            """,
            (
                claim_id,
                subject,
                predicate,
                object,
                evidence,
                source,
                valid_from,
                valid_until,
                now,
                str(trust_tier),
                confidence,
                str(chosen_status),
                int(is_poisoned),
            ),
        )
        self._event(claim_id, "insert", f"status={chosen_status}")
        self.conn.commit()
        return claim_id

    def activate(self, claim_id: str, reason: str = "") -> None:
        self._set_status(claim_id, ClaimStatus.ACTIVE, reason)

    def quarantine(self, claim_id: str, reason: str = "") -> None:
        self._set_status(claim_id, ClaimStatus.QUARANTINED, reason)

    def reject(self, claim_id: str, reason: str = "") -> None:
        self._set_status(claim_id, ClaimStatus.REJECTED, reason)

    def mark_conflict(self, first_claim_id: str, second_claim_id: str, reason: str = "") -> None:
        self._relation(first_claim_id, second_claim_id, RelationKind.CONTRADICTS)
        self._relation(second_claim_id, first_claim_id, RelationKind.CONTRADICTS)
        self._set_status(first_claim_id, ClaimStatus.CONFLICTED, reason)
        self._set_status(second_claim_id, ClaimStatus.CONFLICTED, reason)

    def supersede(self, older_claim_id: str, newer_claim_id: str, reason: str = "") -> None:
        self._relation(newer_claim_id, older_claim_id, RelationKind.SUPERSEDES)
        self._set_status(older_claim_id, ClaimStatus.SUPERSEDED, reason)
        self.activate(newer_claim_id, f"supersedes {older_claim_id}; {reason}".strip())

    def eligible_claims(self, *, at_time: str | None = None) -> EligibilityResult:
        rows = [self._record(row) for row in self.conn.execute("SELECT * FROM claims")]
        active: list[ClaimRecord] = []
        conflicted: list[ClaimRecord] = []
        quarantined: list[ClaimRecord] = []
        rejected: list[ClaimRecord] = []
        for record in rows:
            if record.status == ClaimStatus.ACTIVE and self._valid_at(record, at_time):
                active.append(record)
            elif record.status == ClaimStatus.CONFLICTED:
                conflicted.append(record)
            elif record.status == ClaimStatus.QUARANTINED:
                quarantined.append(record)
            elif record.status == ClaimStatus.REJECTED:
                rejected.append(record)
        return EligibilityResult(active, conflicted, quarantined, rejected)

    def all_claims(self) -> list[ClaimRecord]:
        return [self._record(row) for row in self.conn.execute("SELECT * FROM claims ORDER BY tx_from, claim_id")]

    def get(self, claim_id: str) -> ClaimRecord:
        row = self.conn.execute("SELECT * FROM claims WHERE claim_id = ?", (claim_id,)).fetchone()
        if row is None:
            raise KeyError(claim_id)
        return self._record(row)

    def event_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM claim_events").fetchone()
        return int(row["n"])

    def load_claims(self, claims: Iterable[dict[str, object]]) -> list[str]:
        ids = []
        for claim in claims:
            ids.append(
                self.add_claim(
                    subject=str(claim["subject"]),
                    predicate=str(claim["predicate"]),
                    object=str(claim["object"]),
                    evidence=str(claim["evidence"]),
                    source=str(claim["source"]),
                    valid_from=claim.get("valid_from"),  # type: ignore[arg-type]
                    valid_until=claim.get("valid_until"),  # type: ignore[arg-type]
                    trust_tier=TrustTier(str(claim.get("trust_tier", TrustTier.MEDIUM))),
                    confidence=float(claim.get("confidence", 0.75)),
                    is_poisoned=bool(claim.get("is_poisoned", False)),
                )
            )
        return ids

    def _initial_status(self, trust_tier: TrustTier, confidence: float) -> ClaimStatus:
        if trust_tier == TrustTier.CONFIRMED and confidence >= 0.65:
            return ClaimStatus.ACTIVE
        if trust_tier == TrustTier.HIGH and confidence >= 0.8:
            return ClaimStatus.ACTIVE
        return ClaimStatus.QUARANTINED

    def _set_status(self, claim_id: str, status: ClaimStatus, reason: str) -> None:
        before = self.conn.total_changes
        self.conn.execute("UPDATE claims SET status = ? WHERE claim_id = ?", (str(status), claim_id))
        if self.conn.total_changes == before:
            raise KeyError(claim_id)
        self._event(claim_id, f"status:{status}", reason)
        self.conn.commit()

    def _event(self, claim_id: str, event_type: str, details: str = "") -> None:
        self.conn.execute(
            "INSERT INTO claim_events (claim_id, event_type, event_time, details) VALUES (?, ?, ?, ?)",
            (claim_id, event_type, _now(), details),
        )

    def _relation(self, from_claim_id: str, to_claim_id: str, relation: RelationKind) -> None:
        self.conn.execute(
            """
            INSERT OR IGNORE INTO claim_relations (from_claim_id, to_claim_id, relation, event_time)
            VALUES (?, ?, ?, ?)
            """,
            (from_claim_id, to_claim_id, str(relation), _now()),
        )

    def _record(self, row: sqlite3.Row) -> ClaimRecord:
        return ClaimRecord(
            claim_id=row["claim_id"],
            subject=row["subject"],
            predicate=row["predicate"],
            object=row["object"],
            evidence=row["evidence"],
            source=row["source"],
            valid_from=row["valid_from"],
            valid_until=row["valid_until"],
            tx_from=row["tx_from"],
            tx_until=row["tx_until"],
            trust_tier=TrustTier(row["trust_tier"]),
            confidence=float(row["confidence"]),
            status=ClaimStatus(row["status"]),
            is_poisoned=bool(row["is_poisoned"]),
        )

    def _valid_at(self, record: ClaimRecord, at_time: str | None) -> bool:
        if at_time is None:
            return record.valid_until is None
        if record.valid_from is not None and record.valid_from > at_time:
            return False
        if record.valid_until is not None and record.valid_until <= at_time:
            return False
        return True
