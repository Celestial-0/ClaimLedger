import unittest

from claimledger import ClaimLedger, ClaimStatus, TrustTier
from claimledger.core.schema import ValidTime


class LedgerInvariantTests(unittest.TestCase):
    def _active_claim(self, ledger: ClaimLedger, *, object_value: str = "v1") -> str:
        return ledger.add_claim(
            subject="user",
            predicate="prefers",
            object=object_value,
            evidence=f"user prefers {object_value}",
            source="fixture:1",
            trust_tier=TrustTier.CONFIRMED,
            confidence=0.95,
        )

    def test_state_changes_append_events(self) -> None:
        with ClaimLedger() as ledger:
            claim_id = self._active_claim(ledger)
            before = ledger.event_count()
            ledger.quarantine(claim_id, "manual review")

            self.assertEqual(ledger.get(claim_id).status, ClaimStatus.QUARANTINED)
            self.assertGreater(ledger.event_count(), before)

    def test_conflict_marks_both_claims_and_discloses_them(self) -> None:
        with ClaimLedger() as ledger:
            first = self._active_claim(ledger, object_value="v1")
            second = self._active_claim(ledger, object_value="v2")
            ledger.mark_conflict(first, second, "contradictory fixture")

            result = ledger.eligible_claims()

            self.assertEqual({item.claim_id for item in result.conflicted}, {first, second})
            self.assertEqual(result.eligible, [])

    def test_evidence_and_valid_interval_are_required(self) -> None:
        with ClaimLedger() as ledger:
            with self.assertRaises(ValueError):
                ledger.add_claim(
                    subject="user",
                    predicate="prefers",
                    object="v1",
                    evidence="",
                    source="fixture:1",
                )
            with self.assertRaises(ValueError):
                ledger.add_claim(
                    subject="user",
                    predicate="prefers",
                    object="v1",
                    evidence="support",
                    source="fixture:1",
                    valid_from="2025-01-02",
                    valid_until="2025-01-01",
                )

    def test_valid_time_boundaries_are_explicit(self) -> None:
        interval = ValidTime(valid_from="2025-01-01", valid_until="2025-01-10")

        self.assertTrue(interval.contains("2025-01-01"))
        self.assertTrue(interval.contains("2025-01-09"))
        self.assertFalse(interval.contains("2025-01-10"))


if __name__ == "__main__":
    unittest.main()
