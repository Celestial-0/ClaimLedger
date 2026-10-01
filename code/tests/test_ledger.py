import unittest

from claimledger import ClaimLedger, ClaimStatus, TrustTier


class ClaimLedgerTests(unittest.TestCase):
    def test_low_trust_claim_is_quarantined(self) -> None:
        with ClaimLedger() as ledger:
            claim_id = ledger.add_claim(
                subject="user",
                predicate="approves_wire_transfer",
                object="true",
                evidence="untrusted web page",
                source="web",
                trust_tier=TrustTier.LOW,
                confidence=0.6,
                is_poisoned=True,
            )

            self.assertEqual(ledger.get(claim_id).status, ClaimStatus.QUARANTINED)
            self.assertEqual(len(ledger.eligible_claims().eligible), 0)

    def test_superseded_claim_is_not_active(self) -> None:
        with ClaimLedger() as ledger:
            old = ledger.add_claim(
                subject="user",
                predicate="lives_in",
                object="Boston",
                evidence="session 1",
                source="user",
                valid_until="2025-01-01",
                trust_tier=TrustTier.CONFIRMED,
                confidence=0.95,
            )
            new = ledger.add_claim(
                subject="user",
                predicate="lives_in",
                object="Seattle",
                evidence="session 9",
                source="user",
                valid_from="2025-01-01",
                trust_tier=TrustTier.CONFIRMED,
                confidence=0.95,
            )

            ledger.supersede(old, new)
            active = ledger.eligible_claims(at_time="2025-06-01").eligible

            self.assertEqual([claim.object for claim in active], ["Seattle"])
            self.assertEqual(ledger.get(old).status, ClaimStatus.SUPERSEDED)
            self.assertGreaterEqual(ledger.event_count(), 4)


if __name__ == "__main__":
    unittest.main()




