import unittest

from claimledger.attacks import run_trust_escalation_probe, sweep_trust_escalation
from claimledger.attacks.trust_escalation import PAYLOAD_PREDICATE


class TrustEscalationTests(unittest.TestCase):
    def test_direct_confirmed_payload_activates(self) -> None:
        # An attacker who can mint confirmed-standing writes bypasses the
        # quarantine gate: the unlabelled payload becomes eligible. This is the
        # adaptive-attack failure the manuscript must disclose.
        outcome = run_trust_escalation_probe(mode="direct", n_benign=0, seed=0)
        self.assertTrue(outcome.payload_active)
        self.assertFalse(outcome.payload_flagged)
        self.assertEqual(outcome.payload_status, "active")

    def test_escalation_from_low_trust_is_quarantined(self) -> None:
        # Under the current static policy, source trust never promotes, so a
        # LOW-trust payload is quarantined regardless of benign history length.
        for n_benign in (0, 1, 10, 50):
            outcome = run_trust_escalation_probe(mode="escalation", n_benign=n_benign, seed=0)
            self.assertFalse(outcome.payload_active, msg=f"n_benign={n_benign}")
            self.assertTrue(outcome.payload_flagged, msg=f"n_benign={n_benign}")
            self.assertEqual(outcome.payload_status, "quarantined", msg=f"n_benign={n_benign}")

    def test_benign_history_does_not_promote_low_trust(self) -> None:
        # The defence holds only because no promotion mechanism exists yet: even
        # a long benign history leaves the attacker's payload quarantined.
        outcome = run_trust_escalation_probe(mode="escalation", n_benign=50, seed=1)
        self.assertFalse(outcome.payload_active)

    def test_payload_uses_expected_predicate(self) -> None:
        # Sanity: the payload is the unsafe-instruction claim, not a benign one.
        self.assertEqual(PAYLOAD_PREDICATE, "approves_unsafe_memory_instruction")

    def test_sweep_covers_direct_and_escalation(self) -> None:
        outcomes = sweep_trust_escalation(n_benign_values=[0, 5], seeds=[0])
        modes = {o.mode for o in outcomes}
        self.assertEqual(modes, {"direct", "escalation"})

    def test_unknown_mode_rejected(self) -> None:
        with self.assertRaises(ValueError):
            run_trust_escalation_probe(mode="sideways", n_benign=0, seed=0)

    def test_deterministic_for_fixed_triple(self) -> None:
        first = run_trust_escalation_probe(mode="escalation", n_benign=7, seed=3)
        second = run_trust_escalation_probe(mode="escalation", n_benign=7, seed=3)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
