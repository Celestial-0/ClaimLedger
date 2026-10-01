"""Adversarial and stress-case generators."""

from .adaptive import AdaptiveAttackReport, generate_policy_aware_attacks
from .trust_escalation import (
    TrustEscalationOutcome,
    run_trust_escalation_probe,
    sweep_trust_escalation,
)

__all__ = [
    "AdaptiveAttackReport",
    "generate_policy_aware_attacks",
    "TrustEscalationOutcome",
    "run_trust_escalation_probe",
    "sweep_trust_escalation",
]
