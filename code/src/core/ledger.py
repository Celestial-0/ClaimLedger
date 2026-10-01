"""Public ledger service boundary exposing the default persistence backend."""

from ..storage.sqlite import ClaimLedger

__all__ = ["ClaimLedger"]
