"""Backward-compatible wrapper for `claimledger prepare-realtalk`."""

import sys
from claimledger.cli import main

if __name__ == "__main__":
    main(["prepare-realtalk"] + sys.argv[1:])
