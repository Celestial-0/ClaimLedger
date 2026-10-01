"""Backward-compatible wrapper for `claimledger run-experiment`."""

import sys
from claimledger.cli import main

if __name__ == "__main__":
    main(["run-experiment"] + sys.argv[1:])
