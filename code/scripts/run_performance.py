"""Backward-compatible wrapper for `claimledger perf`."""

import sys
from claimledger.cli import main

if __name__ == "__main__":
    main(["perf"] + sys.argv[1:])
