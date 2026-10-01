"""Backward-compatible wrapper for `claimledger extraction-sweep`."""

import sys
from claimledger.cli import main

if __name__ == "__main__":
    main(["extraction-sweep"] + sys.argv[1:])
