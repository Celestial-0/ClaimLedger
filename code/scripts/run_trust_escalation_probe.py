"""Backward-compatible wrapper for `claimledger trust-probe`."""

import sys
from claimledger.cli import main

if __name__ == "__main__":
    main(["trust-probe"] + sys.argv[1:])
