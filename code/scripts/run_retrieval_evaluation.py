"""Backward-compatible wrapper for `claimledger retrieval`."""

import sys
from claimledger.cli import main

if __name__ == "__main__":
    main(["retrieval"] + sys.argv[1:])
