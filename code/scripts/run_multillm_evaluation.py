"""Backward-compatible wrapper for `claimledger multillm`."""

import sys
from claimledger.cli import main

if __name__ == "__main__":
    main(["multillm"] + sys.argv[1:])
