"""Run shared inference and verification on a supplied primitive DSL program."""

import sys

from synthesis.entry.synthesize_cfg import main

if __name__ == "__main__":
    raise SystemExit(main(["--mode", "verify", *sys.argv[1:]]))
