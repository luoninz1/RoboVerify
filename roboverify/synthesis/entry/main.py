"""Compatibility shim for the instrumented relational CFG synthesis driver."""

from synthesis.entry.synthesize_cfg import main

if __name__ == "__main__":
    raise SystemExit(main())
