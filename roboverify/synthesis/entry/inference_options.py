"""Shared command-line options for partition-based invariant inference."""

from synthesis.inference_lib.minimization import (
    DEFAULT_INVARIANT_MINIMIZER,
    INVARIANT_MINIMIZERS,
)


def add_inference_options(parser):
    parser.add_argument(
        "--invariant-minimizer",
        choices=INVARIANT_MINIMIZERS,
        default=DEFAULT_INVARIANT_MINIMIZER,
        help="Truth-table minimizer: sympy (default) or pyeda (Espresso)",
    )
