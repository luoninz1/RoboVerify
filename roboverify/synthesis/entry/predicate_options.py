"""Shared configuration for interpreting observed block heights."""

from synthesis.util.on import DEFAULT_HIGHER_TOLERANCE, validate_higher_tolerance


def add_predicate_options(parser):
    parser.add_argument(
        "--higher-tolerance",
        type=validate_higher_tolerance,
        default=DEFAULT_HIGHER_TOLERANCE,
        metavar="METRES",
        help="Higher height tolerance in metres (default 0.001; 0 uses exact ordering)",
    )
