"""Shared explicit opt-in bounds for motion verification entry points."""

from synthesis.verification_lib.bmc_lib import NoiseSpec


def add_motion_options(parser):
    parser.add_argument(
        "--motion-noise",
        type=float,
        nargs=3,
        default=None,
        metavar=("GRASP", "MOVE", "RELEASE"),
        help="Opt into per-axis bounded errors in metres; omitted means noiseless.",
    )
    parser.add_argument(
        "--motion-timeout-ms",
        type=int,
        default=5000,
        help="Timeout for each motion obligation (unknown fails closed).",
    )


def motion_noise_from_args(parser, args):
    if args.motion_timeout_ms <= 0:
        parser.error("--motion-timeout-ms must be positive")
    if args.motion_noise is None:
        return None
    try:
        return NoiseSpec(*args.motion_noise)
    except ValueError as error:
        parser.error(str(error))
