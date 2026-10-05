"""Learn an invariant from generated counterexample executions of a fixed program."""

import argparse
import json
from dataclasses import asdict

from synthesis.cfg.collection import VideoRecorder
from synthesis.entry.inference_options import add_inference_options
from synthesis.entry.predicate_options import add_predicate_options
from synthesis.experiment.invariant_learning.runner import (
    ExperimentConfig,
    run_experiment,
)
from synthesis.experiment.invariant_learning.tasks import TASKS
from synthesis.experiment.run_logger import RunLogger
from synthesis.util.on import using_higher_tolerance


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASKS), default="stack")
    parser.add_argument(
        "--program",
        required=True,
        help="module:factory or path.py:factory; fixed across block counts",
    )
    parser.add_argument(
        "--verification-level", choices=("symbolic", "both"), default="symbolic"
    )
    parser.add_argument(
        "--max-counterexample-blocks",
        type=int,
        default=4,
        help="Search sizes 1..N; the final symbolic proof is unbounded",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=10,
        help="Maximum learner updates; includes one final verification attempt",
    )
    parser.add_argument(
        "--max-loop-iterations",
        type=int,
        help="Maximum total loop bodies, including a failing body; Stack defaults to num_blocks-1",
    )
    parser.add_argument("--verification-timeout-ms", type=int, default=10000)
    parser.add_argument("--trajectory-timeout-seconds", type=float, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--invariant-relations",
        nargs="+",
        choices=("ON_star", "ON_star_zero", "Higher", "Scattered", "equality"),
        default=["ON_star", "Higher", "Scattered", "equality"],
    )
    parser.add_argument("--invariant-variables", type=int, default=2)
    add_inference_options(parser)
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Record generated executions at 20 FPS",
    )
    parser.add_argument("--output-dir", default="runs", help="Experiment results root")
    parser.add_argument("--run-name", help="Optional readable experiment label")
    parser.add_argument("--supported-towers", action="store_true")
    parser.add_argument("--table-surface-height", type=float)
    parser.add_argument("--initial-arm", type=float, nargs=3)
    parser.add_argument("--motion-timeout-ms", type=int, default=10000)
    parser.add_argument(
        "--motion-noise", type=float, nargs=3, metavar=("GRASP", "MOVE", "RELEASE")
    )
    add_predicate_options(parser)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = ExperimentConfig(
            **{
                name: getattr(args, name)
                for name in ExperimentConfig.__dataclass_fields__
            }
        )
        if args.max_loop_iterations is not None and args.max_loop_iterations < 0:
            raise ValueError("--max-loop-iterations must be nonnegative")
        if args.invariant_variables < 0 or args.motion_timeout_ms <= 0:
            raise ValueError(
                "Invariant variable count must be nonnegative; motion timeout must be positive"
            )
        if args.save_video:
            VideoRecorder.check_available()
        with using_higher_tolerance(args.higher_tolerance):
            task = TASKS[args.task](
                args.program,
                relations=args.invariant_relations,
                variables=args.invariant_variables,
                max_loop_iterations=args.max_loop_iterations,
            )
        from synthesis.verification_lib.motion_verification import NoiseSpec

        motion_options = dict(
            supported_tower_model=args.supported_towers,
            table_surface_height=args.table_surface_height,
            initial_arm=args.initial_arm,
            timeout_ms=args.motion_timeout_ms,
            noise=(
                NoiseSpec(*args.motion_noise) if args.motion_noise is not None else None
            ),
        )
    except (ValueError, TypeError, ImportError, AttributeError) as exc:
        parser.error(str(exc))
    logger = RunLogger(
        args.output_dir, "invariant-learning", vars(args), slug=args.run_name
    )
    try:
        with using_higher_tolerance(args.higher_tolerance):
            result = run_experiment(
                task, config, motion_options=motion_options, logger=logger
            )
        fields = asdict(result)
        status = fields.pop("status")
        logger.finish(status, **fields)
        print(
            json.dumps(
                dict(status=status, run=str(logger.run_dir), reason=result.reason)
            )
        )
        return 0 if result else 2
    except Exception as exc:
        logger.log_exception(exc)
        logger.finish("error", reason=f"{type(exc).__name__}: {exc}")
        print(f"Experiment failed: {exc}; see {logger.run_dir}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
