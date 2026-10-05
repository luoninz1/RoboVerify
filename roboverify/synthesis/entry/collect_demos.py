"""Collect DSL demonstrations after 50 settling steps, with optional 20 FPS videos."""

import argparse
import json
from pathlib import Path

from synthesis.cfg.collection import VideoRecorder, record_execution, validate_trace
from synthesis.cfg.program_source import load_program
from synthesis.cfg.recordings import save_traces
from synthesis.cfg.tasks import task_identity
from synthesis.entry.predicate_options import add_predicate_options
from synthesis.util.on import using_higher_tolerance
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext


def resolve_seeds(args):
    if args.seeds is not None:
        if args.seed_start is not None:
            raise ValueError("Use --seeds or --seed-start, not both")
        if (
            args.num_trajectories is not None
            and len(args.seeds) != args.num_trajectories
        ):
            raise ValueError(
                "--num-trajectories must match the number of explicit seeds"
            )
        seeds = list(args.seeds)
    else:
        count = 5 if args.num_trajectories is None else args.num_trajectories
        first = 0 if args.seed_start is None else args.seed_start
        seeds = list(range(first, first + count))
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(s < 0 or s >= 2**32 for s in seeds)
    ):
        raise ValueError(
            "Supply a positive trajectory count and distinct seeds in [0, 2**32)"
        )
    return seeds


def output_directory(args, count):
    if args.output_dir:
        path = Path(args.output_dir)
        path.mkdir(parents=True, exist_ok=False)
        return path
    base = Path("demos") / args.task / f"{args.num_blocks}-blocks-{count}-trajectories"
    index = 1
    while True:
        path = base if index == 1 else base.with_name(base.name + f"-{index}")
        try:
            path.mkdir(parents=True, exist_ok=False)
            return path
        except FileExistsError:
            index += 1


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--program", required=True, help="module:factory or path/to/program.py:factory"
    )
    parser.add_argument("--task", choices=["stack"], default="stack")
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument(
        "--num-trajectories",
        type=int,
        help="Number of distinct seed runs (5 when omitted; explicit seeds set the count)",
    )
    parser.add_argument(
        "--seed-start", type=int, help="First consecutive seed (0 when omitted)"
    )
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument(
        "--output-dir",
        help="New collection directory; otherwise demos/stack/<blocks>-blocks-<count>-trajectories",
    )
    parser.add_argument(
        "--save-video", action="store_true", help="Save one MP4 per seed at 20 FPS"
    )
    parser.add_argument(
        "--render", action="store_true", help="Show a live simulator window"
    )
    parser.add_argument("--max-loop-iterations", type=int, default=100)
    parser.add_argument("--trajectory-timeout-seconds", type=float, default=60)
    add_predicate_options(parser)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    with using_higher_tolerance(args.higher_tolerance):
        return collect(args, parser)


def collect(args, parser):
    try:
        seeds = resolve_seeds(args)
        if (
            args.num_blocks < 2
            or args.max_loop_iterations < 1
            or not 0 < args.trajectory_timeout_seconds < float("inf")
        ):
            raise ValueError(
                "Block count must be >=2 and execution budgets must be positive and finite"
            )
        if args.save_video:
            VideoRecorder.check_available()
        context = HighLevelContext(mode="declare")
        reference = load_program(args.program, context, args.num_blocks)
        output = output_directory(args, len(seeds))
    except (ValueError, FileExistsError, ImportError, AttributeError, TypeError) as exc:
        parser.error(str(exc))
    report = dict(
        task=task_identity(args.task),
        program=reference.metadata,
        seeds=seeds,
        num_blocks=args.num_blocks,
        higher_tolerance=args.higher_tolerance,
        trajectories=[],
        status="collecting",
    )
    accepted = []
    for seed in seeds:
        # Rebuild for every seed; reject factories that change the executable.
        try:
            definition = load_program(args.program, context, args.num_blocks)
            if definition.metadata["fingerprint"] != reference.metadata["fingerprint"]:
                raise ValueError("Program factory changed between seeds")
            trace = record_execution(
                definition,
                seed=seed,
                task=args.task,
                num_blocks=args.num_blocks,
                max_loop_iterations=args.max_loop_iterations,
                timeout_seconds=args.trajectory_timeout_seconds,
                video_path=(
                    output / "videos" / f"seed_{seed:04d}.mp4"
                    if args.save_video
                    else None
                ),
                render=args.render,
            )
        except Exception as exc:
            from synthesis.cfg.demos import DemoTrace

            trace = DemoTrace(
                (),
                seed=seed,
                task=args.task,
                num_blocks=args.num_blocks,
                metadata=dict(
                    reference.metadata,
                    status="failed",
                    reason=f"{type(exc).__name__}: {exc}",
                ),
            )
        trace.metadata["task_spec"] = task_identity(args.task)
        valid = validate_trace(trace)
        if valid:
            accepted.append(trace)
        elif trace.states:
            diagnostics = output / "diagnostics"
            diagnostics.mkdir(exist_ok=True)
            save_traces(diagnostics / f"seed_{seed:04d}.npz", [trace])
        row = dict(seed=seed, observations=len(trace.states), **trace.metadata)
        if "video" in row:
            row["video"] = dict(
                row["video"], path=str(Path(row["video"]["path"]).relative_to(output))
            )
        report["trajectories"].append(row)
        (output / "collection.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"seed {seed}: {trace.metadata['status']}; {trace.metadata['reason']}",
            flush=True,
        )
    success = len(accepted) == len(seeds)
    if success:
        save_traces(output / "demonstrations.npz", accepted)
    else:
        # Preserve successful trajectories too when the requested batch is rejected.
        diagnostics = output / "diagnostics"
        diagnostics.mkdir(exist_ok=True)
        for trace in accepted:
            save_traces(diagnostics / f"seed_{trace.seed:04d}.npz", [trace])
    video_failed = any(
        r.get("video", {}).get("status") == "failed" or r.get("render_error")
        for r in report["trajectories"]
    )
    report["status"] = "valid" if success else "invalid_demonstrations"
    report["media_status"] = "failed" if video_failed else "complete"
    (output / "collection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Collection: {output}")
    if success:
        import shlex

        archive = shlex.quote(str(output / "demonstrations.npz"))
        common = f"uv run python -m synthesis.entry.synthesize_cfg --task stack --num-blocks {args.num_blocks} --demos {archive} --higher-tolerance {args.higher_tolerance}"
        print(f"Full pipeline: {common} --mode full --quotient")
        print(
            f"Verify program: {common} --mode verify --program {shlex.quote(args.program)}"
        )
    return 0 if success and not video_failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
