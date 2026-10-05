"""Instrumented CFG synthesis; synthesized candidates still require verification."""

import argparse
import random
import signal
from functools import partial

from synthesis.api.instructions import LoopBudgetExceeded, While
from synthesis.api.program import Program, generate_random_program
from synthesis.cfg.artifacts import describe_cfg
from synthesis.cfg.bindings import (
    close_objects,
    mutate_ids,
    mutate_scoped,
    require_closed,
    require_ids,
    seed_from_scope,
)
from synthesis.cfg.candidate_traces import prepare_candidate
from synthesis.cfg.collection import execution_deadline
from synthesis.cfg.demos import DemoSegment
from synthesis.cfg.execute import execute_cfg
from synthesis.cfg.graph import RelationalCFG
from synthesis.cfg.lower import lower, lower_region
from synthesis.cfg.program_adapter import program_to_cfg
from synthesis.cfg.program_source import load_program
from synthesis.cfg.recordings import load_traces
from synthesis.cfg.region import BlockRegion, LoopRegion
from synthesis.cfg.straightline import (
    SearchBudget,
    postcondition_reached,
    segment_rollout,
    straight_line_synthesize,
)
from synthesis.cfg.tasks import task_identity, task_spec
from synthesis.cfg.verified_synthesis import verified_synthesis
from synthesis.entry.inference_options import add_inference_options
from synthesis.entry.motion_options import add_motion_options, motion_noise_from_args
from synthesis.entry.predicate_options import add_predicate_options
from synthesis.experiment.run_logger import RunLogger
from synthesis.inference_lib.minimization import using_invariant_minimizer
from synthesis.mcmc.synthesis import (
    make_roboverify_env,
    preserved_global_rng,
    set_np_seed,
)
from synthesis.predicates.language import Language
from synthesis.util.on import DEFAULT_HIGHER_TOLERANCE, using_higher_tolerance
from synthesis.verification_lib.bmc_lib import NoiseSpec
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext


def run(args, logger):
    with using_invariant_minimizer(
        getattr(args, "invariant_minimizer", None)
    ), using_higher_tolerance(
        getattr(args, "higher_tolerance", DEFAULT_HIGHER_TOLERANCE)
    ):
        return _run(args, logger)


def _run(args, logger):
    context = HighLevelContext(use_tbl=args.task == "unstack")
    traces = load_traces(args.demos, require_valid=True)
    if any(t.metadata.get("task_spec") != task_identity(args.task) for t in traces):
        raise ValueError(
            "Demonstration specification differs from the requested task; recollect demonstrations"
        )
    if any(t.task != args.task or t.num_blocks != args.num_blocks for t in traces):
        raise ValueError(
            "Demonstration task/block count differs from requested synthesis"
        )
    segments = [
        DemoSegment(
            i,
            0,
            len(t.states) - 1,
            t,
            {
                **t.metadata["initial_bindings"],
                **{f"o{j}": j for j in range(t.num_blocks)},
                **({"tbl": "tbl"} if context.use_tbl else {}),
            },
        )
        for i, t in enumerate(traces)
    ]
    pre, post = task_spec(args.task)
    from synthesis.cfg.demo_validation import validate_demonstrations

    validation = validate_demonstrations(segments, pre, post)
    demo_checks = validation.as_dict()
    logger.log_event(
        "demo_validation", f"Task predicate checks: {demo_checks}", force=True
    )
    if not validation:
        logger.finish(
            "invalid_demonstrations",
            demonstration_checks=demo_checks,
            formal_verification="not_run",
            reason="Provide recordings satisfying the initial and final task conditions",
        )
        return 2
    cfg = RelationalCFG.initial(
        segments, pre, post, ("b0", "tbl") if context.use_tbl else ("b0",)
    )
    if args.mode == "verify":
        if args.task != "stack":
            raise ValueError("The supplied-program mode currently supports Stack")
        definition = load_program(args.program, context, args.num_blocks)
        if any(
            t.metadata.get("fingerprint") != definition.metadata["fingerprint"]
            for t in traces
        ):
            raise ValueError(
                "Provided program differs from the demonstration source; recollect demonstrations"
            )
        cfg = program_to_cfg(definition, segments, pre, post)
    cfg.synthesis_approach = args.synthesis_approach
    cfg._task_demos = list(segments)
    budget = SearchBudget(
        iterations=args.iterations,
        cem_iterations=args.cem_iterations,
        cem_samples=args.cem_samples,
        cem_elites=args.cem_elites,
        theta=args.theta,
        distance=args.distance,
        seed=args.seed,
        inner_samples=args.inner_samples,
        final_samples=args.final_samples,
        pool_limit=args.pool_limit,
    )

    def env_factory(trace):
        return make_roboverify_env(trace.task, num_blocks=trace.num_blocks)

    rollout = partial(
        segment_rollout, env_factory=env_factory, reset_mode=args.reset_mode
    )

    def realize(node, demos, post, penalty=None):
        if isinstance(node.region, LoopRegion):
            physical = Program(
                len(lower_region(node.region, context, physical=True)),
                lower_region(node.region, context, physical=True),
            )

            def bound_loops(instructions):
                for instruction in instructions:
                    if isinstance(instruction, While):
                        instruction.max_iters = min(
                            args.max_loop_iterations,
                            (
                                instruction.max_iters
                                if instruction.max_iters is not None
                                else args.max_loop_iterations
                            ),
                        )
                        bound_loops(instruction.body)

            # Bound only the lowered execution copy, not the returned program.
            bound_loops(physical.instructions)
            ok = True
            try:
                for segment in demos:
                    with execution_deadline(args.trajectory_timeout_seconds):
                        ok = postcondition_reached(
                            rollout(physical, segment), segment, post
                        )
                    if not ok:
                        break
            except LoopBudgetExceeded:
                ok = False
            return node.region, ok
        id_first = node.synthesis_approach == "id-first"
        if isinstance(node.region, BlockRegion) and node.region.physical:
            initial = Program(len(node.region.physical), list(node.region.physical))
        elif not id_first and isinstance(node.region, BlockRegion):
            initial = seed_from_scope(
                args.slots, node.available_scope, random.Random(args.seed)
            )
        else:
            initial = generate_random_program(
                args.slots, range(args.num_blocks), random.Random(args.seed)
            )

        def propose(candidate, rng):
            with preserved_global_rng():
                set_np_seed(int(rng.integers(2**31)))
                return (
                    mutate_ids(candidate, args.num_blocks)
                    if id_first
                    else mutate_scoped(candidate, node.available_scope)
                )

        if id_first:
            require_ids(initial.instructions, args.num_blocks)
        else:
            initial = close_objects(
                initial,
                demos,
                node.available_scope,
                context,
                prefix="object_" + node.name.replace(".", "_"),
            )

        result = straight_line_synthesize(
            demos,
            post,
            initial,
            propose,
            rollout=rollout,
            budget=budget,
            logger=logger,
            block_id=node.name,
            penalty=penalty,
        )
        logger.log_event(
            "block_result",
            f"{node.name}: {result.status}",
            force=True,
            block_id=node.name,
            search_approach=node.synthesis_approach,
            distance=result.distance,
            post_score=result.post_score,
            elapsed=result.elapsed,
            postscore_seconds=result.postscore_seconds,
        )
        exports = (
            frozenset()
            if id_first
            else (
                require_closed(result.program.instructions, node.available_scope)
                - node.available_scope
            )
        )
        # No unsupported relational summary is invented for arbitrary physical code.
        return (
            BlockRegion(
                node.region.symbolic if isinstance(node.region, BlockRegion) else None,
                tuple(result.program.instructions),
                exports,
            ),
            result.ok,
        )

    quotient_fn = None
    if args.quotient:
        from synthesis.cfg.quotient import quotient

        quotient_fn = partial(
            quotient, language=Language(timeout_seconds=args.predicate_seconds)
        )
    extra_paths = iter(args.additional_demos)

    def provide_demos(request):
        path = next(extra_paths, None)
        if path is None:
            return None
        extra = load_traces(path, require_valid=True)
        if any(
            t.task != args.task
            or t.num_blocks != args.num_blocks
            or t.metadata.get("task_spec") != task_identity(args.task)
            for t in extra
        ):
            raise ValueError(
                "Additional demonstrations must use the same task, specification, and block count"
            )
        return [
            DemoSegment(
                i,
                0,
                len(t.states) - 1,
                t,
                {
                    **t.metadata["initial_bindings"],
                    **getattr(cfg, "_initial_bindings", {}),
                    **getattr(cfg, "_fixed_id_bindings", {}),
                    **{f"o{j}": j for j in range(t.num_blocks)},
                    **({"tbl": "tbl"} if context.use_tbl else {}),
                },
            )
            for i, t in enumerate(extra, start=len(cfg._task_demos))
        ]

    def repair(node, demos, post, penalty):
        region, ok = realize(node, demos, post, penalty=penalty)
        return region if ok else None

    result = verified_synthesis(
        cfg,
        realize,
        partial(
            execute_cfg,
            context=context,
            env_factory=env_factory,
            reset_mode=args.reset_mode,
        ),
        context,
        quotient=quotient_fn,
        initial_candidate=args.mode == "verify",
        prepare=lambda candidate, revision: prepare_candidate(
            candidate,
            context,
            relations=args.invariant_relations,
            variables=args.invariant_variables,
            max_loop_iterations=args.max_loop_iterations,
            timeout_seconds=args.trajectory_timeout_seconds,
            logger=logger,
            revision=revision,
        ),
        demo_provider=provide_demos,
        repair_motion=repair,
        # Concrete IDs assume the demonstrated universe exists. Checking a
        # smaller universe can make fixed-alias premises inconsistent. The
        # existing verifier still requests its unbounded proof afterward.
        min_blocks=args.num_blocks if args.synthesis_approach == "id-first" else 2,
        max_blocks=(
            max(4, args.num_blocks) if args.synthesis_approach == "id-first" else 4
        ),
        symbolic_iterations=args.symbolic_iterations,
        motion_iterations=args.motion_iterations,
        timeout_ms=args.verification_timeout_ms,
        relations=args.invariant_relations,
        variables=args.invariant_variables,
        motion_options={
            "initial_arm": args.initial_arm,
            "noise": (
                NoiseSpec(*args.motion_noise) if args.motion_noise is not None else None
            ),
            "timeout_ms": args.motion_timeout_ms,
            "table_surface_height": args.table_surface_height,
            "supported_tower_model": args.supported_towers,
        },
        max_refinements=args.refinements,
        language=Language(timeout_seconds=args.predicate_seconds),
        logger=logger,
    )
    import json

    logger.write_artifact("cfg.json", json.dumps(describe_cfg(cfg), indent=2))
    if result:
        program = lower(cfg, context, physical=True)
        logger.write_artifact("program.txt", str(program))
    logger.finish(
        result.status,
        stages=result.history,
        reason=result.reason,
        formal_verification=result.status,
        symbolic=str(result.symbolic),
        motion=str(result.motion),
        demonstration_checks=demo_checks,
        task=args.task,
    )
    return 0 if result else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("stack", "unstack"), default="stack")
    parser.add_argument("--mode", choices=("full", "verify"), default="full")
    parser.add_argument("--program", help="DSL factory required by --mode verify")
    parser.add_argument(
        "--synthesis-approach",
        choices=("relational", "id-first"),
        default="relational",
        help="relational: bind names before search; id-first: search/refine IDs, then quotient and return ByName",
    )
    parser.add_argument("--max-loop-iterations", type=int, default=100)
    parser.add_argument("--trajectory-timeout-seconds", type=float, default=60)
    parser.add_argument(
        "--demos",
        required=True,
        help="Validated archive produced by synthesis.entry.collect_demos",
    )
    parser.add_argument(
        "--output-dir", default="runs", help="Directory for experiment results"
    )
    parser.add_argument(
        "--run-name", default=None, help="Optional readable label for this experiment"
    )
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--refinements", type=int, default=5)
    parser.add_argument("--cem-iterations", type=int, default=2)
    parser.add_argument("--cem-samples", type=int, default=8)
    parser.add_argument("--cem-elites", type=int, default=2)
    parser.add_argument("--inner-samples", type=int, default=128)
    parser.add_argument("--final-samples", type=int, default=2048)
    parser.add_argument("--pool-limit", type=int, default=10)
    parser.add_argument("--theta", type=float, default=1.0)
    parser.add_argument(
        "--distance", choices=("kl_kde", "kl_gmm", "mmd"), default="kl_kde"
    )
    parser.add_argument("--reset-mode", choices=("reset", "replay"), default="replay")
    parser.add_argument("--predicate-seconds", type=float, default=5.0)
    parser.add_argument(
        "--quotient", action="store_true", help="Enable conservative flat-loop folding"
    )
    parser.add_argument(
        "--invariant-relations",
        nargs="+",
        choices=("ON_star", "ON_star_zero", "Higher", "Scattered", "equality"),
    )
    parser.add_argument("--invariant-variables", type=int, default=2)
    add_inference_options(parser)
    add_motion_options(parser)
    add_predicate_options(parser)
    parser.add_argument(
        "--table-surface-height",
        type=float,
        help="Physical table height required for table-placement contracts",
    )
    parser.add_argument(
        "--supported-towers",
        action="store_true",
        help="Use complete uniform towers on a common table; check height, arm-clearance and column invariants",
    )
    parser.add_argument(
        "--additional-demos",
        nargs="*",
        default=[],
        help="Recordings supplied for verification-requested resynthesis",
    )
    parser.add_argument("--symbolic-iterations", type=int, default=10)
    parser.add_argument("--motion-iterations", type=int, default=10)
    parser.add_argument("--verification-timeout-ms", type=int, default=5000)
    parser.add_argument(
        "--initial-arm",
        type=float,
        nargs=3,
        help="Explicit initial arm position for the geometric verification model",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    motion_noise_from_args(parser, args)
    if args.synthesis_approach == "id-first":
        args.quotient = True
    if args.mode == "verify" and not args.program:
        parser.error("--mode verify requires --program module:factory")
    if args.max_loop_iterations < 1 or not 0 < args.trajectory_timeout_seconds < float(
        "inf"
    ):
        parser.error("Execution budgets must be positive and finite")
    if args.smoke:
        args.iterations, args.cem_iterations, args.cem_samples, args.cem_elites = (
            2,
            1,
            4,
            2,
        )
        args.inner_samples, args.final_samples, args.refinements = 32, 128, 1
    if (
        args.num_blocks < 2
        or args.slots < 1
        or args.iterations < 0
        or args.refinements < 0
        or args.symbolic_iterations < 0
        or args.motion_iterations < 0
        or args.verification_timeout_ms <= 0
        or args.invariant_variables < 1
    ):
        parser.error("Invalid block, slot, or iteration budget")

    def deadline(signum, frame):
        raise TimeoutError("Unstack end-to-end 60-second budget exhausted")

    previous = signal.signal(signal.SIGALRM, deadline)
    with RunLogger(
        args.output_dir,
        "cfg",
        vars(args),
        slug=args.run_name or f"{args.task}-{args.mode}",
    ) as logger:
        logger.progress_line(f"CFG run: {logger.run_dir}")
        if args.task == "unstack":
            signal.alarm(60)
        try:
            code = run(args, logger)
        except TimeoutError as exc:
            logger.finish("budget_exhausted", reason=str(exc))
            code = 2
        except Exception as exc:
            logger.log_exception(exc)
            logger.finish("failed", reason=str(exc))
            raise
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
    print(f"Read: uv run python -m synthesis.experiment.report --run {logger.run_dir}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
