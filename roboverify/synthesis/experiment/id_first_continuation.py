"""Manual Stack continuation experiment with supplied MCMC placement candidates.

Run from roboverify/ with the simulator environment configured. Only straight-line
search is replaced: predicate learning, resets, execution, CFG refinement,
quotienting, and candidate replay use their actual implementations. A separate,
explicitly labeled diagnostic calls quotient on complete placement fragments
even if the normal continuation failed. Rejected folds are replayed for diagnosis,
never marked as accepted synthesis results or formal verification.
"""

import argparse
import json
from copy import deepcopy
from functools import partial
from unittest.mock import patch

from synthesis.api.instructions import Move, Pick, Release, Skip, While
from synthesis.api.program import Program
from synthesis.cfg.collection import record_execution, validate_trace
from synthesis.cfg.demo_validation import validate_demonstrations
from synthesis.cfg.demos import DemoSegment
from synthesis.cfg.execute import execute_cfg
from synthesis.cfg.graph import RelationalCFG
from synthesis.cfg.id_first import close_id_candidate
from synthesis.cfg.lower import lower, lower_region
from synthesis.cfg.program_source import (
    ProgramDefinition,
    describe_program,
    load_program,
)
from synthesis.cfg.quotient import _quotient_label, find_repetition, quotient
from synthesis.cfg.recordings import load_traces, save_traces
from synthesis.cfg.refine import refine_cfg, scene_at, transition_witnesses
from synthesis.cfg.region import BlockRegion, LoopRegion
from synthesis.cfg.straightline import postcondition_reached, segment_rollout
from synthesis.cfg.synthesize import synthesize_cfg
from synthesis.cfg.tasks import task_identity, task_spec
from synthesis.cfg.validate import validate_cfg
from synthesis.examples.stack import TRANSFER_HEIGHT
from synthesis.experiment.run_logger import RunLogger
from synthesis.mcmc.synthesis import make_roboverify_env
from synthesis.predicates.guard import loop_guard_synthesis
from synthesis.predicates.language import Language
from synthesis.predicates.scene import evaluate
from synthesis.predicates.term import atom, block_id, ref
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext


def placement(source, target):
    return Program(
        5,
        [
            Pick(source),
            Move(source, source, target, target_offset=[0, 0, TRANSFER_HEIGHT]),
            Move(0, 0, target, target_offset=[0, 0, TRANSFER_HEIGHT]),
            Move(0, 0, target, target_offset=[0, 0, 0.05]),
            Release(source, target_z=0.15),
        ],
    )


def graph_summary(cfg):
    return dict(
        order=list(cfg.order),
        edges=[
            dict(
                source=e.source,
                target=e.target,
                label=str(e.label),
                binds=sorted(e.binds),
            )
            for e in cfg.edges
        ],
        segments={
            name: [
                dict(seed=s.trace.seed, start=s.t_start, end=s.t_end)
                for s in cfg.demos.for_node(name)
            ]
            for name in cfg.order
        },
    )


def run(args, logger):
    context = HighLevelContext()
    expected = load_program(
        "synthesis.examples.stack:build_program", context, args.num_blocks
    )
    traces = load_traces(args.demos, require_valid=True)
    if any(
        t.task != "stack"
        or t.num_blocks != args.num_blocks
        or t.metadata.get("task_spec") != task_identity("stack")
        or t.metadata["initial_bindings"].get("b0") != 0
        for t in traces
    ):
        raise ValueError(
            f"This experiment requires validated {args.num_blocks}-block Stack demos with b0 bound to ID 0"
        )
    if any(
        t.metadata.get("fingerprint") != expected.metadata["fingerprint"]
        for t in traces
    ):
        raise ValueError(
            "Use demonstrations from the current synthesis.examples.stack:build_program; this experiment supplies that program's placement fragments"
        )
    rows = [
        DemoSegment(i, 0, len(t.states) - 1, t, dict(t.metadata["initial_bindings"]))
        for i, t in enumerate(traces)
    ]
    pre, post = task_spec("stack")
    if not validate_demonstrations(rows, pre, post):
        raise ValueError("Demonstrations fail the shared task specification")
    language = Language(timeout_seconds=args.predicate_seconds)
    env_factory = lambda t: make_roboverify_env(t.task, num_blocks=t.num_blocks)
    execute = partial(
        execute_cfg, context=context, env_factory=env_factory, reset_mode="replay"
    )
    rollout = partial(segment_rollout, env_factory=env_factory, reset_mode="replay")
    cfg = RelationalCFG.initial(rows, pre, post, {"b0"}, synthesis_approach="id-first")
    cfg._task_demos = rows
    summary = dict(
        num_blocks=args.num_blocks,
        seeds=[t.seed for t in traces],
        mcmc="supplied candidates, not searched",
        formal_verification="not_run",
        demonstrations=[
            dict(
                seed=t.seed,
                observations=len(t.states),
                heads=[
                    dict(index=e["index"], bindings=e["bindings"])
                    for e in t.events
                    if e["kind"] == "loop_head"
                ],
            )
            for t in traces
        ],
    )

    def save():
        logger.write_artifact("summary.json", json.dumps(summary, indent=2))

    # Establish the starting milestone using real refinement and a failed no-op.
    cfg.nodes["v0"].region = BlockRegion(None, (Skip(0),))
    first = refine_cfg(cfg, "v0", execute(cfg), {"b0"}, language=language)
    summary["first_refinement"] = dict(
        status=first.status, predicate=str(first.term), graph=graph_summary(cfg)
    )
    save()
    if not first or str(first.term) != "ON(1, b0)":
        raise ValueError(
            f"Starting milestone was not recovered: {first.status}: {first.term}"
        )
    first_name, tail_name = cfg.order
    placements = [placement(i, i - 1) for i in range(1, args.num_blocks)]
    remaining = [inst for program in placements[1:] for inst in program.instructions]
    candidates = {
        first_name: placements[0],
        tail_name: Program(len(remaining), remaining),
    }

    def check(program, segments, target):
        results = []
        for segment in segments:
            states = rollout(program, segment)
            results.append(
                dict(
                    seed=segment.trace.seed,
                    reached=postcondition_reached(states, segment, target),
                    at_end=evaluate(target, states[-1]),
                    observations=len(states),
                )
            )
        return results

    first_checks = check(
        candidates[first_name], cfg.demos.for_node(first_name), first.term
    )
    summary["supplied_first_placement"] = dict(
        program=describe_program(candidates[first_name]), checks=first_checks
    )
    if not all(r["at_end"] for r in first_checks):
        save()
        raise ValueError(
            "Supplied first placement did not satisfy ON(1, b0) on every seed"
        )
    cfg.nodes[first_name].region = BlockRegion(
        None, tuple(candidates[first_name].instructions)
    )
    logger.log_event(
        "first_placement",
        "Supplied numeric placement achieves ON(1, b0)",
        force=True,
        successes=len(first_checks),
    )

    boundaries = []
    released_rows = []
    for segment in cfg.demos.for_node(tail_name):
        release_end = next(
            e["index"]
            for e in segment.trace.events
            if e["kind"] == "instruction_end" and e["path"] == "1.4"
        )
        scene = scene_at(segment, segment.t_start)
        boundaries.append(
            dict(
                seed=segment.trace.seed,
                learned_cut=segment.t_start,
                first_release_end=release_end,
                height_gap_at_cut=float(scene.positions[1][2] - scene.positions[0][2]),
            )
        )
        released_rows.append(
            DemoSegment(
                segment.demo_idx,
                release_end,
                segment.t_end,
                segment.trace,
                dict(segment.bindings),
            )
        )
    summary["first_boundary_diagnostics"] = boundaries
    summary["remaining_placements_after_recorded_release"] = check(
        candidates[tail_name], released_rows, post
    )

    # Branch A: the remaining MCMC search fails and leaves a no-op candidate.
    failed = deepcopy(cfg)
    failed.nodes[tail_name].region = BlockRegion(None, (Skip(0),))
    negatives = execute(failed)
    positives = transition_witnesses(failed.demos.for_node(tail_name), post)
    on21 = atom("ON", block_id(2), block_id(1))
    diagnostic = dict(
        positive_count=len(positives),
        negative_count=len(negatives),
        on21_true_positive=sum(evaluate(on21, s) for s in positives),
        on21_true_negative=sum(evaluate(on21, s) for s in negatives),
    )
    second = refine_cfg(failed, tail_name, negatives, {"b0"}, language=language)
    summary["failed_tail_refinement"] = dict(
        status=second.status,
        predicate=str(second.term),
        examined=second.examined,
        diagnostics=diagnostic,
        graph=graph_summary(failed),
    )
    logger.log_event(
        "second_refinement", f"{second.status}: {second.term}", force=True, **diagnostic
    )
    save()

    # With four blocks, ON(2, 1) can be an intermediate milestone. Follow the
    # actual refinement if it discovers that predicate; never insert it by hand.
    follow_second = args.num_blocks == 4 and bool(second) and second.term == on21
    summary["followed_second_refinement"] = follow_second
    if follow_second:
        cfg = failed
        candidates = dict(zip(cfg.order, placements))
    # Supply complete placements for each learned milestone, then let the real
    # synthesis driver decide whether their actual segment rollouts permit a fold.
    checks = {
        name: check(program, cfg.demos.for_node(name), cfg.outgoing(name)[0].label)
        for name, program in candidates.items()
    }
    summary["supplied_continuation"] = {
        name: dict(
            target=str(cfg.outgoing(name)[0].label),
            program=describe_program(program),
            checks=checks[name],
        )
        for name, program in candidates.items()
    }
    summary["continuation_graph"] = graph_summary(cfg)
    numeric_cfg = deepcopy(cfg)
    guard_searches = []
    fold_validations = []
    rejected_programs = []

    def observe_guard(*positional, **kwargs):
        result = loop_guard_synthesis(*positional, **kwargs)
        guard_searches.append(
            dict(
                status=result.status,
                predicate=str(result.term),
                examined=result.examined,
                elapsed=result.elapsed,
                positive_count=len(positional[0]),
                exit_count=len(positional[1]),
            )
        )
        return result

    def observe_validation(candidate):
        valid = validate_cfg(candidate)
        boundaries = []
        for name in candidate.order:
            incoming, outgoing = (
                candidate.incoming(name)[0],
                candidate.outgoing(name)[0],
            )
            for row in candidate.demos.for_node(name):
                start_scene, end_scene = scene_at(row, row.t_start), scene_at(
                    row, row.t_end
                )
                boundaries.append(
                    dict(
                        node=name,
                        seed=row.trace.seed,
                        start=row.t_start,
                        end=row.t_end,
                        pre_holds=evaluate(incoming.label, start_scene),
                        post_holds=evaluate(outgoing.label, end_scene),
                        first_post=next(
                            (
                                t
                                for t in range(row.t_start, row.t_end + 1)
                                if evaluate(outgoing.label, scene_at(row, t))
                            ),
                            None,
                        ),
                        end_positions={
                            str(k): list(map(float, v))
                            for k, v in end_scene.positions.items()
                        },
                        first_task_goal=next(
                            (
                                t
                                for t in range(row.t_start, len(row.trace.states))
                                if evaluate(post, scene_at(row, t))
                            ),
                            None,
                        ),
                    )
                )
        fold_validations.append(dict(valid=valid, boundaries=boundaries))
        if not valid:
            try:
                attempted = lower(candidate, context, physical=True)
            except ValueError as error:
                # A relational fold need not have a physical body seed.
                fold_validations[-1]["physical_lowering_error"] = str(error)
                return valid
            rejected_programs.append(attempted)
            logger.write_artifact(
                "rejected_quotient_program.json",
                json.dumps(describe_program(attempted), indent=2),
            )
            logger.write_artifact("rejected_quotient_program.txt", str(attempted))
        return valid

    def named_oracle(node, demos, target):
        # The supplied candidate replaces the new named MCMC call too. Keep
        # actual per-iteration execution and whole-loop acceptance checks.
        if isinstance(node.region, LoopRegion):
            instructions = lower_region(node.region, context, physical=True)
            for instruction in instructions:
                if isinstance(instruction, While):
                    instruction.max_iters = 10
            program = Program(len(instructions), instructions)
            region = node.region
        else:
            instructions = (
                list(node.region.physical)
                if node.region.physical
                else deepcopy(expected.program.instructions[1].body[:-1])
            )
            program = Program(len(instructions), instructions)
            region = BlockRegion(None, tuple(instructions))
        results = check(program, demos, target)
        summary.setdefault("post_quotient_oracle_calls", []).append(
            dict(
                node=node.name,
                target=str(target),
                search_approach=node.synthesis_approach,
                program=describe_program(program),
                checks=results,
            )
        )
        save()
        return region, all(row["reached"] for row in results)

    def realize(node, demos, target):
        if node.synthesis_approach == "relational":
            return named_oracle(node, demos, target)
        return BlockRegion(None, tuple(candidates[node.name].instructions)), all(
            r["reached"] for r in checks[node.name]
        )

    def fold(candidate):
        letters = [
            _quotient_label(candidate, n, candidate.outgoing(n)[0])
            for n in candidate.order
        ]
        repeated = find_repetition(letters)
        summary["quotient_input"] = dict(
            letters=list(map(str, letters)),
            repetition=(
                None
                if repeated is None
                else dict(
                    start=repeated.start,
                    width=repeated.width,
                    substitutions=[
                        {k: str(v) for k, v in mapping.items()}
                        for mapping in repeated.substitutions
                    ],
                )
            ),
            graph=graph_summary(candidate),
        )
        with patch(
            "synthesis.cfg.quotient.loop_guard_synthesis", side_effect=observe_guard
        ), patch("synthesis.cfg.validate.validate_cfg", side_effect=observe_validation):
            changed = quotient(candidate, language=language)
        summary["quotient"] = dict(
            changed=changed,
            guard_searches=guard_searches,
            fold_validations=fold_validations,
            graph=graph_summary(candidate),
        )
        return changed

    result = synthesize_cfg(
        cfg,
        realize,
        execute,
        quotient=fold,
        max_refinements=0,
        language=language,
        logger=logger,
    )
    summary["synthesis_status"] = result.status
    summary["synthesis_failed_block"] = result.failed_block
    save()

    def replay(program, prefix, graph=cfg):
        replays = []
        for segment in graph._task_demos:
            definition = ProgramDefinition(
                program, dict(segment.bindings), "manual ID-first continuation"
            )
            trace = record_execution(
                definition,
                seed=segment.trace.seed,
                num_blocks=args.num_blocks,
                initial_snapshot=segment.trace.snapshots[0],
                max_loop_iterations=10,
                timeout_seconds=args.trajectory_timeout_seconds,
            )
            trace.metadata["task_spec"] = task_identity("stack")
            valid = validate_trace(trace)
            save_traces(logger.artifact_dir(prefix) / f"seed-{trace.seed}.npz", [trace])
            replays.append(
                dict(
                    seed=trace.seed,
                    status=trace.metadata["status"],
                    valid=valid,
                    reason=trace.metadata["reason"],
                    validation=trace.metadata.get("validation"),
                    heads=[
                        dict(index=e["index"], bindings=e["bindings"])
                        for e in trace.events
                        if e["kind"] == "loop_head"
                    ],
                )
            )
            summary[prefix] = replays
            save()
            logger.log_event(
                "candidate_replay",
                f"{prefix}, seed {trace.seed}: {trace.metadata['status']}",
                force=True,
            )
        return replays

    all_instructions = [inst for program in placements for inst in program.instructions]
    continuous = Program(len(all_instructions), all_instructions)
    logger.write_artifact(
        "continuous_program.json", json.dumps(describe_program(continuous), indent=2)
    )
    replay(continuous, "continuous_program_replays")
    # A second control carries each pending placement across a learned cut.
    # Its concatenation is identical to the complete numeric placement program.
    # These are proposed controller boundaries; actual rollouts still decide success.
    matched = deepcopy(numeric_cfg)
    cuts = (
        [0]
        + [5 * i + 3 for i in range(len(matched.order) - 1)]
        + [len(all_instructions)]
    )
    matched_programs = {
        name: Program(end - start, all_instructions[start:end])
        for name, start, end in zip(matched.order, cuts, cuts[1:])
    }
    matched_checks = {
        name: check(
            program, matched.demos.for_node(name), matched.outgoing(name)[0].label
        )
        for name, program in matched_programs.items()
    }

    def matched_realize(node, demos, target):
        if node.synthesis_approach == "relational":
            return named_oracle(node, demos, target)
        return BlockRegion(None, tuple(matched_programs[node.name].instructions)), all(
            row["reached"] for row in matched_checks[node.name]
        )

    def matched_fold(candidate):
        letters = [
            _quotient_label(candidate, n, candidate.outgoing(n)[0])
            for n in candidate.order
        ]
        guard_start, validation_start = len(guard_searches), len(fold_validations)
        with patch("synthesis.cfg.quotient.loop_guard_synthesis", observe_guard), patch(
            "synthesis.cfg.validate.validate_cfg", observe_validation
        ):
            changed = quotient(candidate, language=language)
        summary["boundary_matched_quotient"] = dict(
            letters=list(map(str, letters)),
            changed=changed,
            guard_searches=guard_searches[guard_start:],
            fold_validations=fold_validations[validation_start:],
        )
        return changed

    matched_result = synthesize_cfg(
        matched,
        matched_realize,
        execute,
        quotient=matched_fold,
        max_refinements=0,
        language=language,
        logger=logger,
    )
    summary["boundary_matched_continuation"] = dict(
        status=matched_result.status,
        checks=matched_checks,
        instruction_counts={
            name: len(program.instructions)
            for name, program in matched_programs.items()
        },
    )
    save()
    if matched_result:
        matched_program = lower(matched, context, physical=True)
        logger.write_artifact("boundary_matched_program.txt", str(matched_program))
        logger.write_artifact(
            "boundary_matched_program.json",
            json.dumps(describe_program(matched_program), indent=2),
        )
        replay(matched_program, "boundary_matched_program_replays", matched)

    summary["quotient_reached_by_synthesis"] = "quotient" in summary
    if not result:
        # Explicit diagnostic only: this is not an automatic synthesis success.
        # Ask what quotient would produce from all supplied complete placements.
        # The driver can stop before installing a later candidate, so install it
        # only here, after recording the real failed continuation above.
        summary["quotient_mode"] = "isolated_manual_call_after_segment_failure"
        for name, candidate in candidates.items():
            cfg.nodes[name].region = BlockRegion(None, tuple(candidate.instructions))
        fold(cfg)
        close_id_candidate(cfg)
    else:
        summary["quotient_mode"] = "automatic_synthesis"
    program = lower(cfg, context, physical=True)
    logger.write_artifact("program.txt", str(program))
    logger.write_artifact(
        "program.json", json.dumps(describe_program(program), indent=2)
    )
    summary["returned_program"] = describe_program(program)
    summary["loops"] = [
        dict(
            guard=str(n.region.guard),
            init=n.region.init,
            update=n.region.update,
            exists_vars=n.region.exists_vars,
            iteration_counts=n.region.iteration_counts,
        )
        for n in cfg.nodes.values()
        if isinstance(n.region, LoopRegion)
    ]
    replays = replay(program, "returned_program_replays")
    for rejected in rejected_programs:
        summary["rejected_quotient_program"] = describe_program(rejected)
        replay(rejected, "rejected_quotient_program_replays")
    logger.finish(
        "completed",
        synthesis_status=result.status,
        quotient_reached_by_synthesis=summary["quotient_reached_by_synthesis"],
        quotient_changed=summary["quotient"]["changed"],
        replay_successes=sum(r["valid"] for r in replays),
        replay_count=len(replays),
        formal_verification="not_run",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demos", required=True)
    parser.add_argument(
        "--num-blocks",
        type=int,
        choices=(3, 4),
        default=3,
        help="Physical block count in the supplied Stack recordings (default: 3)",
    )
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--run-name", default="stack-id-first-continuation")
    parser.add_argument("--predicate-seconds", type=float, default=5)
    parser.add_argument("--trajectory-timeout-seconds", type=float, default=60)
    args = parser.parse_args(argv)
    if args.predicate_seconds <= 0 or args.trajectory_timeout_seconds <= 0:
        parser.error("Budgets must be positive")
    with RunLogger(
        args.output_dir, "id-first-continuation", vars(args), slug=args.run_name
    ) as logger:
        logger.progress_line(f"Experiment: {logger.run_dir}")
        run(args, logger)
    print(
        f"Report: uv run python -m synthesis.experiment.report --run {logger.run_dir}"
    )


if __name__ == "__main__":
    main()
