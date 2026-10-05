"""Execute the current candidate, map its boundaries, and bootstrap invariants."""

import json

import z3

from synthesis.cfg.collection import record_execution
from synthesis.cfg.demos import DemoSegment
from synthesis.cfg.lower import lower_with_locations
from synthesis.cfg.program_source import ProgramDefinition, describe_program
from synthesis.cfg.recordings import loop_store, save_traces
from synthesis.inference_lib.demo_store import InferenceVocabulary, InvInference
from synthesis.verification_lib.cegis import _project
from synthesis.verification_lib.counterexamples import state_holds


class CandidateTraceError(ValueError):
    pass


def block_segments(traces, paths):
    """Match a CFG block to completed executions using explicit instruction paths."""
    rows = []
    for index, trace in enumerate(traces):
        starts = [
            e
            for e in trace.events
            if e["kind"] == "instruction_start" and e["path"] == paths[0]
        ]
        ends = [
            e
            for e in trace.events
            if e["kind"] == "instruction_end" and e["path"] == paths[-1]
        ]
        if len(starts) != len(ends):
            raise CandidateTraceError(f"Incomplete execution of CFG block {paths}")
        for start, end in zip(starts, ends):
            ancestor = paths[0].rsplit(".", 1)[0] if "." in paths[0] else None
            entry = next(
                (
                    e["entry_index"]
                    for e in reversed(trace.events)
                    if e["kind"] == "loop_enter"
                    and e["path"] == ancestor
                    and e["index"] <= start["index"]
                ),
                0,
            )
            rows.append(
                DemoSegment(
                    index,
                    start["index"],
                    end["index"],
                    trace,
                    start["bindings"],
                    entry_index=entry,
                    final_bindings=end["bindings"],
                )
            )
    return rows


def prepare_candidate(
    cfg,
    context,
    *,
    relations=None,
    variables=2,
    max_loop_iterations=100,
    timeout_seconds=60,
    logger=None,
    revision=0,
):
    program, locations = lower_with_locations(cfg, context)
    original = cfg._task_demos
    traces = []
    if logger:
        logger.write_artifact(
            f"candidates/{revision}/program.json",
            json.dumps(describe_program(program), indent=2),
        )
    for demo_index, segment in enumerate(original):
        source = segment.trace
        bindings = dict(source.snapshots[segment.t_start].bindings)
        bindings.update(segment.bindings)
        definition = ProgramDefinition(program, bindings, "candidate")
        trace = record_execution(
            definition,
            seed=source.seed,
            num_blocks=source.num_blocks,
            task=source.task,
            initial_snapshot=source.snapshots[segment.t_start],
            max_loop_iterations=max_loop_iterations,
            timeout_seconds=timeout_seconds,
        )
        traces.append(trace)
        if logger and trace.states:
            save_traces(
                logger.artifact_dir(f"candidates/{revision}")
                / f"demo_{demo_index:04d}-seed_{source.seed}.npz",
                [trace],
            )
        if trace.metadata["status"] != "completed":
            raise CandidateTraceError(
                f"Candidate seed {source.seed}: {trace.metadata['status']}: {trace.metadata['reason']}"
            )
    # Runtime data is separate from the expert targets used to score repairs.
    # A supplied program has no prior CFG segmentation, so partition its original
    # recordings once using its known instruction paths. Synthesized CFGs already
    # carry their expert partitions from search/refinement.
    for path, (graph, name, paths) in locations["blocks"].items():
        candidate_rows = block_segments(traces, paths)
        if not hasattr(graph, "_candidate_demos"):
            graph._candidate_demos = {}
        graph._candidate_demos[name] = candidate_rows
        if hasattr(cfg, "_initial_bindings") and not getattr(
            cfg, "_expert_segments_initialized", False
        ):
            graph.demos.segments[name] = block_segments(
                [segment.trace for segment in original], paths
            )
    cfg._expert_segments_initialized = True
    stores = {}
    for loop_path, (cfg_path, loop) in locations["loops"].items():
        scope = set(loop.body_cfg.initial_scope) - set(loop.exists_vars)
        store = loop_store(traces, loop_id=loop_path, names=scope)
        rows = store.for_loop(loop_path)
        if not rows:
            raise CandidateTraceError(
                f"No observed heads or exits for loop {loop_path}"
            )
        names = tuple(
            sorted(scope & set.intersection(*(set(r.constants) for r in rows)))
        )
        vocabulary = InferenceVocabulary(
            variables,
            tuple(relations or ("ON_star", "Higher", "Scattered", "equality")),
            names,
        )
        for row in store._states:
            row.loop_id = "loop"
        loop.invariant = z3.BoolVal(False)
        if logger:
            logger.log_event(
                "invariant_initial",
                f"{loop_path}: False",
                force=True,
                revision=revision,
            )
            logger.write_artifact(
                f"candidates/{revision}/invariant-{loop_path}-initial.smt2", "false"
            )
        formula = InvInference(store, "loop", vocabulary, context)[0]
        if not all(
            state_holds(formula, _project(row, context.use_tbl))
            for row in store.for_loop("loop")
        ):
            raise CandidateTraceError(
                f"Learned invariant excludes recorded states for loop {loop_path}"
            )
        loop.invariant = formula
        loop.body_demos = tuple(
            tuple(loop.body_cfg.demos.for_node(n)) for n in loop.body_cfg.order
        )
        loop.exit_demos = tuple(
            DemoSegment(
                index,
                e["index"],
                e["index"],
                trace,
                e["bindings"],
                entry_index=e["entry_index"],
            )
            for index, trace in enumerate(traces)
            for e in trace.events
            if e["kind"] == "loop_exit" and e["path"] == loop_path
        )
        stores[id(loop)] = (store, vocabulary)
        if logger:
            logger.write_artifact(
                f"candidates/{revision}/invariant-{loop_path}.smt2", formula.sexpr()
            )
            logger.log_event(
                "invariant_bootstrap",
                f"{cfg_path}: {len(rows)} runtime states",
                force=True,
                revision=revision,
            )
    return stores
