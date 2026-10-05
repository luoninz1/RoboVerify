"""Fixed-program experiment orchestration; independent of Stack initialization."""

import json
from dataclasses import asdict, dataclass, field
from time import perf_counter

import z3
from synthesis.cfg.recordings import save_traces
from synthesis.experiment.invariant_learning.tasks import ExperimentFailure
from synthesis.experiment.invariant_learning.witness import find_witness
from synthesis.inference_lib.demo_store import DemoStore, InvInference
from synthesis.inference_lib.minimization import (
    DEFAULT_INVARIANT_MINIMIZER,
    INVARIANT_MINIMIZERS,
)
from synthesis.verification_lib.cegis import _project
from synthesis.verification_lib.counterexamples import state_holds


@dataclass(frozen=True)
class ExperimentConfig:
    verification_level: str = "symbolic"
    max_counterexample_blocks: int = 4
    max_rounds: int = 10
    verification_timeout_ms: int = 10000
    trajectory_timeout_seconds: float = 60
    seed: int = 0
    save_video: bool = False
    invariant_minimizer: str = DEFAULT_INVARIANT_MINIMIZER

    def __post_init__(self):
        import math

        if self.verification_level not in ("symbolic", "both"):
            raise ValueError("verification_level must be symbolic or both")
        if self.invariant_minimizer not in INVARIANT_MINIMIZERS:
            raise ValueError("invariant_minimizer must be sympy or pyeda")
        if (
            min(
                self.max_counterexample_blocks,
                self.max_rounds,
                self.verification_timeout_ms,
            )
            < 1
        ):
            raise ValueError("Search and solver budgets must be positive")
        if (
            not math.isfinite(self.trajectory_timeout_seconds)
            or self.trajectory_timeout_seconds <= 0
        ):
            raise ValueError("Trajectory timeout must be positive and finite")
        if not 0 <= self.seed < 2**32 or self.seed + self.max_rounds >= 2**32:
            raise ValueError("Execution seeds must fit in [0, 2**32)")


@dataclass
class ExperimentResult:
    status: str
    reason: str = ""
    symbolic_status: str = "not_verified"
    proof_scope: str = "not_established"
    motion_status: str = "not_requested"
    verification_attempts: int = 0
    counterexample_executions: int = 0
    learner_updates: int = 0
    history: list = field(default_factory=list)

    def __bool__(self):
        return self.status in ("verified_symbolic", "verified_model")


def verification_record(result):
    rows = []
    for check in result.checks:
        vc = getattr(check, "vc", None)
        rows.append(
            dict(
                obligation=vc.kind if vc is not None else check.obligation,
                loop_id=vc.loop_id if vc is not None else None,
                status=check.status,
                reason=check.reason,
                elapsed_seconds=getattr(check, "elapsed_seconds", None),
                counterexample=(
                    asdict(check.counterexample)
                    if getattr(check, "counterexample", None) is not None
                    else None
                ),
            )
        )
    return dict(
        scope=getattr(result, "scope", "motion_model"),
        checks=rows,
        mode=getattr(result, "mode", None),
        checked_blocks=getattr(result, "checked_blocks", None),
    )


def run_experiment(task, config, *, motion_options=None, logger=None):
    """Learn only from validated physical traces of the unchanged supplied program."""
    store = DemoStore()
    invariant = z3.BoolVal(False)
    result = ExperimentResult(
        "running",
        motion_status=(
            "not_run" if config.verification_level == "both" else "not_requested"
        ),
    )

    def artifact(path, value):
        if logger:
            logger.write_artifact(
                path,
                (
                    value
                    if isinstance(value, str)
                    else json.dumps(value, indent=2, default=str) + "\n"
                ),
            )

    def finish(status, reason=""):
        result.status, result.reason = status, reason
        artifact("progression.json", asdict(result))
        lines = [
            "# Counterexample-guided invariant learning",
            "",
            f"Result: **{status}**",
            "",
            reason,
            "",
            "| Check | Invariant | Failed obligations | Generated blocks | Added heads/exits |",
            "| --- | --- | --- | --- | --- |",
        ]
        for row in result.history:
            lines.append(
                f"| {row['verification_attempt']} | [I{row['revision']}](invariants/{row['revision']}.txt) | {row.get('failures', '')} | {row.get('num_blocks', '—')} | {row.get('new_states', 0)} |"
            )
        lines += [
            "",
            f"Verification attempts: {result.verification_attempts}; accepted counterexample executions: {result.counterexample_executions}; learner updates: {result.learner_updates}.",
            "",
            f"Symbolic: {result.symbolic_status}. Motion: {result.motion_status}.",
            "",
            "Symbolic success requires an unbounded proof. Counterexample search is bounded by the recorded initial domain, block range, and execution limits. Motion success is within the configured geometric model.",
        ]
        artifact("summary.md", "\n".join(lines) + "\n")
        return result

    artifact(
        "experiment.json",
        dict(
            config=asdict(config), task=task.describe(), motion_options=motion_options
        ),
    )
    artifact("invariants/0.smt2", "false\n")
    artifact("invariants/0.txt", "False\n")
    try:
        for revision in range(config.max_rounds + 1):
            task.set_invariant(invariant)
            if logger:
                logger.set_progress(
                    revision, phase="invariant_learning", n_states=len(store)
                )
                logger.progress_line(
                    f"Invariant {revision}: checking unbounded symbolic obligations ({len(store)} states)"
                )
            started = perf_counter()
            proof = task.verify_symbolic(config.verification_timeout_ms)
            result.verification_attempts += 1
            row = dict(
                revision=revision,
                verification_attempt=result.verification_attempts,
                n_states=len(store),
                symbolic_seconds=perf_counter() - started,
                failures=", ".join(
                    f"{c.vc.kind}:{c.status}"
                    for c in proof.checks
                    if c.status != "valid"
                ),
            )
            result.history.append(row)
            artifact(
                f"verification/{revision}-symbolic.json", verification_record(proof)
            )
            for index, check in enumerate(proof.checks):
                if check.model is not None:
                    artifact(
                        f"verification/{revision}-{index}-{check.vc.kind}-countermodel.txt",
                        str(check.model),
                    )
            if proof:
                if proof.scope != "unbounded":
                    return finish(
                        "bounded_only",
                        "Finite checks do not establish an unbounded invariant",
                    )
                result.symbolic_status = "verified"
                result.proof_scope = "unbounded"
                if config.verification_level == "symbolic":
                    return finish("verified_symbolic")
                if logger:
                    logger.progress_line(
                        "Symbolic proof passed; checking motion obligations"
                    )
                    logger.set_progress(revision, phase="motion")
                motion = task.verify_motion(motion_options or {})
                artifact("verification/motion.json", verification_record(motion))
                result.motion_status = "verified" if motion else "failed"
                if motion:
                    return finish("verified_model")
                return finish(
                    "motion_failed",
                    "; ".join(
                        f"{c.obligation}: {c.status} {c.reason}"
                        for c in motion.checks
                        if c.status != "valid"
                    ),
                )
            invalid = [c for c in proof.checks if c.status == "invalid"]
            if any(c.status == "unknown" for c in proof.checks):
                return finish("unknown", "An unbounded symbolic obligation is unknown")
            if not invalid:
                return finish(
                    "vacuous",
                    "No valid unbounded proof; symbolic premises are inconsistent",
                )
            if any(c.vc.kind == "exit" for c in invalid):
                return finish(
                    "needs_stronger_invariant",
                    "An exit countermodel cannot be excluded by invariant enlargement",
                )
            if revision == config.max_rounds:
                return finish("budget_exhausted", "Maximum learner updates reached")

            def record(size, query, attempt):
                artifact(f"queries/{revision}-{size}.smt2", query.solver.to_smt2())
                artifact(f"queries/{revision}-{size}.json", attempt)

            started = perf_counter()
            search = find_witness(
                lambda size: task.search_query(
                    size, config.verification_timeout_ms, invalid
                ),
                config.max_counterexample_blocks,
                record=record,
            )
            row["search"] = search.attempts
            row["search_seconds"] = perf_counter() - started
            if search.status != "found":
                return finish(search.status, search.reason)
            row["num_blocks"] = search.size
            row["replay"] = search.plan.metadata()
            artifact(f"counterexamples/{revision}-replay.json", search.plan.metadata())
            search.query.solver.push()
            search.query.solver.add(search.plan.condition)
            artifact(
                f"queries/{revision}-{search.size}-selected-path.smt2",
                search.query.solver.to_smt2(),
            )
            search.query.solver.pop()
            scene = (
                asdict(search.witness)
                if hasattr(search.witness, "__dataclass_fields__")
                else search.witness
            )
            artifact(f"counterexamples/{revision}-initial-scene.json", scene)
            if logger:
                logger.progress_line(
                    f"Invariant {revision}: executing generated {search.size}-block environment"
                )
                logger.set_progress(
                    revision, phase="counterexample_execution", num_blocks=search.size
                )
            video_path = (
                logger.artifact_dir("videos")
                / f"counterexample-{revision}-{search.size}-blocks.mp4"
                if logger and config.save_video
                else None
            )
            started = perf_counter()
            trace = task.execute(
                search,
                seed=config.seed + revision,
                timeout_seconds=config.trajectory_timeout_seconds,
                video_path=video_path,
            )
            row["execution_seconds"] = perf_counter() - started
            valid = task.validate(trace, search)
            if logger and trace.states:
                save_traces(
                    logger.artifact_dir("trajectories")
                    / f"counterexample-{revision}-{search.size}-blocks.npz",
                    [trace],
                )
            artifact(f"counterexamples/{revision}-execution.json", trace.metadata)
            if not valid:
                return finish(
                    trace.metadata.get("failure_kind", "execution_failed"),
                    trace.metadata.get("reason", "Invalid pre/post transition"),
                )
            states = task.learning_states(trace)
            uncovered = [
                s
                for s in states
                if not state_holds(invariant, _project(s, task.context.use_tbl))
            ]
            if not uncovered:
                return finish(
                    "no_progress",
                    "Physical execution supplied no state outside the previous invariant",
                )
            for state in states:
                store.add(state)
            result.counterexample_executions += 1
            row.update(new_states=len(states), uncovered_states=len(uncovered))
            if logger:
                store.save_diagnostic(logger.artifact_dir() / "learning-states.json")
            started = perf_counter()
            candidate, _ = InvInference(
                store,
                task.loop_id,
                task.vocabulary,
                task.context,
                minimizer=config.invariant_minimizer,
            )
            row["learning_seconds"] = perf_counter() - started
            artifact(f"invariants/{revision+1}.smt2", candidate.sexpr() + "\n")
            artifact(f"invariants/{revision+1}.txt", str(candidate) + "\n")
            if not all(
                state_holds(candidate, _project(s, task.context.use_tbl))
                for s in store.for_loop(task.loop_id)
            ):
                return finish(
                    "learning_failed",
                    "Learned invariant excludes an accumulated execution state",
                )
            solver = task.context.new_solver(config.verification_timeout_ms)
            solver.add(invariant, z3.Not(candidate))
            enlargement = solver.check()
            artifact(f"invariants/{revision+1}-progress.smt2", solver.to_smt2())
            if enlargement != z3.unsat:
                return finish(
                    "unknown" if enlargement == z3.unknown else "nonmonotone",
                    "Could not prove old invariant implies the learned invariant",
                )
            invariant = candidate
            result.learner_updates += 1
            artifact("progression.json", asdict(result))
            if logger:
                logger.log_metrics(
                    revision,
                    phase="invariant_learning",
                    n_states=len(store),
                    n_clauses=len(candidate.children()) if z3.is_and(candidate) else 1,
                    num_blocks=search.size,
                    new_states=len(states),
                    invariant_sexpr_path=f"artifacts/invariants/{revision+1}.smt2",
                )
    except ExperimentFailure as exc:
        return finish(exc.status, str(exc))
    except TimeoutError as exc:
        return finish("timeout", str(exc))
