"""Bounded symbolic and motion counterexample-guided refinement.

Phase E operates on an existing program. Entry failures require Phase F's
program synthesizer and are surfaced as NeedsResynthesis, never relabeled as
invariant-learning examples.
"""

import json
from copy import deepcopy
from dataclasses import asdict, dataclass, field

import z3

from synthesis.inference_lib.demo_store import DemoStore, InvInference, LoopHeadState
from synthesis.verification_lib.counterexamples import (
    UnrealizableCounterexample,
    state_holds,
    symbolic_successor,
)
from synthesis.verification_lib.symbolic_verify import symbolic_verify


class NeedsResynthesis(RuntimeError):
    def __init__(self, result):
        super().__init__(
            "Entry/body obligation failed; program resynthesis requires Phase F"
        )
        self.result = result
        self.s0 = result.loop_head_state
        self.model = result.model


@dataclass
class CEGISResult:
    status: str
    iterations: int
    invariant: object = None
    program: object = None
    verification: object = None
    reason: str = ""
    history: list = field(default_factory=list)

    def __bool__(self):
        return self.status == "verified"


def _log(logger, iteration, phase, invariant=None, **fields):
    if invariant is not None:
        fields["n_clauses"] = len(invariant.children()) if z3.is_and(invariant) else 1
        if logger:
            path = logger.write_artifact(
                f"invariants/{iteration}.smt2", invariant.sexpr() + "\n"
            )
            fields["invariant_sexpr_path"] = str(path)
    if logger:
        logger.set_progress(iteration, phase=phase, **fields)
        logger.log_metrics(iteration, phase=phase, **fields)
    return dict(
        iteration=iteration,
        phase=phase,
        **fields,
        invariant_sexpr=invariant.sexpr() if invariant is not None else None,
    )


def run_symbolic_cegis(
    store,
    loop_id,
    vocab,
    context,
    build_problem,
    *,
    initial_invariant=None,
    max_iterations=20,
    min_blocks=2,
    max_blocks=4,
    timeout_ms=5000,
    prove_unbounded=True,
    logger=None,
):
    """Use InvInference on supplied demos and successors of preservation failures.

    ``build_problem(invariant, size)`` must instantiate the invariant in the new
    context and return (program, pre, post, context). Single-loop Phase E scope;
    nested/CFG propagation requires Phase F's per-block summaries.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    invariant = z3.BoolVal(False) if initial_invariant is None else initial_invariant
    history = [_log(logger, 0, "symbolic_initial", invariant, n_states=len(store))]
    for iteration in range(1, max_iterations + 1):
        rows = store.for_loop(loop_id)
        uncovered = [
            s for s in rows if not state_holds(invariant, _project(s, context.use_tbl))
        ]
        if uncovered:
            candidate = InvInference(store, loop_id, vocab, context)[0]
            if not all(
                state_holds(candidate, _project(s, context.use_tbl)) for s in rows
            ):
                return CEGISResult(
                    "learning_failed",
                    iteration,
                    invariant,
                    reason="Learner excludes a positive state",
                    history=history,
                )
            solver = context.new_solver(timeout_ms)
            solver.add(invariant, z3.Not(candidate))
            monotone = solver.check()
            if monotone != z3.unsat:
                return CEGISResult(
                    "unknown" if monotone == z3.unknown else "nonmonotone",
                    iteration,
                    invariant,
                    reason="Could not prove old invariant implies learned invariant",
                    history=history,
                )
            # An uncovered positive is a concrete witness of strict enlargement.
            invariant = candidate
        history.append(
            _log(
                logger,
                iteration,
                "symbolic",
                invariant,
                n_states=len(store),
                enlarged=bool(uncovered),
            )
        )
        problems = {}

        def problem(size):
            if size not in problems:
                problems[size] = build_problem(invariant, size)
            return problems[size]

        from synthesis.api.instructions import While

        seed_program = problem(min_blocks)[0]
        loops = [
            (str(i), inst)
            for i, inst in enumerate(seed_program.instructions)
            if isinstance(inst, While)
        ]
        if (
            len(loops) != 1
            or loops[0][0] != loop_id
            or any(isinstance(inst, While) for inst in loops[0][1].body)
        ):
            return CEGISResult(
                "unsupported",
                iteration,
                invariant,
                reason="Phase E symbolic CEGIS requires one non-nested loop",
                history=history,
            )
        result = symbolic_verify(
            problem,
            min_blocks=min_blocks,
            max_blocks=max_blocks,
            timeout_ms=timeout_ms,
            prove_unbounded=prove_unbounded,
            constants=tuple(
                dict.fromkeys((*vocab.constants, *_guard_names(seed_program)))
            ),
        )
        if result:
            return CEGISResult(
                "verified", iteration, invariant, verification=result, history=history
            )
        failure = result.failure
        if logger:
            logger.log_event(
                "symbolic_failure",
                f"{failure.vc.kind}: {failure.status}",
                step=iteration,
                force=True,
            )
            if result.model is not None:
                logger.write_artifact(
                    f"counterexamples/{iteration}.smt2", str(result.model)
                )
            if result.loop_head_state is not None:
                DemoStore([result.loop_head_state]).save_diagnostic(
                    logger.artifact_dir("counterexamples") / f"{iteration}.json"
                )
        if failure.status != "invalid":
            return CEGISResult(
                failure.status,
                iteration,
                invariant,
                verification=result,
                reason=failure.reason,
                history=history,
            )
        if failure.vc.kind in ("establish", "body"):
            raise NeedsResynthesis(result)
        if failure.vc.kind == "exit":
            return CEGISResult(
                "needs_stronger_invariant",
                iteration,
                invariant,
                verification=result,
                reason="Weakening cannot exclude an exit counterexample already covered by the invariant",
                history=history,
            )
        if result.loop_head_state is None:
            return CEGISResult(
                "unrealizable_counterexample",
                iteration,
                invariant,
                verification=result,
                reason=result.reason,
                history=history,
            )
        program, _, _, _ = problem(result.num_blocks)
        loop = _loop(program, failure.vc.loop_id)
        try:
            successor = symbolic_successor(result.loop_head_state, loop.body)
        except UnrealizableCounterexample as exc:
            return CEGISResult(
                "unsupported", iteration, invariant, reason=str(exc), history=history
            )
        successor.loop_id = loop_id
        if state_holds(invariant, _project(successor, context.use_tbl)):
            return CEGISResult(
                "no_progress",
                iteration,
                invariant,
                verification=result,
                reason="Constructed successor does not violate the old invariant",
                history=history,
            )
        store.add(successor)
        if logger:
            store.save_diagnostic(logger.artifact_dir() / "invariant_examples.json")
    return CEGISResult("budget_exhausted", max_iterations, invariant, history=history)


def _project(state, use_tbl):
    if use_tbl:
        return state
    return LoopHeadState(
        state.loop_id,
        {k: v for k, v in state.positions.items() if k != "tbl"},
        {k: v for k, v in state.entry_positions.items() if k != "tbl"},
        {k: v for k, v in state.constants.items() if v != "tbl"},
    )


def _loop(program, loop_id):
    instructions = program.instructions
    for part in loop_id.split("."):
        instruction = instructions[int(part)]
        instructions = instruction.body
    return instruction


def _guard_names(program):
    from synthesis.api.instructions import While

    for instruction in program.instructions:
        if isinstance(instruction, While):
            yield from (str(v) for v in instruction.guard_exists_vars)


@dataclass(frozen=True)
class MotionBlockSpec:
    block_id: str
    conditions: tuple
    constants: tuple
    contract: object
    relational_summary: str

    def fingerprint(self):
        return (
            self.block_id,
            tuple(c.sexpr() for c in self.conditions),
            self.constants,
            repr(self.contract),
            self.relational_summary,
        )


class PenStore:
    """Monotonically accumulated, deduplicated environments per motion block."""

    def __init__(self):
        self._examples = {}
        self._keys = set()

    def add(self, counterexample):
        payload = asdict(counterexample)
        # Multiple failed obligations in the same environment cost one penalty.
        environment = {
            key: payload[key]
            for key in ("block_v", "mu_k", "bindings", "entry_positions", "initial_arm")
        }
        key = json.dumps(environment, sort_keys=True)
        if key in self._keys:
            return False
        self._keys.add(key)
        self._examples.setdefault(counterexample.block_v, []).append(
            deepcopy(counterexample)
        )
        return True

    def for_block(self, block_id):
        return deepcopy(self._examples.get(str(block_id), []))

    def __len__(self):
        return len(self._keys)

    def save(self, path):
        from pathlib import Path

        Path(path).write_text(
            json.dumps(
                {
                    "version": 1,
                    "examples": [
                        asdict(e) for es in self._examples.values() for e in es
                    ],
                },
                indent=2,
            )
            + "\n"
        )


class MotionPenalty:
    """Objective (8): count failing environments, including inconclusive checks.

    Object attributes and frozen geometry are fixed; declared bounded noise is
    still universally checked. This penalty guides search; only the subsequent
    full MotionVerify call can certify a candidate.
    """

    def __init__(self, blocks, pen, *, noise=None, timeout_ms=5000):
        self.blocks, self.pen, self.noise, self.timeout_ms = (
            blocks,
            pen,
            noise,
            timeout_ms,
        )

    def __call__(self, program):
        from synthesis.verification_lib.motion_verification import verify_motion_block

        failures = 0
        for spec in self.blocks:
            for example in self.pen.for_block(spec.block_id):
                result = verify_motion_block(
                    spec.conditions,
                    _loop(program, spec.block_id).body,
                    spec.constants,
                    spec.contract,
                    block_v=spec.block_id,
                    noise=self.noise,
                    timeout_ms=self.timeout_ms,
                    initial_positions=example.mu_k,
                    entry_positions=example.entry_positions,
                    initial_bindings=example.bindings,
                )
                failures += not bool(result)
        return failures


def _control_fingerprint(program):
    from synthesis.api.instructions import Assign, PickPlaceByName, Skip, While

    def shape(instructions):
        result = []
        for inst in instructions:
            if isinstance(inst, While):
                invariant = inst.invariant
                exprs = program._invariant_condition_exprs(invariant)
                result.append(
                    (
                        "while",
                        inst.cond.sexpr(),
                        tuple(
                            z3.BoolVal(e).sexpr() if isinstance(e, bool) else e.sexpr()
                            for e in exprs
                        ),
                        inst.max_iters,
                        shape(inst.body),
                    )
                )
            elif isinstance(inst, Assign):
                result.append(("assign", inst.left, inst.right))
            elif isinstance(inst, PickPlaceByName):
                # Offsets may change, but the declared source/placement remains.
                result.append(
                    (
                        "motion",
                        inst.grab_box_name,
                        tuple(inst.target_box_names),
                        inst.release,
                    )
                )
            elif isinstance(inst, Skip):
                result.append(("skip",))
            else:
                raise ValueError(
                    f"Unsupported motion resynthesis instruction: {type(inst).__name__}"
                )
        return tuple(result)

    return shape(program.instructions)


def run_motion_cegis(
    program,
    blocks,
    resynthesize,
    *,
    noise=None,
    timeout_ms=5000,
    max_iterations=10,
    logger=None,
    pen=None,
):
    """Re-synthesize offsets while preserving the symbolic proof's assumptions."""
    from synthesis.verification_lib.motion_verification import (
        MotionVerificationResult,
        verify_motion_block,
    )

    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    from synthesis.api.instructions import Assign, Skip, While

    expected = {
        str(i) for i, inst in enumerate(program.instructions) if isinstance(inst, While)
    }
    if (
        not expected
        or expected != {spec.block_id for spec in blocks}
        or len(blocks) != len(expected)
    ):
        raise ValueError("Motion blocks must cover every loop exactly once")
    if any(
        not isinstance(inst, (While, Assign, Skip)) for inst in program.instructions
    ):
        raise ValueError("Uncovered straight-line motion")
    for spec in blocks:
        loop = _loop(program, spec.block_id)
        actual = z3.simplify(
            z3.And(
                *program._invariant_condition_exprs(loop.invariant),
                loop.instantiated_cond,
            )
        )
        supplied = z3.simplify(z3.And(*spec.conditions))
        if not actual.eq(supplied):
            raise ValueError(
                "Motion precondition must equal the proved invariant and guard"
            )
    pen = PenStore() if pen is None else pen
    signature = (tuple(s.fingerprint() for s in blocks), _control_fingerprint(program))
    history = []
    for iteration in range(max_iterations + 1):
        if signature != (
            tuple(s.fingerprint() for s in blocks),
            _control_fingerprint(program),
        ):
            raise ValueError(
                "Motion resynthesis changed a relational summary or control structure"
            )
        checks = []
        for spec in blocks:
            checks.append(
                verify_motion_block(
                    spec.conditions,
                    _loop(program, spec.block_id).body,
                    spec.constants,
                    spec.contract,
                    noise=noise,
                    block_v=spec.block_id,
                    timeout_ms=timeout_ms,
                )
            )
        result = MotionVerificationResult(
            [c for r in checks for c in r.checks],
            noise,
            sum(r.checked_blocks for r in checks),
            sum(r.elapsed_seconds for r in checks),
        )
        history.append(
            _log(
                logger, iteration, "motion", n_penalties=len(pen), verified=bool(result)
            )
        )
        if result:
            return CEGISResult(
                "verified",
                iteration,
                program=program,
                verification=result,
                history=history,
            )
        if any(
            c.status in ("unknown", "unsupported", "inconsistent")
            for c in result.checks
        ):
            return CEGISResult(
                "motion_inconclusive",
                iteration,
                program=program,
                verification=result,
                history=history,
            )
        for example in result.counterexamples:
            pen.add(example)
        if logger:
            pen.save(logger.artifact_dir() / "penalties.json")
        if iteration == max_iterations:
            return CEGISResult(
                "budget_exhausted",
                iteration,
                program=program,
                verification=result,
                history=history,
            )
        penalty = MotionPenalty(blocks, pen, noise=noise, timeout_ms=timeout_ms)
        program = resynthesize(deepcopy(program), penalty, iteration)
        if logger:
            logger.write_artifact(f"motion/{iteration}.txt", str(program))


def optimize_motion_parameters(
    program,
    penalty,
    *,
    base_score=None,
    weight=1.0,
    seed=0,
    iterations=3,
    samples=16,
    elites=4,
    std=0.03,
):
    """Serial CEM for solver-bearing motion candidates (Z3 objects aren't picklable).

    With base_score, maximizes base_score - weight * failures (objective 8).
    Without it this is explicitly penalty-only repair, useful for model-level
    diagnostics; it does not claim to optimize demonstration distance.
    """
    import numpy as np

    if not (
        iterations >= 0
        and 1 <= elites <= samples
        and std > 0
        and np.isfinite(std)
        and np.isfinite(weight)
        and weight >= 0
    ):
        raise ValueError("Invalid motion optimizer budget or penalty weight")
    rng = np.random.default_rng(seed)
    mean = np.asarray(program.register_trainable_parameter(), dtype=float)
    sigma = np.full(len(mean), std)
    best = deepcopy(program)

    def score(candidate):
        return (
            0.0 if base_score is None else base_score(candidate)
        ) - weight * penalty(candidate)

    best_score = score(best)
    for _ in range(iterations if len(mean) else 0):
        population = [mean] + list(rng.normal(mean, sigma, (samples, len(mean))))
        ranked = []
        for values in population:
            candidate = deepcopy(program)
            candidate.update_trainable_parameter(list(values))
            value = score(candidate)
            ranked.append((value, values))
            if value > best_score:
                best, best_score = candidate, value
        elite = np.asarray(
            [
                values
                for _, values in sorted(ranked, key=lambda item: item[0], reverse=True)[
                    :elites
                ]
            ]
        )
        mean, sigma = elite.mean(axis=0), np.maximum(elite.std(axis=0), 1e-6)
    return best
