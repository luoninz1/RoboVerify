"""Placement, frame, and swept-cube obligations for a lowered basic block.

This is an idealized waypoint model, not a model of gripper dynamics or settling.
A block supported by the manipulated block is allowed arbitrary displacement;
we refuse to certify such a manipulation rather than silently freezing it.
"""

from dataclasses import dataclass, field, replace
from itertools import product
from time import perf_counter
from typing import Optional

import z3

from synthesis.api.instructions import Assign, PickPlaceByName, Skip
from synthesis.util.symbols import fresh_const
from synthesis.verification_lib.bmc_lib import NoiseSpec, bounded_noise
from synthesis.verification_lib.lowlevel_verification_lib import (
    UNHANDLED_MOTION_INSTRUCTIONS,
    LowLevelContext,
    UnsupportedMotionInstruction,
)


@dataclass(frozen=True)
class MotionContract:
    source: str
    target: str
    frame_base: Optional[str] = None
    table_surface_height: Optional[float] = None


@dataclass(frozen=True)
class MotionCounterexample:
    block_v: str
    mu_k: dict
    obligation: str
    final_positions: dict = field(default_factory=dict)
    noise_values: dict = field(default_factory=dict)
    bindings: dict = field(default_factory=dict)
    entry_positions: dict = field(default_factory=dict)
    initial_arm: object = None


@dataclass(frozen=True)
class MotionCheck:
    obligation: str
    status: str
    elapsed_seconds: float = 0.0
    counterexample: Optional[MotionCounterexample] = None
    reason: str = ""


@dataclass
class MotionVerificationResult:
    checks: list
    noise: Optional[NoiseSpec] = None
    checked_blocks: int = 0
    elapsed_seconds: float = 0.0

    @property
    def mode(self):
        return "noiseless" if self.noise is None else "bounded-noise"

    @property
    def ok(self):
        return (
            self.checked_blocks > 0
            and bool(self.checks)
            and all(c.status == "valid" for c in self.checks)
        )

    @property
    def counterexamples(self):
        return [c.counterexample for c in self.checks if c.counterexample is not None]

    def __bool__(self):
        return self.ok

    def __str__(self):
        failed = [
            f"{c.obligation}:{c.status}" for c in self.checks if c.status != "valid"
        ]
        return f"{self.ok} (mode={self.mode}, checked_blocks={self.checked_blocks}, failures={failed})"


def _number(model, term):
    value = model.eval(term, model_completion=True)
    if z3.is_rational_value(value):
        return float(value.as_fraction())
    return float(value.approx(30).as_fraction())


def _different(a, b):
    return z3.Or(*(x != y for x, y in zip(a, b)))


def _on_star(a, b, length):
    return z3.And(
        z3.Abs(a[0] - b[0]) < length / 2, z3.Abs(a[1] - b[1]) < length / 2, a[2] >= b[2]
    )


class MotionProblem:
    def __init__(
        self,
        context,
        initial_condition,
        constants,
        contract,
        noise,
        block_v,
        timeout_ms,
        initial_positions=None,
        entry_positions=None,
        initial_bindings=None,
    ):
        if timeout_ms <= 0:
            raise ValueError("Motion solver timeout must be positive")
        if "sym" in constants:
            raise ValueError("sym is reserved for the arbitrary other block")
        required = {contract.source, contract.target}
        if not required.issubset(set(constants)):
            raise ValueError(
                f"Contract constants missing from layout: {required - set(constants)}"
            )
        self.context = context
        self.initial_condition = tuple(initial_condition)
        self.contract = contract
        self.noise = noise
        self.block_v = str(block_v)
        self.timeout_ms = timeout_ms
        self.solver = z3.Solver()
        self.solver.set(timeout=timeout_ms)
        self.solver.add(context.L == z3.RealVal(str(context.default_L)))
        self.constants = context.translate_condition(
            self.solver, constants, initial_condition, track=False
        )
        self.constants["sym"] = context.get_consts("sym")
        if context.use_tbl:
            self.solver.add(self.constants["sym"] != context.table_const())
        self.initial = {
            name: (context.X(obj), context.Y(obj), context.Z(obj))
            for name, obj in self.constants.items()
            if name != "tbl"
        }
        self.current = dict(self.initial)
        self.bindings = {name: name for name in self.constants}
        self.errors = {}
        self.checks = []
        self.released = False
        self.grasp_error = (0, 0, 0)
        self.held = False
        for name, xyz in (initial_positions or {}).items():
            if name not in self.initial or len(xyz) != 3:
                raise ValueError(f"Unknown or nonphysical scene object: {name!r}")
            self.solver.add(self.physical(name))
            self.solver.add(
                *(a == z3.RealVal(str(b)) for a, b in zip(self.initial[name], xyz))
            )

        for name, xyz in (entry_positions or {}).items():
            if name not in self.initial or len(xyz) != 3:
                raise ValueError(f"Unknown entry-scene object: {name!r}")
            obj = self.constants[name]
            self.solver.add(
                *(
                    axis(obj) == z3.RealVal(str(value))
                    for axis, value in zip((context.X0, context.Y0, context.Z0), xyz)
                )
            )
        for name, target in (initial_bindings or {}).items():
            if name not in self.constants or target not in self.constants:
                raise ValueError("Unknown counterexample alias")
            self.solver.add(
                context.lowlevel_box_equal(self.constants[name], self.constants[target])
            )

    def same(self, a, b):
        # The translator represents aliases by initial coordinate equality.
        return self.context.lowlevel_box_equal(self.constants[a], self.constants[b])

    def physical(self, name):
        return z3.Not(self.context._is_table(self.constants[name]))

    def error(self, epsilon, prefix):
        terms, bounds = bounded_noise(epsilon, prefix)
        self.solver.add(*bounds)
        self.errors.update({str(term): term for term in terms})
        return terms

    def counterexample(self, model, obligation):
        physical_names = [
            name
            for name in self.initial
            if z3.is_true(model.eval(self.physical(name), model_completion=True))
        ]
        bindings = {}
        for name in physical_names:
            bindings[name] = next(
                (
                    bindings[other]
                    for other in bindings
                    if z3.is_true(
                        model.eval(self.same(name, other), model_completion=True)
                    )
                ),
                name,
            )
        if self.context.use_tbl:
            bindings["tbl"] = "tbl"
        return MotionCounterexample(
            self.block_v,
            {
                name: [_number(model, v) for v in self.initial[name]]
                for name in physical_names
            },
            obligation,
            {
                name: [_number(model, v) for v in self.current[name]]
                for name in physical_names
            },
            {name: _number(model, v) for name, v in self.errors.items()},
            bindings,
            {
                name: [
                    _number(model, axis(self.constants[name]))
                    for axis in (self.context.X0, self.context.Y0, self.context.Z0)
                ]
                for name in physical_names
            },
        )

    def finite_consistency_witness(self, candidates):
        """SAT over an explicit finite domain witnesses unbounded consistency.

        Expand every Box quantifier over that domain and assert domain closure.
        This is only a SAT shortcut: finite UNSAT/UNKNOWN is never a proof of
        inconsistency and never a proof of a universally quantified obligation.
        """
        cache = {}
        expansions = 0

        def expand(expr):
            nonlocal expansions
            if expr in cache:
                return cache[expr]
            if z3.is_quantifier(expr):
                if any(
                    expr.var_sort(i) != self.context.BoxSort
                    for i in range(expr.num_vars())
                ):
                    return expr
                terms = []
                for values in product(candidates, repeat=expr.num_vars()):
                    expansions += 1
                    if expansions > 10000:
                        raise ValueError(
                            "Finite consistency expansion budget exhausted"
                        )
                    terms.append(
                        expand(z3.substitute_vars(expr.body(), *reversed(values)))
                    )
                result = (z3.And if expr.is_forall() else z3.Or)(*terms)
            elif z3.is_app(expr) and expr.num_args():
                result = expr.decl()(*(expand(c) for c in expr.children()))
            else:
                result = expr
            cache[expr] = result
            return result

        finite = z3.Solver()
        finite.set(timeout=self.timeout_ms)
        try:
            finite.add(*(expand(a) for a in self.solver.assertions()))
        except ValueError:
            return False
        obj = fresh_const(self.context.BoxSort, "consistency_object")
        finite.add(z3.ForAll([obj], z3.Or(*(obj == c for c in candidates))))
        return finite.check() == z3.sat

    def check(self, obligation, violation=None):
        start = perf_counter()
        self.solver.push()
        if violation is not None:
            self.solver.add(violation)
        answer = None
        candidates = getattr(self, "consistency_candidates", ())
        if (
            violation is None
            and candidates
            and self.finite_consistency_witness(candidates)
        ):
            answer = z3.sat
        if answer is None:
            answer = self.solver.check()
        counterexample = None
        reason = ""
        if answer == z3.unknown:
            status = "unknown"
            reason = self.solver.reason_unknown()
        elif violation is None:
            status = "valid" if answer == z3.sat else "inconsistent"
        else:
            status = "valid" if answer == z3.unsat else "refuted"
            if answer == z3.sat:
                counterexample = self.counterexample(self.solver.model(), obligation)
        self.solver.pop()
        check = MotionCheck(
            obligation, status, perf_counter() - start, counterexample, reason
        )
        self.checks.append(check)
        return check

    def execute(self, body):
        source = self.contract.source
        if source not in self.current:
            raise ValueError("Contract source must name a physical constant")
        moved_count = 0
        for index, instruction in enumerate(body):
            if isinstance(instruction, Skip):
                continue
            if isinstance(instruction, Assign):
                # Assign is geometrically inert but subsequent operands see the new binding.
                self.bindings[instruction.left] = self.bindings[instruction.right]
                continue
            if not isinstance(instruction, PickPlaceByName):
                if isinstance(instruction, UNHANDLED_MOTION_INSTRUCTIONS):
                    self.checks.append(
                        MotionCheck(
                            f"instruction_{index}",
                            "unsupported",
                            reason=f"No motion encoding for {type(instruction).__name__}",
                        )
                    )
                    return
                raise UnsupportedMotionInstruction(type(instruction).__name__)
            grab = self.bindings[instruction.grab_box_name]
            targets = [self.bindings[name] for name in instruction.target_box_names]
            if grab == "tbl" or "tbl" in targets:
                self.checks.append(
                    MotionCheck(
                        f"instruction_{index}",
                        "unsupported",
                        reason="The relational table has no waypoint coordinates",
                    )
                )
                return
            if (
                self.check(f"source_{index}", z3.Not(self.same(grab, source))).status
                != "valid"
            ):
                return  # A block contract covers one declared manipulated object.
            self.check(
                f"physical_{index}",
                z3.Not(z3.And(*(self.physical(n) for n in [grab, *targets]))),
            )
            start = self.current[grab]
            if not self.held:
                self.grasp_error = (
                    (0, 0, 0)
                    if self.noise is None
                    else self.error(self.noise.eps_grasp, f"motion_grasp_{index}")
                )
            endpoint = tuple(
                self.current[name][axis]
                + z3.RealVal(
                    instruction.target_offset[axis].concrete_float("motion offset")
                )
                for axis, name in enumerate(targets)
            )
            if self.noise is not None:
                error = self.error(self.noise.eps_move, f"motion_move_{index}")
                endpoint = tuple(
                    p + e - g for p, e, g in zip(endpoint, error, self.grasp_error)
                )
            # Check the same swept-cube formula against the arbitrary other block
            # and every named block, using the current (not stale entry) geometry.
            for name, position in self.current.items():
                self.check(
                    f"collision_{index}_{name}",
                    z3.And(
                        self.physical(name),
                        z3.Not(self.same(name, grab)),
                        self.context.encode_collision_at(position, start, endpoint),
                    ),
                )
            before = dict(self.current)
            for name, position in before.items():
                supported = z3.And(
                    self.physical(name),
                    z3.Not(self.same(name, grab)),
                    _on_star(position, start, self.context.L),
                    position[2] > start[2],
                )
                # No sound rigid-stack dynamics is available. Permit arbitrary
                # displacement, and separately refuse to certify this case.
                disturbed = tuple(
                    fresh_const(z3.RealSort(), f"disturbed_{index}_{name}_{axis}")
                    for axis in "xyz"
                )
                self.current[name] = tuple(
                    z3.If(self.same(name, grab), end, z3.If(supported, unknown, old))
                    for end, unknown, old in zip(endpoint, disturbed, position)
                )
                self.check(f"support_{index}_{name}", supported)
            names = list(self.current)
            for i, name in enumerate(names):
                for other in names[i + 1 :]:
                    self.solver.add(
                        z3.Implies(
                            self.same(name, other),
                            z3.Not(_different(self.current[name], self.current[other])),
                        )
                    )
            if instruction.release and self.noise is not None:
                release_error = self.error(
                    self.noise.eps_release, f"motion_release_{index}"
                )
                release_end = tuple(p + e for p, e in zip(endpoint, release_error))
                for name, position in self.current.items():
                    self.check(
                        f"release_collision_{index}_{name}",
                        z3.And(
                            self.physical(name),
                            z3.Not(self.same(name, grab)),
                            self.context.encode_collision_at(
                                position, endpoint, release_end
                            ),
                        ),
                    )
                    self.current[name] = tuple(
                        z3.If(self.same(name, grab), end, old)
                        for end, old in zip(release_end, position)
                    )
            self.held = not instruction.release
            self.released = instruction.release
            moved_count += 1
        if moved_count == 0:
            self.checks.append(
                MotionCheck(
                    "coverage", "unsupported", reason="No motion instructions checked"
                )
            )


def check_contract_realization(problem):
    """Discharge transitions => released and direct-on (or physical table height)."""
    contract = problem.contract
    source = problem.current[contract.source]
    if contract.target == "tbl":
        if contract.table_surface_height is None:
            check = MotionCheck(
                "contract",
                "unsupported",
                reason="Table placement requires table_surface_height",
            )
            problem.checks.append(check)
            return check
        goal = (
            source[2]
            == z3.RealVal(str(contract.table_surface_height)) + problem.context.L / 2
        )
    else:
        target = problem.current[contract.target]
        goal = z3.And(
            _on_star(source, target, problem.context.L),
            source[2] - target[2] < z3.RealVal("1.5") * problem.context.L,
            problem.physical(contract.target),
            z3.Not(problem.same(contract.source, contract.target)),
        )
    return problem.check(
        "contract",
        z3.Not(z3.And(goal, problem.released, problem.physical(contract.source))),
    )


def check_abstract_effects(problem):
    """Check the entire Put interpretation, not just its local placement atom.

    Every relation involving the manipulated object is checked against each
    named object and the arbitrary object sym. Relations between other objects
    are unchanged by successful, support-free execution. Exact quantified
    translation is necessary here: weakening under an equivalence is unsound.
    """
    from synthesis.api.instructions import Put
    from synthesis.api.program import wp
    from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext

    problem.solver.push()
    try:
        # Quantified effect checks may need information discarded by the finite
        # assumption expansion. Reassert the actual supplied premises exactly;
        # these are not new assumptions inferred from demonstration geometry.
        problem.solver.add(
            *(
                problem.context.translate_exact(c, problem.constants)
                for c in problem.initial_condition
            )
        )
        high = HighLevelContext(use_tbl=problem.context.use_tbl)
        put = Put(problem.contract.source, problem.contract.target)
        names = list(problem.constants)
        source = problem.contract.source
        # Include source/other in both directions, plus unchanged named pairs.
        pairs = [(a, b) for a in names for b in names if a == source or b == source]
        replacements = [
            (axis(problem.constants[name]), point[i])
            for name, point in problem.current.items()
            for i, axis in enumerate(
                (problem.context.X, problem.context.Y, problem.context.Z)
            )
        ]
        for relation in ("ON_star", "Higher", "Scattered"):
            violations = []
            for left, right in pairs:
                atom = getattr(high, relation)(
                    high.get_consts(left), high.get_consts(right)
                )
                if hasattr(problem, "effect_entry_fields"):
                    expected = problem.at(
                        wp(put, atom, high),
                        fields=problem.effect_entry_fields,
                        resolve=False,
                    )
                    actual = problem.at(atom, resolve=False)
                else:
                    expected = problem.context.translate_exact(
                        wp(put, atom, high), problem.constants
                    )
                    before = problem.context.translate_exact(atom, problem.constants)
                    actual = z3.substitute(before, *replacements)
                violations.append(actual != expected)
            problem.check(f"effect_{relation}", z3.Or(*violations))
    finally:
        problem.solver.pop()


def check_frame_preservation(problem):
    """Keep all non-manipulated objects fixed, including the proved root."""
    violations = []
    for name, initial in problem.initial.items():
        protected = z3.And(
            problem.physical(name),
            z3.Not(problem.same(name, problem.contract.source)),
        )
        violations.append(z3.And(protected, _different(initial, problem.current[name])))
    return problem.check("frame", z3.Or(*violations))


def _aligned(point, root, length):
    # Strict root radius delta=L/4 implies strict pairwise bound N=L/2.
    return z3.And(*(z3.Abs(point[i] - root[i]) < length / 4 for i in (0, 1)))


def assume_input_alignment(problem, roots):
    """Declared input-tower assumption, applied only to fresh entry geometry.

    The quantified member covers unnamed blocks. Later placements must establish
    alignment by VCs; they must never add this assumption to their output state.
    """
    assumed = getattr(problem, "assumed_alignment_roots", set())
    problem.assumed_alignment_roots = assumed
    for name in roots:
        if name in assumed:
            continue
        assumed.add(name)
        root = problem.constants[name]
        member = fresh_const(
            problem.context.BoxSort,
            prefix="aligned_input",
            avoid=problem.constants.values(),
        )
        point = tuple(
            axis(member)
            for axis in (problem.context.X, problem.context.Y, problem.context.Z)
        )
        origin = problem.initial[name]
        problem.solver.add(
            z3.ForAll(
                [member],
                z3.Implies(
                    z3.And(
                        z3.Not(problem.context._is_table(member)),
                        problem.context.lowlevel_on_star(member, root),
                    ),
                    _aligned(point, origin, problem.context.L),
                ),
            )
        )


def prepare_alignment(
    problem,
    root_context,
    candidates,
    *,
    target=None,
    continuation=(),
    postcondition=None,
):
    """Select/prove a root before motion; an unproved hint is never a fallback."""
    if problem.contract.target == "tbl":
        problem.alignment_root = None  # A separated table placement starts a singleton.
        return
    from synthesis.api.instructions import Put

    # Get/Assign may expose another input root without changing object positions.
    # Once a Put has occurred, alignment must follow from checked motion, never
    # a newly inserted assumption on an already-built tower.
    if not any(isinstance(i, Put) for i in root_context.history):
        assume_input_alignment(
            problem,
            [problem.bindings[name] for name in root_context.initial_roots(candidates)],
        )
        problem.check("alignment_assumption_consistency")
    result = root_context.select(
        target or problem.contract.target,
        candidates,
        continuation=continuation,
        postcondition=postcondition,
    )
    problem.checks.append(
        MotionCheck(
            "root_selection",
            (
                "valid"
                if result.status == "valid"
                else "unsupported" if result.status == "unproved" else result.status
            ),
            reason=("root=" + result.root + "; " if result.root else "")
            + result.reason,
        )
    )
    problem.alignment_root = problem.bindings[result.root] if result.root else None
    if result.root is not None:
        problem.contract = replace(problem.contract, frame_base=problem.alignment_root)
        root = problem.current[problem.alignment_root]
        violations = [
            z3.And(
                problem.physical(name),
                _on_star(point, root, problem.context.L),
                z3.Not(_aligned(point, root, problem.context.L)),
            )
            for name, point in problem.current.items()
        ]
        problem.check("alignment_entry", z3.Or(*violations))


def check_alignment(problem):
    """The placed block must meet the tight bound relative to the proved base."""
    if problem.contract.target == "tbl" or problem.alignment_root is None:
        return
    root = problem.current[problem.alignment_root]
    problem.check(
        "alignment",
        z3.Not(
            _aligned(
                problem.current[problem.contract.source],
                root,
                problem.context.L,
            )
        ),
    )


def verify_motion_block(
    initial_condition,
    body,
    constants,
    contract,
    *,
    context=None,
    noise=None,
    block_v="0",
    timeout_ms=5000,
    initial_positions=None,
    entry_positions=None,
    initial_bindings=None,
    initial_arm=None,
    symbolic_context=None,
    symbolic_body=None,
    postcondition=None,
):
    start = perf_counter()
    if contract is None:
        return MotionVerificationResult(
            [
                MotionCheck(
                    "contract",
                    "unsupported",
                    reason="Explicit placement contract required",
                )
            ],
            noise,
        )
    context = context or LowLevelContext(default_L=0.05, use_tbl="tbl" in constants)
    from synthesis.api.instructions import Get, MoveByName, PickByName, ReleaseByName
    from synthesis.verification_lib.primitive_motion import PrimitiveMotionProblem

    primitive = any(
        isinstance(i, (Get, PickByName, MoveByName, ReleaseByName)) for i in body
    )
    problem_type = PrimitiveMotionProblem if primitive else MotionProblem
    options = {"initial_arm": initial_arm} if primitive else {}
    problem = problem_type(
        context,
        initial_condition,
        constants,
        contract,
        noise,
        block_v,
        timeout_ms,
        initial_positions,
        entry_positions,
        initial_bindings,
        **options,
    )
    from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext
    from synthesis.verification_lib.root_selection import RootContext

    high = symbolic_context or HighLevelContext(
        sort_name=context.sort_name, use_tbl=context.use_tbl
    )
    roots = RootContext(high, z3.And(*initial_condition), timeout_ms)
    assume_input_alignment(problem, roots.initial_roots(constants))
    if problem.check("initial_consistency").status == "valid":
        prepare_alignment(
            problem,
            roots,
            constants,
            continuation=symbolic_body or (),
            postcondition=postcondition,
        )
        problem.execute(body)
        # Include a second consistency check: transition equations or aliases must
        # never make all postcondition queries vacuously true.
        if problem.check("transition_consistency").status == "valid":
            check_contract_realization(problem)
            check_frame_preservation(problem)
            check_abstract_effects(problem)
            check_alignment(problem)
    return MotionVerificationResult(problem.checks, noise, 1, perf_counter() - start)


MotionVerify = verify_motion_block
