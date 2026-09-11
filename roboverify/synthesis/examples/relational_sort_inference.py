"""Learn relational invariants for the pointer-free, two-loop robotic sorter.

Unary features are atomic observations with one program argument fixed; none
evaluates sortedness or the requested invariant. Targets are used only AFTER
inference. All formulas and clause provenance come from RoboVerify unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Sequence

import z3

from synthesis.inference_lib.relational import (
    RelationSpec, RelationalDomain, RelationalSnapshot, RelationalInferenceResult,
    infer_relational_invariants,
)
from synthesis.examples.robotic_sort_inference import evaluate_formula


OUTER_NAMES = ("NextStar", "KeyLess", "BeforeB")
INNER_NAMES = (*OUTER_NAMES, "BeforeR", "IsB", "IsX", "AfterX", "SelectedLess")
PROJECTIONS = {
    "outer": (("prefix_order", ("BeforeB", "NextStar", "KeyLess"), 2),),
    "inner": (
        ("boundary", ("BeforeB", "BeforeR", "IsX"), 1),
        ("distinct_cursors", ("IsB", "IsX"), 1),
        ("order_except_selected", ("BeforeR", "IsX", "NextStar", "KeyLess"), 2),
        ("passed_blocks", ("BeforeR", "AfterX", "IsX", "SelectedLess"), 1),
    ),
}


def relation_value(name: str, snapshot: RelationalSnapshot, args: tuple[str, ...]) -> bool:
    state = snapshot.payload
    u = args[0]
    # Ranks belong to the perception adapter, not the sorting program.
    rank = {block: position for position, block in enumerate(state.order)}
    if name == "NextStar":
        return rank[u] <= rank[args[1]]
    if name == "KeyLess":
        return state.key_map[u] < state.key_map[args[1]]
    if name == "BeforeB":
        return rank[u] <= rank[state.b]
    if name == "BeforeR":
        return rank[u] <= rank[state.r]
    if name == "IsB":
        return u == state.b
    if name == "IsX":
        return u == state.x
    if name == "AfterX":
        return rank[state.x] <= rank[u]
    if name == "SelectedLess":
        return state.key_map[state.x] < state.key_map[u]
    raise KeyError(name)


def mathematical_axioms(domain: RelationalDomain) -> tuple[z3.BoolRef, ...]:
    """Linear reachability and strict key order; never a sorting specification."""
    a, b, c = z3.Consts("axiom_a axiom_b axiom_c", domain.object_sort)
    axioms = []
    if "NextStar" in domain.relations:
        reach = domain.relations["NextStar"]
        axioms.extend((
            z3.ForAll(a, reach(a, a)),
            z3.ForAll([a, b], z3.Implies(z3.And(reach(a, b), reach(b, a)), a == b)),
            z3.ForAll([a, b], z3.Or(reach(a, b), reach(b, a))),
            z3.ForAll([a, b, c], z3.Implies(z3.And(reach(a, b), reach(b, c)), reach(a, c))),
        ))
    if "KeyLess" in domain.relations:
        less = domain.relations["KeyLess"]
        axioms.extend((
            z3.ForAll(a, z3.Not(less(a, a))),
            z3.ForAll([a, b, c], z3.Implies(z3.And(less(a, b), less(b, c)), less(a, c))),
            # Negative transitivity allows ties, and excludes incomparable keys.
            z3.ForAll([a, b, c], z3.Implies(z3.And(z3.Not(less(a, b)), z3.Not(less(b, c))),
                                          z3.Not(less(a, c)))),
        ))
    return tuple(axioms)


def definitional_axioms(domain: RelationalDomain) -> tuple[z3.BoolRef, ...]:
    """Expand each unary view into its primitive relation/program constant."""
    u = domain.constant("definition_u")
    b, x, r = (domain.constant(name) for name in ("b", "x", "r"))
    rel = domain.relations
    definitions = []
    if "NextStar" in rel:
        reach = rel["NextStar"]
        for name, rhs in (("BeforeB", reach(u, b)), ("BeforeR", reach(u, r)),
                          ("AfterX", reach(x, u))):
            if name in rel:
                definitions.append(z3.ForAll(u, rel[name](u) == rhs))
    for name, rhs in (("IsB", u == b), ("IsX", u == x)):
        if name in rel:
            definitions.append(z3.ForAll(u, rel[name](u) == rhs))
    if "SelectedLess" in rel and "KeyLess" in rel:
        definitions.append(z3.ForAll(u, rel["SelectedLess"](u) == rel["KeyLess"](x, u)))
    return tuple(definitions)


def build_domain(names: Sequence[str], *, definitions: bool = False) -> RelationalDomain:
    def axioms(domain):
        return (*mathematical_axioms(domain), *(definitional_axioms(domain) if definitions else ()))
    return RelationalDomain(
        name="RelationalInsertionBlock",
        relation_specs=tuple(RelationSpec(name, 2 if name in ("NextStar", "KeyLess") else 1,
            lambda snapshot, args, name=name: relation_value(name, snapshot, args)) for name in names),
        include_equality=True,
        axiom_builder=axioms,
    )


def adapt_snapshot(state: Any, *, constants: bool = False) -> RelationalSnapshot:
    bindings = {}
    if constants:
        bindings["b"] = state.b
        if state.x is not None:
            bindings.update(x=state.x, r=state.r)
    return RelationalSnapshot(objects=state.objects, payload=state, constant_bindings=bindings)


def target_formulas(domain: RelationalDomain, phase: str) -> dict[str, z3.BoolRef]:
    """The formulas from the conversation, expressed only in primitives."""
    u, v = z3.Consts("theory_u theory_v", domain.object_sort)
    b, x, r = (domain.constant(name) for name in ("b", "x", "r"))
    reach, less = domain.relations["NextStar"], domain.relations["KeyLess"]
    if phase == "outer":
        return {"sorted_prefix": z3.ForAll([u, v], z3.Implies(
            z3.And(reach(u, v), reach(v, b)), z3.Not(less(v, u))))}
    if phase != "inner":
        raise ValueError(phase)
    return {
        "distinct_b_x": b != x,
        "prefix_boundary": z3.ForAll(u, reach(u, r) == z3.Or(reach(u, b), u == x)),
        "sorted_except_x": z3.ForAll([u, v], z3.Implies(
            z3.And(u != x, v != x, reach(u, v), reach(v, r)), z3.Not(less(v, u)))),
        "passed_blocks_strictly_larger": z3.ForAll(u, z3.Implies(
            z3.And(u != x, reach(x, u), reach(u, r)), less(x, u))),
    }


@dataclass
class LearnedLoop:
    phase: str
    domain: RelationalDomain
    projections: dict[str, RelationalInferenceResult]
    invariant: z3.BoolRef
    elapsed_seconds: float

    def summary(self) -> dict:
        return {
            "phase": self.phase, "elapsed_seconds": self.elapsed_seconds,
            "invariant": self.invariant.sexpr(),
            "projections": {name: {
                "vocabulary": [str(atom) for atom in result.vocabulary],
                "distinct_truth_rows": len(result.truth_rows),
                "clauses": [{"expression": clause.expr.sexpr(),
                             "target_atom": clause.target_predicate, "learned_via": clause.learned_via,
                             "omega_index": clause.omega_index} for clause in result.clauses],
            } for name, result in self.projections.items()},
        }


def phase_states(states: Sequence[Any], phase: str) -> tuple[Any, ...]:
    return tuple(state for state in states if state.program_point in (phase + "_head", phase + "_exit"))


def infer_loop(states: Sequence[Any], phase: str) -> LearnedLoop:
    selected = phase_states(states, phase)
    if not selected:
        raise ValueError(f"No {phase} snapshots")
    results = {}
    started = perf_counter()
    for name, names, k in PROJECTIONS[phase]:
        domain = build_domain(names)
        snapshots = tuple(adapt_snapshot(state) for state in selected)
        for snapshot in snapshots:
            if not all(evaluate_formula(ax, domain, snapshot) for ax in mathematical_axioms(domain)):
                raise AssertionError(f"Invalid observation in {snapshot.payload.case_name}")
        print(f"Infer {phase}/{name}: {len(snapshots)} physical snapshots, k={k}", flush=True)
        results[name] = infer_relational_invariants(domain, snapshots, k=k, verbose=False)
    full = build_domain(OUTER_NAMES if phase == "outer" else INNER_NAMES, definitions=True)
    return LearnedLoop(phase, full, results, z3.And(*(result.invariant for result in results.values())),
                       perf_counter() - started)


def compare_with_theory(loop: LearnedLoop, *, timeout_ms: int = 30_000) -> dict:
    """Distinguish consistency, entailment, and equivalence; UNKNOWN is no proof."""
    targets = target_formulas(loop.domain, loop.phase)
    theory = z3.And(*targets.values())
    axioms = (*mathematical_axioms(loop.domain), *definitional_axioms(loop.domain))

    def query(*conditions):
        solver = z3.Solver()
        solver.set(timeout=timeout_ms)
        solver.add(*axioms, *conditions)
        status = solver.check()
        answer = {"status": str(status)}
        if status == z3.sat:
            answer["model"] = str(solver.model())
        elif status == z3.unknown:
            answer["reason"] = solver.reason_unknown()
        return answer

    result = {
        "scope": "First-order relational models under mathematical and feature-definition axioms; no fixed block count.",
        "axioms_satisfiable": query(),
        "candidate_satisfiable": query(loop.invariant),
        "theory_satisfiable": query(theory),
        "joint_satisfiable": query(loop.invariant, theory),
        "learned_implies_each_target": {name: query(loop.invariant, z3.Not(target))
                                        for name, target in targets.items()},
        "theory_implies_learned": query(theory, z3.Not(loop.invariant)),
        "target_formulas": {name: target.sexpr() for name, target in targets.items()},
        # A target must not already be an axiom disguised as a feature definition.
        "targets_not_assumed": {name: query(z3.Not(target)) for name, target in targets.items()},
    }
    result["consistent"] = result["joint_satisfiable"]["status"] == "sat"
    result["entails_theory"] = (result["candidate_satisfiable"]["status"] == "sat" and
        all(item["status"] == "unsat" for item in result["learned_implies_each_target"].values()))
    result["equivalent"] = result["entails_theory"] and result["theory_implies_learned"]["status"] == "unsat"
    return result


def validate_snapshots(loop: LearnedLoop, states: Sequence[Any]) -> dict:
    selected = phase_states(states, loop.phase)
    targets = target_formulas(loop.domain, loop.phase)
    violations = []
    for index, state in enumerate(selected):
        snapshot = adapt_snapshot(state, constants=True)
        checks = {"actual_learned_formula": loop.invariant, **targets}
        for name, formula in checks.items():
            if not evaluate_formula(formula, loop.domain, snapshot):
                violations.append({"snapshot": index, "case": state.case_name,
                                   "program_point": state.program_point, "step": state.step, "check": name})
        for axiom in (*mathematical_axioms(loop.domain), *definitional_axioms(loop.domain)):
            if not evaluate_formula(axiom, loop.domain, snapshot):
                violations.append({"snapshot": index, "case": state.case_name, "check": "domain_axiom"})
    return {"snapshots_checked": len(selected), "violations": violations,
            "all_passed": bool(selected) and not violations}
