"""RoboVerify relational inference over measured robotic sorting snapshots.

Targets are used only to assess the learned result, never as learning features
or axioms. Small vocabulary projections are learned independently by the existing
backend and conjoined, keeping its Boolean search tractable.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from time import perf_counter
from typing import Any, Sequence

import z3

from synthesis.inference_lib.relational import (
    RelationSpec, RelationalDomain, RelationalSnapshot, RelationalInferenceResult,
    infer_relational_invariants,
)


OUTER_RELATIONS = ("Processed", "Before", "KeyLE", "InRow", "InBuffer", "AtAssignedSlot")
INNER_RELATIONS = ("Active", "Shifted", "Before", "KeyLE", "InRow", "InBuffer", "AtAssignedSlot")


def relation_value(name: str, snapshot: RelationalSnapshot, args: tuple[str, ...]) -> bool:
    """Evaluate primitive features; no evaluator tests whether a prefix is sorted."""
    state = snapshot.payload
    x = args[0]
    slots = state.slot_map
    if name == "KeyLE":
        return bool(state.key_map[x] <= state.key_map[args[1]])
    if name == "Before":
        a, b = slots[x], slots[args[1]]
        return bool(a is not None and b is not None and a < b)
    if name == "Processed":
        return x in state.order[:state.i]
    if name == "InRow":
        return slots[x] is not None
    if name == "InBuffer":
        return x in state.buffered
    if name == "AtAssignedSlot":
        return bool(x in state.order and slots[x] == state.order.index(x))
    if name == "Active":
        return bool(slots[x] is not None and 0 <= slots[x] <= state.i)
    if name == "Shifted":
        if state.j is None:
            raise ValueError("Shifted requires an inner-loop snapshot")
        return bool(slots[x] is not None and state.j + 2 <= slots[x] <= state.i)
    raise KeyError(name)


def domain_axioms(domain: RelationalDomain) -> tuple[z3.BoolRef, ...]:
    """Only mathematical/definitional properties, never the sorting targets."""
    r = domain.relations
    x, y, z = z3.Consts("axiom_x axiom_y axiom_z", domain.object_sort)
    axioms = []
    if "Before" in r:
        b = r["Before"]
        axioms += [z3.ForAll(x, z3.Not(b(x, x))),
                   z3.ForAll([x, y, z], z3.Implies(z3.And(b(x, y), b(y, z)), b(x, z)))]
        if "InRow" in r:
            row = r["InRow"]
            axioms += [z3.ForAll([x, y], z3.Implies(b(x, y), z3.And(row(x), row(y)))),
                       z3.ForAll([x, y], z3.Implies(z3.And(row(x), row(y)),
                                                   z3.Or(x == y, b(x, y), b(y, x))))]
    if "KeyLE" in r:
        le = r["KeyLE"]
        axioms += [z3.ForAll(x, le(x, x)),
                   z3.ForAll([x, y], z3.Or(le(x, y), le(y, x))),
                   z3.ForAll([x, y, z], z3.Implies(z3.And(le(x, y), le(y, z)), le(x, z)))]
    if {"InRow", "InBuffer"} <= r.keys():
        axioms.append(z3.ForAll(x, z3.Not(z3.And(r["InRow"](x), r["InBuffer"](x)))))
    if {"InRow", "AtAssignedSlot"} <= r.keys():
        axioms.append(z3.ForAll(x, z3.Implies(r["AtAssignedSlot"](x), r["InRow"](x))))
    return tuple(axioms)


def build_domain(names: Sequence[str]) -> RelationalDomain:
    return RelationalDomain(
        name="RoboticSortBlock",
        relation_specs=tuple(RelationSpec(
            name, 2 if name in ("Before", "KeyLE") else 1,
            lambda state, args, name=name: relation_value(name, state, args),
        ) for name in names),
        include_equality=True, axiom_builder=domain_axioms,
    )


def adapt_snapshot(state: Any, *, selected_constant: bool = False) -> RelationalSnapshot:
    bindings = {}
    if selected_constant:
        if state.selected is None:
            raise ValueError("Inner-loop inference requires a selected block")
        bindings["selected"] = state.selected
    return RelationalSnapshot(objects=state.objects, payload=state, constant_bindings=bindings)


def evaluate_formula(expr: z3.ExprRef, domain: RelationalDomain,
                     snapshot: RelationalSnapshot) -> bool:
    """Evaluate a learned formula on every finite object assignment independently of SMT."""
    def visit(node: z3.ExprRef, bound: tuple[str, ...] = ()):
        if z3.is_quantifier(node):
            values = (visit(node.body(), tuple(reversed(assignment)) + bound)
                      for assignment in product(snapshot.objects, repeat=node.num_vars()))
            return all(values) if node.is_forall() else any(values)
        if z3.is_var(node):
            return bound[z3.get_var_index(node)]
        if z3.is_true(node):
            return True
        if z3.is_false(node):
            return False
        kind = node.decl().kind()
        if kind == z3.Z3_OP_UNINTERPRETED:
            name = str(node.decl().name())
            if node.num_args() == 0:
                if name in snapshot.constant_bindings:
                    return snapshot.constant_bindings[name]
                if name in snapshot.objects:
                    return name
                raise ValueError(f"Unbound object constant: {name}")
            return domain.evaluate(name, snapshot, tuple(visit(arg, bound) for arg in node.children()))
        args = [visit(arg, bound) for arg in node.children()]
        if kind == z3.Z3_OP_AND:
            return all(args)
        if kind == z3.Z3_OP_OR:
            return any(args)
        if kind == z3.Z3_OP_NOT:
            return not args[0]
        if kind == z3.Z3_OP_IMPLIES:
            return not args[0] or args[1]
        if kind == z3.Z3_OP_EQ:
            return args[0] == args[1]
        if kind == z3.Z3_OP_DISTINCT:
            return len(set(args)) == len(args)
        raise ValueError(f"Unsupported formula node: {node}")
    return bool(visit(expr))


def target_formulas(domain: RelationalDomain, phase: str) -> dict[str, z3.BoolRef]:
    r = domain.relations
    x, y = z3.Consts("target_x target_y", domain.object_sort)
    region = r["Processed" if phase == "outer" else "Active"]
    targets = {"sorted_prefix" if phase == "outer" else "sorted_occupied_active_region":
               z3.ForAll([x, y], z3.Implies(z3.And(region(x), region(y), r["Before"](x, y)),
                                            r["KeyLE"](x, y)))}
    if phase == "outer":
        targets["all_blocks_in_assigned_row_slots"] = z3.ForAll(x, z3.And(r["InRow"](x), r["AtAssignedSlot"](x)))
        targets["empty_buffer"] = z3.ForAll(x, z3.Not(r["InBuffer"](x)))
    else:
        selected = domain.constant("selected")
        targets["shifted_keys_strictly_greater"] = z3.ForAll(x, z3.Implies(r["Shifted"](x), z3.Not(r["KeyLE"](x, selected))))
        targets["selected_is_the_only_buffered_block"] = z3.ForAll(x, r["InBuffer"](x) == (x == selected))
        targets["other_blocks_in_assigned_row_slots"] = z3.ForAll(x, z3.Implies(x != selected, z3.And(r["InRow"](x), r["AtAssignedSlot"](x))))
    return targets


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
                "clauses": [clause.expr.sexpr() for clause in result.clauses],
            } for name, result in self.projections.items()},
        }


def infer_loop(states: Sequence[Any], phase: str) -> LearnedLoop:
    if phase not in ("outer", "inner"):
        raise ValueError(phase)
    wanted = ("outer_head", "outer_exit") if phase == "outer" else ("inner_head",)
    states = tuple(state for state in states if state.program_point in wanted)
    if not states:
        raise ValueError(f"No {phase} snapshots")
    projections = [
        ("ordering", ("Processed" if phase == "outer" else "Active", "Before", "KeyLE", "InRow"), 2, False),
        ("physical", ("InRow", "InBuffer", "AtAssignedSlot"), 1, phase == "inner"),
    ]
    if phase == "inner":
        projections.append(("shifted_comparison", ("Shifted", "KeyLE"), 1, True))
    results = {}
    start = perf_counter()
    for name, names, k, selected in projections:
        domain = build_domain(names)
        snapshots = tuple(adapt_snapshot(state, selected_constant=selected) for state in states)
        for snapshot in snapshots:
            if not all(evaluate_formula(axiom, domain, snapshot) for axiom in domain_axioms(domain)):
                raise ValueError(f"Concrete axiom failure in {phase}/{name}: {snapshot.payload}")
        print(f"Inferring {phase}/{name}: {len(snapshots)} snapshots, k={k}", flush=True)
        results[name] = infer_relational_invariants(domain, snapshots, k=k,
                           constant_names=("selected",) if selected else (), verbose=False)
    return LearnedLoop(phase, build_domain(OUTER_RELATIONS if phase == "outer" else INNER_RELATIONS),
                       results, z3.And(*(result.invariant for result in results.values())), perf_counter() - start)


def validate_learned(loop: LearnedLoop, states: Sequence[Any]) -> dict:
    """Check nonvacuous entailment, actual learned formulas, and targets on held-out states."""
    domain = loop.domain
    solver = z3.Solver()
    solver.set(timeout=30000)
    domain.add_axioms(solver)
    axioms_result = str(solver.check())
    solver.add(loop.invariant)
    consistency = str(solver.check())
    targets = target_formulas(domain, loop.phase)
    entailment = {}
    for name, target in targets.items():
        solver.push()
        solver.add(z3.Not(target))
        entailment[name] = str(solver.check())
        solver.pop()
    wanted = ("outer_head", "outer_exit") if loop.phase == "outer" else ("inner_head",)
    snapshots = tuple(adapt_snapshot(state, selected_constant=loop.phase == "inner")
                      for state in states if state.program_point in wanted)
    violations = []
    for index, snapshot in enumerate(snapshots):
        if not evaluate_formula(loop.invariant, domain, snapshot):
            violations.append({"index": index, "kind": "learned_formula", "step": snapshot.payload.step})
        for name, target in targets.items():
            if not evaluate_formula(target, domain, snapshot):
                violations.append({"index": index, "kind": name, "step": snapshot.payload.step})
    return {"axioms_satisfiable": axioms_result, "axioms_and_candidate_satisfiable": consistency,
            "target_entailment": entailment, "snapshots_checked": len(snapshots), "violations": violations,
            "all_passed": axioms_result == consistency == "sat" and bool(snapshots) and not violations
                          and all(value == "unsat" for value in entailment.values())}
