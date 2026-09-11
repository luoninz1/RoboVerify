"""Symbolic checks for the two-loop, relation-selected robotic insertion sort.

The actual candidate formulas supplied by the caller are grounded over ``n``
node identities. Positions range over all row permutations; keys are arbitrary
mathematical integers, so duplicate keys are included. No candidate clause is
silently replaced by a hand-written invariant.

This is a fixed-size transition-model proof under a successful pick/move/release
contract. It does not verify Python translation, robot control, collision
avoidance, or arbitrary list sizes. Invariants describe completed loop heads,
not the intermediate states while a block occupies the temporary buffer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from typing import Any, Sequence

import z3

from synthesis.inference_lib.relational import RelationalDomain


@dataclass(frozen=True)
class _State:
    positions: tuple[z3.ArithRef, ...]
    b: z3.ArithRef
    x: z3.ArithRef
    r: z3.ArithRef


def _lookup(values: Sequence[z3.ArithRef], index: z3.ArithRef) -> z3.ArithRef:
    result = values[-1]
    for node in reversed(range(len(values) - 1)):
        result = z3.If(index == node, values[node], result)
    return result


def _put(values: Sequence[z3.ArithRef], node: z3.ArithRef,
         value: z3.ArithRef) -> tuple[z3.ArithRef, ...]:
    return tuple(z3.If(node == k, value, old) for k, old in enumerate(values))


def _interpret(formula: z3.ExprRef, domain: RelationalDomain,
               state: _State, keys: Sequence[z3.ArithRef]) -> z3.BoolRef:
    """Ground object quantifiers and give every supported relation its meaning."""
    n = len(keys)

    def reaches(left: z3.ArithRef, right: z3.ArithRef) -> z3.BoolRef:
        return _lookup(state.positions, left) <= _lookup(state.positions, right)

    def relation(name: str, args: list[z3.ArithRef]) -> z3.BoolRef:
        u = args[0]
        if name == "NextStar":
            return reaches(u, args[1])
        if name == "KeyLess":
            return _lookup(keys, u) < _lookup(keys, args[1])
        if name == "BeforeB":
            return reaches(u, state.b)
        if name == "BeforeR":
            return reaches(u, state.r)
        if name == "IsB":
            return u == state.b
        if name == "IsX":
            return u == state.x
        if name == "AfterX":
            return reaches(state.x, u)
        if name == "SelectedLess":
            return _lookup(keys, state.x) < _lookup(keys, u)
        raise ValueError(f"No symbolic interpretation for relation {name!r}.")

    def visit(expr: z3.ExprRef, bound: tuple[z3.ArithRef, ...] = ()) -> z3.ExprRef:
        if z3.is_var(expr):
            return bound[z3.get_var_index(expr)]
        if z3.is_quantifier(expr):
            if any(expr.var_sort(k) != domain.object_sort for k in range(expr.num_vars())):
                raise ValueError("Only finite node-object quantifiers are supported.")
            instances = [visit(expr.body(), tuple(z3.IntVal(k) for k in reversed(values)) + bound)
                         for values in product(range(n), repeat=expr.num_vars())]
            if expr.is_forall():
                return z3.And(*instances)
            if expr.is_exists():
                return z3.Or(*instances)
            raise ValueError("Lambda expressions are not invariants.")
        if z3.is_true(expr) or z3.is_false(expr):
            return expr
        if z3.is_const(expr) and expr.sort() == domain.object_sort:
            name = str(expr.decl().name())
            if name in ("b", "x", "r"):
                return getattr(state, name)
            raise ValueError(f"Unbound object constant in invariant: {expr}.")
        args = [visit(child, bound) for child in expr.children()]
        name = str(expr.decl().name())
        if name in domain.relations:
            return relation(name, args)
        kind = expr.decl().kind()
        if kind == z3.Z3_OP_EQ:
            return args[0] == args[1]
        if kind == z3.Z3_OP_DISTINCT:
            return z3.Distinct(*args)
        if kind == z3.Z3_OP_NOT:
            return z3.Not(args[0])
        if kind == z3.Z3_OP_AND:
            return z3.And(*args)
        if kind == z3.Z3_OP_OR:
            return z3.Or(*args)
        if kind == z3.Z3_OP_IMPLIES:
            return z3.Implies(*args)
        if kind == z3.Z3_OP_XOR:
            return z3.Xor(*args)
        if kind == z3.Z3_OP_ITE:
            return z3.If(*args)
        raise ValueError(f"Unsupported invariant expression: {expr}.")

    return z3.simplify(visit(formula))


def theoretical_formulas(domain: RelationalDomain) -> dict[str, z3.BoolRef]:
    """The user's hand-written targets, for comparison and verifier fixtures."""
    u, v = z3.Consts("theory_u theory_v", domain.object_sort)
    b, x, r = (domain.constant(name) for name in ("b", "x", "r"))
    nxt, less = domain.relation("NextStar"), domain.relation("KeyLess")
    return {
        "outer": z3.ForAll([u, v], z3.Implies(z3.And(nxt(u, v), nxt(v, b)),
                                              z3.Not(less(v, u)))),
        "inner": z3.And(
            b != x,
            z3.ForAll([u], nxt(u, r) == z3.Or(nxt(u, b), u == x)),
            z3.ForAll([u, v], z3.Implies(
                z3.And(u != x, v != x, nxt(u, v), nxt(v, r)), z3.Not(less(v, u)))),
            z3.ForAll([u], z3.Implies(z3.And(u != x, nxt(x, u), nxt(u, r)), less(x, u))),
        ),
    }


def verify_sort_invariants(outer_domain: RelationalDomain, outer_invariant: z3.BoolRef,
                           inner_domain: RelationalDomain, inner_invariant: z3.BoolRef,
                           n: int = 4, timeout_ms: int = 30_000) -> dict[str, Any]:
    """Check actual candidate invariants against all symbolic fixed-size states.

    Every obligation checks both a satisfiable premise and an unsatisfiable
    negated conclusion. UNKNOWN, counterexamples, and vacuous premises fail.
    Additional two-way comparisons to the theoretical formulas are reported
    separately; equivalence to those formulas is not required for inductiveness.
    """
    return _verify(outer_domain, outer_invariant, inner_domain, inner_invariant,
                   n=n, timeout_ms=timeout_ms)


def _verify(outer_domain: RelationalDomain, outer_invariant: z3.BoolRef,
            inner_domain: RelationalDomain, inner_invariant: z3.BoolRef,
            n: int = 4, timeout_ms: int = 30_000, *,
            wrong_r_update: bool = False, reverse_comparator: bool = False) -> dict[str, Any]:
    """Private mutation switches exercise regression tests, never normal runs."""
    if n < 2:
        raise ValueError("Use n >= 2 so loop-entry and swap premises can be nonvacuous.")
    keys = tuple(z3.Int(f"relational_verify_key_{k}") for k in range(n))
    state = _State(tuple(z3.Int(f"relational_verify_pos_{k}") for k in range(n)),
                   z3.Int("relational_verify_b"), z3.Int("relational_verify_x"),
                   z3.Int("relational_verify_r"))
    p = z3.Int("relational_verify_p")

    def position(s: _State, node: z3.ArithRef) -> z3.ArithRef:
        return _lookup(s.positions, node)

    def valid_node(node: z3.ArithRef) -> z3.BoolRef:
        return z3.And(node >= 0, node < n)

    def structure(s: _State) -> z3.BoolRef:
        return z3.And(*(z3.And(v >= 0, v < n) for v in s.positions),
                      z3.Distinct(*s.positions), *(valid_node(v) for v in (s.b, s.x, s.r)))

    def outer(s: _State) -> z3.BoolRef:
        return z3.And(structure(s), _interpret(outer_invariant, outer_domain, s, keys))

    def inner(s: _State) -> z3.BoolRef:
        return z3.And(structure(s), _interpret(inner_invariant, inner_domain, s, keys))

    def outer_guard(s: _State, selected: z3.ArithRef) -> z3.BoolRef:
        # Exact forall form from the pseudocode, not a pointer dereference.
        return z3.And(valid_node(selected), s.b != selected,
                      position(s, s.b) <= position(s, selected),
                      *(z3.Implies(z3.And(s.b != y, position(s, s.b) <= s.positions[y]),
                                   position(s, selected) <= s.positions[y]) for y in range(n)))

    def adjacent_predecessor(s: _State, predecessor: z3.ArithRef) -> z3.BoolRef:
        return z3.And(valid_node(predecessor), predecessor != s.x,
                      position(s, predecessor) <= position(s, s.x),
                      *(z3.Implies(z3.And(position(s, predecessor) <= s.positions[z],
                                         s.positions[z] <= position(s, s.x)),
                                   z3.Or(predecessor == z, s.x == z)) for z in range(n)))

    def inner_guard(s: _State, predecessor: z3.ArithRef) -> z3.BoolRef:
        selected_key, preceding_key = _lookup(keys, s.x), _lookup(keys, predecessor)
        comparison = preceding_key < selected_key if reverse_comparator else selected_key < preceding_key
        return z3.And(adjacent_predecessor(s, predecessor), comparison)

    def witness(model: z3.ModelRef, s: _State) -> dict[str, Any]:
        def integer(value: z3.ArithRef) -> int:
            return model.eval(value, model_completion=True).as_long()
        positions = [integer(v) for v in s.positions]
        return {
            "keys_by_node": [integer(k) for k in keys], "positions_by_node": positions,
            "row_node_ids": sorted(range(n), key=lambda node: positions[node]),
            "b": integer(s.b), "x": integer(s.x), "r": integer(s.r), "p": integer(p),
        }

    checks: list[dict[str, Any]] = []

    def check(name: str, premise: z3.BoolRef, conclusion: z3.BoolRef,
              successor: _State | None = None, *, destination: list | None = None) -> None:
        solver = z3.Solver()
        solver.set(timeout=timeout_ms)
        solver.add(premise)
        nonvacuity = solver.check()
        nonvacuity_reason = solver.reason_unknown() if nonvacuity == z3.unknown else None
        solver.add(z3.Not(conclusion))
        outcome = solver.check()
        item: dict[str, Any] = {
            "name": name, "status": str(outcome), "antecedent_status": str(nonvacuity),
            "passed": outcome == z3.unsat and nonvacuity == z3.sat,
        }
        if nonvacuity_reason:
            item["antecedent_reason_unknown"] = nonvacuity_reason
        if outcome == z3.sat:
            item["counterexample"] = witness(solver.model(), state)
            if successor is not None:
                item["counterexample_successor"] = witness(solver.model(), successor)
        elif outcome == z3.unknown:
            item["reason_unknown"] = solver.reason_unknown()
        (checks if destination is None else destination).append(item)

    check("outer_initialization", z3.And(structure(state), position(state, state.b) == 0), outer(state))
    entry = z3.And(outer(state), outer_guard(state, state.x))
    entered = replace(state, r=state.x)
    check("outer_to_inner_r_equals_x", entry, inner(entered), entered)

    # Three successful pick/move/release sequences, one temporarily empty row
    # slot. -1 is the reserved buffer, initially empty at every loop head.
    source_x, source_p = position(state, state.x), position(state, p)
    saved = replace(state, positions=_put(state.positions, state.x, z3.IntVal(-1)))
    shifted = replace(saved, positions=_put(saved.positions, p, source_x))
    placed = replace(shifted, positions=_put(shifted.positions, state.x, source_p))
    after_swap = replace(placed, r=state.x if wrong_r_update else state.b)
    swapping = z3.And(inner(state), inner_guard(state, p))

    def transfer_pre(s: _State, node: z3.ArithRef, source: z3.ArithRef,
                     destination: z3.ArithRef) -> z3.BoolRef:
        return z3.And(position(s, node) == source,
                      *(v != destination for v in s.positions))

    check("transfer_x_to_empty_buffer", swapping,
          transfer_pre(state, state.x, source_x, z3.IntVal(-1)))
    check("transfer_p_to_vacated_x_slot", swapping,
          transfer_pre(saved, p, source_p, source_x), saved)
    check("transfer_x_from_buffer_to_vacated_p_slot", swapping,
          transfer_pre(shifted, state.x, z3.IntVal(-1), source_p), shifted)
    check("three_transfers_realize_adjacent_swap", swapping, z3.And(
        structure(placed), position(placed, state.x) == source_p,
        position(placed, p) == source_x,
        *(z3.Implies(z3.And(k != state.x, k != p), placed.positions[k] == state.positions[k])
          for k in range(n)),
        *(v != -1 for v in placed.positions),
        z3.Distinct(*saved.positions), z3.Distinct(*shifted.positions)), placed)
    check("inner_swap_preservation", swapping, inner(after_swap), after_swap)

    exiting_inner = z3.And(inner(state),
                           z3.Not(z3.Or(*(inner_guard(state, z3.IntVal(k)) for k in range(n)))))
    after_inner = replace(state, b=state.r)
    check("inner_exit_restores_outer", exiting_inner, outer(after_inner), after_inner)
    exiting_outer = z3.And(outer(state),
                           z3.Not(z3.Or(*(outer_guard(state, z3.IntVal(k)) for k in range(n)))))
    globally_sorted = z3.And(*(z3.Implies(state.positions[u] <= state.positions[v], keys[u] <= keys[v])
                               for u, v in product(range(n), repeat=2)))
    check("outer_exit_global_sorted", exiting_outer, globally_sorted)

    # The enlarged prefix boundary r has one more position at entry; swaps
    # preserve its position; b <- r installs it. These checks compose with
    # strict descent of x's nonnegative position to justify termination.
    check("outer_entry_extends_prefix_by_one", entry,
          position(entered, entered.r) == position(state, state.b) + 1, entered)
    check("inner_swap_decreases_selected_position", swapping,
          z3.And(position(after_swap, after_swap.x) == source_x - 1,
                 position(after_swap, after_swap.x) >= 0), after_swap)
    check("inner_swap_preserves_enlarged_prefix_size", swapping,
          position(after_swap, after_swap.r) == position(state, state.r), after_swap)
    check("outer_update_installs_enlarged_prefix", exiting_inner,
          position(after_inner, after_inner.b) == position(state, state.r), after_inner)

    comparisons: list[dict[str, Any]] = []
    for point, domain, formula, learned in (
        ("outer", outer_domain, outer_invariant, outer(state)),
        ("inner", inner_domain, inner_invariant, inner(state)),
    ):
        theory = z3.And(structure(state), _interpret(theoretical_formulas(domain)[point], domain, state, keys))
        check(f"{point}_learned_implies_theory", learned, theory, destination=comparisons)
        check(f"{point}_theory_implies_learned", theory,
              _interpret(formula, domain, state, keys), destination=comparisons)

    return {
        "all_passed": all(item["passed"] for item in checks), "block_count": n,
        "key_domain": "arbitrary mathematical integers, including duplicate keys",
        "method": "finite object grounding of actual learned formulas; symbolic transition induction",
        "structural_assumptions": [
            "positions form a permutation of row slots 0..n-1 at each loop head",
            "b, x, r identify existing nodes; each node and its key are conserved",
            "NextStar is reflexive position order; the row is a finite single chain",
        ],
        "transfer_contract": [
            "each completed successful pick/move/release moves its named node to an empty destination",
            "other node positions and every key are unchanged by each completed transfer",
            "x goes to buffer -1, p goes to x's old slot, x goes to p's old slot",
            "each transfer finishes with an empty gripper; buffer is empty at loop heads",
        ],
        "termination_argument": "x's position decreases each inner step; the enlarged prefix has exactly one additional position per completed outer step",
        "guarantee_boundary": (
            "Fixed-n abstract correctness and termination under successful-transfer contracts. "
            "This does not prove source-to-model translation, continuous motion/grasp success, "
            "physical deployment, or arbitrary n. Loop-head formulas are not asserted mid-transfer."
        ),
        "checks": checks, "semantic_comparison": comparisons,
    }


__all__ = ["verify_sort_invariants", "theoretical_formulas"]
