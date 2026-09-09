"""Finite-object inductiveness checks for the robotic insertion-sort adapter.

The input formulas are the *actual* outputs of relational inference. They are
interpreted over separate logical row assignments and physical occupancy, then
checked against every symbolic state/transition for a fixed block count. Keys
are arbitrary mathematical integers, including equal keys.

This is a task-specific abstraction, not a proof of the MuJoCo controller or
automatic Python-source verification. A successful transfer moves its named
block to an empty destination, leaves all other block occupancies unchanged,
and returns with an empty gripper. Reachability/grasp success are assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
from typing import Any, Sequence

import z3

from synthesis.inference_lib.relational import RelationalDomain


@dataclass(frozen=True)
class _State:
    """Row IDs and physical slots are independent; -1 denotes vacancy/buffer."""

    row: tuple[z3.ArithRef, ...]
    physical: tuple[z3.ArithRef, ...]
    i: z3.ArithRef
    j: z3.ArithRef
    selected: z3.ArithRef


def _lookup(values: Sequence[z3.ArithRef], index: z3.ArithRef) -> z3.ArithRef:
    """Finite array access; callers establish that the index is in range."""
    result = values[-1]
    for position in reversed(range(len(values) - 1)):
        result = z3.If(index == position, values[position], result)
    return result


def _put(
    values: Sequence[z3.ArithRef], index: z3.ArithRef, value: z3.ArithRef
) -> tuple[z3.ArithRef, ...]:
    return tuple(z3.If(index == position, value, old) for position, old in enumerate(values))


def _logical_slot(state: _State, block: z3.ArithRef) -> z3.ArithRef:
    result = z3.IntVal(-1)
    for position in reversed(range(len(state.row))):
        result = z3.If(state.row[position] == block, position, result)
    return result


def _interpret(
    formula: z3.ExprRef,
    domain: RelationalDomain,
    state: _State,
    keys: Sequence[z3.ArithRef],
) -> z3.BoolRef:
    """Expand finite object quantifiers and interpret named predicates exactly.

    No inferred clause is removed or replaced by a hand-written target. Domain
    axioms need not be assumed: the concrete predicate definitions implement
    key order, physical row order, and program regions directly.
    """
    n = len(state.row)

    def relation(name: str, arguments: list[z3.ArithRef]) -> z3.BoolRef:
        block = arguments[0]
        physical = _lookup(state.physical, block)
        logical = _logical_slot(state, block)
        in_row = z3.And(physical >= 0, physical < n)
        if name == "KeyLE":
            return _lookup(keys, block) <= _lookup(keys, arguments[1])
        if name == "Before":
            other = _lookup(state.physical, arguments[1])
            return z3.And(in_row, other >= 0, other < n, physical < other)
        if name == "Processed":
            return z3.And(logical >= 0, logical < state.i)
        if name == "Active":
            return z3.And(in_row, physical <= state.i)
        if name == "Shifted":
            return z3.And(in_row, physical >= state.j + 2, physical <= state.i)
        if name == "InRow":
            return in_row
        if name == "InBuffer":
            return physical == -1
        if name == "AtAssignedSlot":
            # A buffered block has no assigned *row* slot, so this is false.
            return z3.And(logical >= 0, physical == logical)
        raise ValueError(f"No symbolic interpretation for relation {name!r}.")

    def visit(expression: z3.ExprRef, bound: tuple[z3.ArithRef, ...] = ()) -> z3.ExprRef:
        if z3.is_var(expression):
            return bound[z3.get_var_index(expression)]
        if z3.is_quantifier(expression):
            if any(expression.var_sort(k) != domain.object_sort for k in range(expression.num_vars())):
                raise ValueError("Only finite block-object quantifiers are supported.")
            instances = [
                visit(expression.body(), tuple(z3.IntVal(value) for value in reversed(values)) + bound)
                for values in product(range(n), repeat=expression.num_vars())
            ]
            if expression.is_forall():
                return z3.And(*instances)
            if expression.is_exists():
                return z3.Or(*instances)
            raise ValueError("Lambda expressions are not invariant formulas.")
        if z3.is_true(expression) or z3.is_false(expression):
            return expression
        if z3.is_const(expression) and expression.sort() == domain.object_sort:
            if str(expression.decl().name()) == "selected":
                return state.selected
            raise ValueError(f"Unbound object constant in invariant: {expression}.")
        arguments = [visit(child, bound) for child in expression.children()]
        name = str(expression.decl().name())
        if name in domain.relations:
            return relation(name, arguments)
        kind = expression.decl().kind()
        if kind == z3.Z3_OP_EQ:
            return arguments[0] == arguments[1]
        if kind == z3.Z3_OP_DISTINCT:
            return z3.Distinct(*arguments)
        if kind == z3.Z3_OP_NOT:
            return z3.Not(arguments[0])
        if kind == z3.Z3_OP_AND:
            return z3.And(*arguments)
        if kind == z3.Z3_OP_OR:
            return z3.Or(*arguments)
        if kind == z3.Z3_OP_IMPLIES:
            return z3.Implies(*arguments)
        if kind == z3.Z3_OP_XOR:
            return z3.Xor(*arguments)
        if kind == z3.Z3_OP_ITE:
            return z3.If(*arguments)
        raise ValueError(f"Unsupported invariant expression: {expression}.")

    return z3.simplify(visit(formula))


def verify_sort_invariants(
    outer_domain: RelationalDomain,
    outer_invariant: z3.BoolRef,
    inner_domain: RelationalDomain,
    inner_invariant: z3.BoolRef,
    n: int = 4,
    timeout_ms: int = 30_000,
) -> dict[str, Any]:
    """Check learned invariants across all abstract states for a fixed ``n``.

    Structural assertions supply identity conservation, one inner-loop vacancy
    at ``j + 1``, index bounds, and unique physical occupancy. They do *not*
    assume sortedness or logical/physical agreement. Those must follow from the
    inferred formulas and are checked through initialization and preservation.

    The report is JSON serializable. ``all_passed`` requires every negated
    obligation to be UNSAT and every branch antecedent to be SAT. UNKNOWN and
    inconsistent/vacuous premises count as failures. SAT counterexamples retain
    the actual symbolic state's keys, row, physical occupancy, and indices.
    """
    if n < 2:
        raise ValueError("Use n >= 2 so both loop entry and skip branches exist.")
    keys = tuple(z3.Int(f"verify_key_{block}") for block in range(n))
    state = _State(
        row=tuple(z3.Int(f"verify_row_{slot}") for slot in range(n)),
        physical=tuple(z3.Int(f"verify_physical_{block}") for block in range(n)),
        i=z3.Int("verify_i"),
        j=z3.Int("verify_j"),
        selected=z3.Int("verify_selected"),
    )

    def physical_structure(s: _State) -> z3.BoolRef:
        return z3.And(
            *(z3.And(position >= -1, position < n) for position in s.physical),
            z3.Distinct(*s.physical),
        )

    def outer_structure(s: _State) -> z3.BoolRef:
        return z3.And(
            s.i >= 1, s.i <= n,
            *(z3.And(block >= 0, block < n) for block in s.row),
            z3.Distinct(*s.row), physical_structure(s),
        )

    def inner_structure(s: _State) -> z3.BoolRef:
        return z3.And(
            s.i >= 1, s.i < n, s.j >= -1, s.j < s.i,
            s.selected >= 0, s.selected < n,
            *(z3.And(block >= -1, block < n, block != s.selected,
                     (block == -1) == (slot == s.j + 1))
              for slot, block in enumerate(s.row)),
            z3.Distinct(*s.row), physical_structure(s),
        )

    def outer(s: _State) -> z3.BoolRef:
        return z3.And(outer_structure(s), _interpret(outer_invariant, outer_domain, s, keys))

    def inner(s: _State) -> z3.BoolRef:
        return z3.And(inner_structure(s), _interpret(inner_invariant, inner_domain, s, keys))

    def transfer_precondition(s: _State, block: z3.ArithRef,
                              source: z3.ArithRef, destination: z3.ArithRef) -> z3.BoolRef:
        return z3.And(
            _lookup(s.physical, block) == source,
            *(position != destination for position in s.physical),
        )

    def counterexample(model: z3.ModelRef, s: _State = state) -> dict[str, Any]:
        def integer(value: z3.ArithRef) -> int:
            return model.eval(value, model_completion=True).as_long()
        return {
            "keys_by_block_id": [integer(value) for value in keys],
            "row_block_ids_minus_one_is_vacancy": [integer(value) for value in s.row],
            "physical_slots_by_block_minus_one_is_buffer": [integer(value) for value in s.physical],
            "i": integer(s.i), "j": integer(s.j),
            "selected_block_id": integer(s.selected),
        }

    checks: list[dict[str, Any]] = []

    def check(name: str, premise: z3.BoolRef, conclusion: z3.BoolRef,
              successor: _State | None = None) -> None:
        solver = z3.Solver()
        solver.set(timeout=timeout_ms)
        solver.add(premise)
        nonvacuity = solver.check()
        solver.add(z3.Not(conclusion))
        outcome = solver.check()
        item: dict[str, Any] = {
            "name": name, "status": str(outcome),
            "antecedent_status": str(nonvacuity),
            "passed": outcome == z3.unsat and nonvacuity == z3.sat,
        }
        if outcome == z3.sat:
            item["counterexample"] = counterexample(solver.model())
            if successor is not None:
                item["counterexample_successor"] = counterexample(solver.model(), successor)
            if name.startswith("outer_") or name == "save_transfer_precondition":
                item["counterexample_note"] = (
                    "At outer heads, j and selected are not live program variables. "
                    "The insertion branch selects row[i]; see the successor when present."
                )
        elif outcome == z3.unknown:
            item["reason_unknown"] = solver.reason_unknown()
        checks.append(item)

    # Initially the row is an arbitrary permutation, physically realized, with
    # arbitrary integer keys. Its one-element prefix is necessarily sorted.
    initial = z3.And(
        outer_structure(state), state.i == 1,
        *(state.physical[block] == _logical_slot(state, z3.IntVal(block)) for block in range(n)),
    )
    check("outer_initialization", initial, outer(state))

    selected = _lookup(state.row, state.i)
    preceding = _lookup(state.row, state.i - 1)
    needs_insertion = _lookup(keys, preceding) > _lookup(keys, selected)
    outer_entry = z3.And(outer(state), state.i < n, needs_insertion)
    check("save_transfer_precondition", outer_entry,
          transfer_precondition(state, selected, state.i, z3.IntVal(-1)))
    buffered = replace(
        state, row=_put(state.row, state.i, z3.IntVal(-1)),
        physical=_put(state.physical, selected, z3.IntVal(-1)),
        selected=selected, j=state.i - 1,
    )
    check("outer_to_inner_after_buffering", outer_entry, inner(buffered), buffered)

    shifted = _lookup(state.row, state.j)
    shift_guard = z3.And(state.j >= 0, _lookup(keys, shifted) > _lookup(keys, state.selected))
    inner_shift = z3.And(inner(state), shift_guard)
    check("shift_transfer_precondition", inner_shift,
          transfer_precondition(state, shifted, state.j, state.j + 1))
    after_shift = replace(
        state,
        row=_put(_put(state.row, state.j + 1, shifted), state.j, z3.IntVal(-1)),
        physical=_put(state.physical, shifted, state.j + 1), j=state.j - 1,
    )
    check("inner_shift_preservation", inner_shift, inner(after_shift), after_shift)

    inner_exit = z3.And(inner(state), z3.Not(shift_guard))
    check("insert_transfer_precondition", inner_exit,
          transfer_precondition(state, state.selected, z3.IntVal(-1), state.j + 1))
    after_insertion = replace(
        state, row=_put(state.row, state.j + 1, state.selected),
        physical=_put(state.physical, state.selected, state.j + 1), i=state.i + 1,
    )
    check("inner_exit_restores_outer", inner_exit, outer(after_insertion), after_insertion)

    skipped = replace(state, i=state.i + 1)
    check("outer_skip_preservation", z3.And(outer(state), state.i < n, z3.Not(needs_insertion)),
          outer(skipped), skipped)

    postcondition = z3.And(
        *(state.physical[block] == _logical_slot(state, z3.IntVal(block)) for block in range(n)),
        *(_lookup(keys, state.row[slot]) <= _lookup(keys, state.row[slot + 1]) for slot in range(n - 1)),
        *(position >= 0 for position in state.physical),
    )
    check("program_exit_sorted_and_physically_realized", z3.And(outer(state), state.i == n), postcondition)

    return {
        "all_passed": all(item["passed"] for item in checks),
        "block_count": n,
        "key_domain": "unbounded mathematical integers; duplicate keys included",
        "method": "finite object grounding of the actual learned formulas; symbolic induction",
        "structural_invariant": [
            "outer row is a permutation of all block identities; 1 <= i <= n",
            "inner row contains every block except selected once, with one vacancy at j + 1",
            "inner bounds: 1 <= i < n and -1 <= j < i",
            "distinct physical occupancies in row slots 0..n-1 or buffer -1",
        ],
        "transfer_contract": [
            "source contains the named block and destination is empty (checked before each transfer)",
            "successful transfer places that block at destination and frames other block occupancies",
            "gripper is empty at sampled loop boundaries; safe reach and grasp success assumed",
        ],
        "guarantee_boundary": (
            "Fixed-size abstract partial correctness under successful-transfer contracts; "
            "does not prove the Python-to-model translation, MuJoCo controller, physical robot, "
            "arbitrary block counts, or stability of equal-key identities."
        ),
        "checks": checks,
    }


__all__ = ["verify_sort_invariants"]
