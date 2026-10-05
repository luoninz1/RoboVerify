"""Concretize finite relational models without silently changing their meaning."""

import itertools
from copy import deepcopy
from functools import lru_cache

import z3

from synthesis.inference_lib.demo_store import LoopHeadState
from synthesis.util import on
from synthesis.util.symbols import fresh_const


class UnrealizableCounterexample(ValueError):
    pass


def stacks_to_positions(stacks, *, block_length=0.05, table_height=0.4):
    """Canonical bottom-to-top towers, keyed by stable object IDs."""
    positions = {}
    for column, stack in enumerate(stacks):
        for level, name in enumerate(stack):
            if name in positions:
                raise ValueError(f"Duplicate block {name!r}")
            positions[name] = [
                1.2 + column * 3 * block_length,
                0.75,
                table_height + (level + 0.5) * block_length,
            ]
    return positions


@lru_cache(maxsize=256)
def _compile_state_predicate(expr):
    """Compile syntax once; each call still reads the current scene and aliases."""

    def compile_term(term):
        if isinstance(term, bool):
            return lambda state, bound: term
        if z3.is_true(term):
            return lambda state, bound: True
        if z3.is_false(term):
            return lambda state, bound: False
        if z3.is_var(term):
            index = z3.get_var_index(term)
            return lambda state, bound: bound[index]
        if z3.is_quantifier(term):
            body = compile_term(term.body())
            count = term.num_vars()
            reduce = all if term.is_forall() else any
            return lambda state, bound: reduce(
                body(state, tuple(reversed(values)) + bound)
                for values in itertools.product(state.positions, repeat=count)
            )
        args = [compile_term(a) for a in term.children()]
        if z3.is_and(term):
            return lambda state, bound: all(a(state, bound) for a in args)
        if z3.is_or(term):
            return lambda state, bound: any(a(state, bound) for a in args)
        if z3.is_not(term):
            return lambda state, bound: not args[0](state, bound)
        if z3.is_implies(term):
            return lambda state, bound: not args[0](state, bound) or args[1](
                state, bound
            )
        if z3.is_eq(term):
            return lambda state, bound: args[0](state, bound) == args[1](state, bound)
        if z3.is_distinct(term):
            return lambda state, bound: len({a(state, bound) for a in args}) == len(
                args
            )
        name = str(term.decl().name())
        if not args:
            return lambda state, bound: state.constants[name]
        predicates = {
            "ON_star": on.on_star_implementation,
            "ON_star_zero": on.on_star_implementation,
            "Higher": on.higher_implementation,
            "Scattered": on.scattered_implementation,
        }
        if name not in predicates:
            raise ValueError(f"Unsupported predicate: {name}")
        predicate = predicates[name]

        def evaluate_relation(state, bound):
            positions = (
                state.entry_positions if name == "ON_star_zero" else state.positions
            )
            return bool(predicate(*(positions[a(state, bound)] for a in args)))

        return evaluate_relation

    return compile_term(expr)


def state_holds(expr, state):
    """Evaluate the finite tower predicate language on a complete saved scene."""
    return bool(_compile_state_predicate(expr)(state, ()))


def model_to_loop_head(context, model, loop_id, constants, *, timeout_ms=5000):
    """Realize *all* relation tables, including frozen ON*, or refuse the model.

    The abstract axioms allow non-total Higher and arbitrary Scattered tables.
    Merely drawing ON* towers can therefore turn an SMT counterexample into a
    different, non-counterexample scene. A bounded geometric query prevents that.
    """
    if context.mode != "enum":
        raise UnrealizableCounterexample("A finite model is required")
    objects = context.enum_blocks
    table = (
        model.eval(context.get_consts("tbl"), model_completion=True)
        if context.use_tbl
        else None
    )
    names = {
        str(obj): ("tbl" if table is not None and obj.eq(table) else f"x{i+1}")
        for i, obj in enumerate(objects)
    }
    physical = [obj for obj in objects if names[str(obj)] != "tbl"]
    solver = z3.Solver()
    solver.set(timeout=timeout_ms)
    xyz = {
        str(obj): tuple(fresh_const(z3.RealSort(), f"current_{axis}") for axis in "xyz")
        for obj in physical
    }
    entry = {
        str(obj): tuple(fresh_const(z3.RealSort(), f"entry_{axis}") for axis in "xyz")
        for obj in physical
    }
    length = z3.RealVal("0.05")
    for coords in (xyz, entry):
        for pos in coords.values():
            solver.add(pos[2] >= z3.RealVal("0.425"))
        for a, b in itertools.combinations(coords.values(), 2):
            solver.add(z3.Or(*(z3.Abs(x - y) >= length for x, y in zip(a, b))))
    for a, b in itertools.product(physical, repeat=2):
        p, q = xyz[str(a)], xyz[str(b)]
        p0, q0 = entry[str(a)], entry[str(b)]

        def on_star(u, v):
            return z3.And(
                z3.Abs(u[0] - v[0]) < length / 2,
                z3.Abs(u[1] - v[1]) < length / 2,
                u[2] >= v[2],
            )

        formulas = (
            on_star(p, q),
            on_star(p0, q0),
            on.higher_z3(p[2], q[2]),
            z3.Or(z3.Abs(p[0] - q[0]) >= 2 * length, z3.Abs(p[1] - q[1]) >= 2 * length),
        )
        for relation, formula in zip(
            (context.ON_star, context.ON_star_zero, context.Higher, context.Scattered),
            formulas,
        ):
            solver.add(formula == model.eval(relation(a, b), model_completion=True))
    answer = solver.check()
    if answer != z3.sat:
        reason = (
            solver.reason_unknown()
            if answer == z3.unknown
            else "relation tables have no non-overlapping geometric realization"
        )
        raise UnrealizableCounterexample(reason)
    geometric = solver.model()

    def positions(coords):
        result = {
            names[key]: [float(geometric.eval(v).as_fraction()) for v in values]
            for key, values in coords.items()
        }
        if context.use_tbl:
            result["tbl"] = on.TABLE
        return result

    bindings = {
        name: names[str(model.eval(context.get_consts(name), model_completion=True))]
        for name in constants
    }
    if context.use_tbl:
        bindings["tbl"] = "tbl"
    state = LoopHeadState(
        loop_id or "entry", positions(xyz), positions(entry), bindings
    )
    # Float conversion must not cross a strict predicate boundary.
    for relation in (
        context.ON_star,
        context.ON_star_zero,
        context.Higher,
        context.Scattered,
    ):
        for a, b in itertools.product(objects, repeat=2):
            aliases = dict(state.constants, _lhs=names[str(a)], _rhs=names[str(b)])
            row = LoopHeadState(
                state.loop_id, state.positions, state.entry_positions, aliases
            )
            actual = state_holds(
                relation(context.get_consts("_lhs"), context.get_consts("_rhs")), row
            )
            if actual != z3.is_true(model.eval(relation(a, b), model_completion=True)):
                raise UnrealizableCounterexample(
                    "Float conversion changed a predicate boundary"
                )
    return state


def symbolic_successor(state, instructions, *, table_surface_height=0.4):
    """Execute the supported straight-line symbolic body under placement contracts.

    This constructs an abstract successor scene, not a simulator replay. Frozen
    loop-entry positions are preserved. The caller must check that it actually
    violates the old invariant before adding it to D_V.
    """
    from synthesis.api.instructions import Assign, Put, Skip

    row = deepcopy(state)
    # deepcopy of the sentinel must preserve identity.
    for positions in (row.positions, row.entry_positions):
        if "tbl" in positions:
            positions["tbl"] = on.TABLE
    for instruction in instructions:
        if isinstance(instruction, Assign):
            row.constants[instruction.left] = row.constants[instruction.right]
        elif isinstance(instruction, Put):
            source = row.constants[instruction.upper_block]
            target = row.constants[instruction.base_block]
            if target == "tbl":
                old = row.positions[source]
                # Place on a fresh, separated table column.
                x = (
                    max(p[0] for name, p in row.positions.items() if name != "tbl")
                    + 0.15
                )
                row.positions[source] = [x, old[1], table_surface_height + 0.025]
            else:
                x, y, z = row.positions[target]
                row.positions[source] = [x, y, z + on.BLOCK_LENGTH]
        elif not isinstance(instruction, Skip):
            raise UnrealizableCounterexample(
                f"Unsupported symbolic replay: {type(instruction).__name__}"
            )
    return LoopHeadState(row.loop_id, row.positions, row.entry_positions, row.constants)
