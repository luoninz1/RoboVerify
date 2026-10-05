"""Shared guard evaluation for While and Get, including legacy Z3 guards."""

import itertools
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy

import z3

from synthesis.predicates.scene import Scene, evaluate, scene_from_obs
from synthesis.predicates.term import Term
from synthesis.util import on


class NoGuardWitness(RuntimeError):
    pass


_guard_replay = ContextVar("guard_replay", default=None)


@contextmanager
def replay_guard_choices(choices):
    """Replay a validated prefix of enabled choices, then resume normal selection.

    Scoped to this execution: never change a program or the default ID policy.
    Unconsumed or disabled choices fail rather than silently taking another path.
    """
    pending = None if choices is None else deque(deepcopy(choices))
    token = _guard_replay.set(pending)
    try:
        yield
        if pending:
            raise NoGuardWitness(f"Execution left {len(pending)} solver choices unused")
    finally:
        _guard_replay.reset(token)


def evaluate_z3(expr, scene, bindings=None):
    env = dict(scene.bindings, **(bindings or {}))

    def term(node, bound):
        if z3.is_var(node):
            return bound[z3.get_var_index(node)]
        return env[str(node.decl().name())]

    def ev(node, bound=()):
        if isinstance(node, bool):
            return node
        if z3.is_true(node):
            return True
        if z3.is_false(node):
            return False
        if z3.is_quantifier(node):
            values = (
                ev(node.body(), tuple(reversed(combo)) + bound)
                for combo in itertools.product(scene.positions, repeat=node.num_vars())
            )
            return all(values) if node.is_forall() else any(values)
        if z3.is_and(node):
            return all(ev(a, bound) for a in node.children())
        if z3.is_or(node):
            return any(ev(a, bound) for a in node.children())
        if z3.is_not(node):
            return not ev(node.arg(0), bound)
        if z3.is_implies(node):
            return not ev(node.arg(0), bound) or ev(node.arg(1), bound)
        if z3.is_eq(node):
            if node.arg(0).sort().kind() == z3.Z3_BOOL_SORT:
                return ev(node.arg(0), bound) == ev(node.arg(1), bound)
            return term(node.arg(0), bound) == term(node.arg(1), bound)
        if z3.is_distinct(node):
            values = [term(a, bound) for a in node.children()]
            return len(set(values)) == len(values)
        name = str(node.decl().name())
        predicates = {
            "ON_star": on.on_star_implementation,
            "ON_star_zero": on.on_star_implementation,
            "Higher": on.higher_implementation,
            "Scattered": on.scattered_implementation,
        }
        if name not in predicates:
            raise TypeError(f"Unsupported guard predicate: {name}")
        positions = scene.entry_positions if name == "ON_star_zero" else scene.positions
        return bool(
            predicates[name](*(positions[term(a, bound)] for a in node.children()))
        )

    return ev(expr)


def find_and_bind(instruction, env, traj):
    """Replay an enabled prescribed choice, or choose the first matching binding."""
    mapping = getattr(env, "symbolic_name_to_box_id", None)
    if not isinstance(mapping, dict):
        raise ValueError("Guard execution requires symbolic_name_to_box_id")
    count = instruction._get_num_blocks(env, traj[-1])
    scene = scene_from_obs(traj[-1], count, mapping, include_table="tbl" in mapping)
    entry = getattr(instruction, "_guard_entry_positions", None)
    if entry is not None:
        scene.entry_positions = dict(entry)
    names = [
        str(v.decl().name()) if hasattr(v, "decl") else str(v)
        for v in instruction.guard_exists_vars
    ]
    condition = getattr(instruction, "guard_term", None)
    condition = instruction.instantiated_cond if condition is None else condition
    pending = _guard_replay.get()
    prescribed = pending[0] if pending else None
    if prescribed is not None:
        selected = prescribed["bindings"]
        if prescribed["kind"] != type(instruction).__name__ or set(selected) != set(
            names
        ):
            raise NoGuardWitness(
                "Solver choice does not match the next guard instruction"
            )
        if any(value not in scene.positions for value in selected.values()):
            raise NoGuardWitness("Solver choice refers to a nonexistent block")
        candidates = [tuple(selected[name] for name in names)]
    else:
        candidates = itertools.product(scene.positions, repeat=len(names))
    for values in candidates:
        bindings = dict(mapping, **dict(zip(names, values)))
        holds = (
            evaluate(condition, scene, bindings)
            if isinstance(condition, Term)
            else evaluate_z3(condition, scene, bindings)
        )
        if holds:
            mapping.update((name, bindings[name]) for name in names)
            if prescribed is not None:
                pending.popleft()
            return True
    if prescribed is not None:
        raise NoGuardWitness(
            f"Solver choice is not enabled in the actual scene: {prescribed}"
        )
    return False
