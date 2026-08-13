"""Insertion-sort snapshots and relational invariant-inference configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import z3

from synthesis.inference_lib.relational import (
    RelationalDomain,
    RelationalInferenceResult,
    RelationalSnapshot,
    RelationSpec,
    infer_relational_invariants,
)


class Pair:
    """A sortable key and its associated value."""

    def __init__(self, key: int, value: str):
        self.key = key
        self.value = value

    def __repr__(self) -> str:
        return f"Pair(key={self.key}, value={self.value!r})"


class Solution:
    """The supplied stable, slice-shifting insertion-sort implementation."""

    def insertionSort(self, pairs: List[Pair]) -> List[List[Pair]]:
        list_of_lists: List[List[Pair]] = []

        if pairs is None or pairs == []:
            return list_of_lists

        list_of_lists.append(pairs.copy())

        for i in range(1, len(pairs)):
            pair = pairs[i]
            for j in range(i):
                idx = i - j - 1
                if pairs[idx].key <= pair.key:
                    pairs[idx + 2 : i + 1] = pairs[idx + 1 : i]
                    pairs[idx + 1] = pair
                    break
                if idx == 0:
                    pairs[1 : i + 1] = pairs[0:i]
                    pairs[0] = pair
            list_of_lists.append(pairs.copy())

        return list_of_lists


@dataclass(frozen=True)
class InsertionSortSnapshot:
    """One outer-loop-head state represented with stable object names."""

    outer_index: int
    order: Tuple[str, ...]
    processed: frozenset[str]
    keys: Dict[str, int]
    values: Dict[str, str]


@dataclass(frozen=True)
class InsertionSortRun:
    """Sorted output plus all outer-loop-head/exit snapshots."""

    sorted_pairs: Tuple[Pair, ...]
    snapshots: Tuple[InsertionSortSnapshot, ...]


def trace_insertion_sort(pairs: Sequence[Pair]) -> InsertionSortRun:
    """Run the supplied algorithm and record only outer-loop boundaries."""
    working = list(pairs)
    object_ids = [id(pair) for pair in working]
    if len(set(object_ids)) != len(object_ids):
        raise ValueError("Each input position must contain a distinct Pair object.")

    object_names = {id(pair): f"item_{index}" for index, pair in enumerate(working)}
    keys = {object_names[id(pair)]: pair.key for pair in working}
    values = {object_names[id(pair)]: pair.value for pair in working}

    concrete_snapshots = Solution().insertionSort(working)
    snapshots = []
    for prefix_length, current_order in enumerate(concrete_snapshots, start=1):
        named_order = tuple(object_names[id(pair)] for pair in current_order)
        snapshots.append(
            InsertionSortSnapshot(
                outer_index=prefix_length,
                order=named_order,
                processed=frozenset(named_order[:prefix_length]),
                keys=dict(keys),
                values=dict(values),
            )
        )

    return InsertionSortRun(
        sorted_pairs=tuple(working),
        snapshots=tuple(snapshots),
    )


def to_relational_snapshot(snapshot: InsertionSortSnapshot) -> RelationalSnapshot:
    """Adapt one insertion-sort state to the generic relational interface."""
    return RelationalSnapshot(
        objects=tuple(snapshot.keys),
        payload=snapshot,
    )


def _insertion_sort_axioms(domain: RelationalDomain) -> Iterable[z3.BoolRef]:
    """Axiomatize current position as a strict total order and keys as a total preorder."""
    before = domain.relation("Before")
    key_le = domain.relation("KeyLE")
    x, y, z = z3.Consts("axiom_x axiom_y axiom_z", domain.object_sort)

    return (
        z3.ForAll([x], z3.Not(before(x, x))),
        z3.ForAll(
            [x, y, z],
            z3.Implies(z3.And(before(x, y), before(y, z)), before(x, z)),
        ),
        z3.ForAll([x, y], z3.Or(x == y, before(x, y), before(y, x))),
        z3.ForAll([x], key_le(x, x)),
        z3.ForAll(
            [x, y, z],
            z3.Implies(z3.And(key_le(x, y), key_le(y, z)), key_le(x, z)),
        ),
        z3.ForAll([x, y], z3.Or(key_le(x, y), key_le(y, x))),
    )


def build_insertion_sort_domain() -> RelationalDomain:
    """Create the relational vocabulary and concrete semantics for insertion sort."""

    def processed(snapshot: RelationalSnapshot, arguments: Tuple[str, ...]) -> bool:
        (item,) = arguments
        return item in snapshot.payload.processed

    def before(snapshot: RelationalSnapshot, arguments: Tuple[str, ...]) -> bool:
        left, right = arguments
        positions = {item: index for index, item in enumerate(snapshot.payload.order)}
        return positions[left] < positions[right]

    def key_le(snapshot: RelationalSnapshot, arguments: Tuple[str, ...]) -> bool:
        left, right = arguments
        return snapshot.payload.keys[left] <= snapshot.payload.keys[right]

    return RelationalDomain(
        name="InsertionSortItem",
        relation_specs=(
            RelationSpec("Processed", 1, processed),
            RelationSpec("Before", 2, before),
            RelationSpec("KeyLE", 2, key_le),
        ),
        include_equality=True,
        axiom_builder=_insertion_sort_axioms,
    )


WALKTHROUGH_KEYS = (5, 2, 9)
DEFAULT_TRAINING_KEYS = (
    (1,),
    (1, 2),
    (2, 1),
    (1, 2, 3),
    (3, 2, 1),
    (1, 3, 2),
    (3, 4, 1),
    (2, 1, 2),
    (2, 2, 2),
    (4, 1, 3, 2),
)


def pairs_from_keys(keys: Sequence[int], case_name: str = "case") -> List[Pair]:
    """Create distinct Pair objects with deterministic, human-readable values."""
    return [Pair(key, f"{case_name}_value_{index}") for index, key in enumerate(keys)]


def build_training_snapshots(
    key_sequences: Sequence[Sequence[int]] | None = None,
    include_walkthrough: bool = True,
) -> Tuple[RelationalSnapshot, ...]:
    """Generate a diverse corpus of outer-loop snapshots."""
    cases = list(key_sequences or DEFAULT_TRAINING_KEYS)
    if include_walkthrough:
        cases.insert(0, WALKTHROUGH_KEYS)

    snapshots = []
    for case_index, keys in enumerate(cases):
        run = trace_insertion_sort(pairs_from_keys(keys, f"case_{case_index}"))
        snapshots.extend(to_relational_snapshot(state) for state in run.snapshots)
    return tuple(snapshots)


def sorted_prefix_target(
    domain: RelationalDomain,
    variable_names: Tuple[str, str] = ("target_x", "target_y"),
) -> z3.ExprRef:
    """Return the intended nondecreasing processed-prefix invariant."""
    x, y = z3.Consts(" ".join(variable_names), domain.object_sort)
    processed = domain.relation("Processed")
    before = domain.relation("Before")
    key_le = domain.relation("KeyLE")
    return z3.ForAll(
        [x, y],
        z3.Implies(
            z3.And(processed(x), processed(y), before(x, y)),
            key_le(x, y),
        ),
    )


def check_sorted_prefix_entailment(
    domain: RelationalDomain,
    learned_invariant: z3.ExprRef,
) -> z3.CheckSatResult:
    """Check whether the learned invariant entails the intended target under axioms."""
    solver = z3.Solver()
    domain.add_axioms(solver)
    solver.add(learned_invariant)
    solver.add(z3.Not(sorted_prefix_target(domain)))
    return solver.check()


def snapshots_have_sorted_prefix(
    snapshots: Sequence[RelationalSnapshot],
) -> bool:
    """Concretely validate sortedness of every processed prefix."""
    for snapshot in snapshots:
        state: InsertionSortSnapshot = snapshot.payload
        prefix = state.order[: state.outer_index]
        prefix_keys = [state.keys[item] for item in prefix]
        if any(left > right for left, right in zip(prefix_keys, prefix_keys[1:])):
            return False
    return True


def infer_default_insertion_sort_invariant(
    verbose: bool = False,
) -> Tuple[RelationalDomain, RelationalInferenceResult]:
    """Run the complete default insertion-sort invariant-inference example."""
    domain = build_insertion_sort_domain()
    result = infer_relational_invariants(
        domain,
        build_training_snapshots(),
        k=2,
        verbose=verbose,
    )
    return domain, result


__all__ = [
    "DEFAULT_TRAINING_KEYS",
    "WALKTHROUGH_KEYS",
    "InsertionSortRun",
    "InsertionSortSnapshot",
    "Pair",
    "Solution",
    "build_insertion_sort_domain",
    "build_training_snapshots",
    "check_sorted_prefix_entailment",
    "infer_default_insertion_sort_invariant",
    "pairs_from_keys",
    "snapshots_have_sorted_prefix",
    "sorted_prefix_target",
    "to_relational_snapshot",
    "trace_insertion_sort",
]
