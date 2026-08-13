"""Generic relational front end for RoboVerify invariant inference.

The existing learner operates on Boolean truth rows. This module supplies the
domain-facing layer that declares predicates, grounds them over concrete object
snapshots, and forwards the resulting dataset to the Boolean learner.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, Iterable, Mapping, Sequence, Tuple

import z3

from synthesis.inference_lib.inference import (
    ProvenancedClause,
    infer_boolean_invariants,
)

ObjectArguments = Tuple[str, ...]
RelationEvaluator = Callable[["RelationalSnapshot", ObjectArguments], bool]
AxiomBuilder = Callable[["RelationalDomain"], Iterable[z3.BoolRef]]


@dataclass(frozen=True)
class RelationSpec:
    """Describe one unary or binary relation in a relational domain."""

    name: str
    arity: int
    evaluator: RelationEvaluator
    allow_same_terms: bool = False

    def __post_init__(self) -> None:
        if not self.name or not self.name.isidentifier():
            raise ValueError(f"Invalid relation name: {self.name!r}")
        if self.arity not in (1, 2):
            raise ValueError(
                f"Relation {self.name!r} must have arity 1 or 2, got {self.arity}."
            )
        if not callable(self.evaluator):
            raise TypeError(f"Evaluator for relation {self.name!r} must be callable.")


@dataclass(frozen=True)
class RelationalSnapshot:
    """A concrete state over stable object names.

    ``payload`` is domain-specific data consumed by relation evaluators.
    ``constant_bindings`` maps symbolic program-constant names to object names.
    """

    objects: Tuple[str, ...]
    payload: Any
    constant_bindings: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.objects:
            raise ValueError("A relational snapshot must contain at least one object.")
        if len(set(self.objects)) != len(self.objects):
            raise ValueError("Relational snapshot object names must be unique.")
        if any(not isinstance(name, str) or not name for name in self.objects):
            raise ValueError("Every relational snapshot object name must be non-empty.")


@dataclass
class RelationalDomain:
    """Z3 declarations, concrete evaluators, and axioms for one object domain."""

    name: str
    relation_specs: Sequence[RelationSpec]
    include_equality: bool = True
    axiom_builder: AxiomBuilder | None = None
    object_sort: z3.SortRef = field(init=False)
    relations: Dict[str, z3.FuncDeclRef] = field(init=False)

    def __post_init__(self) -> None:
        if not self.name or not self.name.isidentifier():
            raise ValueError(f"Invalid domain name: {self.name!r}")
        if not self.relation_specs:
            raise ValueError("A relational domain must define at least one relation.")

        names = [spec.name for spec in self.relation_specs]
        if len(set(names)) != len(names):
            raise ValueError("Relation names must be unique within a domain.")

        self.object_sort = z3.DeclareSort(self.name)
        self.relations = {}
        for spec in self.relation_specs:
            argument_sorts = [self.object_sort] * spec.arity
            self.relations[spec.name] = z3.Function(
                spec.name, *argument_sorts, z3.BoolSort()
            )

    def relation(self, name: str) -> z3.FuncDeclRef:
        """Return the declared Z3 relation named ``name``."""
        try:
            return self.relations[name]
        except KeyError as error:
            raise KeyError(
                f"Unknown relation {name!r} in domain {self.name!r}."
            ) from error

    def constant(self, name: str) -> z3.ExprRef:
        """Create an object-sorted symbolic program constant."""
        if not name or not name.isidentifier():
            raise ValueError(f"Invalid constant name: {name!r}")
        return z3.Const(name, self.object_sort)

    def add_axioms(self, solver: z3.Solver) -> None:
        """Add domain axioms to ``solver``; a domain may intentionally have none."""
        if self.axiom_builder is not None:
            solver.add(*list(self.axiom_builder(self)))

    def evaluate(
        self,
        relation_name: str,
        snapshot: RelationalSnapshot,
        object_arguments: ObjectArguments,
    ) -> bool:
        """Evaluate a declared relation on concrete object names."""
        spec = next(
            (
                candidate
                for candidate in self.relation_specs
                if candidate.name == relation_name
            ),
            None,
        )
        if spec is None:
            raise KeyError(
                f"Unknown relation {relation_name!r} in domain {self.name!r}."
            )
        if len(object_arguments) != spec.arity:
            raise ValueError(
                f"Relation {relation_name!r} expects {spec.arity} arguments, "
                f"got {len(object_arguments)}."
            )
        missing = [name for name in object_arguments if name not in snapshot.objects]
        if missing:
            raise KeyError(
                f"Relation {relation_name!r} references unknown objects: {missing}."
            )
        value = spec.evaluator(snapshot, object_arguments)
        if not isinstance(value, bool):
            raise TypeError(
                f"Evaluator for relation {relation_name!r} must return bool, "
                f"got {type(value).__name__}."
            )
        return value


@dataclass(frozen=True)
class RelationalInferenceResult:
    """Inspectable intermediate data and final output from relational inference."""

    universal_variables: Tuple[z3.ExprRef, ...]
    vocabulary: Tuple[z3.ExprRef, ...]
    truth_rows: FrozenSet[Tuple[bool, ...]]
    invariant: z3.ExprRef
    clauses: Tuple[ProvenancedClause, ...]


def build_relational_vocabulary(
    domain: RelationalDomain,
    k: int,
    constant_names: Sequence[str] = (),
) -> Tuple[Tuple[z3.ExprRef, ...], Tuple[z3.ExprRef, ...]]:
    """Construct universal variables and the finite atomic vocabulary."""
    if k < 1:
        raise ValueError(
            "Relational inference requires at least one universal variable."
        )
    if len(set(constant_names)) != len(constant_names):
        raise ValueError("Program constant names must be unique.")

    universal_variables = tuple(
        domain.constant(f"ux{index}") for index in range(1, k + 1)
    )
    constants = tuple(domain.constant(name) for name in constant_names)
    terms = universal_variables + constants

    vocabulary = []
    for spec in domain.relation_specs:
        relation = domain.relation(spec.name)
        if spec.arity == 1:
            vocabulary.extend(relation(term) for term in terms)
            continue
        for left, right in itertools.product(terms, repeat=2):
            if not spec.allow_same_terms and z3.eq(left, right):
                continue
            vocabulary.append(relation(left, right))

    if domain.include_equality:
        vocabulary.extend(
            left == right for left, right in itertools.combinations(terms, 2)
        )

    return universal_variables, tuple(vocabulary)


def _validate_constant_bindings(
    snapshot: RelationalSnapshot, constant_names: Sequence[str]
) -> None:
    expected = set(constant_names)
    actual = set(snapshot.constant_bindings)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(
            "Snapshot constant bindings do not match the inference constants: "
            f"missing={missing}, unexpected={unexpected}."
        )
    unknown_objects = sorted(
        object_name
        for object_name in snapshot.constant_bindings.values()
        if object_name not in snapshot.objects
    )
    if unknown_objects:
        raise KeyError(
            f"Snapshot constants reference unknown objects: {unknown_objects}."
        )


def ground_relational_row(
    domain: RelationalDomain,
    snapshot: RelationalSnapshot,
    vocabulary: Sequence[z3.ExprRef],
    universal_variables: Sequence[z3.ExprRef],
    assignment: Sequence[str],
    constant_names: Sequence[str] = (),
) -> Tuple[bool, ...]:
    """Evaluate every vocabulary atom for one universal-variable assignment."""
    if len(assignment) != len(universal_variables):
        raise ValueError(
            f"Expected {len(universal_variables)} assigned objects, got {len(assignment)}."
        )
    unknown = [name for name in assignment if name not in snapshot.objects]
    if unknown:
        raise KeyError(f"Universal variables reference unknown objects: {unknown}.")
    _validate_constant_bindings(snapshot, constant_names)

    variable_bindings = dict(zip(universal_variables, assignment))
    constant_bindings = {
        domain.constant(name): snapshot.constant_bindings[name]
        for name in constant_names
    }

    def resolve(term: z3.ExprRef) -> str:
        if term in variable_bindings:
            return variable_bindings[term]
        if term in constant_bindings:
            return constant_bindings[term]
        raise KeyError(f"Unable to ground symbolic term {term}.")

    row = []
    for atom in vocabulary:
        if z3.is_eq(atom):
            row.append(resolve(atom.arg(0)) == resolve(atom.arg(1)))
            continue
        relation_name = str(atom.decl().name())
        object_arguments = tuple(
            resolve(atom.arg(index)) for index in range(atom.num_args())
        )
        row.append(domain.evaluate(relation_name, snapshot, object_arguments))
    return tuple(row)


def build_relational_dataset(
    domain: RelationalDomain,
    snapshots: Sequence[RelationalSnapshot],
    vocabulary: Sequence[z3.ExprRef],
    universal_variables: Sequence[z3.ExprRef],
    constant_names: Sequence[str] = (),
) -> FrozenSet[Tuple[bool, ...]]:
    """Ground all universal-variable assignments in every snapshot."""
    if not snapshots:
        raise ValueError("Relational inference requires at least one snapshot.")

    rows = set()
    for snapshot in snapshots:
        _validate_constant_bindings(snapshot, constant_names)
        for assignment in itertools.product(
            snapshot.objects, repeat=len(universal_variables)
        ):
            rows.add(
                ground_relational_row(
                    domain,
                    snapshot,
                    vocabulary,
                    universal_variables,
                    assignment,
                    constant_names,
                )
            )
    return frozenset(rows)


def infer_relational_invariants(
    domain: RelationalDomain,
    snapshots: Sequence[RelationalSnapshot],
    k: int,
    constant_names: Sequence[str] = (),
    verbose: bool = False,
) -> RelationalInferenceResult:
    """Infer universally quantified invariants from relational snapshots."""
    universal_variables, vocabulary = build_relational_vocabulary(
        domain, k, constant_names
    )
    truth_rows = build_relational_dataset(
        domain,
        snapshots,
        vocabulary,
        universal_variables,
        constant_names,
    )
    invariant, clauses = infer_boolean_invariants(
        set(truth_rows),
        list(vocabulary),
        list(universal_variables),
        axiom_adder=domain.add_axioms,
        verbose=verbose,
    )
    return RelationalInferenceResult(
        universal_variables=universal_variables,
        vocabulary=vocabulary,
        truth_rows=truth_rows,
        invariant=invariant,
        clauses=tuple(clauses),
    )


__all__ = [
    "RelationSpec",
    "RelationalDomain",
    "RelationalInferenceResult",
    "RelationalSnapshot",
    "build_relational_dataset",
    "build_relational_vocabulary",
    "ground_relational_row",
    "infer_relational_invariants",
]
