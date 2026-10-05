"""Bounded bottom-up predicate search over increasing depth and binder count."""

import itertools
from dataclasses import dataclass
from time import monotonic

from synthesis.predicates.language import Language
from synthesis.predicates.scene import evaluate
from synthesis.predicates.term import (
    Term,
    atom,
    conjunction,
    disjunction,
    exists,
    forall,
    negate,
    ref,
)


@dataclass(frozen=True)
class SearchNode:
    term: object
    depth: int
    variables: int


@dataclass
class SearchResult:
    status: str
    term: object = None
    examined: int = 0
    elapsed: float = 0.0

    def __bool__(self):
        return self.status == "found"


def enumerate_separator(
    examples, constants, *, mode="classifier", language=None, accept=None
):
    """Examples are (Scene, bool); no approximate classifier is returned as exact.

    FO formulas are enumerated in prenex form. Classifiers allow an optional outer
    existential prefix and a quantifier-free matrix. Guard search permits any
    quantifier prefix. Observational pruning compares all binder assignments,
    not just the closed formula's output, so it remains valid under quantifiers.
    """
    language = language or Language()
    constants = tuple(c if isinstance(c, Term) else ref(c) for c in constants)
    if any(c.op not in ("ref", "id") for c in constants):
        raise ValueError("Constants must be object references or block IDs")
    reserved_names = {c.value for c in constants if c.op == "ref"}
    if mode not in ("classifier", "guard") or not examples:
        raise ValueError("A supported mode and nonempty examples are required")
    if language.max_candidates < 1 or language.timeout_seconds <= 0:
        raise ValueError("Search budget must be positive")
    start, examined = monotonic(), 0

    def exhausted():
        return (
            examined >= language.max_candidates
            or monotonic() - start >= language.timeout_seconds
        )

    for depth in range(1, language.max_depth + 1):
        for count in range(language.max_variables + 1):
            matrix_depth = depth - count
            if matrix_depth < 1:
                continue
            # Binder names must stay distinct from the program's free variables.
            candidates = (f"v{i}" for i in itertools.count())
            variables = tuple(
                itertools.islice(
                    (name for name in candidates if name not in reserved_names), count
                )
            )
            references = [*constants, *(ref(name) for name in variables)]
            scenes = [
                (scene, dict(zip(variables, assignment)))
                for scene, _ in examples
                for assignment in itertools.product(scene.positions, repeat=count)
            ]
            levels, signatures = {}, set()
            for level in range(1, matrix_depth + 1):
                if level == 1:
                    candidates = (
                        atom(rel, a, b)
                        for rel in language.relations
                        for a in references
                        for b in references
                    )
                else:
                    previous = [t for d in levels for t in levels[d]]
                    top = levels.get(level - 1, [])
                    candidates = itertools.chain(
                        (negate(t) for t in top),
                        (
                            op(a, b)
                            for op in (conjunction, disjunction)
                            for a in top
                            for b in previous
                        ),
                    )
                kept = []
                for matrix in candidates:
                    if exhausted():
                        return SearchResult(
                            "budget_exhausted",
                            examined=examined,
                            elapsed=monotonic() - start,
                        )
                    examined += 1
                    signature = tuple(
                        evaluate(matrix, scene, values) for scene, values in scenes
                    )
                    if signature in signatures:
                        continue
                    signatures.add(signature)
                    kept.append(matrix)
                    if level != matrix_depth:
                        continue
                    prefixes = (
                        [("exists",) * count]
                        if mode == "classifier"
                        else itertools.product(("exists", "forall"), repeat=count)
                    )
                    for prefix in prefixes:
                        term = matrix
                        for quantifier, variable in reversed(
                            tuple(zip(prefix, variables))
                        ):
                            term = (exists if quantifier == "exists" else forall)(
                                [variable], term
                            )
                        if all(
                            evaluate(term, scene) == label for scene, label in examples
                        ) and (accept is None or accept(term)):
                            return SearchResult(
                                "found", term, examined, monotonic() - start
                            )
                levels[level] = kept
    return SearchResult("no_separator", examined=examined, elapsed=monotonic() - start)
