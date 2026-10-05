"""Learn guards from demonstrated witnesses and witness-free exit scenes."""

import itertools

from synthesis.predicates.enumerate import SearchResult, enumerate_separator
from synthesis.predicates.scene import Scene, evaluate
from synthesis.predicates.term import free_names


def loop_guard_synthesis(
    positive, exits, variables, scope, *, language=None, candidates=()
):
    """Demonstrated bindings are positive; unselected bindings are unlabeled.

    Every binding at a recorded exit is negative. Additional guard witnesses at
    continuing heads are permitted; verification must check the body for all of
    them, not assume that matching the demonstrated choice establishes safety.
    """
    if not positive or not exits:
        return SearchResult("insufficient_examples")
    examples = []
    for scene, chosen in positive:
        if set(chosen) != set(variables):
            raise ValueError("Every guard witness must bind all guard variables")
        examples.append(
            (
                Scene(
                    scene.positions,
                    dict(scene.bindings, **chosen),
                    scene.entry_positions,
                ),
                True,
            )
        )
    for scene in exits:
        for assignment in itertools.product(scene.positions, repeat=len(variables)):
            examples.append(
                (
                    Scene(
                        scene.positions,
                        dict(scene.bindings, **dict(zip(variables, assignment))),
                        scene.entry_positions,
                    ),
                    False,
                )
            )
    for candidate in candidates:
        if free_names(candidate) <= set(scope) | set(variables) and all(
            evaluate(candidate, scene) == label for scene, label in examples
        ):
            return SearchResult("found", candidate, examined=1)
    return enumerate_separator(
        examples,
        tuple(sorted(set(scope) | set(variables))),
        mode="guard",
        language=language,
    )


LoopGuardSynthesis = loop_guard_synthesis
