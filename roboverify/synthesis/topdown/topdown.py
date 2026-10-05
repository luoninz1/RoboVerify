"""Compatibility entry for the retired BFS; new search uses canonical terms."""

from synthesis.predicates.datasets import (
    Box,
    run_reverse_hardcoded_dataset,
    run_stack_hardcoded_dataset,
    run_unstack_hardcoded_dataset,
)
from synthesis.predicates.enumerate import enumerate_separator
from synthesis.predicates.language import Language
from synthesis.predicates.scene import Scene


def topdown_synthesize(
    all_inputs, max_programs=100000, target_accuracy=1.0, time_limit_sec=60
):
    examples = []
    for row in all_inputs:
        positions = {box.id: (box.x, box.y, box.z) for box in row["all_box"]}
        aliases = {name: box.id for name, box in row["constants"].items()}
        aliases["target"] = row["target"].id
        examples.append((Scene(positions, aliases), row["result"]))
    names = tuple(examples[0][0].bindings)
    result = enumerate_separator(
        examples,
        names,
        mode="guard",
        language=Language(max_candidates=max_programs, timeout_seconds=time_limit_sec),
    )
    return result.term
