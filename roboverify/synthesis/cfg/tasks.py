"""Task conditions shared by collection, search, and verification."""

from synthesis.predicates.term import (
    atom,
    conjunction,
    disjunction,
    forall,
    implies,
    negate,
    ref,
)


def task_spec(task):
    x, y, b0 = ref("x"), ref("y"), ref("b0")
    if task == "stack":
        different = negate(atom("eq", x, y))
        return (
            conjunction(
                forall(
                    ["x", "y"],
                    implies(
                        different,
                        conjunction(
                            negate(atom("ON_star", x, y)), atom("Scattered", x, y)
                        ),
                    ),
                ),
                # Both ordered pairs enforce one initial height level; concrete
                # observations use the configured Higher tolerance.
                forall(["x", "y"], atom("Higher", x, y)),
            ),
            forall(["x"], atom("ON_star", x, b0)),
        )
    if task == "unstack":
        return (
            forall(
                ["x"], disjunction(atom("eq", x, ref("tbl")), atom("ON_star", x, b0))
            ),
            forall(["x", "y"], implies(atom("ON_star", x, y), atom("eq", x, y))),
        )
    raise ValueError(f"Unsupported task: {task}")


def task_identity(task):
    pre, post = task_spec(task)
    return {"task": task, "precondition": str(pre), "postcondition": str(post)}
