"""Stack scattered blocks using only explicit physical primitives."""

import z3

from synthesis.api.instructions import (
    Assign,
    MoveByName,
    PickByName,
    ReleaseByName,
    While,
)
from synthesis.api.program import Program

TRANSFER_HEIGHT = 0.10


def build_program(context, *, num_blocks):
    current, selected, other = z3.Consts("b b_prime n", context.BoxSort)
    guard = z3.And(
        selected != current,
        z3.ForAll(
            [other], z3.Or(other == selected, z3.Not(context.ON_star(other, selected)))
        ),
    )
    # Keep transport above the tower without the former 20 cm excursion.
    # Pick uses 10 mm; Move/Release use 2 mm. Each retains its 50-step budget.
    body = [
        PickByName("b_prime"),
        MoveByName("b_prime", "b_prime", "b", target_offset=[0, 0, TRANSFER_HEIGHT]),
        MoveByName("b0", "b0", "b", target_offset=[0, 0, TRANSFER_HEIGHT]),
        MoveByName("b0", "b0", "b", target_offset=[0, 0, 0.05]),
        ReleaseByName("b_prime", target_z=0.15),
        Assign("b", "b_prime"),
    ]
    return Program(
        2,
        [
            Assign("b", "b0"),
            While(guard, [selected], body, invariant=None, max_iters=None),
        ],
    )
