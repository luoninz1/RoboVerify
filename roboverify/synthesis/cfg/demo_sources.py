"""Named historical oracle fixtures; never used as synthesized answers."""

import z3

from synthesis.api import program
from synthesis.verification_lib import highlevel_verification_lib


def unstack_oracle():
    context = highlevel_verification_lib.HighLevelContext(mode="declare")
    b_prime, b, n, b0 = z3.Consts("b_prime b n b0", context.BoxSort)
    loop_program = program.Program(2)
    loop_program.instructions = [
        program.Assign(left="b", right="b0"),
        program.While(
            instantiated_cond=z3.And(
                z3.ForAll(
                    [n],
                    z3.Or(
                        b_prime == n,
                        z3.Not(context.ON_star(n, b_prime)),
                    ),
                ),
                b_prime != b,
            ),
            guard_exists_vars=[b_prime],
            body=[
                program.PickByName(grab_box_name="b_prime"),
                program.MoveByName(
                    target_box_name_x="b_prime",
                    target_box_name_y="b_prime",
                    target_box_name_z="b",
                    target_offset=[0.0, 0.0, 0.15],
                ),
                program.MoveByName(
                    target_box_name_x="b",
                    target_box_name_y="b",
                    target_box_name_z="b",
                    target_offset=[0.0, 0.0, 0.15],
                ),
                program.MoveByName(
                    target_box_name_x="b",
                    target_box_name_y="b",
                    target_box_name_z="b",
                    target_offset=[0.0, 0.0, 0.05],
                ),
                program.ReleaseByName(release_box_name="b_prime", target_z=0.15),
                program.Assign("b", "b_prime"),
            ],
            invariant=None,
        ),
    ]
    return loop_program
