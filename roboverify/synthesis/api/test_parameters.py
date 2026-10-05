"""The optimizer must be able to change every Move offset independently."""

import unittest

from synthesis.api.instructions import Move, Pick, Release
from synthesis.api.program import Program


class MoveParameters(unittest.TestCase):
    def test_all_axes_are_registered_and_updated(self):
        move = Move(target_offset=[0.1, -0.2, 0.3])
        parameters = []
        move.register_trainable_parameter(parameters)
        self.assertEqual(parameters, [0.1, -0.2, 0.3])
        move.update_trainable_parameter([0.4, 0.5, -0.6])
        self.assertEqual([p.val for p in move.target_offset], [0.4, 0.5, -0.6])

    def test_program_moves_have_independent_slots(self):
        first, second = Move(), Move()
        program = Program(4, instructions=[Pick(0), first, second, Release()])
        first, second = program.instructions[1:3]
        self.assertEqual(program.register_trainable_parameter(), [0.0] * 7)
        program.update_trainable_parameter([0.1, 0.2, 0.3, -0.1, -0.2, -0.3, 0.4])
        self.assertEqual(program.instructions[3].target_z_offset.val, 0.4)
        self.assertEqual([p.val for p in first.target_offset], [0.1, 0.2, 0.3])
        self.assertEqual([p.val for p in second.target_offset], [-0.1, -0.2, -0.3])


if __name__ == "__main__":
    unittest.main()
