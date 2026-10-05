"""Task entry facts used by both concrete validation and symbolic verification."""

import unittest

import z3

from synthesis.api.instructions import Assign
from synthesis.api.program import wp
from synthesis.cfg.tasks import task_spec
from synthesis.predicates.term import to_z3
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext
from synthesis.verification_lib.symbolic_verify import VC, discharge_vc


class StackTaskTests(unittest.TestCase):
    def test_entry_establishes_both_height_bounds_without_vacuity(self):
        context = HighLevelContext()
        pre, _ = task_spec("stack")
        b, b0, x = z3.Consts("b b0 x", context.BoxSort)
        height_bounds = z3.ForAll(
            [x], z3.And(context.Higher(b, x), context.Higher(x, b0))
        )
        result = discharge_vc(
            VC(
                "establish",
                "1",
                z3.Implies(
                    to_z3(pre, context), wp(Assign("b", "b0"), height_bounds, context)
                ),
            ),
            context,
            timeout_ms=5000,
        )
        self.assertEqual(result.status, "valid", result.reason)

    def test_final_tower_does_not_require_equal_heights(self):
        context = HighLevelContext()
        _, post = task_spec("stack")
        b0, top = z3.Consts("b0 top", context.BoxSort)
        solver = context.new_solver(5000)
        solver.add(
            to_z3(post, context),
            top != b0,
            context.Higher(top, b0),
            z3.Not(context.Higher(b0, top)),
        )
        self.assertEqual(solver.check(), z3.sat)


if __name__ == "__main__":
    unittest.main()
