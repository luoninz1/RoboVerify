"""Higher must agree across runtime, inference and geometric verification."""

import argparse
import contextlib
import io
import unittest

import z3

from synthesis.api.guard_eval import evaluate_z3
from synthesis.entry.predicate_options import add_predicate_options
from synthesis.inference_lib.demo_store import LoopHeadState
from synthesis.inference_lib.inference import compute_data
from synthesis.predicates.scene import Scene, evaluate
from synthesis.predicates.term import atom, ref
from synthesis.util import on
from synthesis.verification_lib.counterexamples import model_to_loop_head, state_holds
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext
from synthesis.verification_lib.lowlevel_verification_lib import LowLevelContext


class HigherToleranceTests(unittest.TestCase):
    def test_boundary_and_distinct_levels(self):
        tolerance = 1 / 1024  # Binary-exact inclusive boundary.
        a, b = [0, 0, 0.5], [0.2, 0, 0.5 + tolerance]
        self.assertTrue(on.higher_implementation(a, b, tolerance=tolerance))
        self.assertTrue(on.higher_implementation(b, a, tolerance=tolerance))
        self.assertFalse(on.higher_implementation(a, b, tolerance=0))
        b[2] += tolerance / 1024
        self.assertFalse(on.higher_implementation(a, b, tolerance=tolerance))
        for level in range(4):
            for other in range(4):
                first = [0, 0, 0.425 + level * on.BLOCK_LENGTH]
                second = [0, 0, 0.425 + other * on.BLOCK_LENGTH]
                self.assertEqual(
                    on.higher_implementation(first, second), level >= other
                )

    def test_validation_and_scope_restoration(self):
        default = on.get_higher_tolerance()
        for invalid in (-1, float("nan"), float("inf"), 0.025):
            with self.assertRaises(ValueError):
                with on.using_higher_tolerance(invalid):
                    pass
        with self.assertRaises(RuntimeError):
            with on.using_higher_tolerance(0):
                self.assertEqual(on.get_higher_tolerance(), 0)
                with on.using_higher_tolerance(0.002):
                    self.assertEqual(on.get_higher_tolerance(), 0.002)
                raise RuntimeError("restore even on failure")
        self.assertEqual(on.get_higher_tolerance(), default)
        parser = argparse.ArgumentParser()
        add_predicate_options(parser)
        self.assertEqual(
            parser.parse_args(["--higher-tolerance", "0.002"]).higher_tolerance, 0.002
        )

    def test_shared_evaluation_paths(self):
        high = HighLevelContext()
        a, b = high.get_consts("a"), high.get_consts("b")
        formula = high.Higher(a, b)
        positions = {"x1": [0, 0, 0.42475], "x2": [0.2, 0, 0.425]}
        bindings = {"a": "x1", "b": "x2"}
        scene = Scene(positions, bindings)
        row = LoopHeadState("loop", positions, positions, bindings)
        for tolerance, expected in ((0, False), (0.0001, False), (0.001, True)):
            with self.subTest(tolerance=tolerance), on.using_higher_tolerance(
                tolerance
            ):
                self.assertEqual(
                    evaluate(atom("Higher", ref("a"), ref("b")), scene), expected
                )
                self.assertEqual(evaluate_z3(formula, scene), expected)
                self.assertEqual(state_holds(formula, row), expected)
                with contextlib.redirect_stdout(io.StringIO()):
                    data = compute_data(
                        positions, positions, [formula], {}, {a: "x1", b: "x2"}
                    )
                self.assertEqual(bool(data[0]), expected)
                low = LowLevelContext()
                lhs, rhs = low.get_consts("a"), low.get_consts("b")
                solver = z3.Solver()
                solver.add(
                    low.Z(lhs) == z3.RealVal("0.42475"),
                    low.Z(rhs) == z3.RealVal("0.425"),
                )
                solver.add(low.lowlevel_higher(lhs, rhs) != expected)
                self.assertEqual(solver.check(), z3.unsat)
                self.assertFalse(on.higher_implementation(positions["x1"], on.TABLE))
                self.assertTrue(on.higher_implementation(on.TABLE, on.TABLE))

    def test_counterexample_realization_respects_tolerance(self):
        high = HighLevelContext(
            mode="enum", num_blocks=2, sort_name="ToleranceCounterexample"
        )
        a, b = high.enum_blocks
        solver = high.new_solver(5000)
        for left in (a, b):
            for right in (a, b):
                solver.add(high.ON_star(left, right) == left.eq(right))
                solver.add(high.ON_star_zero(left, right) == left.eq(right))
                solver.add(high.Scattered(left, right) == (not left.eq(right)))
                solver.add(high.Higher(left, right) == (left.eq(right) or left.eq(b)))
        solver.add(high.get_consts("a") == a, high.get_consts("b") == b)
        self.assertEqual(solver.check(), z3.sat)
        with on.using_higher_tolerance(0.002):
            row = model_to_loop_head(high, solver.model(), "loop", ("a", "b"))
            self.assertFalse(
                state_holds(
                    high.Higher(high.get_consts("a"), high.get_consts("b")), row
                )
            )

    def test_arbitrary_heights_need_not_be_transitive(self):
        a, b, c = ([0, 0, z] for z in (0, 0.00075, 0.0015))
        self.assertTrue(on.higher_implementation(a, b))
        self.assertTrue(on.higher_implementation(b, c))
        self.assertFalse(on.higher_implementation(a, c))


if __name__ == "__main__":
    unittest.main()
