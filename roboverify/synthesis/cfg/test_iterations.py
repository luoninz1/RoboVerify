import unittest

from synthesis.cfg.demos import DemoSegment, DemoTrace
from synthesis.cfg.iterations import extract_iterations
from synthesis.predicates.scene import Scene
from synthesis.predicates.term import atom, ref


class IterationExtractionTests(unittest.TestCase):
    def test_extracts_all_iterations_without_a_presegmented_cfg(self):
        # Two recordings contain different numbers of iteration boundaries.
        post = atom("ON", ref("next"), ref("current"))
        for count in (2, 4):
            positions = {i: (0, 0, 0.05 * i) for i in range(count + 1)}
            scene = Scene(positions, {"current": 0})
            trace = DemoTrace(tuple(scene for _ in range(count + 1)))
            segment = DemoSegment(0, 0, count, trace, {"current": 0})
            result = extract_iterations(
                segment, [post], ("next",), (("current", "next"),)
            )
            self.assertEqual(len(result.bodies), count)
            self.assertEqual(
                [b[0].bindings["next"] for b in result.bodies],
                list(range(1, count + 1)),
            )
            self.assertEqual(result.terminal.bindings["current"], count)
            self.assertTrue(all(b[0].entry_index == 0 for b in result.bodies))

    def test_multiblock_iteration_requires_every_milestone_in_order(self):
        from synthesis.predicates.term import negate

        states = [
            Scene({0: (0, 0, 0), 1: (0.2, 0, 0)}, {"current": 0}),
            Scene({0: (0, 0, 0), 1: (0.2, 0, 0.1)}, {"current": 0}),
            Scene({0: (0, 0, 0), 1: (0, 0, 0.05)}, {"current": 0}),
        ]
        first = negate(atom("Higher", ref("current"), ref("next")))
        second = atom("ON", ref("next"), ref("current"))
        trace = DemoTrace(tuple(states))
        result = extract_iterations(
            DemoSegment(0, 0, 2, trace, {"current": 0}),
            [first, second],
            ("next",),
            (("current", "next"),),
        )
        self.assertEqual(
            [(s.t_start, s.t_end) for s in result.bodies[0]], [(0, 1), (1, 2)]
        )

    def test_terminal_heads_and_frozen_geometry_reach_invariant_learning(self):
        from synthesis.cfg.invariants import loop_learning_data

        positions = {0: (0, 0, 0), 1: (0, 0, 0.05), 2: (0, 0, 0.1)}
        scene = Scene(positions, {"current": 0})
        trace = DemoTrace((scene, scene, scene))
        result = extract_iterations(
            DemoSegment(0, 0, 2, trace, {"current": 0}),
            [atom("ON", ref("next"), ref("current"))],
            ("next",),
            (("current", "next"),),
        )
        rows = [body[0] for body in result.bodies] + [result.terminal]
        store, vocab = loop_learning_data(
            rows,
            {"current"},
            relations=("ON_star", "ON_star_zero", "Higher", "Scattered", "eq"),
        )
        learned_rows = store.for_loop("loop")
        self.assertEqual(len(learned_rows), 3)
        self.assertNotEqual(
            learned_rows[0].constants["current"], learned_rows[-1].constants["current"]
        )
        self.assertEqual(
            learned_rows[0].entry_positions, learned_rows[-1].entry_positions
        )
        self.assertIn("ON_star_zero", vocab.relations)
        self.assertIn("equality", vocab.relations)
        self.assertNotIn("next", vocab.constants)


class InvariantAdapterResultTests(unittest.TestCase):
    def test_inference_returns_the_formula_not_provenance_tuple(self):
        from unittest.mock import patch

        import z3

        from synthesis.cfg.invariants import infer_loop_invariant
        from synthesis.verification_lib.highlevel_verification_lib import (
            HighLevelContext,
        )

        context = HighLevelContext()
        with patch(
            "synthesis.cfg.invariants.loop_learning_data",
            return_value=(object(), object()),
        ), patch(
            "synthesis.cfg.invariants.InvInference",
            return_value=(z3.BoolVal(True), ["provenance"]),
        ):
            result = infer_loop_invariant([object()], None, (), context=context)
        self.assertTrue(z3.is_true(result))
