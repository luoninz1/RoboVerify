import unittest
from unittest.mock import patch

from synthesis.cfg.demos import DemoSegment, DemoTrace
from synthesis.cfg.graph import RelationalCFG
from synthesis.cfg.refine import refine_cfg, transition_witnesses
from synthesis.predicates.enumerate import SearchResult
from synthesis.predicates.scene import Scene
from synthesis.predicates.term import atom, boolean, ref


class RefineTests(unittest.TestCase):
    def test_all_transition_witnesses_and_lockstep_absolute_partition(self):
        scattered = Scene({0: (0, 0, 0.425), 1: (0.2, 0, 0.425)}, {"a": 1, "b": 0})
        high = Scene({0: (0, 0, 0.425), 1: (0.2, 0, 0.525)}, {"a": 1, "b": 0})
        stack = Scene({0: (0, 0, 0.425), 1: (0, 0, 0.475)}, {"a": 1, "b": 0})
        states = (scattered, high, stack, high, stack)
        segment = DemoSegment(0, 0, 4, DemoTrace(states), {"a": 1, "b": 0})
        post = atom("ON", ref("a"), ref("b"))
        self.assertEqual(len(transition_witnesses([segment], post)), 2)
        cfg = RelationalCFG.initial([segment], boolean(True), post, ("a", "b"))
        feature = atom("Higher", ref("a"), ref("b"))
        # Equality of heights would hold at entry; use the strict reverse negation.
        from synthesis.predicates.term import negate

        feature = negate(atom("Higher", ref("b"), ref("a")))
        with patch(
            "synthesis.cfg.refine.learn_classifier",
            return_value=SearchResult("found", feature),
        ) as learner:
            before_get = Scene(scattered.positions, {"b": 0})
            result = refine_cfg(cfg, "v0", [before_get, scattered], {"a", "b"})
            self.assertEqual(learner.call_args.args[1], [scattered])
        self.assertTrue(result)
        self.assertEqual(cfg.order, ["v0.0", "v0.1"])
        self.assertEqual(cfg.demos.for_node("v0.1")[0].t_start, 1)
        self.assertEqual(cfg.demos.for_node("v0.0")[0].t_end, 1)
        cfg.validate_structure()

    def test_invalid_split_leaves_graph_and_demos_untouched(self):
        scene = Scene({0: (0, 0, 0)}, {"b": 0})
        segment = DemoSegment(0, 0, 1, DemoTrace((scene, scene)))
        cfg = RelationalCFG.initial([segment], boolean(True), boolean(True))
        with patch(
            "synthesis.cfg.refine.learn_classifier",
            return_value=SearchResult("found", boolean(True)),
        ):
            result = refine_cfg(cfg, "v0", [scene], set())
        self.assertEqual(result.status, "validate_reject")
        self.assertEqual(cfg.order, ["v0"])
        self.assertIs(cfg.demos.for_node("v0")[0], segment)

    def test_existential_split_binds_witness_for_suffix_get(self):
        from synthesis.predicates.term import conjunction, exists, negate

        low = Scene({0: (0, 0, 0.425), 1: (0.2, 0, 0.425)}, {"b": 0})
        high = Scene({0: (0, 0, 0.425), 1: (0.2, 0, 0.525)}, {"b": 0})
        trace = DemoTrace((low, high, high))
        cfg = RelationalCFG.initial(
            [DemoSegment(0, 0, 2, trace, {"b": 0})],
            boolean(True),
            boolean(True),
            ("b",),
        )
        feature = exists(["x"], negate(atom("Higher", ref("b"), ref("x"))))
        with patch(
            "synthesis.cfg.refine.learn_classifier",
            return_value=SearchResult("found", feature),
        ):
            result = refine_cfg(cfg, "v0", [low], {"b"})
        self.assertTrue(result)
        edge = cfg.incoming("v0.1")[0]
        self.assertEqual(len(edge.binds), 1)
        name = next(iter(edge.binds))
        self.assertEqual(cfg.demos.for_node("v0.1")[0].bindings[name], 1)


class WholeCFGValidationTests(unittest.TestCase):
    def test_neighbor_boundary_mismatch_is_rejected(self):
        from synthesis.cfg.demos import DemoAssignment
        from synthesis.cfg.graph import Edge, Node
        from synthesis.cfg.validate import validate_cfg

        low = Scene({0: (0, 0, 0), 1: (0.2, 0, 0)}, {"a": 1, "b": 0})
        high = Scene({0: (0, 0, 0), 1: (0, 0, 0.05)}, low.bindings)
        trace = DemoTrace((low, high, high, high))
        post = atom("ON", ref("a"), ref("b"))
        cfg = RelationalCFG(
            {"left": Node("left"), "right": Node("right")},
            [
                Edge("entry", "left", boolean(True)),
                Edge("left", "right", post),
                Edge("right", "exit", boolean(True)),
            ],
            ["left", "right"],
            DemoAssignment(
                {
                    "left": [DemoSegment(0, 0, 1, trace)],
                    "right": [DemoSegment(0, 2, 3, trace)],
                }
            ),
        )
        self.assertFalse(validate_cfg(cfg))
        cfg.demos.segments["right"] = [DemoSegment(0, 1, 3, trace)]
        self.assertTrue(validate_cfg(cfg))

    def test_persistent_previous_milestone_does_not_prevent_progress(self):
        from synthesis.cfg.demos import DemoAssignment
        from synthesis.cfg.graph import Edge, Node
        from synthesis.predicates.term import negate

        low = Scene({0: (0, 0, 0), 1: (0.2, 0, 0)}, {"a": 1, "b": 0})
        lifted = Scene({0: (0, 0, 0), 1: (0.2, 0, 0.1)}, low.bindings)
        placed = Scene({0: (0, 0, 0), 1: (0, 0, 0.05)}, low.bindings)
        trace = DemoTrace((low, lifted, placed, placed))
        first = negate(atom("Higher", ref("b"), ref("a")))
        second = atom("ON", ref("a"), ref("b"))
        cfg = RelationalCFG(
            {"left": Node("left"), "right": Node("right")},
            [
                Edge("entry", "left", boolean(True)),
                Edge("left", "right", first),
                Edge("right", "exit", boolean(True)),
            ],
            ["left", "right"],
            DemoAssignment(
                {
                    "left": [DemoSegment(0, 0, 1, trace)],
                    "right": [DemoSegment(0, 1, 3, trace)],
                }
            ),
        )
        with patch(
            "synthesis.cfg.refine.learn_classifier",
            return_value=SearchResult("found", second),
        ):
            result = refine_cfg(cfg, "right", [low], {"a", "b"})
        self.assertTrue(result)
        self.assertEqual(cfg.order, ["left", "right.0", "right.1"])
