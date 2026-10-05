"""A new cut is compared with the original target on the original segment."""

import unittest
from unittest.mock import patch

from synthesis.cfg.demos import DemoAssignment, DemoSegment, DemoTrace
from synthesis.cfg.graph import Edge, Node, RelationalCFG
from synthesis.cfg.refine import refine_cfg
from synthesis.cfg.validate import validate_cfg, validate_split
from synthesis.predicates.enumerate import SearchResult
from synthesis.predicates.scene import Scene
from synthesis.predicates.term import atom, boolean, negate, ref

SCOPE = {"a", "b", "c", "d"}
P = negate(atom("Higher", ref("b"), ref("c")))
C = atom("ON", ref("a"), ref("b"))
Q = negate(atom("Higher", ref("b"), ref("d")))


def scene(incoming, cut, target):
    return Scene(
        {
            0: (0, 0, 0),
            1: ((0, 0, 0.05) if cut else (0.2, 0, 0)),
            2: (0.4, 0, 0.05 if incoming else 0),
            3: (0.6, 0, 0.05 if target else 0),
        },
        {"a": 1, "b": 0, "c": 2, "d": 3},
    )


def example_cfg(sequence):
    trace = DemoTrace(tuple(scene(*flags) for flags in sequence))
    return RelationalCFG(
        {"left": Node("left"), "right": Node("right")},
        [
            Edge("entry", "left", boolean(True)),
            Edge("left", "right", P),
            Edge("right", "exit", Q),
        ],
        ["left", "right"],
        DemoAssignment(
            {
                "left": [DemoSegment(0, 0, 1, trace)],
                "right": [DemoSegment(0, 1, len(sequence) - 1, trace)],
            }
        ),
    )


def refine(cfg, node="right"):
    with patch(
        "synthesis.cfg.refine.learn_classifier", return_value=SearchResult("found", C)
    ):
        return refine_cfg(cfg, node, [], SCOPE)


class TemporalAcceptanceTests(unittest.TestCase):
    def assert_rejected_without_mutation(self, cfg, node="right"):
        before = (cfg.nodes, cfg.edges, cfg.order, cfg.demos.segments)
        result = refine(cfg, node)
        self.assertEqual(result.status, "validate_reject")
        for old, new in zip(
            before, (cfg.nodes, cfg.edges, cfg.order, cfg.demos.segments)
        ):
            self.assertIs(old, new)

    def test_new_cut_must_precede_or_equal_last_original_target(self):
        self.assertTrue(validate_split({0: 10}, {0: 20}))
        self.assertTrue(validate_split({0: 10}, {0: 10}))
        self.assertFalse(validate_split({0: 11}, {0: 10}))

    def test_requires_occurrences_in_every_original_demonstration(self):
        for cuts, targets in (
            ({}, {}),
            ({0: None}, {0: 10}),
            ({0: 10}, {0: None}),
            ({0: 10}, {1: 10}),
            ({0: 10, 1: 11}, {0: 20, 1: 10}),
        ):
            with self.subTest(cuts=cuts, targets=targets):
                self.assertFalse(validate_split(cuts, targets))

    def test_complete_cfg_allows_incoming_condition_to_be_destroyed(self):
        cfg = example_cfg(
            [
                (False, False, False),
                (True, False, False),
                (False, False, False),
                (False, False, True),
            ]
        )
        self.assertTrue(validate_cfg(cfg))

    def test_refinement_allows_both_incoming_and_cut_conditions_to_be_destroyed(self):
        cfg = example_cfg(
            [
                (False, False, False),
                (True, False, False),
                (False, False, False),
                (False, True, False),
                (False, False, True),
            ]
        )
        self.assertTrue(refine(cfg))
        self.assertEqual(cfg.order, ["left", "right.0", "right.1"])
        self.assertEqual(cfg.demos.for_node("right.0")[0].t_end, 3)
        self.assertEqual(cfg.demos.for_node("right.1")[0].t_start, 3)
        self.assertTrue(validate_cfg(cfg))

    def test_initial_and_offset_entry_segments_use_the_same_rule(self):
        states = (
            scene(True, False, False),
            scene(False, False, False),
            scene(False, True, False),
            scene(False, False, True),
        )
        for start in (0, 5):
            with self.subTest(start=start):
                trace = DemoTrace((states[0],) * start + states)
                cfg = RelationalCFG.initial(
                    [DemoSegment(0, start, start + 3, trace)], P, Q, SCOPE
                )
                self.assertTrue(refine(cfg, "v0"))
                self.assertEqual(cfg.demos.for_node("v0.0")[0].t_end, start + 2)
                self.assertTrue(validate_cfg(cfg))

    def test_target_only_before_cut_or_absent_rejects_atomically(self):
        for old_target in (True, False):
            with self.subTest(old_target=old_target):
                cfg = example_cfg(
                    [
                        (False, False, False),
                        (True, False, False),
                        (False, False, old_target),
                        (False, True, False),
                        (False, False, False),
                    ]
                )
                self.assert_rejected_without_mutation(cfg)

    def test_target_can_hold_before_cut_if_it_is_reestablished_at_end(self):
        cfg = example_cfg(
            [
                (False, False, False),
                (True, False, False),
                (False, False, True),
                (False, True, False),
                (False, False, True),
            ]
        )
        self.assertTrue(refine(cfg))
        self.assertTrue(validate_cfg(cfg))

    def test_target_after_cut_but_false_at_end_still_rejects_atomically(self):
        cfg = example_cfg(
            [
                (False, False, False),
                (True, False, False),
                (False, True, False),
                (False, False, True),
                (False, False, False),
            ]
        )
        self.assert_rejected_without_mutation(cfg)

    def test_new_cut_at_entry_or_end_is_rejected_even_when_inequality_holds(self):
        for cuts in ((True, False, False), (False, False, True)):
            with self.subTest(cuts=cuts):
                trace = DemoTrace(
                    tuple(scene(i == 0, cut, i == 2) for i, cut in enumerate(cuts))
                )
                cfg = RelationalCFG.initial([DemoSegment(0, 0, 2, trace)], P, Q, SCOPE)
                self.assert_rejected_without_mutation(cfg, "v0")

    def test_missing_entry_condition_rejects_an_otherwise_valid_split(self):
        trace = DemoTrace(
            (
                scene(False, False, False),
                scene(False, True, False),
                scene(False, False, True),
            )
        )
        cfg = RelationalCFG.initial([DemoSegment(0, 0, 2, trace)], P, Q, SCOPE)
        self.assert_rejected_without_mutation(cfg, "v0")

    def test_bad_target_in_one_demo_rejects_the_whole_cfg(self):
        cfg = example_cfg(
            [(False, False, False), (True, False, False), (False, False, True)]
        )
        bad = example_cfg(
            [(False, False, False), (True, False, False), (False, False, False)]
        )
        for name in cfg.order:
            cfg.demos.segments[name].extend(bad.demos.segments[name])
        self.assertFalse(validate_cfg(cfg))
