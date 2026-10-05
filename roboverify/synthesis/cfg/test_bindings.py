import unittest
from unittest.mock import Mock, patch

import numpy as np
from synthesis.api.instructions import (
    Assign,
    Get,
    Move,
    Pick,
    PickByName,
    Release,
    Skip,
)
from synthesis.api.program import Program
from synthesis.cfg.bindings import (
    close_objects,
    mutate_scoped,
    require_closed,
    seed_from_scope,
)
from synthesis.cfg.demos import DemoSegment, DemoTrace
from synthesis.cfg.graph import RelationalCFG
from synthesis.cfg.region import BlockRegion
from synthesis.cfg.scope import scope
from synthesis.cfg.straightline import _features, postcondition_reached, segment_rollout
from synthesis.predicates.scene import Scene
from synthesis.predicates.term import atom, boolean, ref
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext


class BindingTests(unittest.TestCase):
    def demo(self):
        s = Scene({0: (0.0, 0.0, 0.0), 1: (0.2, 0.0, 0.0)}, {"b0": 0, "o1": 1})
        return DemoSegment(0, 0, 1, DemoTrace((s, s), num_blocks=2), dict(s.bindings))

    def test_numeric_closure_does_not_capture_demo_only_alias(self):
        p = Program(
            3,
            [
                Pick(1),
                Move(0, 0, 1, target_offset=[0, 0, 0.1]),
                Release(1, target_z=0.1),
            ],
        )
        closed = close_objects(p, [self.demo()], {"b0"}, HighLevelContext())
        self.assertIsInstance(closed.instructions[0], Get)
        self.assertNotIn("o1", require_closed(closed.instructions, {"b0"}))
        self.assertEqual(closed.instructions[2].target_box_name_x, "b0")
        self.assertEqual(
            closed.instructions[1].grab_box_name,
            closed.instructions[3].release_box_name,
        )
        self.assertEqual(p.instructions[0].grab_box_id, 1)
        with self.assertRaisesRegex(ValueError, "Out-of-scope"):
            close_objects(
                Program(1, [PickByName("o1")]),
                [self.demo()],
                {"b0"},
                HighLevelContext(),
            )

    def test_named_mutation_preserves_get_and_excludes_demo_aliases(self):
        p = close_objects(
            Program(1, [Pick(1)]), [self.demo()], {"b0"}, HighLevelContext()
        )
        get = str(p.instructions[0])
        for _ in range(40):
            p = mutate_scoped(p, {"b0"})
            self.assertEqual(str(p.instructions[0]), get)
            self.assertNotIn("o1", require_closed(p.instructions, {"b0"}))
        cfg = RelationalCFG.initial([self.demo()], boolean(True), boolean(True), {"b0"})
        cfg.nodes["v0"].region = BlockRegion(
            None, tuple(p.instructions), frozenset({"object"})
        )
        self.assertIn("object", scope(cfg)[cfg.exit])

    def test_new_loop_body_seed_uses_only_its_existing_names(self):
        import random

        available = {"base", "top", "selected", "tbl"}
        first = seed_from_scope(30, available, random.Random(7))
        second = seed_from_scope(30, available, random.Random(7))
        from synthesis.cfg.program_source import describe_program

        self.assertEqual(describe_program(first), describe_program(second))
        self.assertEqual(len(first.instructions), 30)
        names = set()
        for instruction in first.instructions:
            self.assertNotIsInstance(instruction, Get)
            for operand in instruction.get_operand():
                self.assertEqual(operand["type"], "BoxName")
                self.assertIn(operand["val"], available - {"tbl"})
                names.add(operand["val"])
        self.assertEqual(names, available - {"tbl"})
        require_closed(first.instructions, available)
        with self.assertRaisesRegex(ValueError, "in-scope"):
            seed_from_scope(1, {"tbl"}, random.Random(7))

    def test_rollout_scores_bindings_after_assignment(self):
        obs = np.zeros(43)
        obs[10:13], obs[22:25] = (0, 0, 0), (0.2, 0, 0)
        trace = DemoTrace((obs, obs), num_blocks=2)
        segment = DemoSegment(0, 0, 1, trace, {"b": 0, "other": 1})
        env = Mock(symbolic_name_to_box_id=dict(segment.bindings))
        with patch("synthesis.cfg.straightline.reset_segment", return_value=obs):
            states = segment_rollout(
                Program(1, [Assign("b", "other")]), segment, lambda _: env
            )
        self.assertEqual(states[0].bindings["b"], 0)
        self.assertEqual(states[-1].bindings["b"], 1)
        self.assertTrue(
            postcondition_reached(states, segment, atom("eq", ref("b"), ref("other")))
        )
        env.close.assert_called_once()

    def test_imitation_samples_match_recording_without_losing_binding_states(self):
        initial = np.zeros(43)
        initial[10:13], initial[22:25] = (0, 0, 0), (0.2, 0, 0)
        moved, final = initial.copy(), initial.copy()
        moved[0], final[0] = 0.1, 0.2
        # Repeated observations produced by the program remain real samples.
        observations = (initial, moved, moved, final)
        trace = DemoTrace(observations, num_blocks=2)
        segment = DemoSegment(0, 0, 3, trace, {"b": 0, "other": 1})
        env = Mock(symbolic_name_to_box_id=dict(segment.bindings))
        env.flatten_observation.side_effect = observations[1:]
        program = Program(3, [Skip(2), Assign("b", "other"), Skip(1)])
        with patch("synthesis.cfg.straightline.reset_segment", return_value=initial):
            states = segment_rollout(program, segment, lambda _: env)
        np.testing.assert_array_equal(_features(states, 2), _features(observations, 2))
        self.assertTrue(
            any(s.bindings["b"] == 1 and s.observation[0] == 0.1 for s in states)
        )
        self.assertEqual(states[0].bindings["b"], 0)
        self.assertTrue(
            postcondition_reached(states, segment, atom("eq", ref("b"), ref("other")))
        )
        env.close.assert_called_once()
