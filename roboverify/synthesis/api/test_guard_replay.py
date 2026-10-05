"""Prescribed counterexample witnesses remain enabled runtime choices."""

import unittest
from types import SimpleNamespace

import numpy as np
import z3

from synthesis.api.guard_eval import NoGuardWitness, find_and_bind, replay_guard_choices
from synthesis.api.instructions import While
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext


class GuardReplayTests(unittest.TestCase):
    def setUp(self):
        context = HighLevelContext()
        selected, current = context.get_consts("selected"), context.get_consts(
            "current"
        )
        self.loop = While(selected != current, [selected], [], z3.BoolVal(True))
        self.env = SimpleNamespace(num_blocks=3, symbolic_name_to_box_id={"current": 0})
        self.traj = [np.zeros(46)]

    def choice(self, value):
        return {"kind": "While", "bindings": {"selected": value}}

    def test_higher_id_can_be_replayed_and_default_policy_is_restored(self):
        with replay_guard_choices([self.choice(2)]):
            self.assertTrue(find_and_bind(self.loop, self.env, self.traj))
            self.assertEqual(self.env.symbolic_name_to_box_id["selected"], 2)
        self.assertTrue(find_and_bind(self.loop, self.env, self.traj))
        self.assertEqual(self.env.symbolic_name_to_box_id["selected"], 1)

    def test_replay_is_only_a_prefix_then_normal_execution_continues(self):
        with replay_guard_choices([self.choice(2)]):
            find_and_bind(self.loop, self.env, self.traj)
            self.env.symbolic_name_to_box_id["current"] = 2
            find_and_bind(self.loop, self.env, self.traj)
            self.assertEqual(self.env.symbolic_name_to_box_id["selected"], 0)

    def test_disabled_or_nonexistent_choice_fails_without_fallback(self):
        for selected in (0, 3):
            with self.subTest(selected=selected), self.assertRaises(NoGuardWitness):
                with replay_guard_choices([self.choice(selected)]):
                    find_and_bind(self.loop, self.env, self.traj)
        self.assertTrue(find_and_bind(self.loop, self.env, self.traj))

    def test_unconsumed_choices_and_wrong_instruction_are_errors(self):
        with self.assertRaises(NoGuardWitness):
            with replay_guard_choices([self.choice(2)]):
                pass
        wrong = self.choice(2)
        wrong["kind"] = "Get"
        with self.assertRaises(NoGuardWitness):
            with replay_guard_choices([wrong]):
                find_and_bind(self.loop, self.env, self.traj)


if __name__ == "__main__":
    unittest.main()
