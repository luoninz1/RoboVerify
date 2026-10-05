import unittest

import z3

from synthesis.api.instructions import MoveByName, PickByName, ReleaseByName
from synthesis.verification_lib.lowlevel_verification_lib import LowLevelContext
from synthesis.verification_lib.motion_verification import (
    MotionContract,
    verify_motion_block,
)
from synthesis.verification_lib.primitive_motion import PrimitiveMotionProblem
from synthesis.verification_lib.test_motion_verification import rooted_tower_conditions


def primitives(height=0.05):
    return [
        PickByName("a"),
        MoveByName("a", "a", "a", target_offset=[0, 0, 0.3]),
        MoveByName("b", "b", "b", target_offset=[0, 0, 0.2]),
        MoveByName("b", "b", "b", target_offset=[0, 0, height]),
        ReleaseByName("a", target_z=0.15),
    ]


class PrimitiveMotionTests(unittest.TestCase):
    scene = {"a": [0.3, 0, -0.1], "b": [0, 0, 0], "b0": [0, 0, 0], "sym": [2, 2, 0]}

    def verify(self, body):
        return verify_motion_block(
            rooted_tower_conditions(),
            body,
            ["a", "b", "b0"],
            MotionContract("a", "b"),
            initial_positions=self.scene,
            initial_arm=[0.3, 0, 0.2],
        )

    def test_pick_move_release_realizes_contract(self):
        result = self.verify(primitives())
        self.assertTrue(result, str(result))

    def test_pick_checks_horizontal_approach_then_vertical_descent(self):
        # The diagonal crosses the obstacle; the actual two-segment Pick clears
        # it at z=.2. Lowering the approach to z=.1 must detect a collision.
        for height, expected in ((0.2, "valid"), (0.1, "refuted")):
            with self.subTest(height=height):
                problem = PrimitiveMotionProblem(
                    LowLevelContext(default_L=0.05),
                    [],
                    ["a", "b", "obstacle"],
                    MotionContract("a", "b"),
                    None,
                    "pick",
                    3000,
                    initial_positions={
                        "a": [0.3, 0, 0],
                        "b": [0, 0, 0],
                        "obstacle": [0.15, 0, 0.1],
                        "sym": [2, 2, 0],
                    },
                    initial_arm=[0, 0, height],
                )
                problem.execute([PickByName("a")])
                approach = next(
                    c
                    for c in problem.checks
                    if c.obligation == "collision_1_approach_obstacle"
                )
                self.assertEqual(approach.status, expected)
                descent = next(
                    c
                    for c in problem.checks
                    if c.obligation == "collision_1_descend_obstacle"
                )
                self.assertEqual(descent.status, "valid")

    def test_unsupported_release_cannot_freeze_the_payload(self):
        result = self.verify(primitives(0.1))
        self.assertFalse(result)
        self.assertTrue(
            any(
                c.obligation.startswith("release_support") and c.status == "refuted"
                for c in result.checks
            )
        )

    def test_state_and_held_object_thread_across_blocks(self):
        p = PrimitiveMotionProblem(
            LowLevelContext(),
            [],
            ["a", "b", "b0"],
            MotionContract("a", "b"),
            None,
            "0",
            3000,
            initial_positions=self.scene,
            initial_arm=[0.3, 0, 0.2],
        )
        body = primitives()
        p.execute(body[:2])
        first = p.current["a"][2]
        p.execute([MoveByName("a", "a", "a", target_offset=[0, 0, 0.1])])
        p.check("composed", p.current["a"][2] != first + z3.RealVal(".1"))
        self.assertEqual(p.checks[-1].status, "valid")
        self.assertEqual(p.held_name, "a")

    def test_empty_gripper_motion_is_checked(self):
        result = self.verify(
            [MoveByName("b", "b", "b", target_offset=[0, 0, 0]), *primitives()]
        )
        self.assertFalse(result)
        self.assertTrue(
            any(
                c.obligation == "collision_1_b" and c.status == "refuted"
                for c in result.checks
            )
        )
