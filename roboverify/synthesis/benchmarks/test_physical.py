"""Simulator-backed adapter regression tests, independent of any local model."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from synthesis.benchmarks.physical import PhysicalSuite
from synthesis.benchmarks.specs import Case
from synthesis.cap.test_row_verification import SWAP
from synthesis.cap.verification import verify_tower_trace

STACK = """
def policy():
    top = base
    for obj in objects():
        if obj != base:
            pick(obj)
            move(obj,obj,obj,(0,0,.15))
            move(base,base,obj,(0,0,0))
            move(base,base,top,(0,0,.05))
            release(obj)
            top = obj
"""


class PhysicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.suite = PhysicalSuite()

    @classmethod
    def tearDownClass(cls):
        cls.suite.close()

    def test_stack_and_existing_motion_backend(self):
        result = self.suite.run(Case("stack", "train", 0, 3), STACK)
        self.assertTrue(result["success"], result.get("error"))
        self.assertEqual(verify_tower_trace(result)["status"], "valid")
        incomplete = {**result, "primitive_calls": result["primitive_calls"] + 1}
        self.assertEqual(verify_tower_trace(incomplete)["status"], "unsupported")

    def test_row_buffered_sort_and_key_witness(self):
        case = Case("insertion_sort", "train", 0, 4)
        result = self.suite.run(case, SWAP)
        self.assertTrue(result["success"], result.get("error"))
        self.assertGreater(result["physics_steps"], 0)
        keys = dict(zip(result["scene"]["initial_order"], [3, 2, 1, 0]))
        noop = self.suite.run(case, "def policy():\n    pass\n", keys_override=keys)
        self.assertFalse(noop["success"])

    def test_five_block_initial_tower_is_clear_of_gripper(self):
        env, scene = self.suite.prepare(Case("unstack", "test", 1001, 5, "reverse"))
        grip = env.sim.data.get_site_xpos("robot0:grip")
        self.assertGreater(
            grip[2], max(point[2] for point in scene.positions.values()) + 0.09
        )
        self.assertTrue(
            all(
                np.linalg.norm(np.array(point)[:2] - grip[:2]) > 0.10
                for point in scene.positions.values()
            )
        )

    def test_exact_full_snapshot_restore_and_archive(self):
        case = Case("stack", "train", 0, 3)
        env, scene = self.suite.prepare(case)
        before = env.sim.get_state().flatten().copy()
        before_inputs = {
            name: value.copy() for name, value in self.suite.starts[case.name][1].inputs
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "start.npz"
            self.suite.save_start(case, path)
            self.suite.run(case, STACK)
            self.suite.load_start(case, scene, path)
            self.assertTrue(np.array_equal(before, env.sim.get_state().flatten()))
            for name, value in before_inputs.items():
                self.assertTrue(
                    np.array_equal(value, getattr(env.sim.data, name)), name
                )


if __name__ == "__main__":
    unittest.main()
