"""Stack resets retain task geometry while excluding the far workspace edge."""

import unittest
from unittest.mock import patch

import numpy as np
from synthesis.environment.stack_reset import sample_stack_xy
from synthesis.util.on import scattered_implementation


class StackResetTests(unittest.TestCase):
    def setUp(self):
        state = np.random.get_state()
        self.addCleanup(np.random.set_state, state)
        self.base = np.array([0.69189994, 0.74409998])
        self.gripper = self.base + [0.65, 0.005]

    def assert_layout(self, positions, base, gripper):
        positions = np.asarray(positions)
        self.assertTrue(np.all(np.linalg.norm(positions - base, axis=1) <= 0.70))
        self.assertTrue(np.all(np.linalg.norm(positions - gripper, axis=1) >= 0.10))
        for i, position in enumerate(positions):
            for other in positions[:i]:
                self.assertTrue(
                    scattered_implementation(
                        np.r_[position, 0.425], np.r_[other, 0.425]
                    )
                )

    def test_many_seeds_preserve_separation_and_distance_bounds(self):
        for count in (2, 3, 4, 6):
            for seed in range(100):
                with self.subTest(count=count, seed=seed):
                    np.random.seed(seed)
                    positions = sample_stack_xy(count, self.base, self.gripper)
                    self.assertEqual(len(positions), count)
                    self.assert_layout(positions, self.base, self.gripper)

    def test_layout_is_seeded_and_relative_to_base(self):
        np.random.seed(73)
        original = np.asarray(sample_stack_xy(4, self.base, self.gripper))
        np.random.seed(73)
        repeated = np.asarray(sample_stack_xy(4, self.base, self.gripper))
        np.testing.assert_array_equal(original, repeated)
        offset = np.array([0.25, -0.1])
        np.random.seed(73)
        shifted = sample_stack_xy(4, self.base + offset, self.gripper + offset)
        np.testing.assert_allclose(shifted, original + offset, atol=1e-14, rtol=0)
        np.random.seed(74)
        self.assertFalse(
            np.array_equal(sample_stack_xy(4, self.base, self.gripper), original)
        )

    def test_exhaustion_never_relaxes_radius_or_returns_partial_layout(self):
        # The far corner belongs to the sampling rectangle but exceeds 70 cm.
        with patch(
            "synthesis.environment.stack_reset.np.random.uniform",
            return_value=np.array([0.70, 0.20]),
        ), self.assertRaisesRegex(ValueError, "within 0.70 m"):
            sample_stack_xy(4, self.base, self.gripper)
        # Repeated positions cannot satisfy Scattered, even inside the radius.
        with patch(
            "synthesis.environment.stack_reset.np.random.uniform",
            return_value=np.array([0.55, -0.15]),
        ), self.assertRaisesRegex(ValueError, "within 0.70 m"):
            sample_stack_xy(4, self.base, self.gripper)

    def test_simulator_reset_uses_bounded_layout_and_keeps_base_binding(self):
        from synthesis.mcmc.synthesis import make_roboverify_stack_env

        env = make_roboverify_stack_env(num_blocks=4)
        self.addCleanup(env.close)
        inner = env.env
        for seed in (0, 3, 38, 46, 73, 85, 99):
            with self.subTest(seed=seed):
                np.random.seed(seed)
                obs, _ = env.reset()
                xyz = np.array([obs[10 + 12 * i : 13 + 12 * i] for i in range(4)])
                self.assert_layout(
                    xyz[:, :2], inner.robot_base_xy, inner.initial_gripper_xpos[:2]
                )
                np.testing.assert_allclose(
                    xyz[:, 2], inner.height_offset, atol=1e-7, rtol=0
                )
                self.assertEqual(inner.symbolic_name_to_box_id, {"b0": 0})


if __name__ == "__main__":
    unittest.main()
