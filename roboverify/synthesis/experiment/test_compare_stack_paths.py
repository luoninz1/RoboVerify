"""Separate saturation from changed direction, physical drift, and waypoint turns."""

import unittest

import numpy as np

from synthesis.api.control import PrimitiveController
from synthesis.api.test_control import ServoEnvironment
from synthesis.experiment.compare_stack_paths import (
    capture_phases,
    cross_track,
    direction_angles,
    phase_metrics,
)


class StackPathTests(unittest.TestCase):
    def test_saturation_does_not_always_change_direction(self):
        raw = np.array([[4, 1, 0], [0, 0, 4], [4, 4, 0], [0, 0, 0]])
        angles = direction_angles(raw, np.clip(raw, -1, 1))
        self.assertAlmostEqual(angles[0], 30.963756532, places=7)
        np.testing.assert_allclose(angles[1:], 0, atol=2e-6)

    def test_cross_track_does_not_count_endpoint_shortfall_or_phase_turn(self):
        points = [[0, 0, 0], [0.05, 0, 0], [0.09, 0, 0]]
        np.testing.assert_allclose(cross_track(points, [0.1, 0, 0]), 0)
        second = [[0.09, 0, 0], [0.09, 0.05, 0], [0.09, 0.1, 0]]
        np.testing.assert_allclose(cross_track(second, second[-1]), 0)
        self.assertAlmostEqual(
            cross_track([[0, 0, 0], [0.05, 0.02, 0]], [0.1, 0, 0])[1], 0.02
        )

    def test_capture_preserves_baseline_and_uniform_trial_is_straight(self):
        target = np.array([0.3, 0.15, 0.6])
        reference = ServoEnvironment()
        PrimitiveController(reference, []).move(target, phase="approach")
        results = []
        for uniform in (False, True):
            env, rows = ServoEnvironment(), []
            with capture_phases(rows, uniform=uniform):
                PrimitiveController(env, []).move(target, phase="approach")
            if uniform:
                np.testing.assert_array_equal(env.actions, reference.actions)
            self.assertTrue(rows[0]["converged"])
            results.append(rows[0])
            # Gripper command must retain its strength under XYZ scaling.
            np.testing.assert_array_equal(np.array(env.actions)[:, 3], -0.2)
        self.assertGreater(results[0]["max_cross_track_mm"], 30)
        self.assertLess(results[1]["max_cross_track_mm"], 1e-10)
        self.assertGreater(results[0]["direction_changed_steps"], 0)
        self.assertEqual(results[1]["direction_changed_steps"], 0)

    def test_empty_phase_and_patch_restoration(self):
        original = PrimitiveController.move
        env, rows = ServoEnvironment(), []
        with capture_phases(rows):
            PrimitiveController(env, []).move(env.obs[:3], phase="approach")
        self.assertIs(PrimitiveController.move, original)
        self.assertEqual(rows[0]["steps"], 0)
        self.assertEqual(phase_metrics(rows[0])["max_cross_track_mm"], 0)


if __name__ == "__main__":
    unittest.main()
