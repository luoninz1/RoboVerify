"""Check that diagnostics preserve execution and use finite path segments."""

import unittest
from contextlib import nullcontext
from dataclasses import replace

import numpy as np

from synthesis.cfg.collection import record_execution, validate_trace
from synthesis.cfg.program_source import load_program
from synthesis.experiment.compare_stack_control import measure, segment_distance
from synthesis.experiment.controller_baseline import (
    previous_release_control,
    stack_env_with_head_contacts,
)
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext


class ComparisonTests(unittest.TestCase):
    def test_segment_distance_counts_overshoot_and_handles_zero_length(self):
        np.testing.assert_allclose(
            segment_distance(
                [[0, 0, 0], [0.5, 0.2, 0], [1.3, 0, 0]], [0, 0, 0], [1, 0, 0]
            ),
            [0, 0.2, 0.3],
        )
        np.testing.assert_allclose(
            segment_distance([[1, 2, 2]], [0, 0, 0], [0, 0, 0]), [3]
        )

    def test_measurement_preserves_updated_and_previous_rollouts(self):
        for previous in (False, True):
            with self.subTest(previous=previous):
                definition = load_program(
                    "synthesis.examples.stack:build_program", HighLevelContext(), 4
                )
                if previous:
                    pick = definition.program.instructions[1].body[0]
                    pick.control = replace(pick.control, position_tolerance=0.01)
                factory = stack_env_with_head_contacts if previous else None
                with previous_release_control() if previous else nullcontext():
                    reference = record_execution(
                        definition,
                        seed=9,
                        num_blocks=4,
                        env_factory=factory,
                        max_loop_iterations=3,
                    )
                    rows = []
                    with measure(rows):
                        measured = record_execution(
                            definition,
                            seed=9,
                            num_blocks=4,
                            env_factory=factory,
                            initial_snapshot=reference.snapshots[0],
                            max_loop_iterations=3,
                        )
                self.assertTrue(validate_trace(reference), reference.metadata)
                self.assertTrue(validate_trace(measured), measured.metadata)
                np.testing.assert_allclose(
                    reference.actions[0], measured.actions[0], rtol=0, atol=1e-8
                )
                np.testing.assert_allclose(
                    reference.states, measured.states, rtol=0, atol=1e-8
                )
                self.assertEqual(len(rows), 18)
                self.assertEqual(sum(bool(p["payload"]) for p in rows), 9)
                for p in rows:
                    if p["phase"] == "retreat":
                        np.testing.assert_array_equal(
                            p["reference_start"][:2], p["target"][:2]
                        )


if __name__ == "__main__":
    unittest.main()
