"""Independent placement oracle must expose drift and incorrect placements."""

import unittest

import numpy as np

from synthesis.cfg.demos import DemoTrace
from synthesis.experiment.compare_stack_heights import compare_traces


def stack_trace():
    initial = np.zeros(46)
    for i, p in enumerate(((1.2, 0.6, 0.425), (1.4, 0.6, 0.425), (1.2, 0.8, 0.425))):
        initial[10 + 12 * i : 13 + 12 * i] = p
    middle = initial.copy()
    middle[12] -= 0.00025
    middle[22:25] = (1.2, 0.6, 0.4745)
    final = middle.copy()
    final[34:37] = (1.2, 0.6, 0.524)
    events = tuple(
        dict(
            kind=kind,
            path="1",
            invocation=0,
            iteration=i,
            index=i,
            bindings=dict(b0=0, b=i, b_prime=min(i + 1, 2)),
        )
        for i, kind in enumerate(("loop_head", "loop_head", "loop_exit"))
    )
    return DemoTrace(
        (initial, middle, final), task="stack", num_blocks=3, events=events
    )


class StackHeightComparisonTests(unittest.TestCase):
    def test_contact_drift_matches_ideal_levels_with_tolerance(self):
        result = compare_traces([stack_trace()], tolerance=0.001)
        self.assertTrue(result["matches_ideal"])
        self.assertEqual(result["continuing_heads"], 2)
        self.assertEqual(result["normal_exits"], 1)
        self.assertEqual(result["relations"]["Higher"]["evaluations"], 27)
        self.assertEqual(result["strict_higher_mismatches"], 1)
        self.assertEqual(result["higher_order_violation_states"], 0)
        strict = compare_traces([stack_trace()], tolerance=0)
        self.assertEqual(strict["relations"]["Higher"]["mismatches"], 1)

    def test_oracle_is_independent_of_observed_height(self):
        trace = stack_trace()
        trace.states[1][
            24
        ] = 0.425  # Failed first placement; metadata alone cannot hide it.
        result = compare_traces([trace])
        self.assertGreater(result["relations"]["Higher"]["mismatches"], 0)
        self.assertFalse(result["matches_ideal"])

    def test_separation_mismatch_is_reported_separately(self):
        trace = stack_trace()
        trace.states[1][23] = 0.705
        result = compare_traces([trace])
        self.assertEqual(result["relations"]["Higher"]["mismatches"], 0)
        self.assertGreater(result["relations"]["Scattered"]["mismatches"], 0)
        self.assertFalse(result["matches_ideal"])

    def test_incomplete_or_unexpected_program_is_rejected(self):
        trace = stack_trace()
        trace.events = trace.events[:-1]
        with self.assertRaisesRegex(ValueError, "normal exit"):
            compare_traces([trace])
        trace = stack_trace()
        trace.events[1]["bindings"]["b"] = 0
        with self.assertRaisesRegex(ValueError, "convention"):
            compare_traces([trace])


if __name__ == "__main__":
    unittest.main()
