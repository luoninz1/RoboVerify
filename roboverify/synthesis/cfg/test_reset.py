import unittest

import numpy as np
from synthesis.api.instructions import MoveByName, PickByName
from synthesis.api.program import Program
from synthesis.cfg.demos import DemoSegment, DemoTrace
from synthesis.cfg.reset import capture, collect_recording, reset_segment
from synthesis.mcmc.synthesis import (
    make_roboverify_stack_env,
    preserved_global_rng,
    set_np_seed,
)


class SegmentResetTests(unittest.TestCase):
    def test_reset_matches_action_replay_and_next_action_while_holding(self):
        with preserved_global_rng():
            set_np_seed(0)
            env = make_roboverify_stack_env(num_blocks=2)
            try:
                program = Program(
                    2,
                    [
                        PickByName("b0"),
                        MoveByName("b0", "b0", "b0", target_offset=[0, 0, 0.1]),
                    ],
                )
                states, record = collect_recording(program, env)
                trace = DemoTrace(
                    states,
                    tuple(record.snapshots),
                    (tuple(record.actions), tuple(record.action_indices)),
                    num_blocks=2,
                )
                for t in sorted(set((0, len(states) // 2, len(states) - 1))):
                    segment = DemoSegment(0, t, len(states) - 1, trace, {"b0": 0})
                    reset_segment(env, segment, mode="reset")
                    direct = capture(env)
                    action = np.array([0.0, 0.0, 0.001, -0.2])
                    next_direct = env.step(action)[0]
                    reset_segment(env, segment, mode="replay")
                    replay = capture(env)
                    next_replay = env.step(action)[0]
                    np.testing.assert_allclose(
                        direct.gt_state, replay.gt_state, atol=1e-8, rtol=0
                    )
                    np.testing.assert_allclose(
                        next_direct, next_replay, atol=1e-8, rtol=0
                    )
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()


class RecordingTests(unittest.TestCase):
    def test_archive_roundtrip_and_observation_only_rejected(self):
        import tempfile
        from pathlib import Path

        from synthesis.api.instructions import Skip
        from synthesis.cfg.recordings import load_traces, save_traces

        with preserved_global_rng():
            set_np_seed(0)
            env = make_roboverify_stack_env(num_blocks=2)
            try:
                program = Program(1, [Skip(3)])
                states, record = collect_recording(program, env)
                trace = DemoTrace(
                    states,
                    tuple(record.snapshots),
                    (tuple(record.actions), tuple(record.action_indices)),
                    num_blocks=2,
                )
                with tempfile.TemporaryDirectory() as root:
                    path = Path(root) / "record.npz"
                    save_traces(path, [trace])
                    loaded = load_traces(path)[0]
                    np.testing.assert_array_equal(
                        loaded.snapshots[0].gt_state, trace.snapshots[0].gt_state
                    )
                    segment = DemoSegment(0, 1, 3, loaded, {"b0": 0})
                    np.testing.assert_allclose(
                        reset_segment(env, segment, mode="reset"), states[1]
                    )
                observations = DemoTrace(states, num_blocks=2)
                with self.assertRaisesRegex(ValueError, "recollect"):
                    reset_segment(env, DemoSegment(0, 1, 3, observations, {"b0": 0}))
            finally:
                env.close()
