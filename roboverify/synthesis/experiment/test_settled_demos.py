"""Settled demo starts survive disk storage, candidate tracing and search resets."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from synthesis.api.instructions import Skip
from synthesis.api.program import Program
from synthesis.cfg.collection import record_execution, validate_trace
from synthesis.cfg.demos import DemoSegment
from synthesis.cfg.program_source import load_program
from synthesis.cfg.recordings import load_traces, save_traces
from synthesis.cfg.reset import capture, inner_env, reset_segment
from synthesis.cfg.straightline import _features, segment_rollout
from synthesis.mcmc import synthesis
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext

SOURCE = "synthesis.examples.stack:build_program"


class SettledDemoTests(unittest.TestCase):
    def test_collection_excludes_fifty_holding_actions(self):
        observed = {}

        def factory():
            env = synthesis.make_roboverify_stack_env(num_blocks=4)
            original_step = env.step
            observed["steps"] = 0

            def step(action):
                result = original_step(action)
                observed["steps"] += 1
                if observed["steps"] == 50:
                    observed["settled"] = capture(env)
                    inner = inner_env(env)
                    observed["observation"] = inner.flatten_observation(
                        inner._get_obs()
                    )
                return result

            env.step = step
            return env

        definition = load_program(SOURCE, HighLevelContext(), 4)
        trace = record_execution(
            definition,
            seed=38,
            num_blocks=4,
            env_factory=factory,
            max_loop_iterations=3,
        )
        self.assertTrue(validate_trace(trace), trace.metadata)
        self.assertEqual(
            trace.metadata["initialization"], {"source": "reset", "settling_steps": 50}
        )
        self.assertEqual(observed["steps"], 50 + len(trace.actions[0]))
        self.assertEqual(trace.actions[1][0], 0)
        np.testing.assert_array_equal(trace.states[0], observed["observation"])
        self.assert_snapshot_equal(trace.snapshots[0], observed["settled"])
        self.assertEqual(
            [
                e["bindings"]["b_prime"]
                for e in trace.events
                if e["kind"] == "loop_head"
            ],
            [1, 2, 3],
        )

    def assert_snapshot_equal(self, actual, expected):
        np.testing.assert_array_equal(actual.gt_state, expected.gt_state)
        self.assertEqual(actual.bindings, expected.bindings)
        self.assertEqual(actual.arrays.keys(), expected.arrays.keys())
        for name in actual.arrays:
            np.testing.assert_array_equal(actual.arrays[name], expected.arrays[name])

    def test_saved_start_is_used_without_reset_or_resettling(self):
        definition = load_program(SOURCE, HighLevelContext(), 4)
        original = record_execution(
            definition, seed=73, num_blocks=4, max_loop_iterations=3
        )
        self.assertTrue(validate_trace(original), original.metadata)
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "demonstrations.npz"
            save_traces(archive, [original])
            source = load_traces(archive, require_valid=True)[0]
        self.assertEqual(source.metadata["initialization"]["settling_steps"], 50)
        self.assert_snapshot_equal(source.snapshots[0], original.snapshots[0])

        def fresh_env(*args, **kwargs):
            env = synthesis.make_roboverify_stack_env(num_blocks=4)
            # A different raw reset cannot affect restoration. Forbid any later reset.
            env.reset()
            env.reset = Mock(
                side_effect=AssertionError("Saved state must replace reset")
            )
            return env

        restored = record_execution(
            load_program(SOURCE, HighLevelContext(), 4),
            seed=source.seed,
            num_blocks=4,
            env_factory=fresh_env,
            initial_snapshot=source.snapshots[0],
            max_loop_iterations=3,
        )
        self.assertTrue(validate_trace(restored), restored.metadata)
        self.assertEqual(
            restored.metadata["initialization"],
            {"source": "snapshot", "settling_steps": 0},
        )
        self.assert_snapshot_equal(restored.snapshots[0], source.snapshots[0])
        np.testing.assert_array_equal(restored.actions[0], source.actions[0])
        np.testing.assert_allclose(restored.states, source.states, atol=1e-8, rtol=0)

        # Both integrated synthesis reset modes start at S50, including when replaying
        # a later segment. No discarded preparation actions belong to the trace.
        env = fresh_env()
        try:
            for index in (0, len(source.states) // 2):
                segment = DemoSegment(0, index, len(source.states) - 1, source)
                for mode in ("reset", "replay"):
                    first = reset_segment(env, segment, mode=mode)
                    np.testing.assert_allclose(
                        first, source.states[index], atol=1e-8, rtol=0
                    )
        finally:
            env.close()
        segment = DemoSegment(0, 0, len(source.states) - 1, source)
        scenes = segment_rollout(Program(1, [Skip(0)]), segment, fresh_env)
        np.testing.assert_array_equal(scenes[0].observation, source.states[0])

        # Search must score the same sample sequence as collection, including a
        # supplied loop and its binding-only instructions.
        scenes = segment_rollout(definition.program, segment, fresh_env)
        np.testing.assert_allclose(
            _features(scenes, 4), _features(source.states, 4), atol=1e-8, rtol=0
        )

        # Standalone MCMC uses the same archived start instead of recreating seed 73.
        with patch(
            "synthesis.mcmc.synthesis.make_roboverify_env", side_effect=fresh_env
        ):
            trajectories, _, successes, _ = synthesis.rollout_demos(
                load_program(SOURCE, HighLevelContext(), 4).program,
                1,
                num_blocks=4,
                seeds=[source.seed],
                verbose=False,
                initial_snapshots={source.seed: source.snapshots[0]},
            )
        self.assertEqual(successes, [True])
        np.testing.assert_allclose(trajectories[0], source.states, atol=1e-8, rtol=0)

    def test_mcmc_cli_selects_matching_saved_starts(self):
        from synthesis.cfg.test_collection import example_trace
        from synthesis.experiment.mcmc.run import main

        traces = [example_trace(seed) for seed in (73, 38)]
        for trace in traces:
            self.assertTrue(validate_trace(trace))
        result = SimpleNamespace(
            best_cost=0, best_iter=0, iters_completed=0, best_program="Skip"
        )
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "demonstrations.npz"
            save_traces(archive, traces)
            with patch(
                "synthesis.experiment.mcmc.run.MCMC", return_value=result
            ) as search:
                code = main(
                    [
                        "--demos",
                        str(archive),
                        "--num-blocks",
                        "2",
                        "--num-seeds",
                        "1",
                        "--iters",
                        "0",
                        "--run-root",
                        str(Path(directory) / "runs"),
                        "--no-capture",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertEqual(search.call_args.kwargs["seeds"], [73])
            snapshots = search.call_args.kwargs["initial_snapshots"]
            self.assertEqual(set(snapshots), {73})
            self.assert_snapshot_equal(snapshots[73], traces[0].snapshots[0])

    def test_mcmc_cannot_fall_back_to_seed_reset_when_snapshot_missing(self):
        with self.assertRaisesRegex(ValueError, "Missing saved initial snapshots"):
            synthesis.rollout_demos(
                Program(1, [Skip(0)]),
                1,
                seeds=[42],
                initial_snapshots={},
                verbose=False,
            )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Missing saved initial snapshots"):
                synthesis.save_program_seed_videos(
                    Program(1, [Skip(0)]),
                    num_seeds=1,
                    num_block=4,
                    video_dir=directory,
                    seeds=[42],
                    initial_snapshots={},
                )


if __name__ == "__main__":
    unittest.main()
