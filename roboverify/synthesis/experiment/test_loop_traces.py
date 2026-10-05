"""Real primitive Stack collection includes terminal loop heads."""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from synthesis.cfg.collection import record_execution, validate_trace
from synthesis.cfg.program_source import load_program
from synthesis.cfg.recordings import loop_store
from synthesis.inference_lib.demo_store import to_inference_inputs, tower_vocabulary
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext


class SimulatorLoopTraces(unittest.TestCase):
    def test_stack_rollout_has_two_bodies_and_terminal_head(self):
        context = HighLevelContext()
        definition = load_program("synthesis.examples.stack:build_program", context, 3)
        trace = record_execution(definition, seed=0, num_blocks=3)
        self.assertTrue(validate_trace(trace), trace.metadata)
        self.assertEqual(len([e for e in trace.events if e["kind"] == "loop_head"]), 2)
        self.assertEqual(len([e for e in trace.events if e["kind"] == "loop_exit"]), 1)
        store = loop_store([trace])
        rows = store.for_loop("1")
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(r.entry_positions == rows[0].entry_positions for r in rows))
        self.assertNotEqual(rows[0].positions, rows[-1].positions)
        self.assertTrue(all("b_prime" not in r.constants for r in rows))
        zeros, states, mappings = to_inference_inputs(
            store, "1", tower_vocabulary("stack"), context
        )
        self.assertEqual([len(zeros), len(states), len(mappings)], [3, 3, 3])

    @unittest.skipUnless(shutil.which("ffmpeg"), "Video encoder unavailable")
    def test_video_does_not_change_recorded_actions_or_states(self):
        context = HighLevelContext()
        source = "synthesis.examples.stack:build_program"
        plain = record_execution(load_program(source, context, 2), seed=0, num_blocks=2)
        with tempfile.TemporaryDirectory() as directory:
            filmed = record_execution(
                load_program(source, context, 2),
                seed=0,
                num_blocks=2,
                video_path=Path(directory) / "seed_0000.mp4",
            )
            self.assertEqual(
                filmed.metadata["video"]["status"], "saved", filmed.metadata
            )
            self.assertEqual(
                filmed.metadata["video"]["frames"], len(filmed.actions[0]) + 1
            )
        self.assertTrue(validate_trace(plain), plain.metadata)
        self.assertTrue(validate_trace(filmed), filmed.metadata)
        np.testing.assert_array_equal(plain.actions[0], filmed.actions[0])
        np.testing.assert_allclose(plain.states, filmed.states, atol=1e-8, rtol=0)
        self.assertEqual(plain.events, filmed.events)
