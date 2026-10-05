"""Prove Stack using an invariant actually learned by the partition algorithm."""

import contextlib
import io
import unittest

from synthesis.cfg.program_adapter import program_to_cfg
from synthesis.cfg.program_source import load_program
from synthesis.cfg.tasks import task_spec
from synthesis.cfg.verification import propose_summaries, verify_cfg_symbolic
from synthesis.cfg.verified_synthesis import loop_regions
from synthesis.inference_lib.demo_store import (
    DemoStore,
    InferenceVocabulary,
    InvInference,
    LoopHeadState,
)
from synthesis.verification_lib.counterexamples import stacks_to_positions, state_holds
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext


class StackInferenceTests(unittest.TestCase):
    minimizer = "sympy"

    def test_partition_learner_proves_stack_with_on_and_equality(self):
        context = HighLevelContext()
        blocks = [f"x{i}" for i in range(4)]
        entry = stacks_to_positions([[block] for block in blocks])
        store = DemoStore(
            LoopHeadState(
                "1",
                stacks_to_positions(
                    [blocks[: top + 1]] + [[block] for block in blocks[top + 1 :]]
                ),
                entry,
                {"b0": blocks[0], "b": blocks[top]},
            )
            for top in range(4)
        )
        # Three continuing heads plus the normal exit. No handwritten invariant
        # is passed to the verifier, and no particular optimizer tie is required.
        with contextlib.redirect_stdout(io.StringIO()):
            invariant, _ = InvInference(
                store,
                "1",
                InferenceVocabulary(2, ("ON_star", "equality"), ("b", "b0")),
                context,
                minimizer=self.minimizer,
            )
        self.assertTrue(all(state_holds(invariant, row) for row in store.for_loop("1")))
        definition = load_program("synthesis.examples.stack:build_program", context, 4)
        cfg = program_to_cfg(definition, [], *task_spec("stack"))
        propose_summaries(cfg, context)
        next(loop_regions(cfg)).invariant = invariant
        result = verify_cfg_symbolic(cfg, context, max_blocks=6, timeout_ms=10000)
        self.assertTrue(result, str(result))
        self.assertEqual(result.scope, "unbounded")
        self.assertEqual(len(result.checks), 18)
        self.assertEqual(
            {check.vc.kind for check in result.checks},
            {"establish", "preserve", "exit"},
        )


class PyEDAStackInferenceTests(StackInferenceTests):
    minimizer = "pyeda"


if __name__ == "__main__":
    unittest.main()
