"""Symbolic inference has one algorithm and no alternate learner selection."""

import contextlib
import io
import unittest
from copy import deepcopy
from unittest.mock import patch

import z3
from synthesis.api.instructions import Skip, While
from synthesis.api.program import Program
from synthesis.cfg.candidate_traces import prepare_candidate
from synthesis.cfg.demos import DemoSegment
from synthesis.cfg.invariants import infer_loop_invariant
from synthesis.cfg.program_adapter import program_to_cfg
from synthesis.cfg.program_source import prepare_program
from synthesis.cfg.tasks import task_spec
from synthesis.cfg.test_collection import example_trace
from synthesis.cfg.verification import propose_summaries
from synthesis.cfg.verified_synthesis import verified_synthesis
from synthesis.entry.synthesize_cfg import main
from synthesis.inference_lib.minimization import get_invariant_minimizer
from synthesis.verification_lib.cegis import run_symbolic_cegis
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext


class SymbolicInferenceRoutingTests(unittest.TestCase):
    def test_cli_scopes_and_records_minimizer(self):
        for backend in ("sympy", "pyeda"):
            with self.subTest(backend=backend), patch(
                "synthesis.entry.synthesize_cfg.RunLogger"
            ) as logger, patch(
                "synthesis.entry.synthesize_cfg._run"
            ) as run, contextlib.redirect_stdout(
                io.StringIO()
            ):

                def check_backend(args, logger):
                    self.assertEqual(get_invariant_minimizer(), backend)
                    return 0

                run.side_effect = check_backend
                self.assertEqual(
                    main(["--demos", "unused.npz", "--invariant-minimizer", backend]), 0
                )
                run.assert_called_once()
                self.assertEqual(
                    logger.call_args.args[2]["invariant_minimizer"], backend
                )
                self.assertEqual(get_invariant_minimizer(), "sympy")

    def test_cli_rejects_removed_learner_options(self):
        for learner in ("legacy", "monotone"):
            with self.subTest(learner=learner), contextlib.redirect_stderr(
                io.StringIO()
            ) as error:
                with patch("synthesis.entry.synthesize_cfg.run") as run:
                    with self.assertRaises(SystemExit) as exit:
                        main(["--demos", "unused.npz", "--learner", learner])
                    self.assertEqual(exit.exception.code, 2)
                    run.assert_not_called()
                self.assertIn("unrecognized arguments: --learner", error.getvalue())

    def test_symbolic_apis_reject_alternate_learner_injection(self):
        for function, args in (
            (prepare_candidate, (None, None)),
            (infer_loop_invariant, (None, None, None)),
            (verified_synthesis, (None, None, None, None)),
            (run_symbolic_cegis, (None, None, None, None, None)),
        ):
            with self.subTest(function=function.__name__):
                with self.assertRaisesRegex(TypeError, "learner"):
                    function(*args, learner="monotone")

    def test_candidate_bootstrap_passes_runtime_exit_to_inv_inference(self):
        context = HighLevelContext()
        definition = prepare_program(
            Program(1, [While(z3.BoolVal(False), [], [Skip(0)], None)]), 2
        )
        expert = example_trace()
        expert.events = tuple(
            dict(kind=kind, path="0", index=0, entry_index=0, bindings={"b0": 0})
            for kind in ("loop_enter", "loop_exit")
        )
        cfg = program_to_cfg(
            definition, [DemoSegment(0, 0, 1, expert)], *task_spec("stack")
        )
        propose_summaries(cfg, context)
        candidate = deepcopy(expert)
        candidate.metadata["status"] = "completed"
        with patch(
            "synthesis.cfg.candidate_traces.record_execution", return_value=candidate
        ), patch(
            "synthesis.cfg.candidate_traces.InvInference",
            return_value=(z3.BoolVal(True), []),
        ) as infer:
            prepare_candidate(cfg, context)
        infer.assert_called_once()
        store, loop_id, vocabulary, actual_context = infer.call_args.args
        self.assertEqual(len(store.for_loop(loop_id)), 1)
        self.assertEqual(vocabulary.constants, ("b0",))
        self.assertIs(actual_context, context)
        self.assertTrue(z3.is_true(cfg.nodes["v0"].region.invariant))


if __name__ == "__main__":
    unittest.main()
