import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import z3
from synthesis.api.instructions import Assign, While
from synthesis.api.program import Program
from synthesis.experiment.invariant_learning.runner import (
    ExperimentConfig,
    run_experiment,
)
from synthesis.experiment.invariant_learning.witness import (
    WitnessQuery,
    failure_paths,
    find_witness,
)
from synthesis.inference_lib.demo_store import (
    InferenceVocabulary,
    InvInference,
    LoopHeadState,
)
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext
from synthesis.verification_lib.motion_verification import (
    MotionCheck,
    MotionVerificationResult,
)
from synthesis.verification_lib.symbolic_verify import (
    SymbolicVerificationResult,
    discharge_vc,
)


class BindingTask:
    """Small independent adapter: bind a to b, without any Stack names or geometry."""

    loop_id = "0"
    vocabulary = InferenceVocabulary(0, ("equality",), ("a", "b"))

    def __init__(self):
        self.context = HighLevelContext()
        self.invariant = z3.BoolVal(False)
        self.verify_motion = Mock(
            return_value=MotionVerificationResult(
                [MotionCheck("valid_contract", "valid")], checked_blocks=1
            )
        )
        self.describe = Mock(return_value={"task": "binding"})
        self.executions = 0
        self.valid = True

    def set_invariant(self, value):
        self.invariant = value

    def problem(self, size=None):
        from uuid import uuid4

        ctx = (
            self.context
            if size is None
            else HighLevelContext(
                mode="enum", num_blocks=size, sort_name="Binding_" + uuid4().hex
            )
        )
        invariant = ctx.spec_to_expr(
            self.context.expr_to_spec(self.invariant), known_const_names=["a", "b"]
        )
        a, b = ctx.get_consts("a"), ctx.get_consts("b")
        return (
            Program(1, [While(a != b, [], [Assign("a", "b")], invariant)]),
            a != b,
            a == b,
            ctx,
        )

    def verify_symbolic(self, timeout_ms, size=None):
        program, pre, post, ctx = self.problem(size)
        return SymbolicVerificationResult(
            [
                discharge_vc(vc, ctx, timeout_ms)
                for vc in program.VC_gen(pre, post, ctx)
            ],
            scope="unbounded" if size is None else f"finite:{size}",
            num_blocks=size,
        )

    def search_query(self, size, timeout_ms, failures):
        program, pre, post, ctx = self.problem(size)
        selected = {(c.vc.kind, c.vc.loop_id) for c in failures}
        checks = [
            vc
            for vc in program.VC_gen(pre, post, ctx)
            if (vc.kind, vc.loop_id) in selected
        ]
        paths = failure_paths(program, ctx, checks, 1)
        solver = ctx.new_solver(timeout_ms)
        solver.add(pre)
        return WitnessQuery(
            solver,
            z3.Or(*[p.condition for p in paths]),
            lambda model: {"size": size},
            1,
            ctx,
            paths,
        )

    def execute(self, search, **kwargs):
        self.executions += 1
        return SimpleNamespace(states=(), metadata={"status": "completed"})

    def validate(self, trace, search):
        return self.valid

    def learning_states(self, trace):
        positions = {"x1": [0, 0, 0.425], "x2": [0.15, 0, 0.425]}
        return [
            LoopHeadState("0", positions, positions, {"a": "x1", "b": "x2"}),
            LoopHeadState("0", positions, positions, {"a": "x2", "b": "x2"}),
        ]


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def test_empty_start_uses_intended_learner_and_skips_motion(self):
        task = BindingTask()
        with patch(
            "synthesis.experiment.invariant_learning.runner.InvInference",
            wraps=InvInference,
        ) as learner:
            result = run_experiment(task, ExperimentConfig())
        self.assertEqual(result.status, "verified_symbolic", result.reason)
        self.assertEqual(result.verification_attempts, 2)
        self.assertEqual(result.history[0]["n_states"], 0)
        self.assertEqual(result.counterexample_executions, 1)
        learner.assert_called_once()
        task.verify_motion.assert_not_called()

    def test_pyeda_selection_reaches_learner(self):
        task = BindingTask()
        with patch(
            "synthesis.experiment.invariant_learning.runner.InvInference",
            wraps=InvInference,
        ) as learner:
            result = run_experiment(task, ExperimentConfig(invariant_minimizer="pyeda"))
        self.assertEqual(result.status, "verified_symbolic", result.reason)
        self.assertEqual(learner.call_args.kwargs["minimizer"], "pyeda")

    def test_minimizer_cli_and_config_validation(self):
        from synthesis.entry.learn_invariant import build_parser

        parser = build_parser()
        self.assertEqual(
            parser.parse_args(["--program", "unused"]).invariant_minimizer, "sympy"
        )
        self.assertEqual(
            parser.parse_args(
                ["--program", "unused", "--invariant-minimizer", "pyeda"]
            ).invariant_minimizer,
            "pyeda",
        )
        with self.assertRaisesRegex(ValueError, "invariant_minimizer"):
            ExperimentConfig(invariant_minimizer="unknown")

    def test_motion_runs_after_symbolic_success(self):
        task = BindingTask()
        result = run_experiment(task, ExperimentConfig(verification_level="both"))
        self.assertEqual(result.status, "verified_model")
        task.verify_motion.assert_called_once()
        self.assertEqual(result.symbolic_status, "verified")

    def test_invalid_execution_is_not_training_data(self):
        task = BindingTask()
        task.valid = False
        with patch(
            "synthesis.experiment.invariant_learning.runner.InvInference"
        ) as learner:
            result = run_experiment(task, ExperimentConfig(verification_level="both"))
        self.assertEqual(result.status, "execution_failed")
        learner.assert_not_called()
        task.verify_motion.assert_not_called()

    def test_search_exhaustion_is_not_success(self):
        result = run_experiment(
            BindingTask(), ExperimentConfig(max_counterexample_blocks=1)
        )
        self.assertEqual(result.status, "no_reachable_counterexample")
        self.assertFalse(result)

    def test_motion_failure_retains_symbolic_success(self):
        class MotionFailure:
            checks = [
                SimpleNamespace(
                    obligation="clearance", status="invalid", reason="collision"
                )
            ]

            def __bool__(self):
                return False

        task = BindingTask()
        task.verify_motion.return_value = MotionFailure()
        result = run_experiment(task, ExperimentConfig(verification_level="both"))
        self.assertEqual(result.status, "motion_failed")
        self.assertEqual(result.symbolic_status, "verified")
        self.assertEqual(result.motion_status, "failed")

    def test_repeated_covered_execution_does_not_trigger_another_update(self):
        task = BindingTask()
        failure = task.verify_symbolic(1000)
        search = find_witness(
            lambda size: task.search_query(
                size, 1000, [c for c in failure.checks if c.status == "invalid"]
            ),
            2,
        )
        task.verify_symbolic = Mock(return_value=failure)
        with patch(
            "synthesis.experiment.invariant_learning.runner.find_witness",
            return_value=search,
        ):
            result = run_experiment(task, ExperimentConfig())
        self.assertEqual(result.status, "no_progress")
        self.assertEqual(result.learner_updates, 1)

    def test_nonmonotone_candidate_is_rejected_even_if_it_covers_data(self):
        task = BindingTask()
        rows = task.learning_states(None)
        task.learning_states = Mock(side_effect=[[rows[0]], [rows[1]]])
        a, b = task.context.get_consts("a"), task.context.get_consts("b")
        with patch(
            "synthesis.experiment.invariant_learning.runner.InvInference",
            side_effect=[(a != b, None), (task.context.Higher(a, b), None)],
        ):
            result = run_experiment(task, ExperimentConfig())
        self.assertEqual(result.status, "nonmonotone")
        self.assertEqual(result.learner_updates, 1)

    def test_unknown_proof_never_generates_an_execution(self):
        task = BindingTask()
        proof = task.verify_symbolic(1000)
        proof.checks[0].status = "unknown"
        task.verify_symbolic = Mock(return_value=proof)
        result = run_experiment(task, ExperimentConfig())
        self.assertEqual(result.status, "unknown")
        self.assertEqual(task.executions, 0)

    def test_exit_failure_does_not_add_more_positive_states(self):
        task = BindingTask()
        proof = task.verify_symbolic(1000)
        proof.checks[0].vc = type(proof.checks[0].vc)("exit", "0", z3.BoolVal(False))
        task.verify_symbolic = Mock(return_value=proof)
        result = run_experiment(task, ExperimentConfig())
        self.assertEqual(result.status, "needs_stronger_invariant")
        self.assertEqual(task.executions, 0)

    def test_final_verification_attempt_occurs_after_last_allowed_update(self):
        result = run_experiment(BindingTask(), ExperimentConfig(max_rounds=1))
        self.assertEqual(result.status, "verified_symbolic")
        self.assertEqual(result.verification_attempts, 2)

    def test_learner_must_cover_every_accumulated_state(self):
        with patch(
            "synthesis.experiment.invariant_learning.runner.InvInference",
            return_value=(z3.BoolVal(False), None),
        ):
            result = run_experiment(BindingTask(), ExperimentConfig())
        self.assertEqual(result.status, "learning_failed")


if __name__ == "__main__":
    unittest.main()
