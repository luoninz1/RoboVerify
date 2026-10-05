import contextlib
import io
import itertools
import unittest
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import z3

from synthesis.api.instructions import Assign, Skip, While
from synthesis.api.program import Program
from synthesis.inference_lib.demo_store import (
    DemoStore,
    InferenceVocabulary,
    InvInference,
    LoopHeadState,
)
from synthesis.verification_lib.cegis import (
    MotionBlockSpec,
    MotionPenalty,
    NeedsResynthesis,
    PenStore,
    optimize_motion_parameters,
    run_motion_cegis,
    run_symbolic_cegis,
)
from synthesis.verification_lib.counterexamples import state_holds
from synthesis.verification_lib.highlevel_verification_lib import (
    HighLevelContext,
    InvariantSpec,
)
from synthesis.verification_lib.motion_verification import (
    MotionContract,
    verify_motion_block,
)
from synthesis.verification_lib.test_motion_verification import stack_body


class SymbolicCEGISTests(unittest.TestCase):
    ids = itertools.count()

    def setUp(self):
        self.ctx = HighLevelContext()
        positions = {"x1": [0.0, 0.0, 0.425], "x2": [0.15, 0.0, 0.425]}
        self.row = LoopHeadState("0", positions, positions, {"a": "x1", "b": "x2"})
        self.vocab = InferenceVocabulary(0, ("equality",), ("a", "b"))

    def build(self, invariant, size):
        ctx = HighLevelContext(
            mode="enum" if size else "declare",
            num_blocks=size,
            sort_name=f"CEGIS{next(self.ids)}" if size else "Box",
        )
        inv = ctx.spec_to_expr(InvariantSpec({"sexpr": invariant.sexpr()}), ["a", "b"])
        a, b = ctx.get_consts("a"), ctx.get_consts("b")
        loop = While(a != b, [], [Assign("a", "b")], inv)
        return Program(1, [loop]), a != b, a == b, ctx

    def test_false_to_inductive_invariant_strictly_enlarges_and_terminates(self):
        store = DemoStore([self.row])
        with patch(
            "synthesis.verification_lib.cegis.InvInference", wraps=InvInference
        ) as infer:
            result = run_symbolic_cegis(
                store, "0", self.vocab, self.ctx, self.build, max_blocks=2
            )
        # Bootstrap and counterexample refinement must both use the intended algorithm.
        self.assertEqual(infer.call_count, 2)
        self.assertTrue(result, (result.status, result.reason))
        self.assertEqual(result.iterations, 2)
        self.assertEqual(len(store), 2)
        self.assertTrue(all(c.status == "valid" for c in result.verification.checks))
        self.assertEqual(result.verification.scope, "unbounded")
        self.assertTrue(all(step["enlarged"] for step in result.history[1:]))
        self.assertEqual(result.history[0]["invariant_sexpr"], "false")
        self.assertTrue(state_holds(result.invariant, store.for_loop("0")[-1]))

    def test_false_without_demonstrations_needs_resynthesis(self):
        with self.assertRaises(NeedsResynthesis) as error:
            run_symbolic_cegis(
                DemoStore(), "0", self.vocab, self.ctx, self.build, max_blocks=2
            )
        self.assertEqual(error.exception.result.failed_vc_kind, "establish")
        self.assertIsNotNone(error.exception.model)

    def test_budget_is_not_success(self):
        result = run_symbolic_cegis(
            DemoStore([self.row]),
            "0",
            self.vocab,
            self.ctx,
            self.build,
            max_blocks=2,
            max_iterations=1,
        )
        self.assertEqual(result.status, "budget_exhausted")
        self.assertFalse(result)

    def test_exit_failure_cannot_be_fixed_by_weakening(self):
        def build(inv, size):
            program, pre, _, ctx = self.build(inv, size)
            return program, pre, z3.BoolVal(False), ctx

        result = run_symbolic_cegis(
            DemoStore([self.row]), "0", self.vocab, self.ctx, build, max_blocks=2
        )
        self.assertEqual(result.status, "needs_stronger_invariant")
        self.assertFalse(result)


class MotionCEGISTests(unittest.TestCase):
    def setUp(self):
        ctx = HighLevelContext()
        a, b, b0, n = z3.Consts("a b b0 n", ctx.BoxSort)
        self.inv = z3.And(
            a != b,
            b0 == b,
            ctx.Scattered(a, b),
            ctx.Higher(a, b),
            ctx.Higher(b, a),
            z3.ForAll([n], z3.Or(n == a, n == b)),
        )
        self.program = Program(
            1, [While(z3.BoolVal(True), [], stack_body(0.1), self.inv)]
        )
        self.spec = MotionBlockSpec(
            "0",
            (self.inv, z3.BoolVal(True)),
            ("a", "b", "b0"),
            MotionContract("a", "b"),
            "Put(a,b)",
        )
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def test_real_counterexamples_drive_repair_and_full_recheck(self):
        pen = PenStore()
        calls = []

        def repair(program, penalty, iteration):
            calls.append(iteration)
            self.assertGreater(penalty(program), 0)
            program.instructions[0].body = stack_body(0.06)
            self.assertEqual(penalty(program), 0)
            return program

        result = run_motion_cegis(
            self.program, [self.spec], repair, pen=pen, max_iterations=2
        )
        self.assertTrue(result, (result.status, str(result.verification)))
        self.assertEqual(calls, [0])
        self.assertGreater(len(pen), 0)
        self.assertTrue(all(c.status == "valid" for c in result.verification.checks))

    def test_relational_summary_change_is_rejected(self):
        def invalid(program, penalty, iteration):
            program.instructions[0].body.append(Assign("b", "a"))
            return program

        with self.assertRaisesRegex(ValueError, "relational summary"):
            run_motion_cegis(self.program, [self.spec], invalid)

    def test_missing_coverage_and_stronger_precondition_rejected(self):
        with self.assertRaisesRegex(ValueError, "cover every loop"):
            run_motion_cegis(self.program, [], lambda *args: None)
        spec = MotionBlockSpec(
            "0",
            (z3.BoolVal(False),),
            self.spec.constants,
            self.spec.contract,
            "Put(a,b)",
        )
        with self.assertRaisesRegex(ValueError, "precondition"):
            run_motion_cegis(self.program, [spec], lambda *args: None)

    def test_penalty_deduplicates_environment_and_preserves_frozen_scene(self):
        scene = {"a": [0.3, 0, 0], "b": [0, 0, 0], "b0": [0, 0, 0], "sym": [2, 2, 0]}
        result = verify_motion_block(
            [],
            stack_body(0.1),
            self.spec.constants,
            self.spec.contract,
            block_v="0",
            initial_positions=scene,
        )
        pen = PenStore()
        example = result.counterexamples[-1]
        self.assertTrue(pen.add(example))
        self.assertFalse(pen.add(example))
        self.assertEqual(len(pen), 1)
        self.assertEqual(pen.for_block("0")[0].entry_positions, example.entry_positions)

    def test_unknown_never_enters_resynthesis(self):
        with patch("z3.Solver.check", return_value=z3.unknown):
            result = run_motion_cegis(
                self.program, [self.spec], lambda *a: self.fail("must not resynthesize")
            )
        self.assertEqual(result.status, "motion_inconclusive")

    def test_serial_cem_improves_a_penalty_objective(self):
        def penalty(program):
            return abs(
                float(
                    program.instructions[0]
                    .body[-1]
                    .target_offset[2]
                    .concrete_float("test")
                )
                - 0.06
            )

        result = optimize_motion_parameters(
            self.program, penalty, iterations=2, samples=8, elites=2, seed=7
        )
        self.assertLess(penalty(result), penalty(self.program))


if __name__ == "__main__":
    unittest.main()


class PenaltyObjectiveTests(unittest.TestCase):
    def test_both_runners_subtract_weighted_penalty_and_keep_default_score(self):
        from synthesis.experiment.mcmc.search import InstrumentedRunner
        from synthesis.mcmc.synthesis import Runner

        states = np.zeros((2, 60))
        program = Program(1, [Skip()])
        with patch(
            "synthesis.mcmc.synthesis.evaluate_program",
            return_value=(states, [states], []),
        ), patch(
            "synthesis.experiment.mcmc.search.rollout_demos",
            return_value=([states], states, [False], []),
        ), patch(
            "synthesis.mcmc.cost_func.maximum_mean_discrepancy_rbf", return_value=3.0
        ):
            for cls in (Runner, InstrumentedRunner):
                base = cls(program, states, 1, 2)
                penalized = cls(
                    program,
                    states,
                    1,
                    2,
                    motion_penalty=lambda p: 2,
                    motion_penalty_weight=4.0,
                )
                self.assertEqual(base([]), -3.0)
                self.assertEqual(penalized([]), -11.0)
                self.assertEqual(penalized.last_motion_penalty, 2)

    def test_serial_cem_accepts_unpicklable_solver_objective(self):
        from synthesis.mcmc.cem import cem_optimize

        term = z3.Real("serial_only")

        def objective(values):
            solver = z3.Solver()
            solver.add(term == float(values[0]))
            return -abs(float(values[0])) if solver.check() == z3.sat else -100.0

        with contextlib.redirect_stdout(io.StringIO()):
            score, _ = cem_optimize(
                objective, 1, iterations=1, N=2, K=1, init_mu=[0.0], num_workers=0
            )
        self.assertEqual(score, 0.0)
