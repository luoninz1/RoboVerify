import contextlib
import io
import itertools
import unittest
from unittest.mock import Mock

import z3

from synthesis.api.instructions import Assign, Get, Skip, While
from synthesis.api.program import Program
from synthesis.experiment.invariant_learning.tasks import StackExperiment
from synthesis.experiment.invariant_learning.witness import (
    ReachPath,
    WitnessQuery,
    failure_paths,
    find_witness,
    finite_formula,
)
from synthesis.inference_lib.demo_store import DemoStore, InvInference, LoopHeadState
from synthesis.verification_lib.counterexamples import stacks_to_positions
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext
from synthesis.verification_lib.symbolic_verify import (
    discharge_vc,
)


class WitnessTests(unittest.TestCase):
    ids = itertools.count()

    def context(self, n):
        return HighLevelContext(
            mode="enum", num_blocks=n, sort_name=f"WitnessTest{next(self.ids)}"
        )

    def test_finite_expansion_preserves_nested_shadowed_quantifiers(self):
        ctx = self.context(2)
        x, y = ctx.get_consts("x"), ctx.get_consts("y")
        formula = z3.ForAll(
            [x],
            z3.Exists(
                [y],
                z3.And(
                    x == y,
                    z3.ForAll([x], z3.Or(x == y, ctx.Higher(x, y))),
                ),
            ),
        )
        expanded = finite_formula(formula, ctx)
        solver = z3.Solver()
        solver.add(z3.Xor(formula, expanded))
        self.assertEqual(solver.check(), z3.unsat)

    def test_symbolic_choices_allow_higher_id_and_propagate_assignment(self):
        ctx = self.context(3)
        a, choice, base = [ctx.get_consts(s) for s in ("a", "choice", "base")]
        invariant = a != ctx.enum_blocks[2]
        loop = While(choice != a, [choice], [Assign("a", "choice")], invariant)
        program = Program(2, [Assign("a", "base"), loop])
        failures = [
            discharge_vc(vc, ctx, 1000)
            for vc in program.VC_gen(z3.BoolVal(True), z3.BoolVal(True), ctx)
            if vc.kind == "preserve"
        ]
        paths = failure_paths(program, ctx, [c.vc for c in failures], 2)
        self.assertEqual(len(paths), 2)
        # Require one preceding iteration: 0 -> 1 -> 2. The second choice must
        # be 2; default lowest-ID selection at b=1 would instead choose 0.
        path = paths[1]
        solver = ctx.new_solver(1000)
        solver.add(
            base == ctx.enum_blocks[0],
            path.condition,
            path.choices[0].values[0] == ctx.enum_blocks[1],
        )
        self.assertEqual(solver.check(), z3.sat)
        plan = path.decode(solver.model(), ctx)
        self.assertEqual([c["bindings"]["choice"] for c in plan.guard_choices], [1, 2])
        solver.add(path.choices[1].values[0] == ctx.enum_blocks[1])
        self.assertEqual(solver.check(), z3.unsat)  # guard sees updated a=1

    def test_zero_iteration_establishment_includes_guard_false_head(self):
        ctx = self.context(1)
        loop = While(z3.BoolVal(False), [], [Skip()], z3.BoolVal(False))
        program = Program(1, [loop])
        failures = [
            discharge_vc(vc, ctx, 1000)
            for vc in program.VC_gen(z3.BoolVal(True), z3.BoolVal(True), ctx)
            if vc.kind == "establish"
        ]
        paths = failure_paths(program, ctx, [c.vc for c in failures], 0)
        solver = ctx.new_solver(1000)
        solver.add(paths[0].condition)
        self.assertEqual(solver.check(), z3.sat)
        self.assertEqual(paths[0].choices, [])

    def test_prefix_get_is_existential_and_not_lowest_id(self):
        ctx = self.context(3)
        a = ctx.get_consts("a")
        program = Program(
            2,
            [
                Get(a, z3.BoolVal(True)),
                While(z3.BoolVal(False), [], [Skip()], a != ctx.enum_blocks[2]),
            ],
        )
        failures = [
            discharge_vc(vc, ctx, 1000)
            for vc in program.VC_gen(z3.BoolVal(True), z3.BoolVal(True), ctx)
            if vc.kind == "establish"
        ]
        path = failure_paths(program, ctx, [c.vc for c in failures], 0)[0]
        solver = ctx.new_solver(1000)
        solver.add(path.condition)
        self.assertEqual(solver.check(), z3.sat)
        self.assertEqual(
            path.decode(solver.model(), ctx).guard_choices,
            [{"kind": "Get", "bindings": {"a": 2}}],
        )

    def test_unreachable_preservation_is_not_replaced_by_generic_missing_state(self):
        ctx = self.context(3)
        a, zero, one, two = [
            ctx.get_consts(name) for name in ("a", "zero", "one", "two")
        ]
        program = Program(
            2,
            [
                Assign("a", "zero"),
                While(z3.BoolVal(True), [], [Assign("a", "one")], a == two),
            ],
        )
        vc = next(
            vc
            for vc in program.VC_gen(z3.BoolVal(True), z3.BoolVal(True), ctx)
            if vc.kind == "preserve"
        )
        paths = failure_paths(program, ctx, [vc], 3)
        solver = ctx.new_solver(1000)
        solver.add(
            zero == ctx.enum_blocks[0],
            one == ctx.enum_blocks[1],
            two == ctx.enum_blocks[2],
        )
        # Preservation fails abstractly at a=two, but execution only visits zero
        # then one. Both are missing invariant states, neither reproduces this VC.
        solver.push()
        solver.add(z3.Not(vc.expr))
        self.assertEqual(solver.check(), z3.sat)
        solver.pop()
        solver.add(z3.Or(*[path.condition for path in paths]))
        self.assertEqual(solver.check(), z3.unsat)

    def test_unknown_reachability_stops_before_larger_size(self):
        solver = Mock()
        solver.check.side_effect = [z3.sat, z3.unknown]
        solver.reason_unknown.return_value = "timeout"
        build = Mock(return_value=WitnessQuery(solver, z3.BoolVal(True), None, 0))
        result = find_witness(build, 4)
        self.assertEqual(result.status, "unknown")
        build.assert_called_once_with(1)

    def test_search_minimizes_reachable_failures_directly(self):
        def build(n):
            solver = z3.Solver()
            path = ReachPath(z3.BoolVal(n == 3), [], "establish", "0", 0)
            return WitnessQuery(
                solver, path.condition, lambda model: n, 0, self.context(n), [path]
            )

        result = find_witness(build, 4)
        self.assertEqual(result.size, 3)
        self.assertEqual(
            [r["status"] for r in result.attempts], ["unsat", "unsat", "sat"]
        )

    def test_two_block_invariant_requires_reachable_three_block_head(self):
        task = StackExperiment("synthesis.examples.stack:build_program")
        import numpy as np

        task._geometry = (np.array([0.0, 0.0]), np.array([0.6, 0.0, 0.5]), 0.425)
        store = DemoStore()
        for size in (1, 2):
            blocks = [f"x{i+1}" for i in range(size)]
            entry = stacks_to_positions([[b] for b in blocks])
            for top in range(size):
                store.add(
                    LoopHeadState(
                        task.loop_id,
                        stacks_to_positions(
                            [blocks[: top + 1]] + [[b] for b in blocks[top + 1 :]]
                        ),
                        entry,
                        {"b0": blocks[0], "b": blocks[top]},
                    )
                )
        with contextlib.redirect_stdout(io.StringIO()):
            invariant, _ = InvInference(
                store, task.loop_id, task.vocabulary, task.context
            )
        task.set_invariant(invariant)
        failures = [
            c for c in task.verify_symbolic(10000).checks if c.status == "invalid"
        ]
        result = find_witness(lambda size: task.search_query(size, 10000, failures), 3)
        self.assertEqual(result.status, "found", result.reason)
        self.assertEqual(result.size, 3)
        self.assertEqual(
            [r["status"] for r in result.attempts],
            ["unsat", "unsat", "sat"],
        )
        self.assertEqual(len(result.witness.positions), 3)
        self.assertIn(result.plan.failure_kind, ("establish", "preserve"))


if __name__ == "__main__":
    unittest.main()
