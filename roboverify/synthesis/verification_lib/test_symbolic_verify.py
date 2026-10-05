import unittest
from unittest.mock import Mock

import z3

from synthesis.api.instructions import Assign, Skip, While
from synthesis.api.program import Program
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext
from synthesis.verification_lib.symbolic_verify import VC, discharge_vc


class SymbolicVerdictTests(unittest.TestCase):
    def setUp(self):
        self.context = HighLevelContext()

    def test_false_invariant_has_invalid_establishment_and_vacuous_loop_vcs(self):
        loop = While(z3.BoolVal(True), [], [Skip()], z3.BoolVal(False))
        vcs = Program(1, [loop]).VC_gen(
            z3.BoolVal(True), z3.BoolVal(True), self.context
        )
        self.assertEqual([v.kind for v in vcs], ["establish", "preserve", "exit"])
        self.assertEqual(
            [discharge_vc(v, self.context).status for v in vcs],
            ["invalid", "vacuous", "vacuous"],
        )
        self.assertEqual(discharge_vc(vcs[1], self.context).queries, 1)

    def test_semantic_vacuity_and_nonvacuous_validity(self):
        a = self.context.get_consts("a")
        reflexive = self.context.ON_star(a, a)
        self.assertEqual(
            discharge_vc(
                VC("body", None, z3.Implies(z3.Not(reflexive), False)), self.context
            ).status,
            "vacuous",
        )
        verdict = discharge_vc(
            VC("body", None, z3.Implies(True, reflexive)), self.context
        )
        self.assertEqual(verdict.status, "valid")
        self.assertEqual(verdict.queries, 2)

    def test_unknown_is_not_a_counterexample(self):
        context = Mock()
        context.new_solver.return_value = Mock(spec=z3.Solver())
        context.new_solver.return_value.check.return_value = z3.unknown
        context.new_solver.return_value.reason_unknown.return_value = "timeout"
        verdict = discharge_vc(VC("body", None, z3.BoolVal(True)), context)
        self.assertEqual(verdict.status, "unknown")
        self.assertIsNone(verdict.model)
        self.assertEqual(verdict.reason, "timeout")

    def test_nonminimal_core_falls_back_to_premise_check(self):
        context = Mock()
        context.new_solver.return_value = Mock(spec=z3.Solver())
        solver = context.new_solver.return_value
        solver.check.side_effect = [z3.unsat, z3.unsat]
        solver.unsat_core.side_effect = lambda: [
            solver.assert_and_track.call_args.args[1]
        ]
        verdict = discharge_vc(VC("preserve", "0", z3.Implies(False, False)), context)
        self.assertEqual(verdict.status, "vacuous")
        self.assertEqual(verdict.queries, 2)

    def test_assignment_establishment_and_nested_loop_ids(self):
        a, b = z3.Consts("a b", self.context.BoxSort)
        inner = While(a != b, [], [Skip()], z3.BoolVal(True))
        outer = While(a != b, [a], [inner], a == b)
        vcs = Program(2, [Assign("a", "b"), outer]).VC_gen(
            z3.BoolVal(True), z3.BoolVal(True), self.context
        )
        self.assertEqual([v.loop_id for v in vcs], ["1", "1.0", "1.0", "1", "1"])
        self.assertEqual(discharge_vc(vcs[0], self.context).status, "valid")
        # Exit has no free chosen witness: it negates Exists(a, a != b).
        self.assertTrue(z3.is_quantifier(vcs[-1].expr.arg(0).arg(0).arg(0)))


if __name__ == "__main__":
    unittest.main()


class CounterexampleTests(unittest.TestCase):
    counter = 0

    def context(self, count, use_tbl=False):
        CounterexampleTests.counter += 1
        return HighLevelContext(
            mode="enum",
            num_blocks=count,
            sort_name=f"Scene{self.counter}",
            use_tbl=use_tbl,
        )

    def test_positions_and_aliases_preserve_current_and_entry_geometry(self):
        from synthesis.verification_lib.counterexamples import (
            model_to_loop_head,
            state_holds,
        )

        ctx = self.context(3, True)
        a, b, table = ctx.enum_blocks
        solver = ctx.new_solver(1000)
        solver.add(
            ctx.get_consts("tbl") == table,
            ctx.get_consts("b") == b,
            ctx.get_consts("b0") == b,
            ctx.get_consts("b_prime") == a,
        )
        for x in (a, b):
            for y in (a, b):
                same = x.eq(y)
                solver.add(ctx.ON_star(x, y) == (same or (x.eq(a) and y.eq(b))))
                solver.add(ctx.ON_star_zero(x, y) == same)
                solver.add(ctx.Higher(x, y) == (same or x.eq(a)))
                solver.add(ctx.Scattered(x, y) == False)
        self.assertEqual(solver.check(), z3.sat)
        row = model_to_loop_head(ctx, solver.model(), "1", ("b", "b0", "b_prime"))
        self.assertEqual(row.constants["b"], row.constants["b0"])
        source, target = ctx.get_consts("b_prime"), ctx.get_consts("b")
        self.assertTrue(state_holds(ctx.ON_star(source, target), row))
        self.assertFalse(state_holds(ctx.ON_star_zero(source, target), row))
        self.assertIn("tbl", row.positions)

    def test_unrealizable_higher_model_is_not_added_as_a_scene(self):
        from synthesis.verification_lib.counterexamples import (
            UnrealizableCounterexample,
            model_to_loop_head,
        )

        ctx = self.context(2)
        a, b = ctx.enum_blocks
        solver = ctx.new_solver(1000)
        solver.add(z3.Not(ctx.Higher(a, b)), z3.Not(ctx.Higher(b, a)))
        self.assertEqual(solver.check(), z3.sat)
        with self.assertRaises(UnrealizableCounterexample):
            model_to_loop_head(ctx, solver.model(), "1", ())

    def test_smallest_counterexample_and_bounded_scope(self):
        from synthesis.verification_lib.symbolic_verify import symbolic_verify

        seen = []

        def build(size):
            seen.append(size)
            ctx = self.context(size)
            a, b, c = z3.Consts("a b c", ctx.BoxSort)
            post = z3.Exists([a, b], z3.ForAll([c], z3.Or(c == a, c == b)))
            return Program(1, [Skip()]), z3.BoolVal(True), post, ctx

        result = symbolic_verify(build, max_blocks=4, prove_unbounded=False)
        self.assertEqual(result.num_blocks, 3)
        self.assertEqual(seen, [2, 3])
        self.assertEqual(result.failed_vc_kind, "body")
        bounded = symbolic_verify(build, max_blocks=2, prove_unbounded=False)
        self.assertTrue(bounded)
        self.assertEqual(bounded.scope, "finite:2..2")
