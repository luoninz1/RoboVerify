"""Fresh update witnesses must not change the meaning of existing variables."""

import unittest

import z3

from synthesis.api.instructions import MarkGoal, MoveDown, MoveRight, Put
from synthesis.api.program import wp
from synthesis.inference_lib.quant_enum_merge import (
    python_expr_to_z3,
    rebuild_exists_forall_right,
)
from synthesis.predicates.term import atom, exists, free_names, open_existentials, ref
from synthesis.util.symbols import fresh_const
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext


class QuantifierHygieneTests(unittest.TestCase):
    def equivalent(self, left, right):
        solver = z3.Solver()
        solver.set(timeout=5000)
        solver.add(left != right)
        self.assertEqual(solver.check(), z3.unsat)

    def test_table_higher_keeps_a_program_block_named_t_free(self):
        names = ["a", "b", "t", "tbl"]
        high = HighLevelContext(
            mode="enum",
            enum_names=["table_" + n for n in names],
            sort_name="TableCapture",
            use_tbl=True,
        )
        objects = {n: high.get_consts(n) for n in names}
        heights = {"a": 0, "b": 0, "t": 1}
        solver = z3.Solver()
        for name, value in zip(names, high.enum_blocks):
            solver.add(objects[name] == value)
        for left in names:
            for right in names:
                relation = (
                    left == right
                    if "tbl" in (left, right)
                    else heights[left] >= heights[right]
                )
                solver.add(high.Higher(objects[left], objects[right]) == relation)
        self.assertEqual(solver.check(), z3.sat)
        for other, expected in (("t", False), ("b", True), ("tbl", False)):
            with self.subTest(other=other):
                solver.push()
                solver.add(
                    wp(Put("a", "tbl"), high.Higher(objects["a"], objects[other]), high)
                    != expected
                )
                self.assertEqual(solver.check(), z3.unsat)
                solver.pop()

    def test_put_rewrites_preserve_shadowing_free_operands_and_other_atoms(self):
        high = HighLevelContext(use_tbl=True)
        a, b, c, u, v = [high.get_consts(n) for n in ("a", "b", "c", "u", "v")]
        for relation in (high.Higher, high.ON_star, high.Scattered):
            # The source's program name a is also used as two shadowed binders.
            original = z3.ForAll(
                [a], z3.And(relation(a, b), a != c, z3.Exists([a], relation(c, a)))
            )
            renamed = z3.ForAll(
                [u], z3.And(relation(u, b), u != c, z3.Exists([v], relation(c, v)))
            )
            for target in ("b", "tbl"):
                with self.subTest(relation=relation, target=target):
                    self.equivalent(
                        wp(Put("a", target), original, high),
                        wp(Put("a", target), renamed, high),
                    )

    def test_mark_goal_keeps_update_target_free_beneath_same_named_binder(self):
        high = HighLevelContext(verification_mode="goals")
        a, u = [high.get_goal_consts(n) for n in ("a", "u")]
        self.equivalent(
            wp(MarkGoal("a"), z3.ForAll([a], high.Mark(a)), high),
            wp(MarkGoal("a"), z3.ForAll([u], high.Mark(u)), high),
        )

    def test_goal_successor_helpers_do_not_capture_their_operands(self):
        high = HighLevelContext(verification_mode="goals")
        a, other = [high.get_goal_consts(n) for n in ("a", "other")]
        for helper, old_name in ((high.f_, "t_f"), (high.f_tot, "t_ft")):
            named = high.get_goal_consts(old_name)
            self.equivalent(
                helper(high.d_star, named, a),
                z3.substitute(helper(high.d_star, other, a), (other, named)),
            )
            self.equivalent(
                helper(high.d_star, a, named),
                z3.substitute(helper(high.d_star, a, other), (other, named)),
            )

    def test_goal_moves_do_not_capture_free_postcondition_names(self):
        high = HighLevelContext(verification_mode="goals")
        current = high.get_goal_consts("a")
        for action, old_name, relation in (
            (MoveRight, "z_r_a", high.rtot),
            (MoveDown, "z_d_a", high.dtot),
        ):
            fixed = high.get_goal_consts(old_name)
            post = z3.And(current != fixed, high.Mark(fixed))
            witness = fresh_const(high.GoalSort, "expected")
            expected = z3.And(
                current != high.null,
                z3.ForAll(
                    [witness],
                    z3.Implies(
                        relation(current, witness),
                        z3.And(witness != fixed, high.Mark(fixed)),
                    ),
                ),
            )
            self.equivalent(wp(action("a"), post, high), expected)

    def test_inference_ast_does_not_capture_free_ux1_or_ex1(self):
        sort = z3.DeclareSort("InferenceCapture")
        for kind, name in (("ForAll", "ux1"), ("Exists", "ex1")):
            fixed = z3.Const(name, sort)
            ast = {
                "op": kind,
                "vars": ["x"],
                "body": {
                    "op": "Eq",
                    "args": [
                        {"op": "Var", "name": "Var(0)"},
                        {"op": "Var", "name": name},
                    ],
                },
            }
            actual, _, _ = python_expr_to_z3(ast, {name: fixed}, {}, sort)
            v = fresh_const(sort, "expected")
            self.equivalent(
                actual, (z3.ForAll if kind == "ForAll" else z3.Exists)([v], v == fixed)
            )

    def test_promotion_keeps_outer_variables_and_free_generated_style_names(self):
        sort = z3.DeclareSort("PromotionCapture")
        x, y, z, ux1, ex1 = z3.Consts("x y z ux1 ex1", sort)
        relation = z3.Function(
            "promotion_relation", sort, sort, sort, sort, sort, z3.BoolSort()
        )
        original = z3.ForAll([x], z3.Exists([y, z], relation(x, y, z, ux1, ex1)))
        actual = rebuild_exists_forall_right(original, 1)
        expected = z3.ForAll(
            [x], z3.Exists([y], z3.ForAll([z], relation(x, y, z, ux1, ex1)))
        )
        self.equivalent(actual, expected)

    def test_opened_classifier_binders_avoid_free_and_in_scope_names(self):
        term = exists(["x"], atom("Higher", ref("x"), ref("g_0")))
        names, opened = open_existentials(term, "g", occupied={"g_0_1"})
        self.assertEqual(names, ("g_0_2",))
        self.assertEqual(free_names(opened), {"g_0", "g_0_2"})
