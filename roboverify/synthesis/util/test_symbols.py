import unittest
from unittest.mock import patch

import z3

from synthesis.util.symbols import (
    fresh_const,
    fresh_name,
    open_quantifier,
    rewrite_quantifier,
)


class SymbolTests(unittest.TestCase):
    def test_allocations_are_distinct_and_respect_sort_and_context(self):
        context = z3.Context()
        sort = z3.DeclareSort("Object", context)
        values = [fresh_const(sort, "same") for _ in range(20)]
        self.assertEqual(len({v.get_id() for v in values}), 20)
        self.assertTrue(all(v.sort() == sort for v in values))

    def test_avoids_existing_free_and_bound_names_even_if_z3_proposes_them(self):
        x, y, result = z3.Ints("aux!0 aux!1 unused")
        formula = z3.And(x > 0, z3.Exists([y], y > x))
        with patch("synthesis.util.symbols.z3.FreshConst", side_effect=[x, y, result]):
            allocated = fresh_const(z3.IntSort(), "aux", avoid=(formula,))
        self.assertTrue(allocated.eq(result))

    def test_opening_preserves_sorts_order_and_nested_shadowing(self):
        x, y = z3.Ints("x y")
        flag = z3.Bool("flag")
        formula = z3.ForAll([x, flag], z3.And(flag == (x > y), z3.Exists([x], x == y)))
        variables, body = open_quantifier(formula)
        self.assertEqual([v.sort() for v in variables], [z3.IntSort(), z3.BoolSort()])
        solver = z3.Solver()
        solver.add(z3.ForAll(variables, body) != formula)
        self.assertEqual(solver.check(), z3.unsat)

    def test_rewriting_does_not_capture_a_free_replacement(self):
        x, y = z3.Ints("x y")
        p = z3.Function("P", z3.IntSort(), z3.BoolSort())
        formula = z3.ForAll([x], p(x))
        rewritten = rewrite_quantifier(
            formula, lambda body: z3.And(body, x == y), avoid=(x, y)
        )
        solver = z3.Solver()
        solver.add(rewritten != z3.And(formula, x == y))
        self.assertEqual(solver.check(), z3.unsat)

    def test_program_names_are_reserved_without_reusing_scope_names(self):
        occupied = {"g", "g_1", "g_2"}
        self.assertEqual(fresh_name("g", occupied), "g_3")
        self.assertEqual(fresh_name("g", occupied), "g_4")
