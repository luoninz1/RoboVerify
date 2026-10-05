"""Check exact truth-table semantics and the normal forms consumed by inference."""

import itertools
import random
import unittest
from unittest.mock import patch

import sympy
import z3
from sympy.logic.boolalg import is_cnf, is_dnf
from synthesis.inference_lib.inference import (
    construct_truth_table_and_extract_expression_for_phi as make_phi,
)
from synthesis.inference_lib.inference import (
    construct_truth_table_and_extract_expression_for_phi_prime as make_phi_prime,
)
from synthesis.inference_lib.inference import sympy_to_z3
from synthesis.inference_lib.minimization import (
    INVARIANT_MINIMIZERS,
    get_invariant_minimizer,
    using_invariant_minimizer,
)


class MinimizationTests(unittest.TestCase):
    def check_table(self, size, positive, negative):
        universe = set(itertools.product((0, 1), repeat=size))
        terms = [z3.Bool(f"p{i}") for i in range(size)]
        for backend, (build, accepted, normal_form) in itertools.product(
            INVARIANT_MINIMIZERS,
            (
                (make_phi, universe - negative, is_dnf),
                (make_phi_prime, positive, is_cnf),
            ),
        ):
            with self.subTest(
                size=size, positive=positive, backend=backend, form=build.__name__
            ):
                with using_invariant_minimizer(backend):
                    expression, symbols = build(positive, negative, size)
                self.assertEqual(symbols, sympy.symbols(f"term0:{size}"))
                self.assertTrue(normal_form(expression), expression)
                expected = z3.Or(
                    *(
                        z3.And(
                            *(
                                term if bit else z3.Not(term)
                                for term, bit in zip(terms, row)
                            )
                        )
                        for row in accepted
                    )
                )
                solver = z3.Solver()
                solver.add(sympy_to_z3(expression, terms) != expected)
                self.assertEqual(solver.check(), z3.unsat, str(expression))

    def test_every_boolean_function_up_to_three_variables(self):
        # Includes empty/full on-sets, zero variables, single literals, XOR,
        # irrelevant variables, and asymmetric assignments (variable ordering).
        for size in range(4):
            rows = list(itertools.product((0, 1), repeat=size))
            for mask in range(1 << len(rows)):
                positive = {row for i, row in enumerate(rows) if mask & (1 << i)}
                self.check_table(size, positive, set(rows) - positive)

    def test_unobserved_rows_keep_phi_and_phi_prime_completion_policies(self):
        self.check_table(3, {(0, 1, 0)}, {(1, 0, 1)})
        self.check_table(2, set(), set())

    def test_larger_tables_and_sympy_cnf_fallback_remain_equivalent(self):
        rng = random.Random(42)
        for size in (4, 5, 6, 7):
            rows = list(itertools.product((0, 1), repeat=size))
            positive = set(rng.sample(rows, 10))
            self.check_table(size, positive, set(rows) - positive)

    def test_pyeda_minimizes_above_sympy_threshold_without_sympy_minimization(self):
        positive = {row for row in itertools.product((0, 1), repeat=6) if row[4]}
        with using_invariant_minimizer("pyeda"), patch.object(
            sympy, "SOPform", side_effect=AssertionError("SymPy SOPform called")
        ), patch.object(
            sympy, "POSform", side_effect=AssertionError("SymPy POSform called")
        ):
            expression, symbols = make_phi_prime(positive, set(), 6)
            self.assertEqual(expression, symbols[4])
            expression, symbols = make_phi(set(), positive, 6)
            self.assertEqual(expression, ~symbols[4])

    def test_selection_is_scoped_and_validated(self):
        self.assertEqual(get_invariant_minimizer(), "sympy")
        with using_invariant_minimizer("pyeda"):
            with using_invariant_minimizer():
                self.assertEqual(get_invariant_minimizer(), "pyeda")
            with self.assertRaisesRegex(
                RuntimeError, "test"
            ), using_invariant_minimizer("sympy"):
                self.assertEqual(get_invariant_minimizer(), "sympy")
                raise RuntimeError("test")
            self.assertEqual(get_invariant_minimizer(), "pyeda")
            with self.assertRaisesRegex(ValueError, "Unknown invariant minimizer"):
                with using_invariant_minimizer("unknown"):
                    self.fail("Invalid selection accepted")
            self.assertEqual(get_invariant_minimizer(), "pyeda")
        self.assertEqual(get_invariant_minimizer(), "sympy")


if __name__ == "__main__":
    unittest.main()
