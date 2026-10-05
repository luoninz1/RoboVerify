"""Predicate compilation must preserve binding and mutable-state semantics."""

import unittest
from copy import deepcopy

import z3

from synthesis.inference_lib.demo_store import LoopHeadState
from synthesis.verification_lib.counterexamples import state_holds
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext


class StatePredicateTests(unittest.TestCase):
    def setUp(self):
        self.context = HighLevelContext()
        self.a, self.x, self.y = z3.Consts("a x y", self.context.BoxSort)
        positions = {
            "p": [0.0, 0.0, 0.425],
            "q": [0.0, 0.0, 0.475],
            "r": [0.2, 0.0, 0.425],
        }
        self.state = LoopHeadState("loop", positions, deepcopy(positions), {"a": "p"})

    def test_cached_formula_observes_alias_and_geometry_changes(self):
        formula = z3.ForAll([self.x], self.context.Higher(self.a, self.x))
        self.assertFalse(state_holds(formula, self.state))
        self.state.constants["a"] = "q"
        self.assertTrue(state_holds(formula, self.state))
        self.state.positions["r"][2] = 0.525
        self.assertFalse(state_holds(formula, self.state))

    def test_current_and_frozen_geometry_remain_independent(self):
        self.state.constants["a"] = "q"
        formula = z3.Exists(
            [self.x],
            z3.And(
                self.x != self.a,
                self.context.ON_star(self.a, self.x),
                self.context.ON_star_zero(self.a, self.x),
            ),
        )
        self.assertTrue(state_holds(formula, self.state))
        self.state.positions["q"] = [0.4, 0.0, 0.425]
        self.assertFalse(state_holds(formula, self.state))
        frozen = z3.Exists(
            [self.x],
            z3.And(self.x != self.a, self.context.ON_star_zero(self.a, self.x)),
        )
        self.assertTrue(state_holds(frozen, self.state))

    def test_quantifier_order_and_shadowed_binders(self):
        higher = self.context.Higher
        self.assertTrue(
            state_holds(
                z3.ForAll([self.x], z3.Exists([self.y], higher(self.y, self.x))),
                self.state,
            )
        )
        self.assertFalse(
            state_holds(
                z3.ForAll(
                    [self.x],
                    z3.Exists(
                        [self.y],
                        z3.And(higher(self.y, self.x), z3.Not(higher(self.x, self.y))),
                    ),
                ),
                self.state,
            )
        )
        self.assertTrue(
            state_holds(
                z3.Exists(
                    [self.x],
                    z3.And(
                        self.x == self.a,
                        z3.ForAll([self.x], z3.Exists([self.y], self.x == self.y)),
                    ),
                ),
                self.state,
            )
        )

    def test_distinct_boolean_structure_and_unsupported_predicate(self):
        formula = z3.ForAll(
            [self.x, self.y],
            z3.Implies(z3.Distinct(self.x, self.y), z3.Not(self.x == self.y)),
        )
        self.assertTrue(state_holds(formula, self.state))
        unknown = z3.Function("unknown", self.context.BoxSort, z3.BoolSort())
        with self.assertRaisesRegex(ValueError, "Unsupported predicate"):
            state_holds(unknown(self.a), self.state)


if __name__ == "__main__":
    unittest.main()
