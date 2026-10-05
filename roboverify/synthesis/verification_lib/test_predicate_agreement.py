"""The two readings of the block algebra have to agree.

An invariant is *learned* under the numeric predicates in ``synthesis/util/on.py``
and *checked* under the Z3 predicates in ``lowlevel_verification_lib``. When the
two drift apart nothing crashes: the invariant is simply discharged against a
relation other than the one it was learned from, and the verdict means nothing.
These tests pin them together, including at the table, which the axioms isolate
from all three relations.
"""

import random
import unittest

from z3 import (
    And,
    BoolVal,
    Consts,
    EnumSort,
    Exists,
    ForAll,
    Not,
    Or,
    RealVal,
    Solver,
    sat,
    unsat,
)
from z3.z3util import get_vars

import synthesis.util.on as on
import synthesis.verification_lib.highlevel_verification_lib as highlevel
import synthesis.verification_lib.lowlevel_verification_lib as lowlevel

# Random coordinates are compared by exact rational arithmetic on the Z3 side and
# by binary floating point on the Python side, so the two can disagree by ~1e-17
# on a configuration that sits exactly on a threshold. That is a property of the
# arithmetic and not of the definitions, so such samples are skipped here and the
# boundary convention is pinned separately by the tests that build it exactly.
TIE_MARGIN = 1e-9


def _pin(context, const, position):
    x, y, z = position
    return [
        context.X(const) == RealVal(repr(float(x))),
        context.Y(const) == RealVal(repr(float(y))),
        context.Z(const) == RealVal(repr(float(z))),
    ]


def _decide(formula, assumptions):
    """Whether *formula* is entailed by *assumptions*, refuted by them, or open."""
    solver = Solver()
    for assumption in assumptions:
        solver.add(assumption)

    solver.push()
    solver.add(Not(formula))
    negation = solver.check()
    solver.pop()
    if negation == unsat:
        return True

    solver.push()
    solver.add(formula)
    positive = solver.check()
    solver.pop()
    if positive == unsat:
        return False

    raise AssertionError(f"assumptions leave {formula} undecided")


def _on_a_threshold(first, second):
    """Whether this pair sits on a boundary of any of the four relations."""
    length = on.BLOCK_LENGTH
    x1, y1, z1 = first
    x2, y2, z2 = second
    candidates = [
        (abs(x1 - x2), length / 2),
        (abs(y1 - y2), length / 2),
        (z1 - z2, 0.0),
        (z1 - z2, -on.get_higher_tolerance()),
        (z1 - z2, 1.5 * length),
        (abs(x1 - x2), 2 * length),
        (abs(y1 - y2), 2 * length),
    ]
    return any(abs(value - threshold) < TIE_MARGIN for value, threshold in candidates)


def _sample_pairs(count, seed):
    """Pairs covering stacked, nearby and well-separated configurations."""
    rng = random.Random(seed)
    pairs = []
    while len(pairs) < count:
        base = [round(rng.uniform(-0.2, 0.2), 4) for _ in range(3)]
        shape = rng.choice(("stacked", "nearby", "scattered"))
        if shape == "stacked":
            other = [
                base[0] + round(rng.uniform(-0.03, 0.03), 4),
                base[1] + round(rng.uniform(-0.03, 0.03), 4),
                base[2] + round(rng.uniform(-0.2, 0.2), 4),
            ]
        elif shape == "nearby":
            other = [
                base[0] + round(rng.uniform(-0.12, 0.12), 4),
                base[1] + round(rng.uniform(-0.12, 0.12), 4),
                base[2] + round(rng.uniform(-0.12, 0.12), 4),
            ]
        else:
            other = [round(rng.uniform(-0.6, 0.6), 4) for _ in range(3)]
        if _on_a_threshold(base, other):
            continue
        pairs.append((base, other))
    return pairs


class RelationsAgree(unittest.TestCase):
    """Same verdict from the numeric and the Z3 reading, pair by pair."""

    def setUp(self):
        # No table: check physical block geometry independently of the marker.
        self.context = lowlevel.LowLevelContext(default_L=on.BLOCK_LENGTH)
        self.b1, self.b2 = Consts("agree_b1 agree_b2", self.context.BoxSort)

    def _assumptions(self, first, second):
        return (
            [self.context.L == RealVal(repr(on.BLOCK_LENGTH))]
            + _pin(self.context, self.b1, first)
            + _pin(self.context, self.b2, second)
        )

    def test_relations_agree_on_random_configurations(self):
        cases = [
            ("ON*", on.on_star_implementation, self.context.lowlevel_on_star),
            ("on", on.on, self.context.lowlevel_on_direct),
            ("Higher", on.higher_implementation, self.context.lowlevel_higher),
            ("Scattered", on.scattered_implementation, self.context.lowlevel_scattered),
        ]
        for first, second in _sample_pairs(120, seed=20260919):
            assumptions = self._assumptions(first, second)
            for name, numeric, symbolic in cases:
                with self.subTest(relation=name, first=first, second=second):
                    self.assertEqual(
                        numeric(first, second),
                        _decide(symbolic(self.b1, self.b2), assumptions),
                    )

    def test_scattered_boundary_is_non_strict_on_both_sides(self):
        # Built from the same decimal on both sides so the comparison is exact.
        # `_reset_sim_roboverify_stack` accepts a layout as soon as dx >= 2L, so
        # exactly 2L is a separation the environment can actually hand us.
        first = [0.0, 0.0, 0.0]
        second = [2 * on.BLOCK_LENGTH, 0.0, 0.0]
        self.assertTrue(on.scattered_implementation(first, second))
        self.assertTrue(
            _decide(
                self.context.lowlevel_scattered(self.b1, self.b2),
                self._assumptions(first, second),
            )
        )

    def test_on_direct_is_narrower_than_on_star(self):
        # A block two levels up is on* the base but not directly on it.
        first = [0.0, 0.0, 3 * on.BLOCK_LENGTH]
        second = [0.0, 0.0, 0.0]
        assumptions = self._assumptions(first, second)
        self.assertTrue(
            _decide(self.context.lowlevel_on_star(self.b1, self.b2), assumptions)
        )
        self.assertFalse(
            _decide(self.context.lowlevel_on_direct(self.b1, self.b2), assumptions)
        )

    def test_blocks_below_the_table_plane_stay_in_the_relations(self):
        # `higher` and `scattered` used to carry `z >= 0` conjuncts, which was
        # how the table's sentinel position was excluded and which also dropped
        # any genuine block sitting below z = 0 out of both relations.
        first = [0.0, 0.0, -0.10]
        second = [0.0, 0.0, -0.20]
        self.assertTrue(on.higher_implementation(first, second))
        self.assertTrue(
            _decide(
                self.context.lowlevel_higher(self.b1, self.b2),
                self._assumptions(first, second),
            )
        )

        apart = [0.0, 0.0, -0.10]
        far = [4 * on.BLOCK_LENGTH, 0.0, -0.10]
        self.assertTrue(on.scattered_implementation(apart, far))


class TableIsIsolated(unittest.TestCase):
    """Both readings must match the ``tbl`` axioms rather than the geometry."""

    def setUp(self):
        self.context = lowlevel.LowLevelContext(default_L=on.BLOCK_LENGTH, use_tbl=True)
        self.tbl = self.context.table_const()
        (self.block,) = Consts("isolated_b", self.context.BoxSort)
        # Matching coordinates make ON* and Higher hold geometrically; table
        # identity must still exclude cross-pairs from the relations.
        self.geometry = (
            [self.context.L == RealVal(repr(on.BLOCK_LENGTH))]
            + _pin(self.context, self.block, [0.0, 0.0, 0.0])
            + _pin(self.context, self.tbl, [0.0, 0.0, 0.0])
        )

    def test_numeric_predicates_follow_the_axioms(self):
        block = [0.0, 0.0, 0.0]
        # on_tbl and higher_tbl, with on2 / higher2 reflexivity.
        self.assertFalse(on.on_star_implementation(block, on.TABLE))
        self.assertFalse(on.on_star_implementation(on.TABLE, block))
        self.assertTrue(on.on_star_implementation(on.TABLE, on.TABLE))
        self.assertFalse(on.higher_implementation(block, on.TABLE))
        self.assertFalse(on.higher_implementation(on.TABLE, block))
        self.assertTrue(on.higher_implementation(on.TABLE, on.TABLE))
        # scattered_not_tbl, with no reflexive case.
        self.assertFalse(on.scattered_implementation(block, on.TABLE))
        self.assertFalse(on.scattered_implementation(on.TABLE, on.TABLE))

    def test_numeric_isolation_does_not_depend_on_block_length(self):
        original = on.BLOCK_LENGTH
        try:
            on.BLOCK_LENGTH = 1000.0
            self.assertFalse(on.on_star_implementation([0.0, 0.0, 0.0], on.TABLE))
        finally:
            on.BLOCK_LENGTH = original

    def test_symbolic_predicates_follow_the_axioms(self):
        distinct = self.geometry + [self.block != self.tbl]
        self.assertFalse(
            _decide(self.context.lowlevel_on_star(self.block, self.tbl), distinct)
        )
        self.assertFalse(
            _decide(self.context.lowlevel_higher(self.block, self.tbl), distinct)
        )
        self.assertFalse(
            _decide(self.context.lowlevel_scattered(self.block, self.tbl), distinct)
        )
        self.assertFalse(
            _decide(self.context.lowlevel_on_direct(self.block, self.tbl), distinct)
        )

        identified = self.geometry + [self.block == self.tbl]
        self.assertTrue(
            _decide(self.context.lowlevel_on_star(self.block, self.tbl), identified)
        )
        self.assertTrue(
            _decide(self.context.lowlevel_higher(self.block, self.tbl), identified)
        )
        self.assertFalse(
            _decide(self.context.lowlevel_scattered(self.block, self.tbl), identified)
        )

    def test_table_equality_uses_identity_even_at_equal_coordinates(self):
        equality = self.context.lowlevel_box_equal(self.block, self.tbl)
        self.assertFalse(_decide(equality, self.geometry + [self.block != self.tbl]))
        self.assertTrue(_decide(equality, self.geometry + [self.block == self.tbl]))

    def test_a_task_without_a_table_is_left_alone(self):
        # Stack never names the table, and must not gain an unconstrained sort
        # element that weakens its conditions.
        plain = lowlevel.LowLevelContext(default_L=on.BLOCK_LENGTH)
        (block,) = Consts("isolated_b", plain.BoxSort)
        tbl = plain.get_consts(lowlevel.TABLE_CONST_NAME)
        assumptions = (
            [plain.L == RealVal(repr(on.BLOCK_LENGTH)), block != tbl]
            + _pin(plain, block, [0.0, 0.0, 0.1])
            + _pin(plain, tbl, [0.0, 0.0, 0.0])
        )
        self.assertTrue(_decide(plain.lowlevel_higher(block, tbl), assumptions))


class QuantifierTranslation(unittest.TestCase):
    """Quantifiers are assumptions, so they may only be weakened, never sharpened."""

    def setUp(self):
        self.low = lowlevel.LowLevelContext(default_L=on.BLOCK_LENGTH)
        self.high = highlevel.HighLevelContext(mode="declare")
        self.const_map = {
            name: self.low.get_consts(name) for name in ("b0", "b", "b_prime")
        }
        self.constants = list(self.const_map.values())

    def _translate(self, expr):
        return self.low._translate_expr(expr, self.constants, self.const_map, [])

    def _skolem_names(self, translated):
        return {str(var) for var in get_vars(translated) if str(var).startswith("sk_")}

    def test_exists_is_skolemized_with_a_fresh_witness_each_time(self):
        b0 = self.high.get_consts("b0")
        (x,) = Consts("x", self.high.BoxSort)
        formula = Exists([x], self.high.ON_star(x, b0))

        first = self._skolem_names(self._translate(formula))
        second = self._skolem_names(self._translate(formula))
        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        # Two occurrences must not share a witness, or the translation would
        # assert that one block satisfies both of them.
        self.assertTrue(first.isdisjoint(second))

    def test_witness_does_not_capture_a_named_input_constant(self):
        named = self.high.get_consts("sk_0")
        self.const_map["sk_0"] = self.low.get_consts("sk_0")
        (x,) = Consts("x", self.high.BoxSort)
        translated = self._translate(Exists([x], x != named))
        solver = Solver()
        solver.add(translated)
        self.assertEqual(solver.check(), sat)

    def test_boolean_equality_preserves_boolean_structure(self):
        b0 = self.high.get_consts("b0")
        b = self.high.get_consts("b")
        translated = self._translate(self.high.Higher(b, b0) == BoolVal(True))
        assumptions = _pin(self.low, self.const_map["b"], [0, 0, 1])
        assumptions += _pin(self.low, self.const_map["b0"], [0, 0, 0])
        self.assertTrue(_decide(translated, assumptions))

    def test_equality_from_a_finite_box_sort_reaches_geometry(self):
        finite_sort, _ = EnumSort("PhaseBFiniteBox", ["finite_0", "finite_1"])
        first, second = Consts("b b0", finite_sort)
        assumptions = _pin(self.low, self.const_map["b"], [0, 0, 1])
        assumptions += _pin(self.low, self.const_map["b0"], [0, 0, 0])
        self.assertFalse(_decide(self._translate(first == second), assumptions))
        self.assertTrue(_decide(self._translate(first != second), assumptions))

    def test_quantifier_under_boolean_equality_is_refused(self):
        b0 = self.high.get_consts("b0")
        (x,) = Consts("x", self.high.BoxSort)
        formula = ForAll([x], self.high.ON_star(x, b0)) == BoolVal(False)
        with self.assertRaises(NotImplementedError):
            self._translate(formula)

    def test_exists_translation_reaches_the_geometry(self):
        b0 = self.high.get_consts("b0")
        (x,) = Consts("x", self.high.BoxSort)
        translated = str(self._translate(Exists([x], self.high.ON_star(x, b0))))
        # The old fall-through returned the high-level formula untouched, so the
        # uninterpreted ON_star survived into the low-level solver.
        self.assertNotIn("ON_star", translated)
        self.assertIn("Z(", translated)

    def test_forall_expands_over_the_named_constants(self):
        b0 = self.high.get_consts("b0")
        (x,) = Consts("x", self.high.BoxSort)
        translated = str(self._translate(ForAll([x], self.high.ON_star(x, b0))))
        for name in self.const_map:
            self.assertIn(name, translated)

    def test_negated_universal_keeps_an_unnamed_counterexample_witness(self):
        b0 = self.high.get_consts("b0")
        x = self.high.get_consts("x")
        translated = self._translate(Not(ForAll([x], self.high.ON_star(x, b0))))
        solver = Solver()
        for obj in self.constants:
            solver.add(*_pin(self.low, obj, [0, 0, 0]))
        solver.add(translated)
        self.assertEqual(solver.check(), sat)
        self.assertEqual(len(self._skolem_names(translated)), 1)

    def test_negated_existential_checks_every_named_counterexample(self):
        b0 = self.high.get_consts("b0")
        x = self.high.get_consts("x")
        translated = self._translate(Not(Exists([x], Not(self.high.ON_star(x, b0)))))
        solver = Solver()
        solver.add(*_pin(self.low, self.const_map["b0"], [0, 0, 0]))
        solver.add(*_pin(self.low, self.const_map["b"], [1, 0, 0]))
        solver.add(self.low.L == RealVal("0.05"), translated)
        self.assertEqual(solver.check(), unsat)
        self.assertFalse(self._skolem_names(translated))

    def test_stack_guard_false_translates_nested_quantifiers(self):
        b, picked, other = Consts("b picked other", self.high.BoxSort)
        guard = And(
            picked != b,
            ForAll([other], Or(other == picked, Not(self.high.ON_star(other, picked)))),
        )
        translated = self._translate(Not(Exists([picked], guard)))
        self.assertNotIn("ON_star", str(translated))
        self.assertEqual(len(self._skolem_names(translated)), len(self.constants))


if __name__ == "__main__":
    unittest.main()
