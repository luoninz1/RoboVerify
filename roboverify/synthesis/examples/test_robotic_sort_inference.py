"""Unit tests for the robotic inference adaptor, using explicitly synthetic states.

These exercise predicate and formula semantics; they are not MuJoCo rollouts or
evidence that the physical controller satisfies its transfer contract.
"""

from types import SimpleNamespace
import unittest

import z3

from synthesis.examples.robotic_sort_inference import (
    INNER_RELATIONS,
    OUTER_RELATIONS,
    adapt_snapshot,
    build_domain,
    domain_axioms,
    evaluate_formula,
    infer_loop,
    relation_value,
    target_formulas,
    validate_learned,
)
from synthesis.inference_lib.relational import (
    build_relational_dataset,
    build_relational_vocabulary,
    ground_relational_row,
)


def synthetic_state(**changes):
    """Construct adaptor input without depending on the simulator trace class."""
    values = {
        "objects": ("a", "b", "c"),
        "order": ("a", "b", "c"),
        "key_map": {"a": 3, "b": 1, "c": 2},
        "slot_map": {"a": 0, "b": 1, "c": 2},
        "buffered": frozenset(),
        "i": 2,
        "j": None,
        "selected": None,
        "program_point": "outer_head",
        "step": 0,
    }
    values.update(changes)
    return SimpleNamespace(**values)


class TestRoboticPredicateSemantics(unittest.TestCase):
    def test_structural_validation_reports_missing_inner_index(self):
        from synthesis.examples.robotic_sort_experiment import structural_violations

        state = synthetic_state(
            order=("a", None, "c"), initial_order=("a", "b", "c"),
            locations={"a": 0, "b": "buffer", "c": 2},
            rotation_map={name: ((1, 0, 0), (0, 1, 0), (0, 0, 1))
                          for name in ("a", "b", "c")},
            held=None, selected="b", i=1, j=None, program_point="inner_head",
        )
        self.assertIn("inner index bounds", structural_violations(state))

    def test_unsorted_state_satisfies_axioms_but_fails_sortedness_target(self):
        domain = build_domain(OUTER_RELATIONS)
        snapshot = adapt_snapshot(synthetic_state())
        self.assertTrue(all(evaluate_formula(ax, domain, snapshot)
                            for ax in domain_axioms(domain)))
        self.assertTrue(relation_value("Processed", snapshot, ("a",)))
        self.assertTrue(relation_value("Before", snapshot, ("a", "b")))
        self.assertFalse(relation_value("KeyLE", snapshot, ("a", "b")))
        self.assertFalse(evaluate_formula(target_formulas(domain, "outer")["sorted_prefix"],
                                          domain, snapshot))

    def test_physical_slot_mismatch_is_not_hidden_by_logical_order(self):
        snapshot = adapt_snapshot(synthetic_state(slot_map={"a": 1, "b": 0, "c": None}))
        self.assertFalse(relation_value("AtAssignedSlot", snapshot, ("a",)))
        self.assertFalse(relation_value("AtAssignedSlot", snapshot, ("b",)))
        self.assertFalse(relation_value("AtAssignedSlot", snapshot, ("c",)))
        self.assertFalse(relation_value("InRow", snapshot, ("c",)))
        self.assertTrue(relation_value("Before", snapshot, ("b", "a")))

    def test_off_row_selected_does_not_violate_before_axioms(self):
        domain = build_domain(INNER_RELATIONS)
        snapshot = adapt_snapshot(synthetic_state(
            order=("a", None, "c"), slot_map={"a": 0, "b": None, "c": 2},
            buffered=frozenset({"b"}), selected="b", i=1, j=0,
            program_point="inner_head",
        ), selected_constant=True)
        self.assertFalse(relation_value("Before", snapshot, ("a", "b")))
        self.assertFalse(relation_value("Before", snapshot, ("b", "a")))
        self.assertFalse(relation_value("AtAssignedSlot", snapshot, ("b",)))
        self.assertTrue(all(evaluate_formula(ax, domain, snapshot)
                            for ax in domain_axioms(domain)))

    def test_no_assessment_target_is_already_an_axiom(self):
        for phase, names in (("outer", OUTER_RELATIONS), ("inner", INNER_RELATIONS)):
            domain = build_domain(names)
            for name, target in target_formulas(domain, phase).items():
                with self.subTest(phase=phase, target=name):
                    solver = z3.Solver()
                    solver.set(timeout=5000)
                    domain.add_axioms(solver)
                    solver.add(z3.Not(target))
                    self.assertEqual(solver.check(), z3.sat)


class TestRoboticFormulaEvaluation(unittest.TestCase):
    def test_selected_is_a_bound_program_constant_not_a_universal_variable(self):
        domain = build_domain(("InBuffer",))
        x = domain.constant("x")
        selected = domain.constant("selected")
        formula = z3.ForAll(x, domain.relation("InBuffer")(x) == (x == selected))
        good = adapt_snapshot(synthetic_state(selected="b", buffered=frozenset({"b"})),
                              selected_constant=True)
        wrong_binding = adapt_snapshot(synthetic_state(selected="a", buffered=frozenset({"b"})),
                                       selected_constant=True)
        self.assertTrue(evaluate_formula(formula, domain, good))
        self.assertFalse(evaluate_formula(formula, domain, wrong_binding))

    def test_nested_quantifiers_preserve_outer_binding_and_argument_order(self):
        domain = build_domain(("Before",))
        x, y = z3.Consts("x y", domain.object_sort)
        selected = domain.constant("selected")
        first_in_row = z3.Exists(x, z3.And(
            x == selected,
            z3.ForAll(y, z3.Or(x == y, domain.relation("Before")(x, y))),
        ))
        first = adapt_snapshot(synthetic_state(selected="a"), selected_constant=True)
        last = adapt_snapshot(synthetic_state(selected="c"), selected_constant=True)
        self.assertTrue(evaluate_formula(first_in_row, domain, first))
        self.assertFalse(evaluate_formula(first_in_row, domain, last))

    def test_snapshot_constant_grounding_and_validation(self):
        domain = build_domain(("KeyLE",))
        variables, vocabulary = build_relational_vocabulary(domain, 1, ("selected",))
        snapshot = adapt_snapshot(synthetic_state(selected="b"), selected_constant=True)
        row = ground_relational_row(domain, snapshot, vocabulary, variables, ("a",),
                                    constant_names=("selected",))
        self.assertEqual(row, (False, True, False))
        self.assertEqual(snapshot.constant_bindings, {"selected": "b"})
        with self.assertRaisesRegex(ValueError, "selected block"):
            adapt_snapshot(synthetic_state(), selected_constant=True)
        with self.assertRaisesRegex(ValueError, "constant bindings"):
            build_relational_dataset(domain, (adapt_snapshot(synthetic_state()),),
                                     vocabulary, variables, ("selected",))
        bad = adapt_snapshot(synthetic_state(selected="missing"), selected_constant=True)
        with self.assertRaisesRegex(KeyError, "unknown objects"):
            build_relational_dataset(domain, (bad,), vocabulary, variables, ("selected",))

    def test_unbound_constants_fail_explicitly(self):
        domain = build_domain(("InBuffer",))
        expr = domain.relation("InBuffer")(domain.constant("selected"))
        with self.assertRaisesRegex(ValueError, "Unbound object constant"):
            evaluate_formula(expr, domain, adapt_snapshot(synthetic_state()))


class TestSyntheticInferenceSmoke(unittest.TestCase):
    def test_actual_backend_preserves_selected_binding_in_inner_inference(self):
        # Before and after the only shift of a descending two-block input.
        common = {
            "objects": ("a", "b"), "key_map": {"a": 2, "b": 1}, "i": 1,
            "selected": "b", "buffered": frozenset({"b"}), "program_point": "inner_head",
        }
        states = (
            synthetic_state(**common, order=("a", None), slot_map={"a": 0, "b": None}, j=0),
            synthetic_state(**common, order=(None, "a"), slot_map={"a": 1, "b": None}, j=-1),
        )
        learned = infer_loop(states, "inner")
        self.assertTrue(validate_learned(learned, states)["all_passed"])
        bad = synthetic_state(**{**common, "selected": "a"}, order=(None, "a"),
                              slot_map={"a": 1, "b": None}, j=-1)
        result = validate_learned(learned, (bad,))
        self.assertFalse(result["all_passed"])
        kinds = {failure["kind"] for failure in result["violations"]}
        self.assertIn("selected_is_the_only_buffered_block", kinds)
        self.assertIn("learned_formula", kinds)

    def test_actual_backend_learns_from_small_synthetic_outer_corpus(self):
        # Both possible two-block input orders, plus their exit snapshots.
        common = {"objects": ("a", "b"), "key_map": {"a": 2, "b": 1}}
        states = (
            synthetic_state(**common, order=("a", "b"), slot_map={"a": 0, "b": 1}, i=1),
            synthetic_state(**common, order=("b", "a"), slot_map={"a": 1, "b": 0}, i=1),
            synthetic_state(**common, order=("b", "a"), slot_map={"a": 1, "b": 0},
                            i=2, program_point="outer_exit"),
        )
        learned = infer_loop(states, "outer")
        self.assertTrue(validate_learned(learned, states)["all_passed"])
        bad = synthetic_state(**common, order=("a", "b"), slot_map={"a": 0, "b": 1},
                              i=2, program_point="outer_exit")
        result = validate_learned(learned, (bad,))
        self.assertFalse(result["all_passed"])
        self.assertIn("sorted_prefix", {failure["kind"] for failure in result["violations"]})
        self.assertIn("learned_formula", {failure["kind"] for failure in result["violations"]})


if __name__ == "__main__":
    unittest.main()
