"""Positive symbolic checks and mutations that invalidate the restricted sorter."""

import unittest

import z3

from synthesis.examples.relational_sort_verification import (
    _verify, theoretical_formulas, verify_sort_invariants,
)
from synthesis.inference_lib.relational import RelationSpec, RelationalDomain


def fixture():
    domain = RelationalDomain("RestrictedSortVerificationFixture", [
        RelationSpec(name, arity, lambda snapshot, arguments: False)
        for name, arity in (("NextStar", 2), ("KeyLess", 2), ("BeforeB", 1),
                            ("BeforeR", 1), ("IsB", 1), ("IsX", 1),
                            ("AfterX", 1), ("SelectedLess", 1))
    ])
    theory = theoretical_formulas(domain)
    return domain, theory["outer"], theory["inner"]


class RelationalSortVerificationTests(unittest.TestCase):
    def test_theory_is_inductive_and_transfers_are_safe_for_arbitrary_keys(self):
        domain, outer, inner = fixture()
        result = verify_sort_invariants(domain, outer, domain, inner, n=4)
        self.assertTrue(result["all_passed"], result)
        self.assertTrue(all(c["passed"] for c in result["semantic_comparison"]))
        self.assertTrue(all(c["antecedent_status"] == "sat" for c in result["checks"]))

    def test_wrong_boundary_update_is_detected(self):
        domain, outer, inner = fixture()
        result = _verify(domain, outer, domain, inner, n=3, wrong_r_update=True)
        checks = {c["name"]: c for c in result["checks"]}
        self.assertEqual(checks["inner_swap_preservation"]["status"], "sat")
        self.assertEqual(checks["inner_swap_preserves_enlarged_prefix_size"]["status"], "sat")

    def test_reversed_comparator_is_detected(self):
        domain, outer, inner = fixture()
        result = _verify(domain, outer, domain, inner, n=3, reverse_comparator=True)
        checks = {c["name"]: c for c in result["checks"]}
        self.assertEqual(checks["inner_swap_preservation"]["status"], "sat")
        self.assertEqual(checks["inner_exit_restores_outer"]["status"], "sat")

    def test_missing_inner_boundary_constraint_is_detected(self):
        domain, outer, inner = fixture()
        weak_inner = z3.And(*(inner.children()[0:1] + inner.children()[2:]))
        result = verify_sort_invariants(domain, outer, domain, weak_inner, n=3)
        checks = {c["name"]: c for c in result["checks"]}
        self.assertFalse(result["all_passed"])
        self.assertEqual(checks["inner_swap_preserves_enlarged_prefix_size"]["status"], "sat")

    def test_false_candidates_are_rejected_as_vacuous(self):
        domain, _, _ = fixture()
        result = verify_sort_invariants(domain, z3.BoolVal(False), domain, z3.BoolVal(False), n=3)
        checks = {c["name"]: c for c in result["checks"]}
        self.assertFalse(result["all_passed"])
        self.assertEqual(checks["outer_initialization"]["status"], "sat")
        self.assertEqual(checks["inner_swap_preservation"]["antecedent_status"], "unsat")

    def test_duplicate_key_overfitting_fails_initialization(self):
        domain, outer, inner = fixture()
        u, v = z3.Consts("distinct_u distinct_v", domain.object_sort)
        less = domain.relation("KeyLess")
        distinct_keys = z3.ForAll([u, v], z3.Implies(u != v, z3.Or(less(u, v), less(v, u))))
        result = verify_sort_invariants(domain, z3.And(outer, distinct_keys), domain, inner, n=3)
        initialization = result["checks"][0]
        self.assertEqual(initialization["status"], "sat")
        values = initialization["counterexample"]["keys_by_node"]
        self.assertLess(len(set(values)), len(values))

    def test_projection_predicates_match_raw_relations(self):
        domain, outer, inner = fixture()
        u, v = z3.Consts("projection_u projection_v", domain.object_sort)
        relations = domain.relations
        projected_outer = z3.ForAll([u, v], z3.Implies(
            z3.And(relations["NextStar"](u, v), relations["BeforeB"](v)),
            z3.Not(relations["KeyLess"](v, u))))
        projected_inner = z3.And(
            z3.ForAll([u], z3.Not(z3.And(relations["IsB"](u), relations["IsX"](u)))),
            z3.ForAll([u], relations["BeforeR"](u) == z3.Or(relations["BeforeB"](u), relations["IsX"](u))),
            z3.ForAll([u, v], z3.Implies(z3.And(
                z3.Not(relations["IsX"](u)), z3.Not(relations["IsX"](v)),
                relations["NextStar"](u, v), relations["BeforeR"](v)),
                z3.Not(relations["KeyLess"](v, u)))),
            z3.ForAll([u], z3.Implies(z3.And(z3.Not(relations["IsX"](u)),
                relations["AfterX"](u), relations["BeforeR"](u)), relations["SelectedLess"](u))),
        )
        result = verify_sort_invariants(domain, projected_outer, domain, projected_inner, n=3)
        self.assertTrue(result["all_passed"], result)
        self.assertTrue(all(c["passed"] for c in result["semantic_comparison"]))


if __name__ == "__main__":
    unittest.main()
