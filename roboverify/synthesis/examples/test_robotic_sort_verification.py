"""Positive and deliberately-invalid invariant checks for the abstract verifier."""

import unittest

import z3

from synthesis.examples.robotic_sort_verification import verify_sort_invariants
from synthesis.inference_lib.relational import RelationSpec, RelationalDomain


def example_domain_and_targets():
    """Hand-written fixtures test the verifier, not the inference pipeline."""
    domain = RelationalDomain(
        "RoboticVerificationTestBlock",
        [RelationSpec(name, arity, lambda snapshot, arguments: False) for name, arity in (
            ("Processed", 1), ("Active", 1), ("Shifted", 1), ("Before", 2),
            ("KeyLE", 2), ("InRow", 1), ("InBuffer", 1), ("AtAssignedSlot", 1),
        )],
    )
    x, y = z3.Consts("x y", domain.object_sort)
    selected = domain.constant("selected")
    r = domain.relations
    outer = z3.And(
        z3.ForAll([x], z3.And(r["InRow"](x), r["AtAssignedSlot"](x), z3.Not(r["InBuffer"](x)))),
        z3.ForAll([x, y], z3.Implies(
            z3.And(r["Processed"](x), r["Processed"](y), r["Before"](x, y)), r["KeyLE"](x, y))),
    )
    inner = z3.And(
        z3.ForAll([x], r["InBuffer"](x) == (x == selected)),
        z3.ForAll([x], z3.Implies(r["InRow"](x), r["AtAssignedSlot"](x))),
        z3.ForAll([x, y], z3.Implies(
            z3.And(r["Active"](x), r["Active"](y), r["Before"](x, y)), r["KeyLE"](x, y))),
        z3.ForAll([x], z3.Implies(r["Shifted"](x), z3.Not(r["KeyLE"](x, selected)))),
    )
    return domain, outer, inner


class RoboticSortVerificationTests(unittest.TestCase):
    def test_complete_invariants_are_inductive_for_symbolic_duplicate_keys(self):
        domain, outer, inner = example_domain_and_targets()
        report = verify_sort_invariants(domain, outer, domain, inner)
        self.assertTrue(report["all_passed"], report)
        self.assertEqual(len(report["checks"]), 9)
        self.assertTrue(all(check["antecedent_status"] == "sat" for check in report["checks"]))

    def test_vacuous_invariants_are_rejected(self):
        domain, _, _ = example_domain_and_targets()
        report = verify_sort_invariants(domain, z3.BoolVal(False), domain, z3.BoolVal(False))
        self.assertFalse(report["all_passed"])
        checks = {check["name"]: check for check in report["checks"]}
        self.assertEqual(checks["outer_initialization"]["status"], "sat")
        self.assertEqual(checks["inner_shift_preservation"]["antecedent_status"], "unsat")
        self.assertFalse(checks["inner_shift_preservation"]["passed"])

    def test_missing_ordering_is_rejected_with_concrete_counterexample(self):
        domain, _, inner = example_domain_and_targets()
        x = domain.constant("x")
        physical_only = z3.ForAll([x], z3.And(domain.relation("InRow")(x), domain.relation("AtAssignedSlot")(x)))
        report = verify_sort_invariants(domain, physical_only, domain, inner)
        exit_check = next(check for check in report["checks"] if check["name"] == "program_exit_sorted_and_physically_realized")
        self.assertEqual(exit_check["status"], "sat")
        example = exit_check["counterexample"]
        keys = example["keys_by_block_id"]
        row_keys = [keys[block] for block in example["row_block_ids_minus_one_is_vacancy"]]
        self.assertNotEqual(row_keys, sorted(row_keys))

    def test_overfitting_to_distinct_keys_fails_initialization(self):
        domain, outer, inner = example_domain_and_targets()
        x, y = z3.Consts("x y", domain.object_sort)
        key_le = domain.relation("KeyLE")
        unjustified_distinctness = z3.ForAll([x, y], z3.Implies(z3.And(key_le(x, y), key_le(y, x)), x == y))
        report = verify_sort_invariants(domain, z3.And(outer, unjustified_distinctness), domain, inner)
        check = report["checks"][0]
        self.assertEqual(check["status"], "sat")
        keys = check["counterexample"]["keys_by_block_id"]
        self.assertLess(len(set(keys)), len(keys))

    def test_observed_equal_suffix_overfit_is_rejected_at_buffer_entry(self):
        # This extra clause was actually learned in the first physical corpus.
        # It wrongly forbids two equal-key blocks wholly in the untouched suffix.
        domain, outer, inner = example_domain_and_targets()
        x, y = z3.Consts("x y", domain.object_sort)
        r = domain.relations
        overfit = z3.ForAll([x, y], z3.Or(
            r["Active"](x), z3.Not(r["KeyLE"](x, y)),
            z3.Not(r["Before"](x, y)), z3.Not(r["KeyLE"](y, x)),
        ))
        report = verify_sort_invariants(domain, outer, domain, z3.And(inner, overfit))
        check = next(item for item in report["checks"] if item["name"] == "outer_to_inner_after_buffering")
        self.assertEqual(check["status"], "sat")
        successor = check["counterexample_successor"]
        selected = successor["selected_block_id"]
        self.assertEqual(successor["physical_slots_by_block_minus_one_is_buffer"][selected], -1)
        self.assertEqual(successor["row_block_ids_minus_one_is_vacancy"][successor["i"]], -1)


if __name__ == "__main__":
    unittest.main()
