"""Independent synthetic unit checks; these are not physical training data."""

from types import SimpleNamespace
import unittest

from synthesis.examples.relational_sort_inference import (
    OUTER_NAMES, INNER_NAMES, build_domain, adapt_snapshot, target_formulas,
    mathematical_axioms, definitional_axioms, evaluate_formula, relation_value,
)


def state(order=("a", "b", "c", "d"), values=(1, 3, 2, 4), b="b", x="c", r="c"):
    return SimpleNamespace(objects=tuple("abcd"), order=order,
        key_map=dict(zip(order, values)), b=b, x=x, r=r,
        case_name="synthetic-test", program_point="inner_head", step=0)


class TestRelationalSortTargets(unittest.TestCase):
    def evaluate_targets(self, s, phase="inner"):
        d = build_domain(INNER_NAMES if phase == "inner" else OUTER_NAMES, definitions=True)
        snap = adapt_snapshot(s, constants=True)
        self.assertTrue(all(evaluate_formula(ax, d, snap)
                            for ax in (*mathematical_axioms(d), *definitional_axioms(d))))
        return {name: evaluate_formula(expr, d, snap) for name, expr in target_formulas(d, phase).items()}

    def test_inner_entry_and_after_swap(self):
        self.assertTrue(all(self.evaluate_targets(state()).values()))
        moved = state(order=("a", "c", "b", "d"), values=(1, 2, 3, 4), r="b")
        self.assertTrue(all(self.evaluate_targets(moved).values()))

    def test_boundary_detects_gap_that_rightmost_marker_alone_misses(self):
        bad = state(b="a", x="c", r="c")
        result = self.evaluate_targets(bad)
        self.assertFalse(result["prefix_boundary"])
        self.assertTrue(result["sorted_except_x"])

    def test_sorted_except_selected_and_passed_keys_are_independent(self):
        bad_prefix = state(values=(3, 1, 2, 4))
        self.assertFalse(self.evaluate_targets(bad_prefix)["sorted_except_x"])
        bad_passed = state(order=("a", "c", "b", "d"), values=(1, 3, 2, 4), r="b")
        result = self.evaluate_targets(bad_passed)
        self.assertTrue(result["sorted_except_x"])
        self.assertFalse(result["passed_blocks_strictly_larger"])

    def test_ties_allowed_in_outer_sorted_prefix(self):
        s = state(values=(1, 1, 2, 2), b="d", x=None, r=None)
        self.assertTrue(self.evaluate_targets(s, "outer")["sorted_prefix"])

    def test_reflexive_reachability_and_identity_are_not_key_equality(self):
        s = state(values=(1, 1, 2, 2))
        snap = adapt_snapshot(s)
        self.assertTrue(relation_value("NextStar", snap, ("a", "a")))
        self.assertFalse(relation_value("KeyLess", snap, ("a", "b")))
        self.assertFalse(relation_value("IsX", snap, ("b",)))


class TestCorpusProvenance(unittest.TestCase):
    def corpus(self):
        from synthesis.examples.relational_sort_experiment import experiment_cases, source_hashes
        return {"source_sha256": source_hashes(), "cases": experiment_cases()}

    def test_complete_manifest_and_changed_keys(self):
        from synthesis.examples.relational_sort_experiment import validate_corpus_metadata
        corpus = self.corpus()
        validate_corpus_metadata(corpus)
        corpus["cases"][0]["keys"] = (7, 7, 7, 7)
        with self.assertRaisesRegex(ValueError, "manifest mismatch"):
            validate_corpus_metadata(corpus)

    def test_incomplete_and_duplicate_runs(self):
        from synthesis.examples.relational_sort_experiment import validate_corpus_metadata
        corpus = self.corpus()
        removed = corpus["cases"].pop()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_corpus_metadata(corpus)
        corpus["cases"].append(corpus["cases"][0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_corpus_metadata(corpus)

    def test_source_changes_invalidate_reuse(self):
        from synthesis.examples.relational_sort_experiment import validate_corpus_metadata
        corpus = self.corpus()
        corpus["source_sha256"]["relational_robotic_sort.py"] = "stale"
        with self.assertRaisesRegex(ValueError, "provenance"):
            validate_corpus_metadata(corpus)


if __name__ == "__main__":
    unittest.main()
