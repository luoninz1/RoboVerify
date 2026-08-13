"""Tests for the generic relational front end and insertion-sort example."""

import contextlib
import io
import unittest

import z3

import synthesis.verification_lib.highlevel_verification_lib as highlevel_verification_lib
from synthesis.examples.insertion_sort import (
    Pair,
    build_insertion_sort_domain,
    build_training_snapshots,
    check_sorted_prefix_entailment,
    infer_default_insertion_sort_invariant,
    pairs_from_keys,
    snapshots_have_sorted_prefix,
    to_relational_snapshot,
    trace_insertion_sort,
)
from synthesis.inference_lib.inference import run_proposal_example
from synthesis.inference_lib.relational import (
    RelationalSnapshot,
    RelationSpec,
    build_relational_dataset,
    build_relational_vocabulary,
    ground_relational_row,
)


class TestInsertionSortTrace(unittest.TestCase):
    def test_walkthrough_snapshots_match_outer_loop_boundaries(self):
        run = trace_insertion_sort(
            [Pair(5, "apple"), Pair(2, "banana"), Pair(9, "cherry")]
        )

        self.assertEqual(
            [snapshot.order for snapshot in run.snapshots],
            [
                ("item_0", "item_1", "item_2"),
                ("item_1", "item_0", "item_2"),
                ("item_1", "item_0", "item_2"),
            ],
        )
        self.assertEqual(
            [snapshot.processed for snapshot in run.snapshots],
            [
                frozenset({"item_0"}),
                frozenset({"item_0", "item_1"}),
                frozenset({"item_0", "item_1", "item_2"}),
            ],
        )
        self.assertEqual(
            [pair.value for pair in run.sorted_pairs],
            ["banana", "apple", "cherry"],
        )

    def test_equal_keys_keep_their_original_order(self):
        run = trace_insertion_sort(
            [Pair(2, "first"), Pair(1, "middle"), Pair(2, "second")]
        )
        self.assertEqual(
            [pair.value for pair in run.sorted_pairs],
            ["middle", "first", "second"],
        )


class TestRelationalFrontEnd(unittest.TestCase):
    def setUp(self):
        self.domain = build_insertion_sort_domain()
        run = trace_insertion_sort(
            [Pair(5, "apple"), Pair(2, "banana"), Pair(9, "cherry")]
        )
        self.snapshots = tuple(to_relational_snapshot(state) for state in run.snapshots)

    def test_vocabulary_contains_expected_unary_binary_and_equality_atoms(self):
        universal_variables, vocabulary = build_relational_vocabulary(self.domain, k=2)
        self.assertEqual([str(var) for var in universal_variables], ["ux1", "ux2"])
        self.assertEqual(
            [str(atom) for atom in vocabulary],
            [
                "Processed(ux1)",
                "Processed(ux2)",
                "Before(ux1, ux2)",
                "Before(ux2, ux1)",
                "KeyLE(ux1, ux2)",
                "KeyLE(ux2, ux1)",
                "ux1 == ux2",
            ],
        )

    def test_grounded_truth_row_matches_second_walkthrough_state(self):
        universal_variables, vocabulary = build_relational_vocabulary(self.domain, k=2)
        row = ground_relational_row(
            self.domain,
            self.snapshots[1],
            vocabulary,
            universal_variables,
            assignment=("item_1", "item_0"),
        )
        self.assertEqual(row, (True, True, True, False, True, False, False))

    def test_validation_rejects_bad_relation_and_groundings(self):
        with self.assertRaisesRegex(ValueError, "arity 1 or 2"):
            RelationSpec("Ternary", 3, lambda snapshot, arguments: True)

        universal_variables, vocabulary = build_relational_vocabulary(self.domain, k=2)
        with self.assertRaisesRegex(KeyError, "unknown objects"):
            ground_relational_row(
                self.domain,
                self.snapshots[0],
                vocabulary,
                universal_variables,
                assignment=("item_0", "missing_item"),
            )

        one_variable, with_constant = build_relational_vocabulary(
            self.domain, k=1, constant_names=("cursor",)
        )
        with self.assertRaisesRegex(ValueError, "constant bindings"):
            build_relational_dataset(
                self.domain,
                self.snapshots,
                with_constant,
                one_variable,
                constant_names=("cursor",),
            )

        bad_binding = RelationalSnapshot(
            objects=self.snapshots[0].objects,
            payload=self.snapshots[0].payload,
            constant_bindings={"cursor": "missing_item"},
        )
        with self.assertRaisesRegex(KeyError, "unknown objects"):
            build_relational_dataset(
                self.domain,
                (bad_binding,),
                with_constant,
                one_variable,
                constant_names=("cursor",),
            )

    def test_inferred_formula_entails_sorted_prefix_target(self):
        domain, result = infer_default_insertion_sort_invariant(verbose=False)
        self.assertGreater(len(result.truth_rows), 0)
        self.assertGreater(len(result.clauses), 0)
        self.assertEqual(
            check_sorted_prefix_entailment(domain, result.invariant), z3.unsat
        )

    def test_held_out_traces_have_sorted_processed_prefixes(self):
        held_out = []
        for case_index, keys in enumerate(((7, 3, 7, 1), (-1, 4, 0), (8,))):
            run = trace_insertion_sort(pairs_from_keys(keys, f"held_out_{case_index}"))
            held_out.extend(to_relational_snapshot(state) for state in run.snapshots)
        self.assertTrue(snapshots_have_sorted_prefix(held_out))

    def test_default_training_corpus_has_sorted_processed_prefixes(self):
        self.assertTrue(snapshots_have_sorted_prefix(build_training_snapshots()))


class TestLegacyInferenceRegression(unittest.TestCase):
    def test_block_proposal_example_still_runs(self):
        context = highlevel_verification_lib.HighLevelContext(mode="declare")
        with contextlib.redirect_stdout(io.StringIO()):
            invariant, clauses = run_proposal_example(context=context)
        self.assertFalse(z3.is_false(invariant))
        self.assertGreater(len(clauses), 0)


if __name__ == "__main__":
    unittest.main()
