import unittest

from synthesis.predicates.classifier import learn_classifier
from synthesis.predicates.enumerate import enumerate_separator
from synthesis.predicates.guard import loop_guard_synthesis
from synthesis.predicates.language import Language
from synthesis.predicates.scene import Scene, evaluate
from synthesis.predicates.term import atom, conjunction, free_names, negate, ref


class SearchTests(unittest.TestCase):
    def test_classifier_finds_exact_existential_separator(self):
        positive = Scene({0: (0, 0, 0.425), 1: (0, 0, 0.475)}, {"b": 0})
        negative = Scene({0: (0, 0, 0.425), 1: (0.2, 0, 0.425)}, {"b": 0})
        result = learn_classifier(
            [positive],
            [negative],
            {"b"},
            language=Language(
                relations=("ON_star", "eq"), max_depth=4, max_variables=1
            ),
        )
        self.assertTrue(result, result.status)
        self.assertEqual(result.term.op, "exists")
        self.assertTrue(evaluate(result.term, positive))
        self.assertFalse(evaluate(result.term, negative))

    def test_search_is_invariant_to_program_variable_names(self):
        positions = {0: (0, 0, 0.425), 1: (0, 0, 0.475)}
        for mode in ("classifier", "guard"):
            for name in ("anchor", "v0", "v1"):
                with self.subTest(mode=mode, name=name):
                    examples = [
                        (Scene(positions, {name: 0}), True),
                        (Scene(positions, {name: 1}), False),
                    ]
                    result = enumerate_separator(
                        examples,
                        (name,),
                        mode=mode,
                        language=Language(
                            relations=("Higher",), max_depth=3, max_variables=1
                        ),
                    )
                    self.assertTrue(result, result.status)
                    self.assertEqual(free_names(result.term), {name})
                    for scene, expected in examples:
                        self.assertEqual(evaluate(result.term, scene), expected)

    def test_ground_separator_needs_no_existential_budget(self):
        positive = Scene({0: (0, 0, 0.1), 1: (0, 0, 0)}, {"a": 0, "b": 1})
        negative = Scene({0: (0, 0, 0), 1: (0, 0, 0.1)}, {"a": 0, "b": 1})
        result = learn_classifier(
            [positive],
            [negative],
            {"a", "b"},
            language=Language(relations=("Higher",), max_depth=1, max_variables=0),
        )
        self.assertTrue(result, result.status)
        self.assertEqual(result.term.op, "Higher")
        self.assertEqual(result.term.quantifier_count, 0)

    def test_default_vocabulary_can_distinguish_direct_from_transitive_on(self):
        direct = Scene(
            {0: (0, 0, 0), 1: (0, 0, 0.1), 2: (0.2, 0, 0.05)}, {"a": 1, "b": 0}
        )
        tall = Scene({0: (0, 0, 0), 1: (0, 0, 0.1), 2: (0, 0, 0.05)}, {"a": 1, "b": 0})
        result = learn_classifier(
            [direct],
            [tall],
            {"a", "b"},
            language=Language(max_depth=1, max_variables=0),
        )
        self.assertTrue(result, result.status)
        self.assertEqual(result.term.op, "ON")

    def test_guard_allows_indistinguishable_unselected_witnesses(self):
        # Objects 1 and 2 have identical Higher relations, so demonstrations
        # choosing one cannot justify rejecting the other.
        head = Scene(
            {0: (0, 0, 0.425), 1: (0.2, 0, 0.475), 2: (0.4, 0, 0.475)}, {"b": 0}
        )
        exit_scene = Scene(
            {0: (0, 0, 0.425), 1: (0.2, 0, 0.425), 2: (0.4, 0, 0.425)}, {"b": 0}
        )
        candidate = negate(atom("Higher", ref("b"), ref("x")))
        for candidates in ((), (candidate,)):
            for chosen in (1, 2):
                with self.subTest(candidates=candidates, chosen=chosen):
                    result = loop_guard_synthesis(
                        [(head, {"x": chosen})],
                        [exit_scene],
                        ("x",),
                        {"b"},
                        language=Language(
                            relations=("Higher",), max_depth=2, max_variables=0
                        ),
                        candidates=candidates,
                    )
                    self.assertTrue(result, result.status)
                    for other in (1, 2):
                        self.assertTrue(evaluate(result.term, head, {"x": other}))
                    for other in exit_scene.positions:
                        self.assertFalse(
                            evaluate(result.term, exit_scene, {"x": other})
                        )

    def test_guard_allows_alternative_witness_tuples(self):
        head = Scene(
            {0: (0, 0, 0.425), 1: (0.2, 0, 0.475), 2: (0.4, 0, 0.475)}, {"b": 0}
        )
        exit_scene = Scene(
            {0: (0, 0, 0.425), 1: (0.2, 0, 0.425), 2: (0.4, 0, 0.425)}, {"b": 0}
        )
        candidate = conjunction(
            negate(atom("Higher", ref("b"), ref("x"))),
            negate(atom("Higher", ref("b"), ref("y"))),
        )
        result = loop_guard_synthesis(
            [(head, {"x": 1, "y": 2})],
            [exit_scene],
            ("x", "y"),
            {"b"},
            candidates=(candidate,),
        )
        self.assertTrue(result, result.status)
        self.assertTrue(evaluate(result.term, head, {"x": 2, "y": 1}))

    def test_guard_requires_every_exit_binding_to_fail(self):
        # Checking only the first object at the exit would miss witness 1.
        scene = Scene({0: (0, 0, 0.425), 1: (0.2, 0, 0.475)}, {"b": 0})
        candidate = negate(atom("Higher", ref("b"), ref("x")))
        result = loop_guard_synthesis(
            [(scene, {"x": 1})],
            [scene],
            ("x",),
            {"b"},
            candidates=(candidate,),
            language=Language(relations=("Higher",), max_depth=2, max_variables=0),
        )
        self.assertFalse(result)

    def test_guard_rejects_same_scene_as_both_continuation_and_exit(self):
        scene = Scene({0: (0, 0, 0.425), 1: (0.2, 0, 0.425)}, {"b": 0})
        result = loop_guard_synthesis(
            [(scene, {"x": 0})],
            [scene],
            ("x",),
            {"b"},
            language=Language(max_candidates=100),
        )
        self.assertFalse(result)

    def test_budget_exhaustion_is_explicit(self):
        scene = Scene({0: (0, 0, 0)}, {"b": 0})
        result = learn_classifier(
            [scene], [scene], {"b"}, language=Language(max_candidates=1)
        )
        self.assertEqual(result.status, "budget_exhausted")


if __name__ == "__main__":
    unittest.main()
