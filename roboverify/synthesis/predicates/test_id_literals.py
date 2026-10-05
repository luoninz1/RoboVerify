import unittest

from synthesis.predicates.classifier import learn_ground_classifier
from synthesis.predicates.scene import Scene, evaluate
from synthesis.predicates.term import (
    atom,
    block_id,
    canonical,
    exists,
    free_names,
    ref,
    substitute,
)


class IDLiteralTests(unittest.TestCase):
    def test_ids_survive_binder_renaming_and_free_reference_substitution(self):
        relation = atom("ON", block_id(1), ref("x"))
        self.assertEqual(free_names(relation), {"x"})
        self.assertEqual(
            substitute(relation, {"x": ref("b0")}), atom("ON", block_id(1), ref("b0"))
        )
        quantified = exists(["x"], relation)
        self.assertEqual(
            canonical(quantified),
            exists(["other"], atom("ON", block_id(1), ref("other"))),
        )
        scene = Scene({0: (0, 0, 0.425), 1: (0, 0, 0.475)})
        self.assertTrue(evaluate(quantified, scene))

    def test_ground_search_does_not_accept_different_id_universes(self):
        first = Scene({0: (0, 0, 0.425), 1: (0, 0, 0.475)}, {"b0": 0})
        second = Scene({0: (0, 0, 0.425)}, {"b0": 0})
        with self.assertRaisesRegex(ValueError, "same block IDs"):
            learn_ground_classifier([first], [second], {"b0"})

    def test_ground_literals_distinguish_ids_without_any_aliases(self):
        positive = Scene({0: (0, 0, 0.425), 1: (0, 0, 0.475)})
        negative = Scene({0: (0, 0, 0.425), 1: (0.2, 0, 0.425)})
        result = learn_ground_classifier([positive], [negative], set())
        self.assertTrue(result)
        self.assertEqual(result.term, atom("ON", block_id(1), block_id(0)))
        self.assertFalse(free_names(result.term))


if __name__ == "__main__":
    unittest.main()
