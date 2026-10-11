import unittest

from synthesis.benchmarks.specs import Case, SceneDefinition, evaluate, manifest


class SpecificationTests(unittest.TestCase):
    def scene(self, task):
        positions = {
            "a": (0, 0, 0.425),
            "b": (0, 0, 0.475),
            "c": (0, 0, 0.525),
            "d": (0, 0, 0.575),
        }
        return SceneDefinition(
            Case(task, "test", 1, 4),
            tuple(positions),
            positions,
            tuple(positions),
            {"a": 2, "b": 1, "c": 1, "d": -1},
            {"base": "a", "cut": "c"},
            groups=(("a", "b"), ("c", "d")),
        )

    def test_manifest_disjoint_and_varied_sizes(self):
        cases = manifest("insertion_sort")
        seeds = [
            {case.seed for case in cases if case.split == split}
            for split in ("train", "validation", "test")
        ]
        self.assertFalse(
            seeds[0] & seeds[1] or seeds[0] & seeds[2] or seeds[1] & seeds[2]
        )
        self.assertIn(5, {case.num_blocks for case in cases if case.split == "test"})

    def test_reverse_requires_identity_order_not_only_a_tower(self):
        scene = self.scene("reverse")
        self.assertFalse(evaluate(scene, scene.positions).success)
        positions = {
            obj: (0, 0, 0.425 + 0.05 * (3 - index))
            for index, obj in enumerate(scene.objects)
        }
        self.assertTrue(evaluate(scene, positions).success)

    def test_split_requires_prefix_frame_and_source_order(self):
        scene = self.scene("tower_split")
        positions = {**scene.positions, "c": (0.2, 0, 0.425), "d": (0.2, 0, 0.475)}
        self.assertTrue(evaluate(scene, positions).success)
        positions["a"] = (0.02, 0, 0.425)
        self.assertIn("protected_prefix", evaluate(scene, positions).failed)

    def test_row_stability_and_restore_checkpoint(self):
        scene = self.scene("insertion_sort")
        scene.slots = tuple((0, 0.11 * i, 0.425) for i in range(4))
        scene.buffer = (0.2, 0, 0.425)
        good = dict(zip(("d", "b", "c", "a"), scene.slots))
        self.assertTrue(evaluate(scene, good).success)
        bad = dict(zip(("d", "c", "b", "a"), scene.slots))
        self.assertIn("stable_order", evaluate(scene, bad).failed)
        scene.case = Case("sort_restore", "test", 1, 4)
        original = dict(zip(scene.initial_order, scene.slots))
        self.assertFalse(evaluate(scene, original).success)
        self.assertTrue(evaluate(scene, original, sorted_seen=True).success)

    def test_held_and_tilted_are_failures_even_with_correct_goal(self):
        scene = self.scene("stack")
        self.assertFalse(evaluate(scene, scene.positions, held="b").success)
        self.assertFalse(
            evaluate(scene, scene.positions, upright=dict(a=1, b=1, c=0.9, d=1)).success
        )


if __name__ == "__main__":
    unittest.main()
