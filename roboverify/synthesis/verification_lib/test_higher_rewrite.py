"""Put Higher effects against the agreed discrete physical height model."""

import unittest

import z3

from synthesis.api.instructions import Put
from synthesis.api.program import wp
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext


class HigherPutRuleTwo(unittest.TestCase):
    def test_matches_placement_on_all_integer_height_levels(self):
        high = HighLevelContext()
        a, b, c = [high.get_consts(name) for name in ("a", "b", "c")]
        # Normalize L to one and the common bottom-center height to zero.
        # The source's old height is arbitrary and must not affect this result.
        za = z3.Real("old_source_height")
        zb, zc = z3.Ints("base_level other_level")
        solver = z3.Solver()
        solver.add(z3.Distinct(a, b, c), z3.Not(high.ON_star(b, a)))
        for left, zl in ((a, za), (b, zb), (c, zc)):
            for right, zr in ((a, za), (b, zb), (c, zc)):
                solver.add(high.Higher(left, right) == (zl >= zr))
        self.assertEqual(solver.check(), z3.sat)
        predicted = wp(Put("a", "b"), high.Higher(c, a), high)
        solver.add(predicted != (zc >= zb + 1))
        self.assertEqual(solver.check(), z3.unsat)

    def test_preserves_table_isolation_when_source_is_level_with_base(self):
        high = HighLevelContext(use_tbl=True)
        a, b, tbl = [high.get_consts(name) for name in ("a", "b", "tbl")]
        solver = z3.Solver()
        high.add_axiom_higher(solver)
        solver.add(
            z3.Distinct(a, b, tbl),
            z3.Not(high.ON_star(b, a)),
            high.Higher(a, b),
            high.Higher(b, a),
        )
        self.assertEqual(solver.check(), z3.sat)
        solver.add(wp(Put("a", "b"), high.Higher(tbl, a), high))
        self.assertEqual(solver.check(), z3.unsat)


class HigherPutRuleSix(unittest.TestCase):
    def test_matches_supported_layers_and_preserves_program_object_named_t(self):
        # Complete support levels, with separate peers c and t at the same
        # height. Checking both names also detects auxiliary-variable capture.
        names = ["a", "b", "c", "t", "s0", "s1", "s2", "s3", "s4", "tbl"]
        high = HighLevelContext(
            mode="enum",
            enum_names=["layer_" + name for name in names],
            sort_name="HigherRuleSixLayers",
            use_tbl=True,
        )
        blocks = {name: high.get_consts(name) for name in names}
        binding = [
            blocks[name] == value for name, value in zip(names, high.enum_blocks)
        ]
        predictions = [
            wp(Put("a", "b"), high.Higher(blocks["a"], blocks[n]), high)
            for n in ("c", "t", "tbl")
        ]
        for base_level in range(5):
            for other_level in range(5):
                for source_level in (0, 2, 4):
                    with self.subTest(
                        base=base_level, other=other_level, source=source_level
                    ):
                        heights = dict(
                            a=source_level, b=base_level, c=other_level, t=other_level
                        )
                        heights.update({"s" + str(i): i for i in range(5)})
                        solver = z3.Solver()
                        solver.set(timeout=5000)
                        solver.add(
                            *binding, z3.Not(high.ON_star(blocks["b"], blocks["a"]))
                        )
                        for left in names:
                            for right in names:
                                expected = (
                                    (left == right)
                                    if "tbl" in (left, right)
                                    else heights[left] >= heights[right]
                                )
                                solver.add(
                                    high.Higher(blocks[left], blocks[right]) == expected
                                )
                        self.assertEqual(solver.check(), z3.sat)
                        expected = other_level <= base_level + 1
                        solver.add(
                            z3.Or(
                                predictions[0] != expected,
                                predictions[1] != expected,
                                predictions[2],
                            )
                        )
                        self.assertEqual(solver.check(), z3.unsat)

    def test_quantified_object_named_t_is_not_captured(self):
        high = HighLevelContext()
        a, b, t = [high.get_consts(name) for name in ("a", "b", "t")]
        u = z3.FreshConst(high.BoxSort, prefix="renamed_object")
        left = wp(Put("a", "b"), z3.ForAll([t], high.Higher(a, t)), high)
        right = wp(Put("a", "b"), z3.ForAll([u], high.Higher(a, u)), high)
        solver = z3.Solver()
        solver.set(timeout=5000)
        solver.add(z3.Distinct(a, b), z3.Not(high.ON_star(b, a)), left != right)
        self.assertEqual(solver.check(), z3.unsat)
