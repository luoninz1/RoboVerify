import itertools
import random
import unittest

import z3

from synthesis.predicates.scene import Scene, evaluate
from synthesis.predicates.term import (
    atom,
    boolean,
    conjunction,
    exists,
    forall,
    free_names,
    negate,
    ref,
    substitute,
    to_z3,
)
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext


class TermTests(unittest.TestCase):
    def test_alpha_equivalence_and_shadowing(self):
        a = forall(["x"], exists(["y"], atom("eq", ref("x"), ref("y"))))
        b = forall(["a"], exists(["b"], atom("eq", ref("a"), ref("b"))))
        self.assertEqual(a, b)
        self.assertIs(a, b)
        self.assertEqual(free_names(a), frozenset())
        self.assertEqual(evaluate(a, Scene({0: (0, 0, 0)})), True)

    def test_substitution_does_not_capture_or_replace_bound_variables(self):
        t = forall(["x"], atom("eq", ref("x"), ref("b")))
        replaced = substitute(t, {"b": ref("x")})
        self.assertEqual(free_names(replaced), {"x"})
        self.assertEqual(substitute(t, {"x": ref("c")}), t)

    def test_numeric_and_z3_semantics_agree_on_random_formulas(self):
        rng = random.Random(8)
        ctx = HighLevelContext(mode="enum", num_blocks=3, sort_name="TermDifferential")
        blocks = ctx.enum_blocks
        for trial in range(8):
            scene = Scene(
                {i: (i * 0.15, 0.0, rng.choice((0.425, 0.475))) for i in range(3)},
                {"b": 0},
            )
            solver = ctx.new_solver(1000)
            solver.add(ctx.get_consts("b") == blocks[0])
            for rel in ("ON_star", "ON_star_zero", "Higher", "Scattered"):
                for i, j in itertools.product(range(3), repeat=2):
                    formula = atom(rel, ref("a"), ref("c"))
                    solver.add(
                        getattr(ctx, rel)(blocks[i], blocks[j])
                        == evaluate(formula, scene, {"a": i, "c": j})
                    )
            self.assertEqual(solver.check(), z3.sat)
            for _ in range(12):
                rel = rng.choice(("ON_star", "Higher", "Scattered", "eq", "ON"))
                body = atom(rel, ref("x"), ref("b"))
                if rng.randrange(2):
                    body = negate(body)
                term = (forall if rng.randrange(2) else exists)(["x"], body)
                solver.push()
                solver.add(to_z3(term, ctx) != evaluate(term, scene))
                self.assertEqual(solver.check(), z3.unsat, str(term))
                solver.pop()


if __name__ == "__main__":
    unittest.main()
