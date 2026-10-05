"""No ``Top`` atom may reach a learned invariant.

``Top(x)`` had a declaration, a vocabulary entry and two evaluation branches, but
no entry in the ``rewrite_for_put_for_*`` family, so ``wp`` carried it across a
``Put`` unchanged -- and putting a block on ``b`` is exactly what stops ``b``
being Top. It is gone; these tests keep it from coming back by the same route.
"""

import unittest

from z3 import Z3Exception

import synthesis.inference_lib.inference as inference
import synthesis.verification_lib.highlevel_verification_lib as highlevel


def _mentions_top(atoms):
    return [atom for atom in atoms if str(atom).startswith("Top")]


class NoTopSymbol(unittest.TestCase):
    def setUp(self):
        self.context = highlevel.HighLevelContext(mode="declare", use_tbl=True)
        self.constants = [self.context.get_consts(name) for name in ("b0", "b", "tbl")]

    def test_the_context_declares_no_top_function(self):
        self.assertFalse(hasattr(self.context, "Top"))

    def test_serialized_invariants_cannot_name_top(self):
        # `spec_to_expr` parses a serialized invariant against a fixed decl
        # table; leaving Top out means an old spec that names it fails loudly
        # rather than being re-parsed into an unrewritten predicate.
        spec = highlevel.InvariantSpec(data={"sexpr": "(Top b)"})
        with self.assertRaises(Z3Exception):
            self.context.spec_to_expr(spec, known_const_names=["b"])

    def test_omega_k_builds_no_top_atom(self):
        relations = [
            self.context.ON_star,
            self.context.Higher,
            self.context.Scattered,
            "equality",
        ]
        atoms, _ = inference.compute_omega_k(2, relations, self.constants, self.context)
        self.assertTrue(atoms)
        self.assertEqual([], _mentions_top(atoms))

    def test_forall_exists_omega_builds_no_top_atom(self):
        relations = [self.context.ON_star, self.context.ON_star_zero, "equality"]
        universal = [self.context.get_consts("ux1")]
        existential = [self.context.get_consts("ex1")]
        atoms = inference.forall_exists_compute_omega(
            universal, existential, relations, self.constants
        )
        self.assertTrue(atoms)
        self.assertEqual([], _mentions_top(atoms))


if __name__ == "__main__":
    unittest.main()
