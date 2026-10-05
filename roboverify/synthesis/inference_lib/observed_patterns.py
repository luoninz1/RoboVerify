"""Standalone observed-pattern formulas, separate from symbolic invariant inference.

Verification workflows use demo_store.InvInference and inference.loop_inference.
This utility is retained for other analyses; it is not a verification learner option.
"""

import itertools

import z3

from synthesis.inference_lib.demo_store import LoopHeadState
from synthesis.verification_lib.counterexamples import state_holds


class MonotoneInvariantLearner:
    """Strongest universal Boolean formula over the observed vocabulary rows.

    Uses Phase C's vocabulary and scene representation. The finite set of allowed
    truth-table rows only grows, so the resulting invariant can only weaken.
    This utility has a direct monotonicity guarantee. It is not the symbolic
    invariant inference algorithm and is not selectable by verification drivers.
    """

    def __call__(self, store, loop_id, vocab, context):
        from synthesis.inference_lib.demo_store import to_inference_inputs
        from synthesis.inference_lib.inference import compute_omega_k

        initial, current, mappings = to_inference_inputs(store, loop_id, vocab, context)
        relations = [
            r if r == "equality" else getattr(context, r) for r in vocab.relations
        ]
        constants = [context.get_consts(name) for name in vocab.constants]
        atoms, variables = compute_omega_k(vocab.k, relations, constants, context)
        patterns = set()
        for before, now, mapping in zip(initial, current, mappings):
            aliases = {str(key): value for key, value in mapping.items()}
            for assignment in itertools.product(now, repeat=len(variables)):
                bindings = dict(
                    aliases, **{str(v): obj for v, obj in zip(variables, assignment)}
                )
                if "tbl" in now:
                    bindings["tbl"] = "tbl"
                row = LoopHeadState(loop_id, now, before, bindings)
                patterns.add(tuple(state_holds(atom, row) for atom in atoms))
        body = z3.Or(
            *(
                z3.And(*(a if truth else z3.Not(a) for a, truth in zip(atoms, pattern)))
                for pattern in sorted(patterns)
            )
        )
        expr = z3.ForAll(variables, body) if variables else body
        return z3.simplify(expr)
