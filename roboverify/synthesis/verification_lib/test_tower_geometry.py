"""Geometric solver regressions using an explicit structural invariant fixture.

This fixture tests proof obligations, not acceptance of the inference algorithm.
"""

import unittest

import z3

from synthesis.api.instructions import MoveByName
from synthesis.cfg.program_adapter import program_to_cfg
from synthesis.cfg.program_source import load_program
from synthesis.cfg.tasks import task_spec
from synthesis.cfg.verification import (
    propose_summaries,
    verify_cfg_motion,
    verify_cfg_symbolic,
)
from synthesis.cfg.verified_synthesis import loop_regions
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext


def stack_cfg_with_fixture_invariant():
    context = HighLevelContext()
    definition = load_program("synthesis.examples.stack:build_program", context, 4)
    cfg = program_to_cfg(definition, [], *task_spec("stack"))
    propose_summaries(cfg, context)
    # One tower rooted at b0 and topped by b; all other blocks are singletons.
    # Explicit test input, never injected into production inference.
    b, b0, x, y = z3.Consts("b b0 fixture_x fixture_y", context.BoxSort)
    on = context.ON_star
    invariant = z3.And(
        on(b, b0),
        z3.ForAll([x], z3.Implies(on(b0, x), x == b0)),
        z3.ForAll([x], z3.Implies(on(x, b), x == b)),
        z3.ForAll([x], z3.Implies(on(x, b0), on(b, x))),
        z3.ForAll([x, y], z3.Or(on(x, y), on(y, x), context.Scattered(x, y))),
        z3.ForAll(
            [x, y], z3.Implies(z3.And(x != y, on(x, y)), z3.And(on(x, b0), on(y, b0)))
        ),
    )
    next(loop_regions(cfg)).invariant = invariant
    return context, cfg


class ConsistencyWitnessTests(unittest.TestCase):
    def problem(self):
        from synthesis.verification_lib.lowlevel_verification_lib import LowLevelContext
        from synthesis.verification_lib.motion_verification import (
            MotionContract,
            MotionProblem,
        )

        return MotionProblem(
            LowLevelContext(default_L=0.05),
            [],
            ["a", "b"],
            MotionContract("a", "b"),
            None,
            "test",
            3000,
        )

    def test_supported_towers_do_not_assume_exact_columns(self):
        import z3

        from synthesis.verification_lib.tower_geometry import (
            assume_tower_geometry,
            column_alignment,
        )

        p = self.problem()
        for name, point in {"a": (0.001, 0, 0.475), "b": (0, 0, 0.425)}.items():
            p.solver.add(
                *(v == z3.RealVal(str(n)) for v, n in zip(p.initial[name], point))
            )
        assume_tower_geometry(p, 0.4)
        self.assertEqual(p.check("consistency").status, "valid")
        self.assertEqual(
            p.check("column", z3.Not(column_alignment(p))).status, "refuted"
        )

    def test_finite_unsat_does_not_establish_unbounded_inconsistency(self):
        import z3

        from synthesis.util.symbols import fresh_const

        p = self.problem()
        a = p.constants["a"]
        x = fresh_const(p.context.BoxSort, "extra_object")
        p.solver.add(z3.Exists([x], x != a))
        p.consistency_candidates = (a,)
        self.assertFalse(p.finite_consistency_witness((a,)))
        self.assertEqual(p.check("consistency").status, "valid")

    def test_inconsistent_premises_are_not_given_a_witness(self):
        import z3

        p = self.problem()
        p.solver.add(z3.BoolVal(False))
        p.consistency_candidates = tuple(p.constants.values())
        self.assertFalse(p.finite_consistency_witness(p.consistency_candidates))
        self.assertEqual(p.check("consistency").status, "inconsistent")

    def test_consistency_domain_never_restricts_a_safety_query(self):
        import z3

        from synthesis.util.symbols import fresh_const

        p = self.problem()
        a = p.constants["a"]
        p.consistency_candidates = (a,)
        self.assertEqual(p.check("consistency").status, "valid")
        x = fresh_const(p.context.BoxSort, "arbitrary_object")
        result = p.check("one_object_only", z3.Exists([x], x != a))
        self.assertEqual(result.status, "refuted")


class SupportedStackTests(unittest.TestCase):
    def test_stack_solver_obligations_with_explicit_invariant(self):
        context, cfg = stack_cfg_with_fixture_invariant()
        symbolic = verify_cfg_symbolic(cfg, context, timeout_ms=10000)
        self.assertTrue(symbolic, str(symbolic))
        self.assertEqual(symbolic.scope, "unbounded")
        motion = verify_cfg_motion(
            cfg,
            context,
            supported_tower_model=True,
            table_surface_height=0.4,
            initial_arm=[0.0, 0.0, 0.6],
            timeout_ms=30000,
        )
        self.assertTrue(bool(motion), str(motion))
        for suffix in (
            "arm_clearance_entry",
            "arm_clearance_preserved",
            "column_alignment_entry",
            "column_alignment_preserved",
            "tower_height_entry",
            "tower_height_preserved",
        ):
            checks = [c for c in motion.checks if c.obligation.endswith("/" + suffix)]
            self.assertEqual(len(checks), 1)
            self.assertEqual(checks[0].status, "valid")

    def test_clearance_is_not_assumed_at_program_entry(self):
        context, cfg = stack_cfg_with_fixture_invariant()
        motion = verify_cfg_motion(
            cfg,
            context,
            supported_tower_model=True,
            table_surface_height=0.4,
            initial_arm=[0.0, 0.0, 0.425],
            timeout_ms=10000,
        )
        self.assertFalse(motion)
        entry = next(
            c for c in motion.checks if c.obligation.endswith("/arm_clearance_entry")
        )
        self.assertEqual(entry.status, "refuted")

    def test_offset_placement_fails_column_preservation(self):
        context, cfg = stack_cfg_with_fixture_invariant()
        loop = next(loop_regions(cfg))
        body = loop.body_cfg.nodes[loop.body_cfg.order[0]].region
        changed = list(body.physical)
        changed[3] = MoveByName("b0", "b0", "b", target_offset=[0.001, 0.0, 0.05])
        body.physical = tuple(changed)
        motion = verify_cfg_motion(
            cfg,
            context,
            supported_tower_model=True,
            table_surface_height=0.4,
            initial_arm=[0.0, 0.0, 0.6],
            timeout_ms=10000,
        )
        self.assertFalse(motion)
        column = next(
            c
            for c in motion.checks
            if c.obligation.endswith("/column_alignment_preserved")
        )
        self.assertEqual(column.status, "refuted")

    def test_low_transfer_still_fails_collision_verification(self):
        context, cfg = stack_cfg_with_fixture_invariant()
        loop = next(loop_regions(cfg))
        body = loop.body_cfg.nodes[loop.body_cfg.order[0]].region
        changed = list(body.physical)
        changed[2] = MoveByName("b0", "b0", "b", target_offset=[0.0, 0.0, 0.02])
        body.physical = tuple(changed)
        motion = verify_cfg_motion(
            cfg,
            context,
            supported_tower_model=True,
            table_surface_height=0.4,
            initial_arm=[0.0, 0.0, 0.6],
            timeout_ms=10000,
        )
        self.assertFalse(motion)
        self.assertTrue(
            any(
                c.status == "refuted" and "/collision_" in c.obligation
                for c in motion.checks
            )
        )


if __name__ == "__main__":
    unittest.main()
