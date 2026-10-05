"""Motion contracts must be realized, frame-preserving, and collision-free."""

import argparse
import contextlib
import io
import unittest
from unittest.mock import patch

import z3

from synthesis.api.instructions import Assign, Pick, PickPlaceByName, While
from synthesis.api.program import Program
from synthesis.entry.motion_options import add_motion_options, motion_noise_from_args
from synthesis.verification_lib.bmc_lib import NoiseSpec
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext
from synthesis.verification_lib.lowlevel_verification_lib import LowLevelContext
from synthesis.verification_lib.motion_verification import (
    MotionContract,
    verify_motion_block,
)


def waypoint(grab, x, y, z, offset, *, release):
    return PickPlaceByName(
        grab_box_name=grab,
        target_box_name_x=x,
        target_box_name_y=y,
        target_box_name_z=z,
        target_offset=offset,
        release=release,
    )


def stack_body(release_height=0.05, release=True):
    return [
        waypoint("a", "a", "a", "a", [0, 0, 0.2], release=False),
        waypoint("a", "b", "b", "b", [0, 0, 0.2], release=False),
        waypoint("a", "b", "b", "b", [0, 0, release_height], release=release),
    ]


def rooted_tower_conditions(*, root="b0", target="b", use_tbl=False):
    """Declare the fixture's root, including the absence of unnamed blocks below it."""
    context = HighLevelContext(use_tbl=use_tbl)
    r, y = context.get_consts(root), context.get_consts(target)
    u = z3.FreshConst(context.BoxSort, prefix="fixture_below")
    conditions = [
        context.ON_star(y, r),
        z3.ForAll([u], z3.Implies(context.ON_star(r, u), u == r)),
    ]
    if use_tbl:
        conditions.append(r != context.get_consts("tbl"))
    return conditions


class MotionVerification(unittest.TestCase):
    def setUp(self):
        # Baseline clearance fixture; dedicated tests cover an initially level
        # source and equal-height peers in separate supported towers.
        self.scene = {
            "a": [0.3, 0, -0.1],
            "b": [0, 0, 0],
            "b0": [0, 0, 0],
            "sym": [2, 2, 0],
        }
        self.constants = ["a", "b", "b0"]
        self.contract = MotionContract("a", "b")

    def verify(self, body=None, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return verify_motion_block(
                kwargs.pop(
                    "initial_condition",
                    rooted_tower_conditions(use_tbl="tbl" in self.constants),
                ),
                stack_body() if body is None else body,
                self.constants,
                self.contract,
                block_v="1",
                initial_positions=self.scene,
                **kwargs
            )

    def test_contract_and_frame_hold_in_clear_scene(self):
        result = self.verify()
        self.assertTrue(result, str(result))
        self.assertEqual(result.mode, "noiseless")
        self.assertEqual(result.counterexamples, [])
        self.assertGreater(result.elapsed_seconds, 0)

    def test_root_drift_is_rejected_even_when_local_placement_passes(self):
        self.scene.update(a=[0.3, 0, 0], b=[0.012, 0, 0.05], b0=[0, 0, 0])
        body = [
            waypoint("a", "a", "a", "a", [0, 0, 0.2], release=False),
            waypoint("a", "b", "b", "b", [0.024, 0, 0.2], release=False),
            waypoint("a", "b", "b", "b", [0.024, 0, 0.05], release=True),
        ]
        result = self.verify(body)
        statuses = {c.obligation: c.status for c in result.checks}
        self.assertEqual(statuses["contract"], "valid")
        self.assertEqual(statuses["alignment"], "refuted")
        self.assertEqual(statuses["effect_ON_star"], "refuted")
        self.assertFalse(result)

    def test_unproved_contract_root_hint_is_not_trusted(self):
        self.contract = MotionContract("a", "b", frame_base="a")
        result = self.verify()
        self.assertTrue(result, str(result))
        root = next(c for c in result.checks if c.obligation == "root_selection")
        self.assertIn("root=b0;", root.reason)

    def test_missing_symbolic_root_does_not_fall_back_to_scene_or_name(self):
        result = self.verify(initial_condition=[])
        self.assertFalse(result)
        root = next(c for c in result.checks if c.obligation == "root_selection")
        self.assertEqual(root.status, "unsupported")

    def test_input_outside_declared_tight_alignment_is_inconsistent(self):
        self.scene.update(b=[0.02, 0, 0.05], b0=[0, 0, 0])
        result = self.verify()
        self.assertFalse(result)
        self.assertEqual(result.checks[0].status, "inconsistent")

    def test_new_placement_alignment_is_checked_not_assumed(self):
        body = stack_body(0.06)
        body[-1] = waypoint("a", "b", "b", "b", [0.015, 0, 0.06], release=True)
        result = self.verify(body)
        statuses = {c.obligation: c.status for c in result.checks}
        self.assertEqual(statuses["initial_consistency"], "valid")
        self.assertEqual(statuses["contract"], "valid")
        self.assertEqual(statuses["alignment"], "refuted")

    def test_alignment_covers_bounded_motion_noise(self):
        body = stack_body(0.06)
        body[-1] = waypoint("a", "b", "b", "b", [0.012, 0, 0.06], release=True)
        self.assertTrue(self.verify(body))
        result = self.verify(body, noise=NoiseSpec(0, 0.001, 0))
        statuses = {c.obligation: c.status for c in result.checks}
        self.assertEqual(statuses["contract"], "valid")
        self.assertEqual(statuses["alignment"], "refuted")
        self.assertTrue(
            next(
                c for c in result.counterexamples if c.obligation == "alignment"
            ).noise_values
        )

    def test_table_placement_requires_symbolic_separation_effect(self):
        self.constants.append("tbl")
        self.contract = MotionContract("a", "tbl", table_surface_height=0)
        self.scene.update(a=[0, 0, 0.025], b=[0.075, 0, 0.025], b0=[0.075, 0, 0.025])
        result = self.verify([waypoint("a", "a", "a", "a", [0, 0, 0], release=True)])
        statuses = {c.obligation: c.status for c in result.checks}
        self.assertEqual(statuses["contract"], "valid")
        self.assertEqual(statuses["effect_Scattered"], "refuted")
        self.assertFalse(result)

    def test_level_source_placement_has_correct_higher_effect(self):
        # Rule 2 must not leave a level, stationary block above the lifted source.
        self.scene["a"][2] = 0
        result = self.verify()
        statuses = {c.obligation: c.status for c in result.checks}
        self.assertEqual(statuses["contract"], "valid")
        self.assertEqual(statuses["effect_Higher"], "valid")
        self.assertTrue(result, str(result))

    def test_rule_six_accepts_equal_height_peers_in_supported_towers(self):
        # Two separate supported tower tops share the new source's height.
        # Neither is strictly below the other, so both comparisons must hold.
        self.scene.update(
            a=[0.3, 0, 0],
            c=[1, 1, 0.05],
            cbase=[1, 1, 0],
            sym=[2, 2, 0.05],
            dbase=[2, 2, 0],
        )
        self.constants.extend(["c", "cbase", "dbase"])
        unconstrained = self.verify()
        self.assertEqual(
            next(
                c.status
                for c in unconstrained.checks
                if c.obligation == "effect_Higher"
            ),
            "refuted",
        )
        self.assertFalse(unconstrained)
        high = HighLevelContext()
        u = z3.FreshConst(high.BoxSort, prefix="scene_block")
        # Bound unnamed blocks to this scene's two occupied height levels.
        # Named coordinates alone do not constrain the universal effect query.
        levels = z3.ForAll(
            [u],
            z3.Or(
                *[
                    z3.And(
                        high.Higher(u, high.get_consts(name)),
                        high.Higher(high.get_consts(name), u),
                    )
                    for name in ("b", "c")
                ]
            ),
        )
        result = self.verify(initial_condition=[*rooted_tower_conditions(), levels])
        statuses = {c.obligation: c.status for c in result.checks}
        self.assertEqual(statuses["contract"], "valid")
        self.assertEqual(statuses["effect_Higher"], "valid")
        self.assertTrue(result, str(result))

    def test_table_scattered_wp_preserves_isolation(self):
        from synthesis.api.instructions import Put
        from synthesis.api.program import wp

        high = HighLevelContext(use_tbl=True)
        a, tbl = high.get_consts("a"), high.get_consts("tbl")
        result = z3.simplify(wp(Put("a", "tbl"), high.Scattered(a, tbl), high))
        self.assertTrue(z3.is_false(result))

    def test_large_release_offset_returns_breaking_configuration(self):
        result = self.verify(stack_body(0.1))
        self.assertFalse(result)
        cex = next(c for c in result.counterexamples if c.obligation == "contract")
        self.assertEqual(cex.block_v, "1")
        self.assertEqual(cex.mu_k, self.scene)
        self.assertEqual(cex.bindings["b0"], cex.bindings["b"])
        self.assertEqual(set(cex.entry_positions), set(self.scene))
        self.assertAlmostEqual(cex.final_positions["a"][2], 0.1)
        self.assertEqual(cex.final_positions["b"], self.scene["b"])

    def test_supported_tower_block_is_not_assumed_stationary(self):
        self.scene["b0"] = list(self.scene["a"])
        self.scene["sym"] = [0.3, 0, 0.05]
        result = self.verify(initial_condition=rooted_tower_conditions(root="b"))
        self.assertFalse(result)
        cex = next(c for c in result.counterexamples if c.obligation == "frame")
        self.assertNotEqual(cex.final_positions["sym"], cex.mu_k["sym"])
        self.assertTrue(
            any(
                c.obligation.startswith("support_") and c.status == "refuted"
                for c in result.checks
            )
        )

    def test_collision_with_unnamed_block_is_refuted(self):
        self.scene["sym"] = [0.15, 0, 0.2]
        result = self.verify()
        self.assertFalse(result)
        self.assertTrue(
            any(c.obligation == "collision_1_sym" for c in result.counterexamples)
        )

    def test_no_release_cannot_realize_placement_contract(self):
        result = self.verify(stack_body(release=False))
        self.assertFalse(result)
        self.assertIn("contract", [c.obligation for c in result.counterexamples])

    def test_noise_changes_contract_and_collision_verdict(self):
        # Leave clearance for bounded errors; the final nominal gap is .06.
        self.assertTrue(
            self.verify(stack_body(0.06), noise=NoiseSpec(0.001, 0.001, 0.001))
        )
        result = self.verify(stack_body(0.06), noise=NoiseSpec(0, 0.03, 0))
        self.assertFalse(result)
        cex = next(c for c in result.counterexamples if c.obligation == "contract")
        self.assertTrue(cex.noise_values)
        self.assertEqual(result.mode, "bounded-noise")

    def test_assignments_before_and_after_motion_resolve_operands(self):
        body = [Assign("c", "b"), *stack_body(), Assign("b", "a")]
        body[2].target_box_name_x = "c"
        self.assertTrue(self.verify(body))

    def test_inconsistent_initials_are_not_verified(self):
        result = verify_motion_block(
            [z3.BoolVal(False)], stack_body(), self.constants, self.contract
        )
        self.assertFalse(result)
        self.assertEqual(result.checks[0].status, "inconsistent")
        self.assertEqual(result.counterexamples, [])

    def test_unknown_is_not_a_counterexample_or_proof(self):
        with patch("z3.Solver.check", return_value=z3.unknown):
            result = self.verify()
        self.assertFalse(result)
        self.assertEqual(result.checks[0].status, "unknown")
        self.assertEqual(result.counterexamples, [])

    def test_missing_contract_and_unhandled_primitives_fail_closed(self):
        result = verify_motion_block([], stack_body(), self.constants, None)
        self.assertFalse(result)
        self.assertEqual(result.checked_blocks, 0)
        self.assertFalse(self.verify([Pick(0)]))
        self.assertFalse(self.verify([Assign("b", "a")]))

    def test_program_requires_explicit_contract_and_covers_straight_line_motion(self):
        loop = While(z3.BoolVal(True), [], stack_body(), z3.BoolVal(True))
        program = Program(1, instructions=[loop])
        self.assertFalse(program.lowlevel_verification(constants=self.constants))
        # A verified loop cannot hide uncovered motion before it.
        program = Program(2, instructions=[Pick(0), loop])
        self.assertFalse(program.lowlevel_verification(constants=self.constants))

    def test_table_contract_uses_physical_surface_not_relational_marker(self):
        self.scene["a"][2] = 0
        self.scene["b"][2] = self.scene["b0"][2] = 0.1
        self.scene["sym"][2] = 0.1
        self.constants.append("tbl")
        self.contract = MotionContract("a", "tbl", table_surface_height=-0.025)
        body = [waypoint("a", "a", "a", "a", [0, 0, 0], release=True)]
        self.assertTrue(self.verify(body))
        self.contract = MotionContract("a", "tbl")
        self.assertFalse(self.verify(body))

    def test_self_relative_waypoints_use_updated_positions(self):
        self.scene["a"][2] = 0
        self.scene["b"] = [0.3, 0, -0.05]
        self.scene["b0"] = list(self.scene["b"])
        # Lift .1 then descend .1 relative to the carried block. The original
        # verifier incorrectly interpreted both endpoints against entry geometry.
        body = [
            waypoint("a", "a", "a", "a", [0, 0, 0.1], release=False),
            waypoint("a", "a", "a", "a", [0, 0, -0.1], release=True),
        ]
        result = self.verify(body)
        # This regression concerns state threading. Local placement succeeds;
        # full abstract-effect agreement is checked separately.
        self.assertEqual(
            next(c.status for c in result.checks if c.obligation == "contract"), "valid"
        )

    def test_noise_cli_defaults_off_and_accepts_three_bounds(self):
        parser = argparse.ArgumentParser()
        add_motion_options(parser)
        self.assertIsNone(motion_noise_from_args(parser, parser.parse_args([])))
        args = parser.parse_args(["--motion-noise", ".001", ".002", ".003"])
        self.assertEqual(
            motion_noise_from_args(parser, args), NoiseSpec(0.001, 0.002, 0.003)
        )
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            motion_noise_from_args(
                parser, parser.parse_args(["--motion-noise", "-1", "0", "0"])
            )

    def test_loop_entry_relation_is_not_current_geometry(self):
        high = HighLevelContext(mode="declare")
        a, b = high.get_consts("a"), high.get_consts("b")
        low = LowLevelContext(default_L=0.05)
        solver = z3.Solver()
        with contextlib.redirect_stdout(io.StringIO()):
            low.translate_condition(
                solver,
                ["a", "b"],
                [high.ON_star_zero(a, b), z3.Not(high.ON_star(a, b))],
            )
        solver.add(low.L == 0.05)
        self.assertEqual(solver.check(), z3.sat)


if __name__ == "__main__":
    unittest.main()
