"""
BMC tests. Run from ``roboverify/``:

    uv run python -m unittest synthesis.verification_lib.test_bmc_lib -v
"""

from __future__ import annotations

import unittest

import z3

from synthesis.api.instructions import (
    Move,
    MoveByName,
    Pick,
    PickByName,
    Release,
    ReleaseByName,
)
from synthesis.verification_lib.bmc_lib import (
    bmc_feasible,
    bmc_solve,
    bmc_verify,
    default_table_initial_state,
    goal_on_box_ids,
    goal_on_trace_keys,
    infer_block_layout,
)


class TestBmcStackingProgram(unittest.TestCase):
    """
    Program (id-based; ``b1`` / ``b2`` denote box ids **0** and **1**):

        pick(1);
        move(b1, b1, b2, 0, 0, 1);
        move(b2, b2, b2, 0, 0, 1);
        move(b2, b2, b2, 0, 0, 0.025);
        release()

    Mapped to :class:`Pick` / :class:`Move` / :class:`Release` as in
    ``synthesis.api.instructions`` (``Move`` takes three box ids for the
    target xyz reference frame plus ``target_offset``).
    """

    def _program(self):
        # ``target_offset`` omitted → three ``Parameter()`` (unspecified / unknown
        # in the DSL, not numeric zero). Solve-mode BMC uses fresh Z3 reals, not
        # these fields.
        return [
            Pick(grab_box_id=1),
            Move(0, 0, 1),
            Move(1, 1, 1),
            Move(1, 1, 1),
            Release(release_box_id=0),
        ]

    def _program_without_pick(self):
        """Same moves and release as :meth:`_program`, but no ``Pick``."""
        return [
            Move(0, 0, 1),
            Move(1, 1, 1),
            Move(1, 1, 1),
            Release(release_box_id=0),
        ]

    def _program_pick_block2_moves_use2(self):
        """
        Like :meth:`_program`, but the gripper picks **block 2** and subsequent
        ``Move`` anchors use id **2** where the original used **1** for the
        carried block / z reference.
        """
        return [
            Pick(grab_box_id=2),
            Move(0, 0, 2),
            Move(2, 2, 2),
            Move(2, 2, 2),
            # Operand ``1`` keeps block ``\"1\"`` in the inferred universe so
            # ``goal_on_box_ids(..., 1, 0)`` is well-typed (BMC ignores it for geometry).
            Release(release_box_id=1),
        ]

    def test_infer_layout_two_blocks(self):
        prog = self._program()
        names, m = infer_block_layout(prog)
        self.assertEqual(names, ("0", "1"))
        self.assertEqual(m, {"0": 0, "1": 1})

    def test_bmc_sat_stacking_goal(self):
        prog = self._program()
        infer_block_layout(prog)

        # pick(1): gripper over block 1 at t=0
        def make_init(sym):
            return default_table_initial_state(
                sym,
                xy_positions={"0": (0.0, 0.0), "1": (1.0, 0.0)},
                ee_home=(1.0, 0.0, 0.5),
            )

        # Block 1 (picked first) should be able to end stacked on block 0
        def goal(s):
            return goal_on_box_ids(s, 1, 0)

        sat, model, out_sym = bmc_solve(prog, goal, initial_constraints=make_init)
        self.assertTrue(sat, "expected SAT for stacking goal ON(1, 0)")
        self.assertIsNotNone(model)
        self.assertEqual(out_sym.block_names, ("0", "1"))
        # Witness: final z of upper >= lower (sanity on model)
        t = out_sym.T
        zu = float(model.eval(out_sym.bz["1"][t]).as_fraction())
        zl = float(model.eval(out_sym.bz["0"][t]).as_fraction())
        self.assertGreaterEqual(zu, zl)

    def test_bmc_unsat_cannot_satisfy_on_1_0_with_xy_separation(self):
        """
        ``ON(1, 0)`` needs ``|x1-x0|`` below ``BLOCK_LENGTH/2`` (see ``z3_on``).
        Requiring ``ON(1, 0)`` **and** ``|x1-x0| > 2`` at the final step is
        impossible → ``bmc_solve`` must return unsat.
        """
        prog = self._program()

        def make_init(sym):
            return default_table_initial_state(
                sym,
                xy_positions={"0": (0.0, 0.0), "1": (1.0, 0.0)},
                ee_home=(1.0, 0.0, 0.5),
            )

        def goal(s):
            t = s.T
            sep = z3.Abs(s.bx["1"][t] - s.bx["0"][t]) > z3.RealVal(2.0)
            return z3.And(goal_on_box_ids(s, 1, 0), sep)

        sat, model, out_sym = bmc_solve(prog, goal, initial_constraints=make_init)
        self.assertFalse(sat)
        self.assertIsNone(model)
        self.assertEqual(out_sym.block_names, ("0", "1"))

    def test_bmc_unsat_no_pick_blocks_never_move_goal_on_1_0(self):
        """
        Without ``Pick``, ``holding`` stays empty: ``Move`` only translates the
        EE, so block positions stay at the table layout. ``ON(1, 0)`` cannot be
        met from ``(0,0)`` / ``(1,0)`` table placements → unsat.
        """
        prog = self._program_without_pick()

        def make_init(sym):
            return default_table_initial_state(
                sym,
                xy_positions={"0": (0.0, 0.0), "1": (1.0, 0.0)},
                ee_home=(0.0, 0.0, 0.5),
            )

        def goal(s):
            return goal_on_box_ids(s, 1, 0)

        sat, model, out_sym = bmc_solve(prog, goal, initial_constraints=make_init)
        self.assertFalse(sat)
        self.assertIsNone(model)
        self.assertEqual(out_sym.block_names, ("0", "1"))

    def test_bmc_unsat_pick2_goal_still_on_1_0(self):
        """
        Program manipulates block **2**, but the goal remains ``ON(1, 0)``.
        Block **1** is never grasped, so its ``(x,y)`` never changes from the
        initial separation from block **0** → ``ON(1, 0)`` is unsat.
        """
        prog = self._program_pick_block2_moves_use2()
        names, _ = infer_block_layout(prog)
        self.assertEqual(names, ("0", "1", "2"))

        def make_init(sym):
            return default_table_initial_state(
                sym,
                xy_positions={
                    "0": (0.0, 0.0),
                    "1": (1.0, 0.0),
                    "2": (2.0, 0.0),
                },
                ee_home=(2.0, 0.0, 0.5),
            )

        def goal(s):
            return goal_on_box_ids(s, 1, 0)

        sat, model, out_sym = bmc_solve(prog, goal, initial_constraints=make_init)
        self.assertFalse(sat)
        self.assertIsNone(model)
        self.assertEqual(out_sym.block_names, ("0", "1", "2"))

    def test_bmc_verify_true_when_instruction_offsets_suffice(self):
        """
        With **zero** move offsets, a short pick / move / release program can
        still satisfy ``ON(1, 0)``; **verify** fixes offsets to those zeros and
        should stay SAT (while **solve** would ignore ``Parameter.val`` anyway).
        """
        prog = [
            Pick(grab_box_id=1),
            Move(0, 0, 1, target_offset=[0.0, 0.0, 0.0]),
            Release(release_box_id=0, target_z=0.0),
        ]

        def make_init(sym):
            return default_table_initial_state(
                sym,
                xy_positions={"0": (0.0, 0.0), "1": (1.0, 0.0)},
                ee_home=(1.0, 0.0, 0.5),
            )

        def goal(s):
            return goal_on_box_ids(s, 1, 0)

        self.assertTrue(bmc_feasible(prog, goal, initial_constraints=make_init))
        self.assertTrue(bmc_verify(prog, goal, initial_constraints=make_init))

    def test_bmc_verify_false_feasible_true_when_offsets_are_bad(self):
        """
        First ``Move`` uses a huge fixed ``target_offset`` in x. **Solve** mode
        ignores ``Parameter.val`` and still finds offsets → feasible. **Verify**
        mode fixes offsets to those values → cannot meet ``ON(1, 0)``.
        """
        prog = [
            Pick(grab_box_id=1),
            Move(0, 0, 1, target_offset=[1e6, 0.0, 0.0]),
            Release(release_box_id=0, target_z=0.0),
        ]

        def make_init(sym):
            return default_table_initial_state(
                sym,
                xy_positions={"0": (0.0, 0.0), "1": (1.0, 0.0)},
                ee_home=(1.0, 0.0, 0.5),
            )

        def goal(s):
            return goal_on_box_ids(s, 1, 0)

        self.assertTrue(bmc_feasible(prog, goal, initial_constraints=make_init))
        self.assertFalse(bmc_verify(prog, goal, initial_constraints=make_init))


class TestBmcStackingProgramByName(unittest.TestCase):
    """
    Same stacking pattern as :class:`TestBmcStackingProgram`, but with
    **ByName** instructions and trace keys ``\"b\"`` / ``\"b_prime\"`` (no
    numeric box ids in the program text).

    Layout: ``b`` at the origin, ``b_prime`` at ``(1, 0)`` — isomorphic to
    ``b1`` / ``b2`` as ids 0 / 1 in the id-based test.
    """

    def _program(self):
        return [
            PickByName("b_prime"),
            MoveByName("b", "b", "b_prime", target_offset=[0.0, 0.0, 1.0]),
            MoveByName("b_prime", "b_prime", "b_prime", target_offset=[0.0, 0.0, 1.0]),
            MoveByName(
                "b_prime",
                "b_prime",
                "b_prime",
                target_offset=[0.0, 0.0, 0.025],
            ),
            ReleaseByName("b"),
        ]

    def test_infer_layout_sorted_names(self):
        prog = self._program()
        names, m = infer_block_layout(prog)
        self.assertEqual(names, ("b", "b_prime"))
        self.assertEqual(m, {"b": 0, "b_prime": 1})

    def test_bmc_sat_stacking_goal_on_names(self):
        prog = self._program()
        infer_block_layout(prog)

        def make_init(sym):
            return default_table_initial_state(
                sym,
                xy_positions={"b": (0.0, 0.0), "b_prime": (1.0, 0.0)},
                ee_home=(1.0, 0.0, 0.5),
            )

        def goal(s):
            return goal_on_trace_keys(s, "b_prime", "b")

        sat, model, out_sym = bmc_solve(prog, goal, initial_constraints=make_init)
        self.assertTrue(sat, "expected SAT for stacking goal ON(b_prime, b)")
        self.assertIsNotNone(model)
        self.assertEqual(out_sym.block_names, ("b", "b_prime"))
        t = out_sym.T
        zu = float(model.eval(out_sym.bz["b_prime"][t]).as_fraction())
        zl = float(model.eval(out_sym.bz["b"][t]).as_fraction())
        self.assertGreaterEqual(zu, zl)

    def test_bmc_unsat_cannot_satisfy_on_b_b_prime_with_xy_separation(self):
        """
        Same obstruction as :meth:`TestBmcStackingProgram.test_bmc_unsat_cannot_satisfy_on_1_0_with_xy_separation`,
        but for ``ON(b, b_prime)`` on trace keys ``\"b\"`` / ``\"b_prime\"``.
        """
        prog = self._program()

        def make_init(sym):
            return default_table_initial_state(
                sym,
                xy_positions={"b": (0.0, 0.0), "b_prime": (1.0, 0.0)},
                ee_home=(1.0, 0.0, 0.5),
            )

        def goal(s):
            t = s.T
            sep = z3.Abs(s.bx["b"][t] - s.bx["b_prime"][t]) > z3.RealVal(2.0)
            return z3.And(goal_on_trace_keys(s, "b", "b_prime"), sep)

        sat, model, out_sym = bmc_solve(prog, goal, initial_constraints=make_init)
        self.assertFalse(sat)
        self.assertIsNone(model)
        self.assertEqual(out_sym.block_names, ("b", "b_prime"))


if __name__ == "__main__":
    unittest.main()
