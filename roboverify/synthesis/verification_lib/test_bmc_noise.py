"""Optional bounded-error BMC and explicit proof strength."""

import unittest

import z3

from synthesis.api.instructions import Move, Pick, Release
from synthesis.verification_lib import bmc_lib as bmc


class BMCNoise(unittest.TestCase):
    def program(self):
        return [
            Pick(1),
            Move(0, 0, 0, target_offset=[0, 0, 0.05]),
            Release(0, target_z=0),
        ]

    def initial(self, sym):
        return bmc.default_table_initial_state(
            sym, xy_positions={"0": (0, 0), "1": (1, 0)}
        )

    def verify(self, noise=None):
        return bmc.bmc_verify(
            self.program(),
            lambda s: bmc.goal_on_box_ids(s, 1, 0),
            initial_constraints=self.initial,
            noise=noise,
        )

    def test_default_encoding_unchanged(self):
        # Compare against the original formulas, with shared symbols so enum-sort
        # freshness cannot hide a difference. This pins the no-noise transition.
        sym = bmc.build_trace_symbols(("0", "1"), 3)
        frame = lambda b, t: z3.And(
            sym.bx[b][t + 1] == sym.bx[b][t],
            sym.by[b][t + 1] == sym.by[b][t],
            sym.bz[b][t + 1] == sym.bz[b][t],
        )
        pick = z3.And(
            sym.holding[0] == sym.NONE,
            sym.holding[1] == sym.block_consts["1"],
            sym.ee_x[1] == sym.bx["1"][0],
            sym.ee_y[1] == sym.by["1"][0],
            sym.ee_z[1] == sym.bz["1"][0],
            z3.And(*(frame(b, 0) for b in sym.block_names)),
        )
        release = z3.And(
            sym.holding[2] != sym.NONE,
            sym.holding[3] == sym.NONE,
            sym.ee_x[3] == sym.ee_x[2],
            sym.ee_y[3] == sym.ee_y[2],
            sym.ee_z[3] == sym.bz["0"][2] + z3.RealVal(0),
            z3.And(*(frame(b, 2) for b in sym.block_names)),
        )
        self.assertTrue(pick.eq(bmc._encode_pick(sym, 0, 1)))
        self.assertTrue(release.eq(bmc._encode_release(sym, 2, 0, z3.RealVal(0))))
        tx, ty, tz = (
            sym.bx["0"][1] + 0,
            sym.by["0"][1] + 0,
            sym.bz["0"][1] + z3.RealVal(".05"),
        )
        move = z3.And(
            sym.ee_x[2] == tx,
            sym.ee_y[2] == ty,
            sym.ee_z[2] == tz,
            *(
                z3.If(
                    sym.holding[1] == sym.block_consts[b],
                    z3.And(sym.bx[b][2] == tx, sym.by[b][2] == ty, sym.bz[b][2] == tz),
                    frame(b, 1),
                )
                for b in sym.block_names
            ),
            sym.holding[2] == sym.holding[1]
        )
        actual = bmc._encode_move(
            sym, 1, 0, 0, 0, z3.RealVal(0), z3.RealVal(0), z3.RealVal(".05")
        )
        # D2 deliberately fixes the supported-block frame bug. On the original
        # flat, unsupported scene the nominal transition is still equivalent.
        solver = z3.Solver()
        solver.add(sym.bz["0"][1] == sym.bz["1"][1], z3.Xor(move, actual))
        self.assertEqual(solver.check(), z3.unsat)
        result = self.verify()
        self.assertTrue(result)
        self.assertEqual(result.mode, "noiseless")

    def test_moving_a_support_does_not_freeze_the_upper_block(self):
        sym = bmc.build_trace_symbols(("0", "1"), 1)
        solver = z3.Solver()
        solver.add(
            sym.holding[0] == sym.block_consts["0"],
            sym.bx["0"][0] == 0,
            sym.by["0"][0] == 0,
            sym.bz["0"][0] == 0,
            sym.bx["1"][0] == 0,
            sym.by["1"][0] == 0,
            sym.bz["1"][0] == 0.05,
        )
        solver.add(
            bmc._encode_move(
                sym, 0, 0, 0, 0, z3.RealVal(1), z3.RealVal(0), z3.RealVal(0)
            )
        )
        solver.add(sym.bx["1"][1] != sym.bx["1"][0])
        self.assertEqual(solver.check(), z3.sat)

    def test_verify_fails_when_eps_exceeds_on_tolerance(self):
        for spec in (
            bmc.NoiseSpec(0.03, 0, 0),
            bmc.NoiseSpec(0, 0.03, 0),
            bmc.NoiseSpec(0, 0, 0.03),
        ):
            with self.subTest(noise=spec):
                result = self.verify(spec)
                self.assertFalse(result)
                self.assertEqual(result.status, "refuted")
                self.assertEqual(result.mode, "bounded-noise")
                self.assertIsNotNone(result.model)
                goal = bmc.goal_on_box_ids(result.symbols, 1, 0)
                self.assertTrue(z3.is_false(result.model.eval(goal)))

    def test_small_and_zero_noise_verify(self):
        for spec in (bmc.NoiseSpec(0, 0, 0), bmc.NoiseSpec(0.001, 0.001, 0.001)):
            self.assertTrue(self.verify(spec))

    def test_invalid_bounds_rejected(self):
        for value in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                bmc.NoiseSpec(value, 0, 0)

    def test_infeasible_is_not_a_proof(self):
        result = bmc.bmc_verify(
            self.program(),
            lambda s: z3.BoolVal(True),
            initial_constraints=[z3.BoolVal(False)],
            noise=bmc.NoiseSpec(0, 0.001, 0),
        )
        self.assertEqual(result.status, "infeasible")
        self.assertFalse(result)
        self.assertIsNone(result.model)


if __name__ == "__main__":
    unittest.main()
