"""Trace collection and compatibility with the literal Stack inference inputs."""

import contextlib
import importlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import z3

from synthesis.api.instructions import Assign, LoopBudgetExceeded, While
from synthesis.api.program import Program
from synthesis.inference_lib import golden_tower_fixtures as golden
from synthesis.inference_lib.demo_store import (
    DemoStore,
    InferenceVocabulary,
    InvInference,
    LoopHeadState,
    observation_positions,
    to_inference_inputs,
    tower_vocabulary,
)
from synthesis.inference_lib.inference import compute_data
from synthesis.util import on
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext


class MoveUp:
    """A deterministic test body: move block 1, leaving an unreferenced block alone."""

    def eval(self, env, traj, return_image=False):
        obs = traj[-1].copy()
        obs[24] += 0.05
        traj.append(obs)
        return []


class TraceCollection(unittest.TestCase):
    def setUp(self):
        self.context = HighLevelContext(mode="declare")
        self.obs = np.zeros(13 + 15 * 3)
        self.obs[10:13] = [0, 0, 0]
        self.obs[22:25] = [1, 0, 0]
        self.obs[34:37] = [2, 0, 0.125]
        self.env = SimpleNamespace(
            num_blocks=3, symbolic_name_to_box_id={"b0": 0, "b": 0, "anchor": 2}
        )
        self.env.reset = lambda: (self.obs.copy(), {})
        self.env.set_state_from_observation = lambda obs: None
        b_prime, b = z3.Consts("b_prime b", self.context.BoxSort)
        anchor = self.context.get_consts("anchor")
        self.loop = While(
            z3.And(
                b_prime != b, b_prime != anchor, self.context.Higher(anchor, b_prime)
            ),
            [b_prime],
            [MoveUp()],
            z3.BoolVal(True),
            max_iters=3,
        )

    def test_three_iterations_copy_all_objects_and_guard_bindings(self):
        store = DemoStore()
        traj = [self.obs.copy()]
        self.loop.eval(self.env, traj, on_loop_head=store.add, loop_id="tower")
        rows = store.for_loop("tower")
        self.assertEqual(len(rows), 3)
        self.assertEqual([r.positions["x2"][2] for r in rows], [0, 0.05, 0.1])
        for row in rows:
            self.assertEqual(set(row.positions), {"x1", "x2", "x3", "tbl"})
            self.assertEqual(row.constants["b_prime"], "x2")
            self.assertEqual(row.entry_positions["x2"], [1, 0, 0])
            self.assertIs(row.positions["tbl"], on.TABLE)
        traj[0][22] = 999
        self.env.symbolic_name_to_box_id["b_prime"] = 2
        rows[0].positions["x2"][0] = 999
        self.assertEqual(store.for_loop("tower")[0].positions["x2"], [1, 0, 0])
        self.assertEqual(store.for_loop("tower")[0].constants["b_prime"], "x2")

    def test_false_guard_and_zero_budget_produce_no_rows(self):
        for cond, budget in [(z3.BoolVal(False), 3), (z3.BoolVal(True), 0)]:
            store = DemoStore()
            loop = While(cond, [], [MoveUp()], z3.BoolVal(True), max_iters=budget)
            if budget == 0:
                with self.assertRaises(LoopBudgetExceeded):
                    loop.eval(self.env, [self.obs.copy()], on_loop_head=store.add)
            else:
                loop.eval(self.env, [self.obs.copy()], on_loop_head=store.add)
            self.assertEqual(len(store), 0)

    def test_program_callbacks_and_entry_state_reset_between_rollouts(self):
        # Program deep-copies instructions, so the callback must be passed at eval.
        program = Program(2, instructions=[Assign("b", "b0"), self.loop])
        store = DemoStore()
        plain = program.eval(self.env)
        traced = program.eval(self.env, on_loop_head=store.add)
        np.testing.assert_array_equal(plain, traced)
        self.obs[24] = 0.4
        self.obs[36] = 0.525
        program.eval_from_observation(self.env, self.obs, on_loop_head=store.add)
        rows = store.for_loop("1")
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0].entry_positions["x2"][2], 0)
        self.assertEqual(rows[3].entry_positions["x2"][2], 0.4)
        self.assertEqual(store.for_loop("0"), [])

    def test_diagnostic_export_preserves_table_marker(self):
        import json

        store = DemoStore()
        self.loop.eval(self.env, [self.obs], on_loop_head=store.add)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "diagnostic.json"
            store.save_diagnostic(path)
            payload = json.loads(path.read_text())
        self.assertIsNone(payload["states"][0]["positions"]["tbl"])


class InferenceAdapter(unittest.TestCase):
    def setUp(self):
        self.context = HighLevelContext(mode="declare")
        # Capture exactly what the original example passed to the learner.
        with patch.object(golden, "loop_inference", side_effect=lambda *a, **kw: a):
            self.inputs = golden.run_proposal_example(self.context)
        _, states, _, _, _, mappings = self.inputs
        self.store = DemoStore(
            LoopHeadState(
                "stack", state, state, {str(k): v for k, v in mapping.items()}
            )
            for state, mapping in zip(states, mappings)
        )
        self.vocab = tower_vocabulary("stack")

    def test_adapter_reproduces_consumed_golden_rows_and_bindings(self):
        zeros, states, mappings = to_inference_inputs(
            self.store, "stack", self.vocab, self.context
        )
        self.assertEqual(states, self.inputs[1])
        self.assertEqual(mappings, self.inputs[5])
        self.assertEqual(len(zeros), len(states))
        self.assertEqual(
            zeros, states
        )  # Complete entries replace the old unused {} rows.
        fresh = HighLevelContext(mode="enum", num_blocks=5, visualize_enum_scene=False)
        _, _, fresh_mappings = to_inference_inputs(
            self.store, "stack", self.vocab, fresh
        )
        self.assertEqual(
            set(fresh_mappings[0]), {fresh.get_consts("b0"), fresh.get_consts("b")}
        )

    def test_adapter_learns_an_invariant_equivalent_to_golden(self):
        with contextlib.redirect_stdout(io.StringIO()):
            expected, _ = golden.run_proposal_example(self.context)
            actual, _ = InvInference(self.store, "stack", self.vocab, self.context)
        solver = z3.Solver()
        solver.set(timeout=10000)
        self.context.add_axiom(solver)
        self.context.add_axiom_higher(solver)
        self.context.add_axiom_scattered(solver)
        solver.add(z3.Xor(expected, actual))
        self.assertEqual(solver.check(), z3.unsat)

    def test_reverse_reads_entry_geometry_with_current_bindings(self):
        context = HighLevelContext(mode="declare", use_tbl=True)
        entry = {"x1": [0, 0, 0], "x2": [0, 0, 0.05], "tbl": on.TABLE}
        current = {"x1": [0, 0, 0], "x2": [1, 0, 0], "tbl": on.TABLE}
        store = DemoStore(
            [
                LoopHeadState(
                    "reverse", current, entry, {"b0": "x1", "b": "x2", "tbl": "tbl"}
                )
            ]
        )
        zeros, states, mappings = to_inference_inputs(
            store, "reverse", tower_vocabulary("reverse"), context
        )
        b0, b = context.get_consts("b0"), context.get_consts("b")
        with contextlib.redirect_stdout(io.StringIO()):
            data = compute_data(
                zeros[0],
                states[0],
                [context.ON_star_zero(b, b0), context.ON_star(b, b0)],
                {},
                mappings[0],
            )
        self.assertEqual(data, (True, False))
        self.assertIs(states[0]["tbl"], on.TABLE)

    def test_empty_or_unbound_data_fails_before_learning(self):
        with self.assertRaisesRegex(ValueError, "No loop-head"):
            InvInference(self.store, "absent", self.vocab, self.context)
        bad_vocab = InferenceVocabulary(2, ("ON_star",), ("missing",))
        with self.assertRaisesRegex(ValueError, "no binding"):
            InvInference(self.store, "stack", bad_vocab, self.context)
        with self.assertRaisesRegex(ValueError, "use_tbl"):
            to_inference_inputs(
                self.store, "stack", tower_vocabulary("reverse"), self.context
            )

    def test_inconsistent_snapshots_fail(self):
        with self.assertRaisesRegex(ValueError, "same objects"):
            LoopHeadState("x", {"x1": [0, 0, 0]}, {}, {})
        with self.assertRaisesRegex(ValueError, "Unresolved"):
            LoopHeadState("x", {"x1": [0, 0, 0]}, {"x1": [0, 0, 0]}, {"b": "missing"})
        with self.assertRaisesRegex(ValueError, "Invalid position"):
            observation_positions(np.zeros(10), 3)


class TraceEntryPoints(unittest.TestCase):
    def test_each_tower_entry_uses_explicit_store_and_its_vocabulary(self):
        store = DemoStore()

        class ReachedInference(Exception):
            pass

        for task in ("stack", "unstack", "reverse", "partial"):
            with self.subTest(task=task):
                module = importlib.import_module(
                    f"synthesis.entry.verify_{task}_with_learned_invariant"
                )
                entry = getattr(module, f"verify_{task}_program_with_learned_invariant")
                kwargs = {} if task == "stack" else {"inference_mode": "infinite"}
                with patch.object(
                    module, "InvInference", side_effect=ReachedInference
                ) as infer:
                    with self.assertRaises(ReachedInference):
                        entry(store, loop_id="recorded-loop", **kwargs)
                args = infer.call_args.args
                self.assertIs(args[0], store)
                self.assertEqual(args[1], "recorded-loop")
                self.assertEqual(args[2], tower_vocabulary(task))


if __name__ == "__main__":
    unittest.main()
