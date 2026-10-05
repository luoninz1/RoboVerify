"""Generated scenes only: never depend on saved demonstration archives."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import z3

from synthesis.cfg.collection import record_execution
from synthesis.cfg.recordings import load_traces, save_traces
from synthesis.cfg.reset import restore
from synthesis.experiment.invariant_learning.tasks import StackExperiment
from synthesis.experiment.invariant_learning.witness import WitnessSearch
from synthesis.mcmc.synthesis import make_roboverify_stack_env


class SimulatorTests(unittest.TestCase):
    def test_preservation_failure_replays_descending_ids_and_rejects_wrong_target(self):
        with contextlib.redirect_stdout(io.StringIO()):
            task = StackExperiment("synthesis.examples.stack:build_program")
            ctx = task.context
            x, y, b, base = [ctx.get_consts(name) for name in ("x", "y", "b", "b0")]
            o, h, scattered = ctx.ON_star, ctx.Higher, ctx.Scattered
            # Regression candidate from the two-block stage: its three-block
            # preservation failure must be reproduced by the first placement.
            task.set_invariant(
                z3.ForAll(
                    [x, y],
                    z3.And(
                        z3.Or(o(x, y), h(y, x)),
                        z3.Implies(o(x, b), x == b),
                        z3.Implies(o(b, x), o(x, base)),
                        z3.Or(o(y, x), h(x, b)),
                        z3.Implies(h(y, x), z3.Or(o(y, x), scattered(x, y))),
                    ),
                )
            )
            query = task.search_query(3, 10000)
            path = next(
                p
                for p in query.paths
                if p.failure_kind == "preserve" and p.head_iteration == 0
            )
            query.solver.add(
                path.condition,
                path.choices[0].values[0] == query.context.enum_blocks[2],
            )
            self.assertEqual(query.solver.check(), z3.sat)
            model = query.solver.model()
            search = WitnessSearch(
                "found",
                size=3,
                query=query,
                witness=query.decode(model),
                plan=path.decode(model, query.context),
            )
            self.assertEqual(
                [c["bindings"]["b_prime"] for c in search.plan.guard_choices], [2]
            )
            trace = task.execute(search, seed=0, timeout_seconds=60, video_path=None)
            self.assertTrue(task.validate(trace, search), trace.metadata)
            heads = [e for e in trace.events if e["kind"] == "loop_head"]
            self.assertEqual([e["bindings"]["b_prime"] for e in heads], [2, 1])
            self.assertTrue(trace.metadata["counterexample_reproduced"])
            # A trajectory with some missing state is insufficient: the claimed
            # failure must occur at the planned head, where I is still true.
            search.plan.head_iteration = 1
            trace.metadata["status"] = (
                "completed"  # validate_trace consumes this status
            )
            self.assertFalse(task.validate(trace, search))
            self.assertEqual(
                trace.metadata["failure_kind"], "counterexample_not_reproduced"
            )

    def test_generated_snapshot_preparation_archive_and_replay(self):
        with contextlib.redirect_stdout(io.StringIO()):
            task = StackExperiment("synthesis.examples.stack:build_program")
            query = task.search_query(2, 10000)
            query.solver.add(query.coverage)
            import z3

            self.assertEqual(query.solver.check(), z3.sat)
            search = WitnessSearch(
                "found",
                size=2,
                query=query,
                witness=query.decode(query.solver.model()),
                plan=query.replay_plan(query.solver.model()),
            )
            trace = task.execute(search, seed=0, timeout_seconds=60, video_path=None)
            self.assertTrue(task.validate(trace, search), trace.metadata)
            self.assertEqual(
                trace.metadata["counterexample_initialization"]["settling_steps"], 50
            )
            self.assertEqual(
                trace.metadata["initialization"],
                {"source": "snapshot", "settling_steps": 0},
            )
            self.assertEqual(len(task.learning_states(trace)), 2)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "trace.npz"
                save_traces(path, [trace])
                restored = load_traces(path, require_valid=True)[0]
            env = make_roboverify_stack_env(num_blocks=2)
            try:
                observation = restore(env, restored.snapshots[0])
                np.testing.assert_allclose(
                    observation, trace.states[0], atol=1e-8, rtol=0
                )
            finally:
                env.close()
            replay = record_execution(
                task.definition,
                seed=999,
                num_blocks=2,
                initial_snapshot=restored.snapshots[0],
                max_loop_iterations=1,
            )
            self.assertTrue(task.validate(replay), replay.metadata)
            np.testing.assert_allclose(replay.states, trace.states, atol=1e-8, rtol=0)
            np.testing.assert_allclose(
                replay.actions[0], trace.actions[0], atol=1e-8, rtol=0
            )
            for state in task.learning_states(replay):
                self.assertEqual(
                    state.entry_positions,
                    task.learning_states(trace)[0].entry_positions,
                )


if __name__ == "__main__":
    unittest.main()
