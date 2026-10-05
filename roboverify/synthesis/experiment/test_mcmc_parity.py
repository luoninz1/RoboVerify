"""Pin the instrumented search to the original's behaviour.

The instrumented ``MCMC`` is only trustworthy if it explores the same chain as
``synthesis.mcmc.synthesis.MCMC``. Both draw from the global ``random`` and
``numpy.random`` streams, so any extra RNG consumption added by instrumentation
would silently change the search. This test is what catches that.

Run with::

    uv run python -m unittest synthesis.experiment.test_mcmc_parity -v

The parity case generates a small deterministic simulator trace in memory. It
compares optimizer behavior and makes no claim that the trace solves a tower task.
"""

import os
import tempfile
import unittest

from synthesis.api import program
from synthesis.cfg.collection import record_execution
from synthesis.cfg.program_source import ProgramDefinition
from synthesis.experiment.config import MCMCConfig
from synthesis.experiment.mcmc import search
from synthesis.experiment.run_logger import RunLogger
from synthesis.mcmc import synthesis as original

PARITY_SEED = 4242
NUM_BLOCKS = 4
PROGRAM_SLOTS = 4
ITERS = 3
CEM_KWARGS = {"cem_N": 4, "cem_K": 2, "cem_iterations": 1}


class TestBMCReasonClassification(unittest.TestCase):
    """The reason labels drive the report histogram, so pin the mapping."""

    CASES = (
        ("BMC: disabled (no bmc_goal configured)", "disabled"),
        (
            "BMC: INFEASIBLE — no non-Skip instructions (5 program slots, 0 executable)",
            "empty_program",
        ),
        (
            "BMC: INFEASIBLE — goal references blocks not present in program (3); "
            "instructions: [Pick(0)]",
            "missing_block",
        ),
        (
            "BMC: INFEASIBLE — solver error (bad type); instructions: [Pick(0)]",
            "solver_error",
        ),
        ("BMC: FEASIBLE — 3 executable instruction(s): [Pick(0)]", "feasible"),
        ("BMC: INFEASIBLE — 3 executable instruction(s): [Pick(0)]", "infeasible"),
    )

    def test_each_branch_gets_its_own_label(self):
        for report, expected in self.CASES:
            with self.subTest(expected=expected):
                self.assertEqual(search.classify_bmc_report(report), expected)

    def test_infeasible_is_not_read_as_feasible(self):
        """ "FEASIBLE" is a substring of "INFEASIBLE"; ordering must handle it."""
        self.assertEqual(
            search.classify_bmc_report("BMC: INFEASIBLE — 1 executable"), "infeasible"
        )

    def test_unknown_report_is_labelled_not_crashed(self):
        self.assertEqual(
            search.classify_bmc_report("something else entirely"), "unknown"
        )


def load_expert_states():
    """Fresh motion data; no saved demonstrations or task oracle required."""
    seed = 29
    probe = program.Program(
        2, [program.Pick(0), program.Move(0, 0, 0, target_offset=[0, 0, 0.1])]
    )
    trace = record_execution(
        ProgramDefinition(probe, {"b0": 0}, "parity probe"),
        seed=seed,
        num_blocks=NUM_BLOCKS,
    )
    if trace.metadata["status"] != "completed":
        raise AssertionError(trace.metadata)
    return list(trace.states), [seed], NUM_BLOCKS, {seed: trace.snapshots[0]}


class TestMCMCParity(unittest.TestCase):
    """The instrumented search must reproduce the original's cost sequence."""

    def test_cost_sequence_matches_original(self):
        self.check_cost_sequence(saved_starts=False)

    def test_cost_sequence_matches_with_saved_starts(self):
        self.check_cost_sequence(saved_starts=True)

    def check_cost_sequence(self, *, saved_starts):
        expert_states, seeds, demo_blocks, snapshots = load_expert_states()
        initial_snapshots = snapshots if saved_starts else None
        num_blocks = demo_blocks or NUM_BLOCKS
        operands = {"Box": list(range(num_blocks))}
        instructions = [program.Pick, program.Move, program.Release]

        with tempfile.TemporaryDirectory() as tmp:
            # The original: BMC disabled, videos off, so scoring is the only work.
            original.set_np_seed(PARITY_SEED)
            _samples, original_costs = original.MCMC(
                program.Program(PROGRAM_SLOTS),
                operands,
                instructions,
                ITERS,
                expert_states=expert_states,
                num_seeds=len(seeds),
                num_block=num_blocks,
                save_dir=os.path.join(tmp, "original"),
                seeds=seeds,
                initial_snapshots=initial_snapshots,
                save_candidate_videos=False,
                **CEM_KWARGS,
            )

            # The instrumented version, configured to match: beta=1.0 is the
            # original acceptance rule, video_policy="none" keeps rollouts out of
            # the RNG stream, and refresh_best_metrics=False avoids the extra
            # objective evaluation that the original only performs when a goal
            # feature is configured.
            config = MCMCConfig(
                task="stack",
                num_blocks=num_blocks,
                iters=ITERS,
                program_slots=PROGRAM_SLOTS,
                beta=1.0,
                num_seeds=len(seeds),
                seeds=seeds,
                run_root=os.path.join(tmp, "runs"),
                video_policy="none",
                checkpoint_every=0,
                progress_every=0,
                capture_stdout=False,
                **CEM_KWARGS,
            )
            logger = RunLogger(
                config.run_root,
                "parity",
                config.to_json_dict(),
                slug="parity",
                capture_stdout=False,
            )
            original.set_np_seed(PARITY_SEED)
            result = search.MCMC(
                program.Program(PROGRAM_SLOTS),
                operands,
                instructions,
                config,
                expert_states,
                logger=logger,
                seeds=seeds,
                initial_snapshots=initial_snapshots,
                bmc_goal=None,
                goal_feature=None,
                refresh_best_metrics=False,
            )
            logger.finish("completed")

        self.assertEqual(len(result.costs), len(original_costs))
        for step, (new_cost, old_cost) in enumerate(zip(result.costs, original_costs)):
            with self.subTest(step=step - 1):
                self.assertAlmostEqual(
                    float(new_cost),
                    float(old_cost),
                    places=9,
                    msg=(
                        f"cost diverged at iteration {step - 1}: instrumented "
                        f"{new_cost} vs original {old_cost}. The instrumented "
                        "search consumed a different amount of randomness."
                    ),
                )


if __name__ == "__main__":
    unittest.main()
