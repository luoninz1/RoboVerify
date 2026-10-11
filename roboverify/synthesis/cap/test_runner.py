import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from synthesis.cap.runner import EvaluationConfig, feedback_for, run_evaluation, score


class Model:
    def health(self):
        return {"model": "fake/test"}

    def generate(self, messages, **settings):
        return {"text": "def policy():\n    pass\n"}


class Suite:
    def __init__(self):
        self.starts = {}
        self.calls = []
        self.closed = False

    def run(self, case, source, **settings):
        self.calls.append((case, source, settings))
        return {
            "case": case.as_dict(),
            "status": "goal_failed",
            "success": False,
            "error": None,
            "primitive_calls": 0,
            "physics_steps": 0,
            "elapsed_seconds": 0.01,
            "goal": {"score": 0.5, "failed": ["tower"]},
        }

    def close(self):
        self.closed = True


class RunnerTests(unittest.TestCase):
    def test_complete_paired_pipeline_and_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            suite = Suite()
            config = EvaluationConfig(
                tasks=("stack",),
                train_cases=1,
                validation_cases=1,
                test_cases=2,
                initial_candidates=1,
                repairs=1,
                cem_iterations=0,
                output_root=directory,
            )
            with patch(
                "synthesis.cap.runner.verify_tower_trace",
                return_value={"status": "unsupported"},
            ):
                path, results = run_evaluation(config, Model(), suite=suite)
            self.assertEqual(results["stack"]["cap_only"]["cases"], 2)
            self.assertEqual(results["stack"]["cap_roboverify"]["successes"], 0)
            self.assertTrue((path / "artifacts/episode_index.json").is_file())
            self.assertTrue((path / "artifacts/stack/selection.json").is_file())
            self.assertTrue((path / "result.json").is_file())
            metadata = json.loads((path / "config.json").read_text())["config"]
            self.assertEqual(metadata["implementation"], "google")
            self.assertEqual(metadata["generator"]["implementation"], "google")
            self.assertTrue(suite.closed)
            self.assertEqual(
                len(suite.calls), 4
            )  # identical source/state results shared

    def test_score_exact_success_is_primary(self):
        self.assertEqual(
            score([{"success": False, "goal": {"score": 1.0}}]), (0.0, 1.0)
        )

    def test_feedback_has_training_only_and_bounded_motion_checks(self):
        verification = {
            "status": "refuted",
            "cases": [
                {
                    "status": "refuted",
                    "checks": list(range(100)),
                    "failures": [{"obligation": "bad"}] * 20,
                }
            ],
        }
        feedback = feedback_for([], verification)
        self.assertNotIn("checks", feedback["verification"]["first_failed_trace"])
        self.assertEqual(
            len(feedback["verification"]["first_failed_trace"]["failures"]), 2
        )

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            EvaluationConfig(test_cases=0).validate()
        with self.assertRaises(ValueError):
            EvaluationConfig(implementation="unknown").validate()

    def test_legacy_generator_remains_selectable(self):
        from synthesis.cap.generator import Generator
        from synthesis.cap.runner import Evaluation

        config = EvaluationConfig(implementation="legacy")
        config.validate()
        self.assertIsInstance(
            Evaluation(config, Model(), Suite(), None).generator, Generator
        )

    def test_transport_error_can_enter_repair_without_missing_response_crash(self):
        class BrokenModel(Model):
            def generate(self, *args, **kwargs):
                raise ConnectionError("test transport failure")

        with tempfile.TemporaryDirectory() as directory:
            config = EvaluationConfig(
                tasks=("stack",),
                train_cases=1,
                validation_cases=1,
                test_cases=1,
                initial_candidates=1,
                repairs=1,
                cem_iterations=0,
                output_root=directory,
            )
            _, result = run_evaluation(config, BrokenModel(), suite=Suite())
            self.assertEqual(result["stack"]["cap_only"]["successes"], 0)
            self.assertEqual(result["stack"]["cap_roboverify"]["successes"], 0)


if __name__ == "__main__":
    unittest.main()
