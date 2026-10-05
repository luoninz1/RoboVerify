"""Report verification outcomes without promoting inconclusive runs."""

import json
import tempfile
import unittest
from pathlib import Path

from synthesis.experiment.report import RunSummary


class VerificationReportTests(unittest.TestCase):
    def test_verified_model_is_reported_with_both_proof_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "result.json").write_text(
                json.dumps(
                    {
                        "status": "verified_model",
                        "formal_verification": "verified_model",
                        "symbolic": "True (unbounded)",
                        "motion": "True (mode=noiseless)",
                    }
                )
            )
            summary = RunSummary(path)
            self.assertFalse(summary.looks_unhealthy)
            report = summary.render()
            self.assertNotIn("FAILED", report)
            self.assertIn("True (unbounded)", report)
            self.assertIn("True (mode=noiseless)", report)

    def test_symbolic_only_experiment_is_success_without_motion_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "result.json").write_text(
                json.dumps(
                    dict(
                        status="verified_symbolic",
                        symbolic_status="verified",
                        motion_status="not_requested",
                        verification_attempts=4,
                        counterexample_executions=3,
                        learner_updates=3,
                    )
                )
            )
            summary = RunSummary(path)
            self.assertFalse(summary.looks_unhealthy)
            report = summary.render()
            self.assertIn("not_requested", report)
            self.assertIn("4 proof attempts", report)
            self.assertNotIn("FAILED", report)

    def test_inconclusive_verification_is_still_unsuccessful(self):
        for status in (
            "unknown",
            "needs_demonstrations",
            "motion_unverified",
            "symbolic_budget_exhausted",
        ):
            with self.subTest(
                status=status
            ), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                (path / "result.json").write_text(json.dumps({"status": status}))
                summary = RunSummary(path)
                self.assertTrue(summary.looks_unhealthy)
                self.assertIn(f"FAILED ({status})", summary.render())


if __name__ == "__main__":
    unittest.main()
