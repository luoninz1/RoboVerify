import unittest

from synthesis.cap.report import failure_kind


class ReportTests(unittest.TestCase):
    def test_controller_failure_is_separate_from_codegen(self):
        self.assertEqual(
            failure_kind(
                {"status": "execution_error", "error": "Unconverged MoveByName"}
            ),
            "controller_failure",
        )
        self.assertEqual(
            failure_kind(
                {
                    "status": "execution_error",
                    "error": "Unavailable attribute: position",
                }
            ),
            "language_or_api_violation",
        )

    def test_setup_and_budget_failures_keep_their_status(self):
        for status in (
            "setup_error",
            "timeout",
            "budget_exhausted",
            "generation_error",
            "goal_failed",
        ):
            self.assertEqual(failure_kind({"status": status}), status)


if __name__ == "__main__":
    unittest.main()
