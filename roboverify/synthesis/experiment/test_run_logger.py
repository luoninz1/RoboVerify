"""Tests for the run-directory contract.

These are fast and need no simulator: run with

    uv run python -m unittest synthesis.experiment.test_run_logger -v
"""

import json
import tempfile
import unittest
from pathlib import Path

from synthesis.experiment.run_logger import RollingRate, RunLogger


class TestRollingRate(unittest.TestCase):
    def test_empty_window_has_no_rate(self):
        self.assertIsNone(RollingRate(5).rate)

    def test_window_slides(self):
        rate = RollingRate(4)
        for value in (True, True, True, True):
            rate.add(value)
        self.assertEqual(rate.rate, 1.0)
        for value in (False, False):
            rate.add(value)
        self.assertEqual(rate.rate, 0.5)


class TestRunLogger(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def make_logger(self, **kwargs) -> RunLogger:
        kwargs.setdefault("capture_stdout", False)
        return RunLogger(self.root, "unittest", {"iters": 3}, **kwargs)

    def test_status_stays_a_single_object(self):
        """status.json must be overwritten, never appended.

        This is the property that makes reading a run cost the same at iteration
        10 and 10,000, so it is worth pinning explicitly.
        """
        logger = self.make_logger()
        for step in range(100):
            logger.set_progress(step, 100, best_cost=-float(step))
        path = logger.run_dir / "status.json"
        payload = json.loads(path.read_text())  # raises if more than one object
        self.assertEqual(payload["iter"], 99)
        self.assertEqual(payload["total"], 100)
        self.assertEqual(payload["pct"], 100.0)
        self.assertLess(path.stat().st_size, 2000)
        logger.finish("completed")

    def test_metrics_line_count_matches_calls(self):
        logger = self.make_logger()
        for step in range(25):
            logger.log_metrics(step, cost=-0.5)
        logger.finish("completed")
        lines = (logger.run_dir / "metrics.jsonl").read_text().strip().splitlines()
        self.assertEqual(len(lines), 25)
        self.assertEqual(json.loads(lines[-1])["iter"], 24)

    def test_events_are_rate_limited_per_kind(self):
        logger = self.make_logger(event_min_step_gap=10)
        written = [logger.log_event("noisy", "again", step=step) for step in range(30)]
        # First at step 0, then only once the gap has elapsed.
        self.assertEqual(sum(written), 3)
        self.assertTrue(logger.log_event("noisy", "forced", step=5, force=True))
        logger.finish("completed")

    def test_forced_events_are_never_dropped(self):
        logger = self.make_logger(event_min_step_gap=1000)
        for step in range(5):
            self.assertTrue(logger.log_event("new_best", "best", step=step, force=True))
        logger.finish("completed")

    def test_finish_marks_run_not_alive_and_writes_result(self):
        logger = self.make_logger()
        self.assertTrue(
            json.loads((logger.run_dir / "status.json").read_text())["alive"]
        )
        logger.finish("completed", best_cost=-0.25)
        status = json.loads((logger.run_dir / "status.json").read_text())
        result = json.loads((logger.run_dir / "result.json").read_text())
        self.assertFalse(status["alive"])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["best_cost"], -0.25)

    def test_finish_is_idempotent(self):
        logger = self.make_logger()
        logger.finish("completed")
        logger.finish("failed")  # must not overwrite the recorded outcome
        result = json.loads((logger.run_dir / "result.json").read_text())
        self.assertEqual(result["status"], "completed")

    def test_context_manager_records_an_exception(self):
        with self.assertRaises(ValueError):
            with self.make_logger() as logger:
                run_dir = logger.run_dir
                raise ValueError("boom")
        result = json.loads((run_dir / "result.json").read_text())
        self.assertEqual(result["status"], "failed")
        self.assertIn("boom", result["exit_reason"])
        events = [
            json.loads(line)
            for line in (run_dir / "events.jsonl").read_text().splitlines()
        ]
        exceptions = [event for event in events if event["kind"] == "exception"]
        self.assertEqual(len(exceptions), 1)
        self.assertTrue(Path(exceptions[0]["traceback_path"]).exists())

    def test_latest_symlink_points_at_newest_run(self):
        first = self.make_logger()
        first.finish("completed")
        second = self.make_logger(slug="second")
        second.finish("completed")
        latest = Path(self.root) / "unittest" / "latest"
        self.assertTrue(latest.is_symlink())
        self.assertEqual(latest.resolve(), second.run_dir.resolve())

    def test_config_records_git_and_versions(self):
        logger = self.make_logger()
        config = json.loads((logger.run_dir / "config.json").read_text())
        logger.finish("completed")
        self.assertIn("sha", config["git"])
        self.assertIn("python", config["versions"])
        self.assertEqual(config["config"]["iters"], 3)


if __name__ == "__main__":
    unittest.main()


class FinalizedLoggerExceptionTests(unittest.TestCase):
    def test_finished_context_preserves_original_exception(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RuntimeError, "original failure"):
                with RunLogger(root, "finished", {}, capture_stdout=False) as logger:
                    logger.finish("failed")
                    raise RuntimeError("original failure")
