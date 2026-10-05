import unittest

from synthesis.cfg.demo_validation import validate_demonstrations
from synthesis.cfg.demos import DemoSegment, DemoTrace
from synthesis.predicates.scene import Scene
from synthesis.predicates.term import atom, boolean, ref


class DemoValidationTests(unittest.TestCase):
    def setUp(self):
        self.low = Scene({0: (0, 0, 0), 1: (0.2, 0, 0)}, {"a": 1, "b": 0})
        self.on = Scene({0: (0, 0, 0), 1: (0, 0, 0.05)}, {"a": 1, "b": 0})
        self.post = atom("ON", ref("a"), ref("b"))

    def segment(self, *states):
        return DemoSegment(7, 0, len(states) - 1, DemoTrace(states))

    def test_transient_success_does_not_validate_demo(self):
        result = validate_demonstrations(
            [self.segment(self.low, self.on, self.low)], boolean(True), self.post
        )
        self.assertFalse(result)
        self.assertEqual((result.post_reached, result.post_at_end), (1, 0))
        self.assertEqual(result.issues[0]["demo"], 7)
        self.assertTrue(result.issues[0]["post_reached_transiently"])

    def test_initial_condition_and_empty_input_are_checked(self):
        self.assertFalse(validate_demonstrations([], boolean(True), self.post))
        result = validate_demonstrations(
            [self.segment(self.low, self.on)], self.post, self.post
        )
        self.assertFalse(result)
        self.assertEqual(result.issues[0]["reason"], "Initial task condition is false")

    def test_task_correct_demo_is_accepted_without_oracle(self):
        result = validate_demonstrations(
            [self.segment(self.low, self.on)], boolean(True), self.post
        )
        self.assertTrue(result)
        self.assertEqual(result.post_at_end, 1)

    def test_cli_rejects_bad_demo_before_search(self):
        from types import SimpleNamespace
        from unittest.mock import Mock, patch

        from synthesis.cfg.tasks import task_identity
        from synthesis.entry.synthesize_cfg import run

        trace = DemoTrace(
            (self.low, self.on, self.low),
            task="stack",
            num_blocks=2,
            metadata={
                "task_spec": task_identity("stack"),
                "initial_bindings": {"b0": 0},
            },
        )
        args = SimpleNamespace(demos="unused.npz", task="stack", num_blocks=2)
        logger = Mock()
        with patch(
            "synthesis.entry.synthesize_cfg.load_traces", return_value=[trace]
        ), patch("synthesis.entry.synthesize_cfg.verified_synthesis") as search:
            self.assertEqual(run(args, logger), 2)
        search.assert_not_called()
        self.assertEqual(logger.finish.call_args.args[0], "invalid_demonstrations")
