"""Trace placement and physical-decoding tests; no MuJoCo reliability claims."""

from dataclasses import asdict, FrozenInstanceError
import json
from types import SimpleNamespace
import unittest

import numpy as np

from synthesis.examples.robotic_insertion_sort import insertion_sort_blocks
from synthesis.examples.robotic_sort_traces import (
    PhysicalTraceRecorder, RoboticSortSnapshot, decode_locations,
    capture_simulator_state, restore_simulator_state, INPUT_FIELDS,
)
from synthesis.examples.test_robotic_insertion_sort import OccupancyRobot


class SortTraceTests(unittest.TestCase):
    def trace_algorithm(self, key_values):
        order = [f"b{i}" for i in range(len(key_values))]
        keys = dict(zip(order, key_values))
        slots = np.column_stack((np.ones(len(order)), np.arange(len(order)), np.zeros(len(order))))
        buffer = np.array([2.0, -1.0, 0.0])
        robot = OccupancyRobot(order, slots, buffer)
        events = []

        def observe(point, **state):
            # Every callback must follow completed logical/physical bookkeeping.
            robot.check_slots(list(state["order"]))
            events.append((point, state, dict(robot.occupancy)))

        insertion_sort_blocks(order, keys, slots, buffer, robot, observer=observe)
        return events

    def test_outer_heads_include_continue_and_explicit_exit(self):
        events = self.trace_algorithm([1, 1, 3, 4])
        self.assertEqual([(point, state["i"]) for point, state, _ in events],
                         [("outer_head", 1), ("outer_head", 2), ("outer_head", 3), ("outer_exit", 4)])
        self.assertTrue(all(state["j"] is None and state["selected"] is None for _, state, _ in events))

    def test_inner_heads_include_false_guard_and_correct_hole(self):
        events = self.trace_algorithm([3, 2, 1])
        inner = [state for point, state, _ in events if point == "inner_head"]
        self.assertEqual([(state["i"], state["j"]) for state in inner],
                         [(1, 0), (1, -1), (2, 1), (2, 0), (2, -1)])
        for state in inner:
            self.assertEqual(state["order"].count(None), 1)
            self.assertIsNone(state["order"][state["j"] + 1])
            self.assertNotIn(state["selected"], state["order"])

    def test_false_guard_can_stop_at_nonnegative_index(self):
        events = self.trace_algorithm([1, 3, 2])
        inner = [state for point, state, _ in events if point == "inner_head"]
        self.assertEqual([state["j"] for state in inner], [1, 0])
        self.assertEqual(inner[-1]["order"], ("b0", None, "b1"))

    def test_empty_and_singleton_have_only_exit(self):
        for keys in ([], [9]):
            events = self.trace_algorithm(keys)
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0][0], "outer_exit")
            self.assertEqual(events[0][1]["i"], len(keys))


class PhysicalSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.slots = np.array([[1.0, 0.0, 0.5], [1.0, 0.11, 0.5]])
        self.buffer = np.array([0.82, 0.0, 0.5])
        self.positions = {"a": self.slots[0].copy(), "b": self.slots[1].copy()}
        self.rotations = {name: np.eye(3) for name in self.positions}
        data = SimpleNamespace(get_site_xpos=self.positions.__getitem__, get_site_xmat=self.rotations.__getitem__)
        self.env = SimpleNamespace(object_names=["a", "b"], sim=SimpleNamespace(data=data))
        self.robot = SimpleNamespace(held=None, steps=0)

    def test_snapshot_is_immutable_and_copies_live_simulator_data(self):
        keys = {"a": 2, "b": 1}
        trace = PhysicalTraceRecorder(self.env, self.slots, self.buffer, keys, ("a", "b"), self.robot)
        trace("outer_head", i=1, j=None, selected=None, order=("a", "b"))
        state = trace.snapshots[0]
        self.positions["a"][:] = self.buffer
        self.rotations["a"][:] = 0
        keys["a"] = 99
        self.assertEqual(state.slot_map, {"a": 0, "b": 1})
        self.assertEqual(state.key_map["a"], 2)
        self.assertEqual(state.rotation_map["a"][2][2], 1)
        with self.assertRaises(FrozenInstanceError):
            state.i = 4
        self.assertEqual(RoboticSortSnapshot.from_dict(json.loads(json.dumps(asdict(state)))), state)

    def test_physical_locations_do_not_trust_logical_order(self):
        trace = PhysicalTraceRecorder(self.env, self.slots, self.buffer, {"a": 1, "b": 2}, ("a", "b"), self.robot)
        trace("outer_head", i=1, j=None, selected=None, order=("b", "a"))
        self.assertEqual(trace.snapshots[0].slot_map, {"a": 0, "b": 1})
        self.positions["a"][:] = self.buffer
        trace("inner_head", i=1, j=0, selected="a", order=(None, "b"))
        self.assertEqual(trace.snapshots[1].buffered, frozenset({"a"}))
        self.assertIsNone(trace.snapshots[1].slot_map["a"])

    def test_decoder_rejects_ambiguity_and_invalid_geometry(self):
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            decode_locations({"a": self.slots[0]}, [self.slots[0], self.slots[0]], self.buffer)
        with self.assertRaisesRegex(ValueError, "unlocated"):
            decode_locations({"a": [1.0, 0.0, 0.52]}, self.slots, self.buffer)
        with self.assertRaisesRegex(ValueError, "Invalid measured"):
            decode_locations({"a": [np.nan, 0.0, 0.5]}, self.slots, self.buffer)

    def test_full_baseline_restore_copies_state_and_inputs(self):
        state = {"time": 3.0, "qpos": np.array([1.0])}
        for field in INPUT_FIELDS:
            setattr(self.env.sim.data, field, np.array([2.0]))
        self.env.goal = np.array([4.0])
        self.env.sim.get_state = lambda: state
        restored = []
        self.env.sim.set_state = restored.append
        self.env.sim.forward = lambda: None
        baseline = capture_simulator_state(self.env)
        state["qpos"][0] = 99.0
        self.env.goal[:] = 99.0
        for field in INPUT_FIELDS:
            getattr(self.env.sim.data, field)[:] = 99.0
        restore_simulator_state(self.env, baseline)
        self.assertEqual(restored[0]["time"], 3.0)
        self.assertEqual(restored[0]["qpos"][0], 1.0)
        self.assertEqual(self.env.goal[0], 4.0)
        self.assertTrue(all(getattr(self.env.sim.data, field)[0] == 2.0 for field in INPUT_FIELDS))


if __name__ == "__main__":
    unittest.main()
