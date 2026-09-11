"""Restricted-program semantics and physical occupancy sequencing tests.

These tests use a deterministic fake robot. They validate program/trace
semantics, not the contact/grasp reliability of the MuJoCo controller.
"""

import ast
from dataclasses import FrozenInstanceError
import inspect
import itertools
import json
import unittest

import numpy as np

from synthesis.examples.relational_robotic_sort import (
    RelationalScene, RelationalSortSnapshot, decode_locations,
    insertion_sort_relational,
)


class OccupancyRobot:
    """Realistic vacancy/held-state preconditions for the three primitives."""

    def __init__(self, order, slots, buffer_position):
        self.positions = {name: np.asarray(slot).copy() for name, slot in zip(order, slots)}
        self.slots = np.asarray(slots)
        self.buffer_position = np.asarray(buffer_position)
        self.held = None
        self.steps = 0
        self.description = ""
        self.calls = []
        self.transfers = []

    def position(self, name):
        return self.positions[name].copy()

    def pick(self, name):
        if self.held is not None:
            raise AssertionError("Picking while holding another object")
        self.held = name
        self.source = self.positions[name].copy()
        self.positions[name] = self.source + [0, 0, .2]
        self.calls.append(("pick", name))
        self.steps += 1

    def move(self, target):
        if self.held is None:
            raise AssertionError("Moving without a held object")
        self.positions[self.held] = np.asarray(target) + [0, 0, .2]
        self.calls.append(("move", self.held))
        self.steps += 1

    def release(self, target):
        if self.held is None:
            raise AssertionError("Releasing without a held object")
        target = np.asarray(target)
        if any(name != self.held and np.linalg.norm(pos - target) < .03
               for name, pos in self.positions.items()):
            raise AssertionError("Release destination is already occupied")
        self.positions[self.held] = target.copy()
        self.transfers.append((self.held, self.source.copy(), target.copy()))
        self.calls.append(("release", self.held))
        self.held = None
        self.steps += 1


def make_fake_case(order, keys):
    slots = np.column_stack((np.ones(len(order)), .11 * np.arange(len(order)),
                             np.full(len(order), .4)))
    buffer = np.array([.82, .165, .4])
    robot = OccupancyRobot(order, slots, buffer)
    scene = RelationalScene(tuple(keys), keys, slots, buffer, robot, robot.position,
                            initial_order=order, case_name="fake")
    return robot, scene


def assert_prior_invariants(test, snapshot):
    """Check the previous response's formulas, directly over every u,v binding."""
    names = snapshot.objects
    reachable = snapshot.next_star
    keys = snapshot.key_map
    test.assertCountEqual(snapshot.order, names)
    test.assertIsNone(snapshot.held)
    test.assertEqual(snapshot.buffered, ())
    if snapshot.program_point.startswith("outer"):
        for u, v in itertools.product(names, repeat=2):
            if reachable(u, v) and reachable(v, snapshot.b):
                test.assertGreaterEqual(keys[v], keys[u])
    else:
        b, x, r = snapshot.b, snapshot.x, snapshot.r
        test.assertNotEqual(b, x)
        for u in names:
            test.assertEqual(reachable(u, r), reachable(u, b) or u == x)
            if u != x and reachable(x, u) and reachable(u, r):
                test.assertLess(keys[x], keys[u])
        for u, v in itertools.product(names, repeat=2):
            if u != x and v != x and reachable(u, v) and reachable(v, r):
                test.assertGreaterEqual(keys[v], keys[u])


class RelationalRoboticSortTests(unittest.TestCase):
    def check_case(self, order, keys):
        robot, scene = make_fake_case(order, keys)
        result = insertion_sort_relational(order[0], scene, robot)
        self.assertIsNone(result)
        self.assertEqual(scene.order, tuple(sorted(order, key=keys.__getitem__)))
        self.assertEqual(scene.snapshots[-1].program_point, "outer_exit")
        self.assertEqual(sum(s.program_point == "outer_head" for s in scene.snapshots), len(order) - 1)
        self.assertEqual(sum(s.program_point == "inner_exit" for s in scene.snapshots), len(order) - 1)
        for snapshot in scene.snapshots:
            assert_prior_invariants(self, snapshot)
        self.assertEqual(len(robot.calls) % 9, 0)
        for offset in range(0, len(robot.calls), 9):
            self.assertEqual([name for name, _ in robot.calls[offset:offset+9]],
                             ["pick", "move", "release"] * 3)
            self.assertEqual(robot.calls[offset][1], robot.calls[offset+6][1])
        return robot, scene

    def test_every_distinct_and_tied_four_block_permutation(self):
        for keys in ({"a": 1, "b": 2, "c": 3, "d": 4},
                     {"a": 1, "b": 2, "c": 1, "d": 2},
                     {"a": 2, "b": 2, "c": 2, "d": 2}):
            for order in itertools.permutations(keys):
                with self.subTest(keys=keys, order=order):
                    self.check_case(order, keys)

    def test_all_ternary_key_assignments(self):
        objects = tuple("abcd")
        for values in itertools.product(range(3), repeat=4):
            with self.subTest(keys=values):
                self.check_case(objects, dict(zip(objects, values)))

    def test_singleton_no_transfers(self):
        robot, scene = self.check_case(("a",), {"a": 5})
        self.assertEqual(robot.calls, [])
        self.assertEqual(len(scene.snapshots), 1)

    def test_reverse_order_has_six_swaps_and_eighteen_transfers(self):
        robot, scene = self.check_case(tuple("abcd"), dict(zip("abcd", [4, 3, 2, 1])))
        self.assertEqual(len(robot.transfers), 18)
        self.assertEqual(sum(s.program_point == "inner_head" for s in scene.snapshots), 6)
        for offset in range(0, len(robot.transfers), 3):
            save, predecessor, restore = robot.transfers[offset:offset+3]
            np.testing.assert_array_equal(save[2], scene.buffer_position)
            np.testing.assert_array_equal(predecessor[2], save[1])
            np.testing.assert_array_equal(restore[1], scene.buffer_position)
            np.testing.assert_array_equal(restore[2], predecessor[1])

    def test_guard_reads_actual_poses_not_initial_order(self):
        robot, scene = make_fake_case(tuple("abc"), {"a": 1, "b": 2, "c": 3})
        # External rearrangement before the guard must change its witness.
        robot.positions["b"], robot.positions["c"] = robot.positions["c"], robot.positions["b"]
        self.assertEqual(scene.outer_guard("a"), "c")
        self.assertEqual(scene.snapshots[-1].order, tuple("acb"))

    def test_snapshot_json_roundtrip_and_immutability(self):
        _, scene = self.check_case(tuple("ab"), {"a": 2, "b": 1})
        original = scene.snapshots[1]
        restored = RelationalSortSnapshot.from_dict(json.loads(json.dumps(original.as_dict())))
        self.assertEqual(restored, original)
        with self.assertRaises(FrozenInstanceError):
            restored.b = "other"
        detached = restored.key_map
        detached["a"] = -9
        self.assertEqual(restored.key_map["a"], 2)

    def test_ambiguous_missing_duplicate_and_buffered_occupancy_rejected(self):
        robot, scene = make_fake_case(tuple("ab"), {"a": 2, "b": 1})
        original = robot.positions["a"].copy()
        robot.positions["a"] = np.array([9., 9., 9.])
        with self.assertRaises(ValueError):
            scene.outer_guard("a")
        robot.positions["a"] = robot.positions["b"].copy()
        with self.assertRaises(ValueError):
            scene.outer_guard("a")
        robot.positions["a"] = scene.buffer_position.copy()
        with self.assertRaises(ValueError):
            scene.outer_guard("a")
        with self.assertRaises(ValueError):
            decode_locations({"a": original}, [original, original], scene.buffer_position)

    def test_program_ast_matches_restricted_assignments_and_inline_primitives(self):
        tree = ast.parse(inspect.getsource(insertion_sort_relational))
        self.assertEqual(sum(isinstance(node, ast.While) for node in ast.walk(tree)), 2)
        self.assertFalse(any(isinstance(node, (ast.If, ast.IfExp, ast.For, ast.Subscript))
                             for node in ast.walk(tree)))
        function = tree.body[0]
        assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)]
        self.assertEqual({(node.targets[0].id, node.value.id) for node in assignments},
                         {("b", "head"), ("r", "x"), ("r", "b"), ("b", "r")})
        self.assertEqual(len(assignments), 4)
        self.assertEqual({node.target.id for node in ast.walk(tree) if isinstance(node, ast.NamedExpr)},
                         {"x", "p"})
        outer = next(node for node in function.body if isinstance(node, ast.While))
        inner = next(node for node in outer.body if isinstance(node, ast.While))
        self.assertEqual(ast.unparse(outer.body[-1]), "b = r")
        self.assertEqual(ast.unparse(inner.body[-1]), "r = b")
        methods = [node.value.func.attr for node in inner.body if isinstance(node, ast.Expr)]
        self.assertEqual(methods, ["pick", "move", "release"] * 3)


if __name__ == "__main__":
    unittest.main()
