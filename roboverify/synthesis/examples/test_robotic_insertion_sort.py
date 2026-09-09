"""Algorithm checks using a physical slot/buffer occupancy model.

These tests check sorting and sequencing, not MuJoCo grasp reliability.
"""

import itertools
import unittest

import numpy as np

from synthesis.examples.robotic_insertion_sort import insertion_sort_blocks


class OccupancyRobot:
    """Reject a transfer unless its destination is physically empty."""

    def __init__(self, order, slots, buffer_position):
        self.slots = np.asarray(slots)
        self.buffer_position = np.asarray(buffer_position)
        self.occupancy = {i: name for i, name in enumerate(order)}
        self.occupancy["buffer"] = None
        self.transfers = []
        self.checks = 0

    def _destination(self, target):
        for i, slot in enumerate(self.slots):
            if np.array_equal(slot, target):
                return i
        if np.array_equal(self.buffer_position, target):
            return "buffer"
        raise AssertionError(f"Unknown transfer target: {target}")

    def transfer(self, name, target, description):
        sources = [slot for slot, block in self.occupancy.items() if block == name]
        if len(sources) != 1:
            raise AssertionError(f"Block {name} is missing or duplicated: {self.occupancy}")
        source = sources[0]
        destination = self._destination(target)
        if self.occupancy[destination] is not None:
            raise AssertionError(f"Occupied destination {destination}: {self.occupancy}")
        self.occupancy[source] = None
        self.occupancy[destination] = name
        self.transfers.append((name, source, destination))

    def check_slots(self, order):
        self.checks += 1
        if list(order) != [self.occupancy[i] for i in range(len(self.slots))]:
            raise AssertionError("Logical slot mapping differs from physical occupancy.")
        # A temporary empty row slot must correspond to the buffered selection.
        if order.count(None) != int(self.occupancy["buffer"] is not None):
            raise AssertionError("Buffer and row-vacancy state are inconsistent.")


class InsertionSortBlocksTests(unittest.TestCase):
    def run_sort(self, original, keys):
        order = list(original)
        slots = np.column_stack((np.ones(len(order)), np.arange(len(order)), np.zeros(len(order))))
        buffer_position = np.array([2.0, -1.0, 0.0])
        robot = OccupancyRobot(order, slots, buffer_position)
        result = insertion_sort_blocks(order, keys, slots, buffer_position, robot)
        self.assertIs(result, order)
        self.assertEqual(order, sorted(original, key=keys.__getitem__))
        self.assertEqual(set(order), set(original))
        self.assertIsNone(robot.occupancy["buffer"])
        robot.check_slots(order)
        return robot

    def test_all_24_distinct_key_orders_respect_physical_occupancy(self):
        keys = {f"object{i}": i + 1 for i in range(4)}
        for original in itertools.permutations(keys):
            with self.subTest(order=original):
                self.run_sort(original, keys)

    def test_all_24_duplicate_key_orders_are_stable(self):
        keys = {"a": 2, "b": 1, "c": 2, "d": 1}
        for original in itertools.permutations(keys):
            with self.subTest(order=original):
                # Python's sorted is stable, so equality checks identity order
                # within each equal-key group, not merely sorted key values.
                self.run_sort(original, keys)

    def test_reverse_order_uses_buffer_and_single_vacancy_shifts(self):
        keys = {"a": 4, "b": 3, "c": 2, "d": 1}
        robot = self.run_sort(tuple(keys), keys)
        self.assertEqual(len(robot.transfers), 12)  # Six shifts plus three saves/inserts.
        self.assertEqual(sum(dst == "buffer" for _, _, dst in robot.transfers), 3)
        self.assertEqual(sum(src == "buffer" for _, src, _ in robot.transfers), 3)

    def test_already_sorted_and_trivial_orders_do_not_move(self):
        for original, keys in (([], {}), (["a"], {"a": 1}),
                               (["a", "b", "c"], {"a": 1, "b": 1, "c": 2})):
            with self.subTest(order=original):
                robot = self.run_sort(original, keys)
                self.assertEqual(robot.transfers, [])


if __name__ == "__main__":
    unittest.main()
