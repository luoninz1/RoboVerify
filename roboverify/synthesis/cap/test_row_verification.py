import unittest

from synthesis.cap.row_verification import verify_row_policy

SWAP = """
def exchange(a, b):
    ta, tb = target(a), target(b)
    tmp = buffer_target()
    pick(a)
    move(tmp)
    release(tmp)
    pick(b)
    move(ta)
    release(ta)
    pick(a)
    move(tb)
    release(tb)

def policy():
    order = []
    for obj in objects():
        index = 0
        while index < len(order) and next_star(order[index], obj):
            index += 1
        order.insert(index, obj)
    for i in range(1, len(order)):
        j = i
        while j > 0 and keyless(order[j], order[j-1]):
            exchange(order[j-1], order[j])
            order[j-1], order[j] = order[j], order[j-1]
            j -= 1
""".replace("order.insert(index, obj)", "order = order[:index] + [obj] + order[index:]")


def scene(task="insertion_sort"):
    return {
        "case": {"task": task},
        "objects": ["a", "b", "c"],
        "initial_order": ["c", "a", "b"],
        "roles": {"base": "c"},
    }


class RowVerificationTests(unittest.TestCase):
    def test_sort_arbitrary_keys_and_ties(self):
        result = verify_row_policy(SWAP, scene())
        self.assertEqual(result["status"], "valid", result)
        self.assertGreater(result["completed_paths"], 1)

    def test_noop_has_concrete_counterexample(self):
        result = verify_row_policy("def policy():\n    pass\n", scene())
        self.assertEqual(result["status"], "refuted")
        order = scene()["initial_order"]
        self.assertTrue(
            any(result["keys"][a] > result["keys"][b] for a, b in zip(order, order[1:]))
        )

    def test_occupied_destination_and_buffer_fail(self):
        result = verify_row_policy(
            "def policy():\n    dest = target('b')\n    pick('a')\n    release(dest)\n",
            scene(),
        )
        self.assertEqual(result["status"], "refuted")
        self.assertIn("occupied", result["error"])

    def test_restore_needs_sorted_checkpoint(self):
        result = verify_row_policy("def policy():\n    pass\n", scene("sort_restore"))
        self.assertEqual(result["status"], "refuted")

    def test_exhausted_budget_is_not_valid(self):
        self.assertEqual(
            verify_row_policy(SWAP, scene(), max_paths=1)["status"], "budget_exhausted"
        )


if __name__ == "__main__":
    unittest.main()
