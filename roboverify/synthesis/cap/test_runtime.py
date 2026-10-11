import unittest

from synthesis.cap.runtime import PolicyBudgetExceeded, PolicyError, Runtime


class RuntimeTests(unittest.TestCase):
    def test_pure_python_sorting_and_lambda(self):
        source = """
def policy():
    values = [3, 1]
    values.insert(1, 2)
    values.sort(key=lambda item: -item)
    assert values == [3, 2, 1]
    assert sorted(values) == [1, 2, 3]
    return sum(values)
"""
        self.assertEqual(Runtime({}).run(source), 6)

    def test_formatted_parameter_names(self):
        result = Runtime({}).run(
            'def policy():\n    obj="a"\n    return f"height_{obj}"\n'
        )
        self.assertEqual(result, "height_a")

    def test_helper_feedback_and_comprehension(self):
        state = {"a": 0, "b": 0}
        calls = []

        def pick(obj):
            calls.append(obj)
            state[obj] += 1

        runtime = Runtime({"pick": pick, "done": lambda obj: state[obj] >= 2})
        runtime.run("""
def finish(obj):
    while not done(obj):
        pick(obj)
def policy():
    objects = [x for x in ["a", "b"] if not done(x)]
    for obj in objects:
        finish(obj)
""")
        self.assertEqual(calls, ["a", "a", "b", "b"])

    def test_capabilities_are_not_replaceable(self):
        for source in (
            "pick = 1",
            "def pick():\n    pass",
            "def policy(pick):\n    pass",
        ):
            with self.subTest(source=source), self.assertRaises(PolicyError):
                Runtime({"pick": lambda: None}).run(source)

    def test_no_environment_or_import_access(self):
        for source in (
            "import os",
            "def policy():\n    env.step(0)",
            "def policy():\n    x = (1).__class__",
            'def policy():\n    open("file")',
        ):
            with self.subTest(source=source), self.assertRaises(PolicyError):
                Runtime({}).run(source)

    def test_loop_and_recursive_budgets(self):
        for source in (
            "def policy():\n    while True:\n        pass",
            "def policy():\n    policy()",
        ):
            with self.subTest(source=source), self.assertRaises(PolicyBudgetExceeded):
                Runtime({}, max_steps=50, max_call_depth=5).run(source)

    def test_short_circuit_does_not_call_unreachable_api(self):
        runtime = Runtime({"fail": lambda: self.fail("Should short circuit")})
        self.assertFalse(runtime.run("def policy():\n    return False and fail()"))

    def test_loop_else_and_local_mutation(self):
        result = Runtime({}).run("""
def policy():
    result = []
    for i in range(3):
        if i == 1:
            continue
        result.append(i)
    else:
        result.append(7)
    result[0] += 1
    return result
""")
        self.assertEqual(result, [1, 2, 7])

    def test_generator_short_circuit_and_allocation_budget(self):
        calls = []
        runtime = Runtime({"visit": lambda i: calls.append(i) or True})
        self.assertTrue(
            runtime.run("def policy():\n    return any(visit(i) for i in range(3))")
        )
        self.assertEqual(calls, [0])
        for expression in ("list(range(1000000000))", "[1] * 1000000000"):
            with self.subTest(expression=expression), self.assertRaises(
                PolicyBudgetExceeded
            ):
                Runtime({}, max_steps=100).run(
                    "def policy():\n    return " + expression
                )


if __name__ == "__main__":
    unittest.main()
