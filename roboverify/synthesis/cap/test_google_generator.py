import ast
import contextlib
import hashlib
import io
import json
import unittest

from synthesis.cap.google_generator import (
    PROVENANCE,
    VENDOR,
    GoogleGenerator,
    _load_core,
    normalize_policy,
    provenance,
)
from synthesis.cap.runtime import PolicyBudgetExceeded, Runtime
from synthesis.cap.test_generator import Client


class GoogleGeneratorTests(unittest.TestCase):
    def generate(self, outputs, task="stack", **options):
        client = Client(outputs)
        with contextlib.redirect_stdout(io.StringIO()):
            result = GoogleGenerator(client, **options).generate(task, seed=10)
        return result, client

    def test_verbatim_source_snapshot_and_license(self):
        data = (VENDOR / "google_cap.py").read_text()
        cell = data.split("import ast\n\n", 1)[1][:-1]
        self.assertEqual(
            hashlib.sha256(cell.encode()).hexdigest(), PROVENANCE["source_cell_sha256"]
        )
        self.assertEqual(
            hashlib.sha256((VENDOR / "LICENSE").read_bytes()).hexdigest(),
            PROVENANCE["license_sha256"],
        )
        self.assertEqual(provenance()["implementation"], "google")
        self.assertEqual(provenance()["upstream_commit"], PROVENANCE["commit"])

    def test_upstream_parser_keeps_last_signature_and_assignment(self):
        core = _load_core()
        core.astunparse = type("Unparser", (), {"unparse": staticmethod(ast.unparse)})
        calls, assignments = {}, {}
        core.FunctionParser(calls, assignments).visit(
            ast.parse("a = missing(1)\nb = missing(2)")
        )
        self.assertEqual(calls["missing"], "missing(2)")
        self.assertEqual(assignments["missing"], "b = missing(2)")
        result, client = self.generate(
            [
                "a = missing(1)\nb = missing(2)",
                "def missing(value):\n    return value",
            ]
        )
        self.assertEqual(result["status"], "generated", result.get("error"))
        self.assertIn(
            "# define function: b = missing(2).", client.requests[1][0][1]["content"]
        )

    def test_upstream_recursive_helpers_and_rebinding_execute_in_interpreter(self):
        result, client = self.generate(
            [
                "transfer(base)",
                "def transfer(obj):\n    inspect(obj)",
                "def inspect(obj):\n    pick(obj)\n    move(obj,obj,obj,(0,0,.15))\n    release(obj)",
            ]
        )
        self.assertEqual(result["status"], "generated", result.get("error"))
        self.assertEqual(len(result["generation"]), 3)
        self.assertEqual(list(result["helper_sources"]), ["transfer", "inspect"])
        self.assertEqual([r[1]["seed"] for r in client.requests], [10, 11, 12])
        calls = []
        Runtime(
            {
                "pick": lambda obj: calls.append(("pick", obj)),
                "move": lambda *args: calls.append(("move", args)),
                "release": lambda obj: calls.append(("release", obj)),
            }
        ).run(result["source"], inputs={"base": "b0"})
        self.assertEqual([r[0] for r in calls], ["pick", "move", "release"])
        self.assertEqual(result["raw_source"], "transfer(base)")
        self.assertIn("# task: Solve", client.requests[0][0][1]["content"])

    def test_inline_helpers_local_aliases_and_recursive_self_do_not_regenerate(self):
        result, _ = self.generate(
            [
                "def local(obj):\n    f = lambda value: value\n    return f(obj)\nlocal(base)"
            ]
        )
        self.assertEqual(result["status"], "generated", result.get("error"))
        self.assertEqual(len(result["generation"]), 1)
        result, _ = self.generate(
            [
                "repeat(base)",
                "def repeat(obj):\n    repeat(obj)",
            ]
        )
        self.assertEqual(result["status"], "generated", result.get("error"))
        self.assertEqual(len(result["generation"]), 2)
        with self.assertRaises(PolicyBudgetExceeded):
            Runtime({}, max_call_depth=3).run(result["source"], inputs={"base": "x"})

    def test_existing_policy_and_optional_call_execute_once(self):
        result, _ = self.generate(["def policy():\n    pick(base)\npolicy()"])
        self.assertEqual(result["status"], "generated", result.get("error"))
        calls = []
        Runtime({"pick": calls.append}).run(result["source"], inputs={"base": "x"})
        self.assertEqual(calls, ["x"])
        self.assertEqual(
            normalize_policy("policy()", {"policy": "def policy():\n    pass"}),
            "def policy():\n    pass\n",
        )

    def test_generation_never_executes_primitive_calls_or_helper_top_level_statements(
        self,
    ):
        result, _ = self.generate(["pick(base)"])
        self.assertEqual(result["status"], "generated")
        result, _ = self.generate(
            ["missing(base)", "pick(base)\ndef missing(obj):\n    pass"]
        )
        self.assertEqual(result["status"], "generation_error")
        self.assertIn("only function definitions", result["error"])

    def test_bad_code_and_api_shadowing_are_archived_not_repaired(self):
        for source in (
            "import os",
            "pick = 1",
            "def move(obj):\n    pass",
            "def helper(pick):\n    pass",
            "obj.__class__",
        ):
            with self.subTest(source=source):
                result, _ = self.generate([source])
                self.assertEqual(result["status"], "generation_error")
                self.assertIsNone(result["source"])
                self.assertEqual(len(result["generation"]), 1)
                self.assertEqual(result["generation"][0]["response"]["text"], source)

    def test_budget_and_transport_failure_are_bounded(self):
        result, client = self.generate(["a()", "def a():\n    b()"], helper_budget=1)
        self.assertEqual(result["status"], "generation_error")
        self.assertIn("budget exhausted", result["error"])
        self.assertEqual(len(client.requests), 2)
        with contextlib.redirect_stdout(io.StringIO()):
            result = GoogleGenerator(Client([])).generate("stack")
        self.assertEqual(result["status"], "generation_error")
        self.assertEqual(len(result["generation"]), 1)

    def test_stop_delimiter_applies_before_parsing_and_keeps_raw_response(self):
        result, _ = self.generate(["pick(base)\n# task: next\nnot valid python!"])
        self.assertEqual(result["status"], "generated", result.get("error"))
        self.assertIn("not valid python", result["generation"][0]["response"]["text"])
        self.assertNotIn("not valid python", result["source"])

    def test_feedback_provenance_and_row_api(self):
        client = Client(["pass"])
        with contextlib.redirect_stdout(io.StringIO()):
            result = GoogleGenerator(client).generate(
                "insertion_sort", previous="old()", feedback={"training": "failure"}
            )
        prompt = client.requests[0][0][1]["content"]
        self.assertIn("old()", prompt)
        self.assertIn("keyless", prompt)
        self.assertNotIn("Tower API", prompt)
        self.assertNotIn("put_first_on_second", prompt)
        self.assertEqual(
            result["generator"]["prompt_version"], "roboverify-google-lmp-v1"
        )
        self.assertEqual(json.loads(json.dumps(result))["generator"], provenance())


if __name__ == "__main__":
    unittest.main()
