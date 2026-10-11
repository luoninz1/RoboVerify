import unittest

from synthesis.cap.generator import Generator, extract_code, missing_helpers


class Client:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def generate(self, messages, **settings):
        self.requests.append((messages, settings))
        return {"text": next(self.responses)}


class GeneratorTests(unittest.TestCase):
    def test_recursive_helpers(self):
        client = Client(
            [
                "def policy():\n    transfer(base)\n",
                "def transfer(obj):\n    inspect(obj)\n",
                "def inspect(obj):\n    return higher(obj,obj)\n",
            ]
        )
        result = Generator(client).generate("stack")
        self.assertEqual(result["status"], "generated")
        self.assertEqual(len(result["generation"]), 3)
        self.assertFalse(missing_helpers(result["source"], "stack"))

    def test_bad_model_output_is_recorded_not_manually_fixed(self):
        result = Generator(Client(["import os\ndef policy(): pass"])).generate("stack")
        self.assertEqual(result["status"], "generation_error")
        self.assertIsNone(result["source"])
        self.assertEqual(len(result["generation"]), 1)

    def test_extract_and_helper_budget(self):
        self.assertEqual(
            extract_code("```python\ndef policy(): pass\n```"), "def policy(): pass\n"
        )
        result = Generator(
            Client(["def policy(): missing()"]), helper_budget=0
        ).generate("stack")
        self.assertEqual(result["status"], "generation_error")


if __name__ == "__main__":
    unittest.main()
