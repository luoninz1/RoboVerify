"""Google-style policy generation with recursive missing-helper generation."""

from __future__ import annotations

import ast
import hashlib
import re

from synthesis.benchmarks.specs import ROW_TASKS
from synthesis.cap.prompts import task_prompt
from synthesis.cap.runtime import PolicyError, Runtime


def extract_code(text):
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL)
    source = "\n\n".join(blocks) if blocks else text.strip()
    Runtime.parse(source)
    return source.strip() + "\n"


def missing_helpers(source, task):
    tree = Runtime.parse(source)
    available = set(Runtime({}).builtins) | {
        "objects",
        "parameter",
        "pick",
        "move",
        "release",
    }
    available |= (
        {"next_star", "keyless", "target", "buffer_target"}
        if task in ROW_TASKS
        else {"on_star", "on_star_zero", "higher", "scattered"}
    )
    available |= {
        node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    # Calls through local aliases/arguments are resolved by the interpreter.
    available |= {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    available |= {node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)}
    missing = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id not in available
        ):
            missing.setdefault(node.func.id, ast.unparse(node))
    return missing


class Generator:
    def __init__(self, client, *, helper_budget=8, max_tokens=3072, temperature=0.2):
        self.client = client
        self.helper_budget = helper_budget
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, task, *, seed=0, feedback=None, previous=None):
        prompt = task_prompt(task)
        if feedback is not None:
            prompt += "\nRepair this previous candidate using the reported training failure. " "Keep the same permitted API; return the complete corrected policy and helpers.\n" + str(
                previous
            ) + "\nFailure feedback:\n" + str(
                feedback
            )
        messages = [
            {
                "role": "system",
                "content": "You generate robot policies using only documented primitive and predicate APIs.",
            },
            {"role": "user", "content": prompt},
        ]
        responses = []
        response = self.client.generate(
            messages,
            seed=seed,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        responses.append({"messages": messages, "response": response})
        try:
            source = extract_code(response["text"])
            for helper_index in range(self.helper_budget + 1):
                missing = missing_helpers(source, task)
                if not missing:
                    break
                if helper_index == self.helper_budget:
                    raise PolicyError("Recursive helper-generation budget exhausted")
                name, signature = next(iter(missing.items()))
                query = (
                    task_prompt(task)
                    + "\nCurrent policy:\n"
                    + source
                    + f"\nDefine ONLY the missing helper {name} used as {signature}. "
                    "Use the same API. Do not call policy(). Other missing helpers may be generated recursively."
                )
                helper_messages = [messages[0], {"role": "user", "content": query}]
                helper_response = self.client.generate(
                    helper_messages,
                    seed=seed + helper_index + 1,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                responses.append(
                    {"messages": helper_messages, "response": helper_response}
                )
                helper = extract_code(helper_response["text"])
                definitions = {
                    node.name
                    for node in ast.parse(helper).body
                    if isinstance(node, ast.FunctionDef)
                }
                if name not in definitions:
                    raise PolicyError(f"Helper response did not define {name}")
                source += "\n" + helper
                Runtime.parse(source)
            if not any(
                isinstance(node, ast.FunctionDef) and node.name == "policy"
                for node in ast.parse(source).body
            ):
                raise PolicyError("Generated candidate lacks def policy()")
            return {
                "task": task,
                "seed": seed,
                "source": source,
                "sha256": hashlib.sha256(source.encode()).hexdigest(),
                "generation": responses,
                "status": "generated",
                "error": None,
            }
        except Exception as exc:
            return {
                "task": task,
                "seed": seed,
                "source": None,
                "generation": responses,
                "status": "generation_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
