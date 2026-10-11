"""Google's vendored LMP/LMPFGen, adapted to local inference and restricted APIs.

The trusted upstream module is loaded in an isolated namespace per generation.
Its generation algorithms are unchanged. Its legacy SDK, eval/name lookup and
native exec hooks are replaced; generated programs are never natively executed.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import textwrap
from pathlib import Path
from types import SimpleNamespace

from synthesis.benchmarks.specs import ROW_TASKS
from synthesis.cap.prompts import task_prompt
from synthesis.cap.runtime import PolicyError, Runtime

VENDOR = Path(__file__).with_name("vendor")
PROVENANCE = json.loads((VENDOR / "google_cap.provenance.json").read_text())
PROMPT_VERSION = "roboverify-google-lmp-v2"


def provenance():
    return {
        "implementation": "google",
        "upstream_commit": PROVENANCE["commit"],
        "upstream_source": PROVENANCE["source_url"],
        "upstream_notebook_sha256": PROVENANCE["notebook_sha256"],
        "vendored_file_sha256": hashlib.sha256(
            (VENDOR / "google_cap.py").read_bytes()
        ).hexdigest(),
        "prompt_version": PROMPT_VERSION,
        "execution": "restricted AST interpreter; LMP debug_mode=True",
    }


def _load_core():
    # Normal import execution of a trusted source file, NOT generated code.
    spec = importlib.util.spec_from_file_location(
        "google_cap_snapshot", VENDOR / "google_cap.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _names(task):
    names = set(Runtime({}).builtins) | {
        "objects",
        "parameter",
        "pick",
        "move",
        "release",
    }
    if task in ROW_TASKS:
        names |= {"next_star", "keyless", "target", "buffer_target"}
    else:
        names |= {"on_star", "on_star_zero", "higher", "scattered"}
    return names


def _local_names(tree):
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
    return names


def _validate(source, protected):
    tree = Runtime.parse(source)
    forbidden = _local_names(tree) & protected
    if forbidden:
        raise PolicyError(f"Cannot replace or shadow API/builtin: {sorted(forbidden)}")
    return tree


def normalize_policy(root_source, helpers):
    """Wrap Google's top-level program in the harness's zero-argument entrypoint.

    A chat model may already return def policy(); in that case remove only an
    optional, bare policy() invocation to avoid executing it twice. No algorithmic
    repair is performed. Both the raw completion and this wrapper are archived.
    """
    tree = Runtime.parse(root_source)
    has_policy = any(
        isinstance(node, ast.FunctionDef) and node.name == "policy"
        for node in tree.body
    )
    if has_policy:
        body = []
        for node in tree.body:
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "policy"
                and not node.value.args
                and not node.value.keywords
            ):
                continue
            if not isinstance(node, ast.FunctionDef):
                raise PolicyError(
                    "A def policy() response must contain only definitions and an optional policy() call"
                )
            body.append(node)
        root = ast.unparse(ast.Module(body=body, type_ignores=[])) + "\nreturn policy()"
    else:
        root = root_source.strip() or "pass"
    # All declarations share the same interpreted scope. A recursively generated
    # helper can therefore call an inline helper or read a top-level local value.
    # They must not be separated into global functions and wrapper-local code.
    body = "\n\n".join([*helpers.values(), root])
    source = "def policy():\n" + textwrap.indent(body, "    ") + "\n"
    Runtime.parse(source)
    return source


class GoogleGenerator:
    def __init__(self, client, *, helper_budget=8, max_tokens=3072, temperature=0.2):
        self.client = client
        self.helper_budget = helper_budget
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, task, *, seed=0, feedback=None, previous=None):
        responses, helper_sources = [], {}
        core = _load_core()
        protected = _names(task)
        fixed_vars = {name: object() for name in protected}
        # Roles are input values, not immutable callable capabilities. In
        # particular, def transfer(obj, base) is a legitimate helper signature.
        variable_vars = {"base": object()}
        if task == "tower_merge":
            variable_vars.update(root_a=object(), root_b=object())
        if task == "tower_split":
            variable_vars["cut"] = object()
        description = task_prompt(task).replace(
            "Define def policy(): with no arguments. It must solve the task for any valid\nscene and any provided number of objects, not just one example.",
            "Write a top-level Python program (not a policy() definition) for any valid\nscene and any provided number of objects, not just one example.",
        )
        description += "\nThis is a notebook-cell program, NOT a standalone script. Do not add an if __name__ == '__main__' guard. No underscore-prefixed names, including the throwaway name _. Use ordinary variable names and call the program directly.\n"
        # The few-shot examples and robot vocabulary are adapted to RoboVerify;
        # the original NumPy/Shapely/put_first_on_second APIs are NOT capabilities.
        prompt = description + "\n{variable_vars_imports}\n"
        helper_prompt = (
            description
            + "\nContinue with ONLY the requested function definition, no header comment or call.\n"
        )
        helper_prompt += "\n# define function: total = get_total(xs=numbers).\ndef get_total(xs):\n    return sum(xs)\n"
        core.astunparse = SimpleNamespace(unparse=ast.unparse)
        core.var_exists = lambda name, all_vars: name in all_vars
        core.highlight = lambda value, *args: value
        core.PythonLexer = core.TerminalFormatter = lambda: None
        # Local transport errors deliberately propagate; no unbounded SDK retry.
        core.RateLimitError = type("UnusedLegacyRateLimitError", (Exception,), {})
        core.APIConnectionError = type("UnusedLegacyConnectionError", (Exception,), {})

        def completion(*, prompt, stop, temperature, engine, max_tokens):
            index = len(responses)
            if index > self.helper_budget:
                raise PolicyError("Recursive helper-generation budget exhausted")
            messages = [
                {
                    "role": "system",
                    "content": "Continue the supplied Code-as-Policies prompt with Python code only. Use only the documented robot APIs. Do not repeat the query comment.",
                },
                {"role": "user", "content": prompt},
            ]
            record = {
                "messages": messages,
                "completion_settings": {
                    "stop": list(stop),
                    "temperature": temperature,
                    "engine": engine,
                    "max_tokens": max_tokens,
                },
            }
            responses.append(record)
            response = self.client.generate(
                messages,
                seed=seed + index,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            record["response"] = response
            # Chat backends return a full message rather than a completion.
            # Strip fences, then apply the configured next-query delimiters.
            blocks = re.findall(
                r"```(?:python)?\s*\n(.*?)```", response["text"], flags=re.DOTALL
            )
            code = "\n\n".join(blocks) if blocks else response["text"].strip()
            cuts = [code.find(token) for token in stop if token in code]
            if cuts:
                code = code[: min(cuts)].rstrip() + "\n"
            _validate(code, protected)
            record["completion_text"] = code
            return {"choices": [{"text": code}]}

        core.openai = SimpleNamespace(Completion=SimpleNamespace(create=completion))

        def define_only(source, gvars=None, lvars=None):
            tree = _validate(source, protected)
            if not tree.body or any(
                not isinstance(node, ast.FunctionDef) for node in tree.body
            ):
                raise PolicyError(
                    "Function generation may return only function definitions"
                )
            for node in tree.body:
                helper_sources[node.name] = ast.unparse(node)
                # LMP is in generation-only mode; these are declaration markers,
                # not native Python functions. Actual functions are interpreted
                # later together with the complete program on a physical scene.
                lvars[node.name] = object()

        core.exec_safe = define_only

        class BoundFGen(core.LMPFGen):
            """Lexical-name compatibility for the restricted Python frontend."""

            pending_locals = None

            def create_f_from_sig(
                self, f_name, f_sig, other_vars=None, fix_bugs=False, return_src=False
            ):
                if fix_bugs:
                    raise PolicyError(
                        "The retired OpenAI Edit endpoint is not supported"
                    )
                result, source = super().create_f_from_sig(
                    f_name, f_sig, other_vars, fix_bugs=False, return_src=True
                )
                self.pending_locals = _local_names(Runtime.parse(source))
                return (result, source) if return_src else result

            def create_new_fs_from_code(
                self, code_str, other_vars=None, fix_bugs=False, return_src=False
            ):
                locals_here = _local_names(Runtime.parse(code_str))
                locals_here |= self.pending_locals or set()
                self.pending_locals = None
                bindings = {
                    **(other_vars or {}),
                    **{name: object() for name in locals_here},
                }
                return super().create_new_fs_from_code(
                    code_str, bindings, fix_bugs=fix_bugs, return_src=return_src
                )

        settings = {
            "engine": "local-chat",
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        fgen = BoundFGen(
            {
                **settings,
                "prompt_text": helper_prompt,
                "query_prefix": "# define function: ",
                "query_suffix": ".",
                "stop": ["# define", "# example"],
            },
            fixed_vars,
            variable_vars,
        )
        lmp = core.LMP(
            "roboverify",
            {
                **settings,
                "prompt_text": prompt,
                "query_prefix": "# task: ",
                "query_suffix": ".",
                "stop": ["# task:"],
                "maintain_session": False,
                "include_context": False,
                "debug_mode": True,
                "has_return": False,
            },
            fgen,
            fixed_vars,
            variable_vars,
        )
        query = "Solve the Required final result using only the documented primitives and predicates"
        if feedback is not None:
            query += (
                "\nRepair the previous program using this training-only feedback. Return the complete corrected program.\nPrevious program:\n"
                + str(previous)
                + "\nFeedback:\n"
                + str(feedback)
            )
        result = {
            "task": task,
            "seed": seed,
            "source": None,
            "generation": responses,
            "generator": provenance(),
        }
        try:
            lmp(query)
            raw_source = lmp.exec_hist.lstrip("\n")
            source = normalize_policy(raw_source, helper_sources)
            _validate(source, protected)
            result.update(
                source=source,
                raw_source=raw_source,
                helper_sources=helper_sources,
                sha256=hashlib.sha256(source.encode()).hexdigest(),
                status="generated",
                error=None,
            )
        except Exception as exc:
            result.update(
                status="generation_error", error=f"{type(exc).__name__}: {exc}"
            )
        return result
