"""Load an explicit primitive DSL factory and identify its executable contents."""

import hashlib
import importlib
import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from synthesis.api.instructions import (
    Assign,
    Get,
    Move,
    MoveByName,
    Pick,
    PickByName,
    Release,
    ReleaseByName,
    Skip,
    While,
)
from synthesis.api.program import Program
from synthesis.cfg.physical import name_operands
from synthesis.predicates.term import free_names, from_z3
from synthesis.util.actions import CARTESIAN_ACTION_MODE
from synthesis.util.symbols import fresh_name


def guard_term(instruction):
    term = getattr(instruction, "guard_term", None)
    return term if term is not None else from_z3(instruction.instantiated_cond)


def describe_program(program):
    def describe(i):
        result = {"instruction": type(i).__name__}
        if isinstance(i, While):
            result.update(
                guard=str(guard_term(i)),
                witnesses=list(map(str, i.guard_exists_vars)),
                body=[describe(child) for child in i.body],
                max_iters=i.max_iters,
            )
        elif isinstance(i, Get):
            result.update(
                guard=str(guard_term(i)), witnesses=list(map(str, i.guard_exists_vars))
            )
        elif isinstance(i, Assign):
            result.update(left=i.left, right=i.right)
        elif isinstance(i, Skip):
            result["steps"] = i.skip_steps
        elif type(i) in (Pick, PickByName, Move, MoveByName, Release, ReleaseByName):
            result.update(
                operands=i.get_operand(),
                limit=i.limit,
                control=asdict(i.control),
                action_scaling=CARTESIAN_ACTION_MODE,
            )
            if hasattr(i, "target_offset"):
                result["offsets"] = [p.numeric_val() for p in i.target_offset]
            if hasattr(i, "target_z_offset"):
                result["release_offset"] = i.target_z_offset.numeric_val()
        else:
            raise ValueError(
                f"Unsupported instruction {type(i).__name__}; use explicit Pick/Move/Release primitives"
            )
        return result

    return [describe(i) for i in program.instructions]


def program_fingerprint(program):
    return hashlib.sha256(
        json.dumps(describe_program(program), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


@dataclass
class ProgramDefinition:
    program: Program
    initial_bindings: dict
    source: str

    @property
    def metadata(self):
        return dict(
            source=self.source,
            program=describe_program(self.program),
            fingerprint=program_fingerprint(self.program),
            initial_bindings=self.initial_bindings,
        )


def prepare_program(program, num_blocks, source=""):
    if not isinstance(program, Program):
        raise TypeError("Program factory must return synthesis.api.program.Program")
    describe_program(program)
    occupied, ids = {"b0", "tbl"}, set()

    def inspect(rows, depth=0):
        for i in rows:
            if isinstance(i, (While, Get)):
                occupied.update(free_names(guard_term(i)))
                occupied.update(map(str, i.guard_exists_vars))
            if isinstance(i, While):
                if depth:
                    raise ValueError("Nested loops are not supported")
                inspect(i.body, depth + 1)
            elif isinstance(i, Assign):
                occupied.update((i.left, i.right))
            elif not isinstance(i, (Get, Skip)):
                for op in i.get_operand():
                    if op["type"] == "Box":
                        ids.add(op["val"])
                    else:
                        occupied.add(op["val"])

    inspect(program.instructions)
    if any(
        not isinstance(i, (int, np.integer)) or not 0 <= i < num_blocks for i in ids
    ):
        raise ValueError("Numeric object operand is outside the configured block count")
    aliases = {i: fresh_name(f"object_{i}", occupied) for i in sorted(ids)}

    def normalize(rows):
        output = []
        for i in rows:
            new = name_operands(i, aliases)
            if isinstance(new, While):
                new.body = normalize(i.body)
            if isinstance(new, (While, Get)):
                new.guard_term = guard_term(new)
            output.append(new)
        return output

    return ProgramDefinition(
        Program(len(program.instructions), normalize(program.instructions)),
        {"b0": 0, **{name: int(i) for i, name in aliases.items()}},
        source,
    )


def load_program(source, context, num_blocks):
    module_name, separator, factory_name = source.rpartition(":")
    if not separator or not module_name or not factory_name.isidentifier():
        raise ValueError("Use --program module:factory or path/to/program.py:factory")
    if module_name.endswith(".py"):
        path = Path(module_name).resolve()
        spec = importlib.util.spec_from_file_location("roboverify_user_program", path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load program file: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        source = str(path) + ":" + factory_name
    else:
        module = importlib.import_module(module_name)
    return prepare_program(
        getattr(module, factory_name)(context, num_blocks=num_blocks),
        num_blocks,
        source,
    )
