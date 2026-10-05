"""Loop-head demonstrations and the adapter to the existing invariant learner.

Object IDs are stable names (``x1`` is environment block 0), independent of
symbolic aliases such as ``b`` whose bindings change on every iteration.
Only successful guard checks that actually enter the body produce rows. These
are learning samples, not evidence that an invariant is inductive.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from synthesis.util import on


def _copy_positions(positions):
    result = {}
    for name, pos in positions.items():
        if not isinstance(name, str):
            raise ValueError("Object IDs must be strings")
        if name == "tbl" and on.is_table(pos):
            result[name] = on.TABLE
        else:
            xyz = np.asarray(pos, dtype=float)
            if name == "tbl" or xyz.shape != (3,) or not np.isfinite(xyz).all():
                raise ValueError(f"Invalid position for {name!r}: {pos!r}")
            result[name] = xyz.tolist()
    return result


def observation_positions(obs, num_blocks):
    """Copy every physical block, including blocks with no symbolic alias."""
    positions = {f"x{i + 1}": on.get_block_pos(obs, i) for i in range(num_blocks)}
    positions["tbl"] = on.TABLE
    return _copy_positions(positions)


@dataclass
class LoopHeadState:
    loop_id: str
    positions: dict
    entry_positions: dict
    constants: Mapping[str, str]

    def __post_init__(self):
        if not isinstance(self.loop_id, str) or not self.loop_id:
            raise ValueError("loop_id must be a nonempty string")
        self.positions = _copy_positions(self.positions)
        self.entry_positions = _copy_positions(self.entry_positions)
        if set(self.positions) != set(self.entry_positions):
            raise ValueError(
                "Loop-entry and current states must contain the same objects"
            )
        self.constants = dict(self.constants)
        for name, object_id in self.constants.items():
            if not isinstance(name, str) or object_id not in self.positions:
                raise ValueError(f"Unresolved constant {name!r}: {object_id!r}")
        if "tbl" in self.positions and self.constants.get("tbl") != "tbl":
            raise ValueError("The table constant must resolve to the table marker")

    @classmethod
    def from_observation(cls, loop_id, obs, entry_positions, mapping, num_blocks):
        constants = {"tbl": "tbl"}
        for name, block_id in mapping.items():
            if name == "tbl":
                if block_id != "tbl":
                    raise ValueError("tbl cannot alias a physical block")
                continue
            if (
                not isinstance(block_id, (int, np.integer))
                or not 0 <= block_id < num_blocks
            ):
                raise ValueError(f"Invalid block ID for {name!r}: {block_id!r}")
            constants[str(name)] = f"x{int(block_id) + 1}"
        return cls(
            loop_id, observation_positions(obs, num_blocks), entry_positions, constants
        )


class DemoStore:
    """D_V, retaining loop IDs and each execution's own loop-entry snapshot."""

    def __init__(self, states=()):
        self._states = []
        for state in states:
            self.add(state)

    def add(self, state):
        # Copy at the store boundary as callers may retain and mutate their row.
        self._states.append(
            LoopHeadState(
                state.loop_id, state.positions, state.entry_positions, state.constants
            )
        )

    def for_loop(self, loop_id):
        return [
            LoopHeadState(s.loop_id, s.positions, s.entry_positions, s.constants)
            for s in self._states
            if s.loop_id == loop_id
        ]

    def __len__(self):
        return len(self._states)

    def save_diagnostic(self, path):
        def encode(positions):
            return {
                name: None if on.is_table(pos) else pos
                for name, pos in positions.items()
            }

        rows = [
            dict(
                loop_id=s.loop_id,
                positions=encode(s.positions),
                entry_positions=encode(s.entry_positions),
                constants=s.constants,
            )
            for s in self._states
        ]
        Path(path).write_text(
            json.dumps({"version": 1, "states": rows}, indent=2) + "\n"
        )

    @classmethod
    def from_archive(cls, path):
        from synthesis.cfg.recordings import load_traces, loop_store

        return loop_store(load_traces(path, require_valid=True))


@dataclass(frozen=True)
class InferenceVocabulary:
    k: int
    relations: tuple[str, ...]
    constants: tuple[str, ...]


def tower_vocabulary(task):
    """The existing tower vocabularies, independent of any example dataset."""
    if task == "stack":
        return InferenceVocabulary(
            2, ("ON_star", "Higher", "Scattered", "equality"), ("b0", "b")
        )
    if task == "unstack":
        return InferenceVocabulary(
            2, ("ON_star", "Higher", "Scattered", "equality"), ("b0", "b", "tbl")
        )
    if task == "reverse":
        return InferenceVocabulary(
            2, ("ON_star", "ON_star_zero", "equality"), ("b0", "b", "tbl")
        )
    if task == "partial":
        return InferenceVocabulary(2, ("ON_star", "equality"), ("b0", "b"))
    raise ValueError(f"Unknown tower task: {task!r}")


def to_inference_inputs(store, loop_id, vocab, context):
    """Return aligned ``states_zero, states, constants_mappings`` for compute_dataset.

    The relational table is included only in contexts whose universe contains it.
    Snapshots always retain all physical objects. Missing constants and empty
    datasets fail explicitly rather than producing a vacuous learned invariant.
    """
    rows = store.for_loop(loop_id)
    if not rows:
        raise ValueError(f"No loop-head demonstrations for loop {loop_id!r}")
    include_table = context.use_tbl
    if "tbl" in vocab.constants and not include_table:
        raise ValueError("The vocabulary requires a context with use_tbl=True")
    states_zero, states, mappings = [], [], []
    for row in rows:

        def project(positions):
            result = {
                name: pos
                for name, pos in positions.items()
                if name != "tbl" or include_table
            }
            if include_table:
                result["tbl"] = on.TABLE
            return result

        state = project(row.positions)
        mapping = {}
        for name in vocab.constants:
            object_id = "tbl" if name == "tbl" else row.constants.get(name)
            if object_id not in state:
                raise ValueError(
                    f"Loop {loop_id!r} has no binding for constant {name!r}"
                )
            mapping[context.get_consts(name)] = object_id
        states_zero.append(project(row.entry_positions))
        states.append(state)
        mappings.append(mapping)
    return states_zero, states, mappings


def InvInference(D_V, loop_id, vocab, context, *, minimizer=None):
    """Learn with the existing algorithm; validation remains the verifier's job."""
    from synthesis.inference_lib.inference import loop_inference
    from synthesis.inference_lib.minimization import using_invariant_minimizer

    states_zero, states, mappings = to_inference_inputs(D_V, loop_id, vocab, context)
    relations = [r if r == "equality" else getattr(context, r) for r in vocab.relations]
    constants = [context.get_consts(name) for name in vocab.constants]
    with using_invariant_minimizer(minimizer):
        return loop_inference(
            states_zero,
            states,
            vocab.k,
            relations,
            constants,
            mappings,
            context=context,
        )
