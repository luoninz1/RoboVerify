"""Invariant learning from recovered entry and terminal loop heads."""

from synthesis.cfg.refine import scene_at
from synthesis.inference_lib.demo_store import (
    DemoStore,
    InferenceVocabulary,
    InvInference,
    LoopHeadState,
)
from synthesis.util import on


def loop_learning_data(rows, scope, *, relations=None, variables=2):
    if not rows:
        raise ValueError("Invariant learning requires recorded loop heads")
    names = tuple(
        sorted(
            set(scope)
            & set.intersection(
                *(set(scene_at(row, row.t_start).bindings) for row in rows)
            )
        )
    )
    store = DemoStore()
    for row in rows:
        scene = scene_at(row, row.t_start)
        ids = {
            key: f"x{i}"
            for i, key in enumerate(sorted(scene.positions, key=str))
            if key != "tbl"
        }
        positions = {ids[k]: v for k, v in scene.positions.items() if k != "tbl"}
        initial = {ids[k]: v for k, v in scene.entry_positions.items() if k != "tbl"}
        positions["tbl"] = initial["tbl"] = on.TABLE
        bindings = {
            name: ids[scene.bindings[name]]
            for name in names
            if scene.bindings[name] != "tbl"
        }
        bindings["tbl"] = "tbl"
        store.add(LoopHeadState("loop", positions, initial, bindings))
    relations = tuple(
        "equality" if r == "eq" else r
        for r in (relations or ("ON_star", "Higher", "Scattered", "equality"))
    )
    return store, InferenceVocabulary(variables, relations, names)


def infer_loop_invariant(rows, guard, scope, *, context, relations=None, variables=2):
    store, vocabulary = loop_learning_data(
        rows, scope, relations=relations, variables=variables
    )
    return InvInference(store, "loop", vocabulary, context)[0]
