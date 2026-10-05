"""Legacy literal tower examples, retained only as golden test fixtures.

The original list lengths are intentional: historical compute_dataset zipped
and silently truncated them. Production learning must use DemoStore instead.
"""

from typing import Dict, List, Tuple

import z3

from synthesis.inference_lib.inference import _ensure_context, loop_inference
from synthesis.util import on
from synthesis.verification_lib import highlevel_verification_lib


def run_proposal_example(
    context: highlevel_verification_lib.HighLevelContext,
) -> Tuple[z3.ExprRef, List[z3.ExprRef]]:
    context = _ensure_context(context)
    states_zero: List[Dict] = [{}, {}, {}, {}]
    states: List[Dict] = [
        {
            "x1": [0.0, 0.0, 0.0],
            "x2": [0.0, 0.0, 0.05],
            "x3": [0.0, 0.0, 0.1],
            "x4": [5.0, 5.0, 0.0],
            "x5": [10.0, 10.0, 0.0],
        },
        {"x1": [0.0, 0.0, 0.0], "x2": [5.0, 5.0, 0.0], "x3": [10.0, 10.0, 0.0]},
    ]
    k = 2
    relations = [context.ON_star, context.Higher, context.Scattered, "equality"]
    b0, b = context.get_consts("b0"), context.get_consts("b")
    constants = [b0, b]
    constants_mappings = [
        {b0: "x1", b: "x3"},
        {b0: "x1", b: "x1"},
    ]

    return loop_inference(
        states_zero,
        states,
        k,
        relations,
        constants,
        constants_mappings,
        context=context,
    )


def run_unstack_example(
    context: highlevel_verification_lib.HighLevelContext,
) -> Tuple[z3.ExprRef, List[z3.ExprRef]]:
    context = _ensure_context(context)
    states_zero: List[Dict] = [{}, {}, {}, {}]
    states: List[Dict] = [
        {
            "x1": [0.0, 0.0, 0.0],
            "x2": [0.0, 0.0, 0.05],
            "x3": [0.0, 0.0, 0.1],
            "x4": [0.0, 0.0, 0.15],
            "tbl": on.TABLE,
        },
        {
            "x1": [0.0, 0.0, 0.0],
            "x2": [0.0, 0.0, 0.05],
            "x3": [0.0, 0.0, 0.1],
            "x4": [5.0, 0.0, 0.0],
            "tbl": on.TABLE,
        },
        {
            "x1": [0.0, 0.0, 0.0],
            "x2": [0.0, 0.0, 0.05],
            "x3": [10.0, 0.0, 0.0],
            "x4": [5.0, 0.0, 0.0],
            "tbl": on.TABLE,
        },
        {
            "x1": [0.0, 0.0, 0.0],
            "x2": [15.0, 0.0, 0.0],
            "x3": [10.0, 0.0, 0.0],
            "x4": [5.0, 0.0, 0.0],
            "tbl": on.TABLE,
        },
    ]
    n_forall = 2
    relations = [context.ON_star, context.Higher, context.Scattered, "equality"]
    b0 = context.get_consts("b0")
    b = context.get_consts("b")
    tbl = context.get_consts("tbl")
    constants = [b0, b, tbl]
    constants_mappings = [
        {b0: "x1", b: "x1", tbl: "tbl"},
        {b0: "x1", b: "x4", tbl: "tbl"},
        {b0: "x1", b: "x3", tbl: "tbl"},
        {b0: "x1", b: "x2", tbl: "tbl"},
    ]

    return loop_inference(
        states_zero,
        states,
        n_forall,
        relations,
        constants,
        constants_mappings,
        context=context,
    )


def run_reverse_example(
    context: highlevel_verification_lib.HighLevelContext,
) -> Tuple[z3.ExprRef, List[z3.ExprRef]]:
    context = _ensure_context(context)
    states_zero: List[Dict] = [
        {
            "x1": [0.0, 0.0, 0.0],
            "x2": [0.0, 0.0, 0.05],
            "x3": [0.0, 0.0, 0.1],
            "x4": [0.0, 0.0, 0.15],
            "x5": [0.0, 0.0, 0.20],
            "tbl": on.TABLE,
        },
        {
            "x1": [0.0, 0.0, 0.0],
            "x2": [0.0, 0.0, 0.05],
            "x3": [0.0, 0.0, 0.1],
            "x4": [0.0, 0.0, 0.15],
            "x5": [0.0, 0.0, 0.20],
            "tbl": on.TABLE,
        },
        {
            "x1": [0.0, 0.0, 0.0],
            "x2": [0.0, 0.0, 0.05],
            "x3": [0.0, 0.0, 0.1],
            "x4": [0.0, 0.0, 0.15],
            "x5": [0.0, 0.0, 0.20],
            "tbl": on.TABLE,
        },
        {
            "x1": [0.0, 0.0, 0.0],
            "x2": [0.0, 0.0, 0.05],
            "x3": [0.0, 0.0, 0.1],
            "x4": [0.0, 0.0, 0.15],
            "x5": [0.0, 0.0, 0.20],
            "tbl": on.TABLE,
        },
        {
            "x1": [0.0, 0.0, 0.0],
            "x2": [0.0, 0.0, 0.05],
            "x3": [0.0, 0.0, 0.1],
            "x4": [0.0, 0.0, 0.15],
            "x5": [0.0, 0.0, 0.20],
            "tbl": on.TABLE,
        },
    ]
    states: List[Dict] = [
        {
            "x1": [0.0, 0.0, 0.0],
            "x2": [0.0, 0.0, 0.05],
            "x3": [0.0, 0.0, 0.1],
            "x4": [0.0, 0.0, 0.15],
            "x5": [0.0, 0.0, 0.20],
            "tbl": on.TABLE,
        },
        {
            "x1": [0.0, 0.0, 0.0],
            "x2": [0.0, 0.0, 0.05],
            "x3": [0.0, 0.0, 0.1],
            "x4": [0.0, 0.0, 0.15],
            "x5": [5.0, 0.0, 0.0],
            "tbl": on.TABLE,
        },
        {
            "x1": [0.0, 0.0, 0.0],
            "x2": [0.0, 0.0, 0.05],
            "x3": [0.0, 0.0, 0.1],
            "x4": [5.0, 0.0, 0.05],
            "x5": [5.0, 0.0, 0.0],
            "tbl": on.TABLE,
        },
        {
            "x1": [0.0, 0.0, 0.0],
            "x2": [0.0, 0.0, 0.05],
            "x3": [5.0, 0.0, 0.1],
            "x4": [5.0, 0.0, 0.05],
            "x5": [5.0, 0.0, 0.0],
            "tbl": on.TABLE,
        },
        {
            "x1": [0.0, 0.0, 0.0],
            "x2": [5.0, 0.0, 0.15],
            "x3": [5.0, 0.0, 0.1],
            "x4": [5.0, 0.0, 0.05],
            "x5": [5.0, 0.0, 0.0],
            "tbl": on.TABLE,
        },
    ]
    k = 2
    relations = [context.ON_star, context.ON_star_zero, "equality"]
    b0, b = context.get_consts("b0"), context.get_consts("b")
    tbl = context.get_consts("tbl")
    constants = [b0, b, tbl]
    constants_mappings = [
        {b0: "x1", b: "tbl", tbl: "tbl"},
        {b0: "x1", b: "x5", tbl: "tbl"},
        {b0: "x1", b: "x4", tbl: "tbl"},
        {b0: "x1", b: "x3", tbl: "tbl"},
        {b0: "x1", b: "x2", tbl: "tbl"},
    ]

    return loop_inference(
        states_zero,
        states,
        k,
        relations,
        constants,
        constants_mappings,
        context=context,
    )


def run_partial_stack_example(
    context: highlevel_verification_lib.HighLevelContext,
) -> Tuple[z3.ExprRef, List[z3.ExprRef]]:
    context = _ensure_context(context)
    states_zero: List[Dict] = [{}, {}, {}, {}]
    states: List[Dict] = [
        {
            "x1": [0.0, 0.0, 0.0],
            "x2": [5.0, 0.0, 0.0],
            "x3": [5.0, 0.0, 0.05],
            "x4": [10.0, 0.0, 0.0],
            "x5": [10.0, 0.0, 0.05],
        },
        {
            "x1": [0.0, 0.0, 0.0],
            "x2": [5.0, 0.0, 0.0],
            "x3": [0.0, 0.0, 0.05],
            "x4": [10.0, 0.0, 0.0],
            "x5": [10.0, 0.0, 0.05],
        },
        {
            "x1": [0.0, 0.0, 0.0],
            "x2": [5.0, 0.0, 0.0],
            "x3": [0.0, 0.0, 0.05],
            "x4": [10.0, 0.0, 0.0],
            "x5": [0.0, 0.0, 0.1],
        },
        {
            "x1": [0.0, 0.0, 0.0],
            "x2": [0.0, 0.0, 0.15],
            "x3": [0.0, 0.0, 0.05],
            "x4": [10.0, 0.0, 0.0],
            "x5": [0.0, 0.0, 0.1],
        },
        {
            "x1": [0.0, 0.0, 0.0],
            "x2": [0.0, 0.0, 0.15],
            "x3": [0.0, 0.0, 0.05],
            "x4": [0.0, 0.0, 0.20],
            "x5": [0.0, 0.0, 0.1],
        },
    ]
    k = 2
    relations = [context.ON_star, "equality"]
    b0, b = context.get_consts("b0"), context.get_consts("b")
    constants = [b0, b]
    constants_mappings = [
        {b0: "x1", b: "x1"},
        {b0: "x1", b: "x3"},
        {b0: "x1", b: "x5"},
        {b0: "x1", b: "x2"},
        {b0: "x1", b: "x4"},
    ]

    return loop_inference(
        states_zero,
        states,
        k,
        relations,
        constants,
        constants_mappings,
        context=context,
    )
