import argparse
import os
import time
from copy import deepcopy

import numpy as np
from z3 import And, Consts, ForAll, Implies, Not, Or

import synthesis.verification_lib.highlevel_verification_lib as highlevel_verification_lib
from synthesis.api.instructions import PickPlaceByName
from synthesis.api.program import Assign, Program, Put, While
from synthesis.entry.run_rollouts import run_program_rollouts

# )
# from synthesis.environment.general_env import GymToGymnasium
from synthesis.inference_lib.inference import (
    instantiate_invariant,
    run_unstack_example,
    serialize_invariant,
)

# from synthesis.environment.cee_us_env.fpp_construction_env import (
# FetchPickAndPlaceConstruction,


def verify_unstack_program_with_learned_invariant(
    verification_mode: str = "infinite",
    num_blocks: int = 4,
    visualize_finite_scene: bool = True,
    visualization_prefix: str = "verify_stack",
    inference_mode: str = "finite",
):
    """Infer invariant from examples and verify the stack program."""
    # Inference is always done in the infinite-block (DeclareSort) setting.
    if inference_mode == "finite":
        block_names = [f"b{9 + i}" for i in range(num_blocks)]
        if "tbl" not in block_names:
            block_names = [*block_names, "tbl"]
        inference_context = highlevel_verification_lib.HighLevelContext(
            mode="enum",
            enum_names=block_names,
            use_tbl=True,
            exists_top=True,
            visualize_enum_scene=visualize_finite_scene,
            visualization_prefix=visualization_prefix,
        )
        learned_invariant, learned_invariant_lists = run_unstack_example(
            context=inference_context
        )
    else:
        inference_context = highlevel_verification_lib.HighLevelContext(
            mode="declare", use_tbl=True, exists_top=True
        )
        learned_invariant, learned_invariant_lists = run_unstack_example(
            context=inference_context
        )

    if verification_mode == "finite":
        # Include "tbl" in the finite Box universe so get_consts("tbl") matches an enum member.
        block_names = [f"b{9 + i}" for i in range(num_blocks)]
        if "tbl" not in block_names:
            block_names = [*block_names, "tbl"]
        context = highlevel_verification_lib.HighLevelContext(
            mode="enum",
            enum_names=block_names,
            use_tbl=True,
            exists_top=True,
            visualize_enum_scene=visualize_finite_scene,
            visualization_prefix=visualization_prefix,
        )
        learned_spec = serialize_invariant(learned_invariant, inference_context)
        learned_invariant = instantiate_invariant(
            learned_spec, context, known_const_names=["b0", "b", "tbl"]
        )
    else:
        context = highlevel_verification_lib.HighLevelContext(
            mode="declare", use_tbl=True, exists_top=True
        )
        learned_spec = serialize_invariant(learned_invariant, inference_context)
        learned_invariant = instantiate_invariant(
            learned_spec, context, known_const_names=["b0", "b", "tbl"]
        )

    b_prime, b, c, b0, tbl = Consts("b_prime b c b0 tbl", context.BoxSort)
    instructions = [
        While(
            instantiated_cond=And(
                b_prime != tbl,
                ForAll(
                    [c],
                    Implies(
                        And(
                            c != tbl,
                            Or(context.ON_star(b0, b_prime), context.ON_star(c, b0)),
                        ),
                        context.ON_star(b_prime, c),
                    ),
                ),
            ),
            guard_exists_vars=[b_prime],
            body=[Put("b_prime", "tbl")],
            invariant=learned_invariant,
        ),
    ]
    program = Program(1, instructions=instructions)

    m, n = Consts("m b0", context.BoxSort)
    precondition = And(ForAll([m], context.ON_star(m, b0)))

    m, b0 = Consts("m n", context.BoxSort)
    postcondition = ForAll([m], ForAll([n], Implies(context.ON_star(n, m), n == m)))

    hl_ok = program.highlevel_verification(precondition, postcondition, context=context)
    # ll_ok = ll_program.lowlevel_verification()
    print(f"hl_ok: {hl_ok}")
    return bool(hl_ok)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verification-mode",
        choices=["infinite", "finite"],
        default="infinite",
        help="Use infinite (DeclareSort) or finite (EnumSort) verification.",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=4,
        help="Number of blocks when verification mode is finite.",
    )
    parser.add_argument(
        "--disable-scene-viz",
        action="store_true",
        help="Disable image generation for finite-mode verification.",
    )
    parser.add_argument(
        "--viz-prefix",
        type=str,
        default="verify_stack",
        help="Output prefix for generated finite-mode scene images.",
    )
    args = parser.parse_args()

    verify_unstack_program_with_learned_invariant(
        verification_mode=args.verification_mode,
        num_blocks=args.num_blocks,
        visualize_finite_scene=not args.disable_scene_viz,
        visualization_prefix=args.viz_prefix,
    )
