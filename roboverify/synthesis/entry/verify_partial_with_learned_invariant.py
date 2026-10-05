import argparse
import os
import time
from copy import deepcopy

import numpy as np
import synthesis.verification_lib.highlevel_verification_lib as highlevel_verification_lib
from synthesis.api.instructions import PickPlaceByName
from synthesis.api.program import Assign, Program, Put, While
from synthesis.entry.inference_options import add_inference_options
from synthesis.entry.motion_options import add_motion_options, motion_noise_from_args
from synthesis.entry.run_rollouts import run_program_rollouts
from synthesis.inference_lib.demo_store import DemoStore, InvInference, tower_vocabulary
from synthesis.inference_lib.inference import (
    instantiate_invariant,
    serialize_invariant,
)
from synthesis.verification_lib.motion_verification import (
    MotionCheck,
    MotionVerificationResult,
)
from z3 import And, Consts, ForAll, Implies, Not, Or


def verify_partial_program_with_learned_invariant(
    demo_store: DemoStore,
    loop_id: str = "1",
    verification_mode: str = "infinite",
    num_blocks: int = 4,
    visualize_finite_scene: bool = True,
    visualization_prefix: str = "verify_stack",
    inference_mode: str = "finite",
    noise=None,
    motion_timeout_ms: int = 5000,
    invariant_minimizer=None,
):
    """Infer from recorded loop-head demonstrations and verify the partial program."""
    # Inference is always done in the infinite-block (DeclareSort) setting.
    if inference_mode == "finite":
        block_names = [f"b{9 + i}" for i in range(num_blocks)]
        if "tbl" not in block_names:
            block_names = [*block_names]
        inference_context = highlevel_verification_lib.HighLevelContext(
            mode="enum",
            enum_names=block_names,
            use_tbl=True,
            exists_top=True,
            visualize_enum_scene=visualize_finite_scene,
            visualization_prefix=visualization_prefix,
        )
        learned_invariant, learned_invariant_lists = InvInference(
            demo_store,
            loop_id,
            tower_vocabulary("partial"),
            inference_context,
            minimizer=invariant_minimizer,
        )
    else:
        inference_context = highlevel_verification_lib.HighLevelContext(
            mode="declare", use_tbl=True, exists_top=True
        )
        learned_invariant, learned_invariant_lists = InvInference(
            demo_store,
            loop_id,
            tower_vocabulary("partial"),
            inference_context,
            minimizer=invariant_minimizer,
        )

    if verification_mode == "finite":
        # Include "tbl" in the finite Box universe so get_consts("tbl") matches an enum member.
        # block_names = [f"b{9 + i}" for i in range(num_blocks)]
        # if "tbl" not in block_names:
        #     block_names = [*block_names, "tbl"]
        # context = highlevel_verification_lib.HighLevelContext(
        #     mode="enum",
        #     enum_names=block_names,
        #     use_tbl=True,
        #     exists_top=True,
        #     visualize_enum_scene=visualize_finite_scene,
        #     visualization_prefix=visualization_prefix,
        # )
        # learned_spec = serialize_invariant(learned_invariant, inference_context)
        # learned_invariant = instantiate_invariant(
        #     learned_spec, context, known_const_names=["b0", "b", "tbl"]
        # )
        context = inference_context
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
        Assign("b", "b0"),
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
            body=[Put("b_prime", "tbl"), Assign("b", "b_prime")],
            invariant=learned_invariant,
        ),
    ]
    program = Program(2, instructions=instructions)

    m, n = Consts("m n", context.BoxSort)
    (tbl,) = Consts("tbl", context.BoxSort)
    precondition = And(
        ForAll([m], Implies(m != tbl, context.ON_star(m, b0))),
        b0 != tbl,
    )

    m, b0 = Consts("m n", context.BoxSort)
    postcondition = And(
        ForAll([m], ForAll([n], Implies(context.ON_star(n, m), n == m))),
        ForAll(
            [m], ForAll([n], Implies(And(m != tbl, n != tbl), context.Higher(n, m)))
        ),
        ForAll(
            [m],
            ForAll(
                [n], Implies(And(m != tbl, n != tbl, m != n), context.Scattered(n, m))
            ),
        ),
    )

    hl_ok = program.highlevel_verification(precondition, postcondition, context=context)
    ll_ok = MotionVerificationResult(
        [
            MotionCheck(
                "coverage",
                "unsupported",
                reason="No lowered physical program for this task",
            )
        ],
        noise,
    )
    print(f"hl_ok: {hl_ok}", f"ll_ok: {ll_ok}")
    return bool(hl_ok and ll_ok)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_inference_options(parser)
    add_motion_options(parser)
    parser.add_argument(
        "--demos",
        required=True,
        help="Current full-state demonstration archive with loop events.",
    )
    parser.add_argument(
        "--loop-id",
        default="1",
        help="Instruction path of the recorded loop (default: 1).",
    )
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
        default="verify_partial",
        help="Output prefix for generated finite-mode scene images.",
    )
    args = parser.parse_args()
    noise = motion_noise_from_args(parser, args)

    verify_partial_program_with_learned_invariant(
        invariant_minimizer=args.invariant_minimizer,
        noise=noise,
        motion_timeout_ms=args.motion_timeout_ms,
        demo_store=DemoStore.from_archive(args.demos),
        loop_id=args.loop_id,
        verification_mode=args.verification_mode,
        num_blocks=args.num_blocks,
        visualize_finite_scene=not args.disable_scene_viz,
        visualization_prefix=args.viz_prefix,
    )
