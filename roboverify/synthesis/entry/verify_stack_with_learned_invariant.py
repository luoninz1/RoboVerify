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
from synthesis.inference_lib.inference import (
    instantiate_invariant,
    run_proposal_example,
    serialize_invariant,
)


def verify_stack_program_with_learned_invariant(
    verification_mode: str = "infinite",
    num_blocks: int = 4,
    visualize_finite_scene: bool = True,
    visualization_prefix: str = "verify_stack",
):
    """Infer invariant from examples and verify the stack program."""
    # Inference is always done in the infinite-block (DeclareSort) setting.
    inference_context = highlevel_verification_lib.HighLevelContext(mode="declare")
    learned_invariant, learned_invariant_lists = run_proposal_example(
        context=inference_context
    )

    if verification_mode == "finite":
        context = highlevel_verification_lib.HighLevelContext(
            mode="enum",
            num_blocks=num_blocks,
            visualize_enum_scene=visualize_finite_scene,
            visualization_prefix=visualization_prefix,
        )
        learned_spec = serialize_invariant(learned_invariant, inference_context)
        learned_invariant = instantiate_invariant(
            learned_spec, context, known_const_names=["b0", "b"]
        )
    else:
        context = highlevel_verification_lib.HighLevelContext(mode="declare")

    b_prime, b, n, b0 = Consts("b_prime b n b0", context.BoxSort)
    instructions = [
        Assign("b", "b0"),
        While(
            instantiated_cond=And(
                ForAll(
                    [n],
                    Or(
                        b_prime == n,
                        Not(context.ON_star(n, b_prime)),
                    ),
                ),
                b_prime != b,
            ),
            guard_exists_vars=[b_prime],
            body=[Put("b_prime", "b"), Assign("b", "b_prime")],
            invariant=learned_invariant,
        ),
    ]
    program = Program(2, instructions=instructions)

    BOX_LENGTH = 0.05
    ll_instruction = deepcopy(instructions)
    ll_instruction[1].body = [
        PickPlaceByName(
            grab_box_name="b_prime",
            target_box_name_x="b_prime",
            target_box_name_y="b_prime",
            target_box_name_z="b",
            target_offset=[0.0, 0.0, 4 * BOX_LENGTH],
            release=False,
        ),
        PickPlaceByName(
            grab_box_name="b_prime",
            target_box_name_x="b",
            target_box_name_y="b",
            target_box_name_z="b",
            target_offset=[0.0, 0.0, 4 * BOX_LENGTH],
            release=False,
        ),
        PickPlaceByName(
            grab_box_name="b_prime",
            target_box_name_x="b",
            target_box_name_y="b",
            target_box_name_z="b",
            target_offset=[0.0, 0.0, 1.5 * BOX_LENGTH],
            release=True,
        ),
    ]
    ll_instruction[1].invariant = learned_invariant_lists
    ll_instruction[1].body.append(Assign("b", "b_prime"))
    ll_program = Program(2, instructions=ll_instruction)

    # def _env_factory(seed: int):
    #     # Keep construction local so the rollout utility can be reused elsewhere.
    #     return GymToGymnasium(
    #         FetchPickAndPlaceConstruction(
    #             name=f"roboverify_stack_{num_blocks}",
    #             sparse=False,
    #             shaped_reward=False,
    #             num_blocks=int(num_blocks),
    #             reward_type="sparse",
    #             case="RoboVerifyStack",
    #             visualize_mocap=False,
    #             simple=True,
    #             base_block_id=0,
    #         )
    #     )

    # # Run low-level `ll_program` in a concrete RoboVerify Stack environment.
    # ll_exec = run_program_rollouts(
    #     ll_program,
    #     env_factory=_env_factory,
    #     num_seeds=10,
    #     timesteps=200,
    #     return_img=True,
    #     save_dir="roboverify_stack_rollouts",
    #     fps=20,
    #     save_png_frames=True,
    #     video_filename="000_trajectory.mp4",
    # )
    # print("ll_program RoboVerifyStack success_rate:", ll_exec["success_rate"])
    # print("ll_program RoboVerifyStack per-seed:", ll_exec["results"])
    m, n = Consts("m n", context.BoxSort)
    precondition = And(
        ForAll(
            [m],
            ForAll([n], Or(m == n, Not(context.ON_star(n, m)))),
        ),
        ForAll(
            [m],
            ForAll([n], context.Higher(n, m)),
        ),
        ForAll([m, n], Implies(m != n, context.Scattered(m, n))),
    )

    m, b0 = Consts("m b0", context.BoxSort)
    postcondition = ForAll([m], context.ON_star(m, b0))

    hl_ok = program.highlevel_verification(precondition, postcondition, context=context)
    ll_ok = ll_program.lowlevel_verification(constants=["b0", "b", "b_prime"])
    print(f"hl_ok: {hl_ok}", f"ll_ok: {ll_ok}")
    return bool(hl_ok and ll_ok)


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

    verify_stack_program_with_learned_invariant(
        verification_mode=args.verification_mode,
        num_blocks=args.num_blocks,
        visualize_finite_scene=not args.disable_scene_viz,
        visualization_prefix=args.viz_prefix,
    )
