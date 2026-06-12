import os

import z3
from PIL import Image

import synthesis.verification_lib.highlevel_verification_lib as highlevel_verification_lib
from synthesis.api import program
from synthesis.environment.cee_us_env.fpp_construction_env import (
    FetchPickAndPlaceConstruction,
)
from synthesis.environment.general_env import GymToGymnasium
from synthesis.mcmc import cem, synthesis

if __name__ == "__main__":

    context = highlevel_verification_lib.HighLevelContext(mode="declare")
    b_prime, b, n, b0 = z3.Consts("b_prime b n b0", context.BoxSort)
    loop_program = program.Program(2)
    loop_program.instructions = [
        program.Assign(left="b", right="b0"),
        program.While(
            instantiated_cond=z3.And(
                z3.ForAll(
                    [n],
                    z3.Or(
                        b_prime == n,
                        z3.Not(context.ON_star(n, b_prime)),
                    ),
                ),
                b_prime != b,
            ),
            guard_exists_vars=[b_prime],
            body=[
                program.PickByName(grab_box_name="b_prime"),
                program.MoveByName(
                    target_box_name_x="b_prime",
                    target_box_name_y="b_prime",
                    target_box_name_z="b",
                    target_offset=[0.0, 0.0, 0.15],
                ),
                program.MoveByName(
                    target_box_name_x="b",
                    target_box_name_y="b",
                    target_box_name_z="b",
                    target_offset=[0.0, 0.0, 0.15],
                ),
                program.MoveByName(
                    target_box_name_x="b",
                    target_box_name_y="b",
                    target_box_name_z="b",
                    target_offset=[0.0, 0.0, 0.05],
                ),
                program.ReleaseByName(release_box_name="b_prime", target_z=0.15),
                program.Assign("b", "b_prime"),
            ],
            invariant=None,
        ),
    ]
    expert_states, positive_trajs = synthesis.collect_trajectories(
        loop_program,
        15,
        num_blocks=4,
        save_imgs=True,
        demo_dir="demos",
        img_dir="demo_images",
        verify_reproducible=True,
    )
    synthesis.images_to_video("demo_images", "demo_video.mp4")
    exit()
    # p = program.Program(5)
    # p.instructions = [
    #     program.Pick(
    #         grab_box_id=1
    #     ),
    #     program.Move(
    #         target_box_id_x=1, target_box_id_y=1, target_box_id_z=0, target_offset=[0.0, 0.0, 0.15]
    #     ),
    #     program.Move(
    #         target_box_id_x=0, target_box_id_y=0, target_box_id_z=0, target_offset=[0.0, 0.0, 0.15]
    #     ),
    #     program.Move(
    #         target_box_id_x=0, target_box_id_y=0, target_box_id_z=0, target_offset=[0.0, 0.0, 0.05]
    #     ),
    #     program.Release(
    #         release_box_id=1,
    #         target_z=0.15
    #     ),
    #     program.Pick(
    #         grab_box_id=2
    #     ),
    #     program.Move(
    #         target_box_id_x=2, target_box_id_y=2, target_box_id_z=1, target_offset=[0.0, 0.0, 0.15]
    #     ),
    #     program.Move(
    #         target_box_id_x=1, target_box_id_y=1, target_box_id_z=1, target_offset=[0.0, 0.0, 0.15]
    #     ),
    #     program.Move(
    #         target_box_id_x=1, target_box_id_y=1, target_box_id_z=1, target_offset=[0.0, 0.0, 0.05]
    #     ),
    #     program.Release(
    #         release_box_id=2,
    #         target_z=0.15
    #     ),
    #     program.Pick(
    #         grab_box_id=3
    #     ),
    #     program.Move(
    #         target_box_id_x=3, target_box_id_y=3, target_box_id_z=2, target_offset=[0.0, 0.0, 0.15]
    #     ),
    #     program.Move(
    #         target_box_id_x=2, target_box_id_y=2, target_box_id_z=2, target_offset=[0.0, 0.0, 0.15]
    #     ),
    #     program.Move(
    #         target_box_id_x=2, target_box_id_y=2, target_box_id_z=2, target_offset=[0.0, 0.0, 0.05]
    #     ),
    #     program.Release(
    #         release_box_id=3,
    #         target_z=0.15
    #     ),
    # ]
    available_operands = {
        "Box": [0, 1],
    }

    ############## 3BLOCK TEST ##############
    available_instructions = [program.PickPlace]
    num_seeds = 15
    num_block = 3
    env_name = f"pickmulti{num_block}"
    expert_program = program.Program(3)
    expert_program.instructions = [
        program.PickPlace(grab_box_id=0, target_box_id=0),
        program.PickPlace(grab_box_id=0, target_box_id=1),
        program.PickPlace(grab_box_id=0, target_box_id=1),
    ]
    expert_states, positive_trajs = synthesis.collect_trajectories(
        expert_program, num_seeds, save_imgs=True, demo_dir="demos"
    )
    synthesis.images_to_video("images", "groundtruth.mp4")

    p = program.Program(3)
    p.instructions = [
        program.PickP(grab_box_id=0, target_box_id=0),  # move up with respect to box 0
        program.PickPlace(
            grab_box_id=0, target_box_id=1
        ),  # move horizontally to the top of box 1
        program.PickPlace(
            grab_box_id=0, target_box_id=1
        ),  # move down to place on box 1
    ]
    f = synthesis.Runner(p, expert_states, num_seeds, num_block)
    initial_parameters = p.register_trainable_parameter()
    cem.cem_optimize(f, len(initial_parameters), N=16, K=4, init_mu=initial_parameters)

    # p.update_trainable_parameter(
    #     [
    #         -0.055,
    #         -0.0318,
    #         0.145,
    #         -0.0448,
    #         -0.0214,
    #         0.116,
    #         0.0179,
    #         -0.00182,
    #         0.0428,
    #     ]
    # )

    # iter_folder = "tmp_testing_3block"
    # frames_dir = os.path.join(iter_folder, "frames")
    # os.makedirs(frames_dir, exist_ok=True)

    # # Evaluate new_program and get images
    # _, negative_trajs, imgs = evaluate_program(
    #     p, n=num_seeds, num_block=num_block, return_img=True
    # )

    # # Save each image as PNG
    # for idx, img in enumerate(imgs):
    #     img_path = os.path.join(frames_dir, f"img{idx:04d}.png")
    #     if isinstance(img, Image.Image):
    #         img.save(img_path)
    #     else:  # assume numpy array
    #         Image.fromarray(img).save(img_path)

    # # Generate video
    # output_video_path = os.path.join(iter_folder, "program_video.mp4")
    # images_to_video(frames_dir, output_video_path=output_video_path)

    # # learn the features
    # goal_idxs = [len(traj) for traj in positive_trajs]
    # features = []
    # for b1 in range(0, 3):
    #     for b2 in range(0, 3):
    #         features.append(decision_tree.ON_feature(b1, b2))
    # print(features)
    # decision_tree.learn_features(
    #     num_block, negative_trajs, positive_trajs, goal_idxs, features, num_trees=3
    # )
    ############## END OF 3BLOCK TEST ##############

    import cProfile
    import pstats

    profiler = cProfile.Profile()
    profiler.enable()

    synthesis.MCMC(
        program.Program(3),
        available_operands,
        available_instructions,
        20,
        expert_states=expert_states,
        num_seeds=num_seeds,
        num_block=num_block,
    )

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats(100)

    exit()

    # num_seeds = 15

    # expert_states = collect_trajectories("pickmulti1", num_seeds, save_imgs=True)
    # exit()
    p = Program(3)
    p.instructions = [
        PickPlace(grab_box_id=1, target_box_id=1),  # move up with respect to box 0
        PickPlace(
            grab_box_id=1, target_box_id=0
        ),  # move horizontally to the top of box 1
        PickPlace(grab_box_id=1, target_box_id=0),  # move down to place on box 1
    ]
    p.register_trainable_parameter()
    p.update_trainable_parameter(
        [
            0.0279608,
            0.04278932,
            0.13399421,
            0.02289095,
            -0.04434892,
            0.13288633,
            -0.01862492,
            0.00964001,
            0.05832452,
        ]
    )
    # # p.instructions[0].target_offset = [Parameter(0.), Parameter(0.), Parameter(0.2)]
    # # p.instructions[1].target_offset = [Parameter(0), Parameter(0), Parameter(0.2)]
    # # p.instructions[2].target_offset = [Parameter(0), Parameter(0), Parameter(0.05)]
    # # for _ in range(50):
    # #     print(mutate_program(p))
    # pdb.set_trace()
    iter_folder = "tmp_testing1"
    frames_dir = os.path.join(iter_folder, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Evaluate new_program and get images
    _, imgs = evaluate_program(p, n=num_seeds, return_img=True)

    # Save each image as PNG
    for idx, img in enumerate(imgs):
        img_path = os.path.join(frames_dir, f"img{idx:04d}.png")
        if isinstance(img, Image.Image):
            img.save(img_path)
        else:  # assume numpy array
            Image.fromarray(img).save(img_path)

    # Generate video
    output_video_path = os.path.join(iter_folder, "program_video.mp4")
    images_to_video(frames_dir, output_video_path=output_video_path)
    # states, imgs = evaluate_program(p, 15, True)
    # pdb.set_trace()

    # f = Runner(p, expert_states, num_seeds)
    # initial_parameters = p.register_trainable_parameter()
    # cem.cem_optimize(f, len(initial_parameters), N=16, K=4, init_mu=initial_parameters)
