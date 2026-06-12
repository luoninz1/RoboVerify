import os

import z3
from PIL import Image

import synthesis.verification_lib.highlevel_verification_lib as highlevel_verification_lib
from synthesis.api import program
from synthesis.environment.cee_us_env.fpp_construction_env import (
    FetchPickAndPlaceConstruction,
)
from synthesis.environment.general_env import GymToGymnasium
from synthesis.mcmc import cem, decision_tree, synthesis

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

    # Learn a discriminating ON(...) feature from demos vs a random program.
    num_blocks = 4
    demo_dir = "demos"
    positive_trajs, _, saved_num_blocks = synthesis.load_demo_trajectories(demo_dir)
    if saved_num_blocks is not None:
        num_blocks = saved_num_blocks
    num_demo = len(positive_trajs)
    demo_goal_idxs = [len(traj) for traj in positive_trajs]

    random_program = program.generate_random_program(
        length=8,
        block_ids=list(range(num_blocks)),
    )
    print("=== random program for negative samples ===\n", random_program)
    negative_trajs, _, _, _ = synthesis.rollout_demos(
        random_program,
        num_demo,
        num_blocks=num_blocks,
        save_imgs=False,
        verbose=True,
    )

    on_features = decision_tree.compute_ON_features(num_blocks)
    best_tree, best_feature = decision_tree.learn_features(
        num_blocks,
        negative_trajs,
        positive_trajs,
        demo_goal_idxs,
        on_features,
        num_trees=10,
    )
    if best_feature is not None:
        print(f"Learned discriminating feature: {best_feature}")
        demo_imgs = synthesis.load_demo_images_grouped("demo_images", positive_trajs)
        feature_video_dir = f"demo_splits_with_{best_feature}".replace(" ", "")
        demo_splits = decision_tree.split_demos_by_feature(
            best_feature,
            positive_trajs,
            demo_imgs=demo_imgs,
            save_videos=True,
            video_dir=feature_video_dir,
        )
        print(
            f"Split {len(demo_splits)} demos using {best_feature}; "
            f"videos saved under {feature_video_dir}/"
        )

        valid_splits = [split for split in demo_splits if split["part2"]]
        part2_trajs = [split["part2"] for split in valid_splits]
        checkpoint_states = [split["part1"][-1] for split in valid_splits]
        part2_goal_idxs = [len(traj) for traj in part2_trajs]
        part2_imgs = [
            decision_tree._split_frames(
                demo_imgs[split["demo_idx"]], split["split_idx"]
            )[1]
            for split in valid_splits
        ]

        stage2_random_program = program.generate_random_program(
            length=8,
            block_ids=list(range(num_blocks)),
        )
        stage1_tag = str(best_feature).replace(" ", "")
        random_video_dir = f"random_rollouts_after_{stage1_tag}"
        print(
            "=== stage-2 random program from part-1 checkpoints ===\n",
            stage2_random_program,
        )
        negative_trajs_stage2, _, _, _ = synthesis.rollout_demos_from_initial_states(
            stage2_random_program,
            checkpoint_states,
            num_blocks=num_blocks,
            video_dir=random_video_dir,
            demo_indices=[split["demo_idx"] for split in valid_splits],
        )
        print(f"Random program videos saved under {random_video_dir}/")

        best_tree_stage2, best_feature_stage2 = decision_tree.learn_features(
            num_blocks,
            negative_trajs_stage2,
            part2_trajs,
            part2_goal_idxs,
            on_features,
            num_trees=10,
        )
        if best_feature_stage2 is not None:
            print(f"Stage-2 learned feature: {best_feature_stage2}")
            stage2_video_dir = f"demo_splits_with_{best_feature_stage2}".replace(
                " ", ""
            )
            decision_tree.split_demos_by_feature(
                best_feature_stage2,
                part2_trajs,
                demo_imgs=part2_imgs,
                save_videos=True,
                video_dir=stage2_video_dir,
            )
            print(
                f"Split {len(part2_trajs)} stage-2 demos using "
                f"{best_feature_stage2}; videos saved under {stage2_video_dir}/"
            )

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
