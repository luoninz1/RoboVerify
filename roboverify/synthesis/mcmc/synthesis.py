import math
import os
import pickle
import random
from copy import deepcopy
from typing import Any, Optional, Tuple

import ffmpeg
import imageio
import numpy as np
from PIL import Image  # for saving images as PNG

from synthesis.api import program
from synthesis.environment.cee_us_env.fpp_construction_env import (
    FetchPickAndPlaceConstruction,
)
from synthesis.environment.general_env import GymToGymnasium
from synthesis.mcmc import cem, cost_func, decision_tree
from synthesis.util import on


def sample_proportional(values):
    """
    Given a list of positive integers, sample one index with probability
    proportional to its value.

    Parameters:
    - values: list of positive integers

    Returns:
    - An index sampled according to the proportional distribution
    """
    total = sum(values)
    probs = [v / total for v in values]
    return random.choices(range(len(values)), weights=probs, k=1)[0]


def swap_two_elements_maybe_same(lst):
    """
    Randomly selects two indices (which may be the same) and swaps their elements.

    Parameters:
    - lst: List of elements to mutate (in-place)

    Returns:
    - A new list with two elements swapped (or not, if indices are the same)
    """
    if len(lst) == 0:
        raise ValueError("List must contain at least one element.")

    i = random.randint(0, len(lst) - 1)
    j = random.randint(0, len(lst) - 1)
    print(f"swap instructions {i} and {j}")
    lst[i], lst[j] = lst[j], lst[i]
    return lst


def mutate_program(
    current_program: program.Program,
    available_operands: dict,
    available_instructions: list,
) -> tuple[program.Program, bool]:
    pc = 0  # opcode
    po = 1  # operand
    ps = 1  # swap
    pi = 1  # instruction
    sampled_mutation = sample_proportional([pc, po, ps, pi])
    mutate_type_name = {
        0: "opcode",
        1: "operand",
        2: "swap",
        3: "instruction",
    }[sampled_mutation]
    print("mutation type", mutate_type_name)

    new_program = deepcopy(current_program)
    if sampled_mutation == 0:
        pass
    elif sampled_mutation == 1:
        index = random.randint(0, len(new_program.instructions) - 1)
        old_operands = new_program.instructions[index].get_operand()
        if old_operands:
            operand_index = random.randint(0, len(old_operands) - 1)
            proposed_operand = random.choice(
                available_operands[old_operands[operand_index]["type"]]
            )
            print(
                "old operand",
                old_operands[operand_index],
                "proposed operand",
                proposed_operand,
            )
            if old_operands[operand_index] == proposed_operand:
                return new_program, False
            old_operands[operand_index]["val"] = proposed_operand
            new_program.instructions[index].set_operand(old_operands)
        else:
            return new_program, False
    elif sampled_mutation == 2:
        swap_two_elements_maybe_same(new_program.instructions)
    elif sampled_mutation == 3:
        index = random.randint(0, len(new_program.instructions) - 1)
        pu = 0.25  # probability the SKIP token is proposed
        if random.random() < pu:
            if type(new_program.instructions[index]) == program.Skip:
                print("chaning skip to skip")
                return new_program, False
            new_program.instructions[index] = program.Skip()
        else:
            new_instruction = random.choice(available_instructions)()
            operands = new_instruction.get_operand()
            for i in range(len(operands)):
                operands[i]["val"] = random.choice(
                    available_operands[operands[i]["type"]]
                )
            new_instruction.set_operand(operands)
            new_program.instructions[index] = new_instruction
    else:
        assert False, "unknown mutation"

    return new_program, True


def MCMC(
    current_program: program.Program,
    available_operands: dict,
    available_instructions: list,
    iters: int,
    expert_states,
    num_seeds: int,
    num_block: int,
    cem_N: int = 16,
    cem_K: int = 4,
    cem_iterations: int = 10,
    save_dir: str = "MCMC_results",
):
    # Ensure save_dir exists
    os.makedirs(save_dir, exist_ok=True)

    print("=== initial program ===\n", current_program)
    current_cost, current_program = optimize_program(
        current_program,
        expert_states,
        num_seeds,
        num_block,
        cem_N,
        cem_K,
        cem_iterations,
    )
    print("evaluated with cost", current_cost)

    samples = [deepcopy(current_program)]
    costs = [current_cost]

    for i in range(iters):
        iter_folder = os.path.join(save_dir, f"iter{i}")
        os.makedirs(iter_folder, exist_ok=True)

        new_program, changed = mutate_program(
            current_program, available_operands, available_instructions
        )
        print(f"=== iter {i} program ===\n", new_program)

        ins1 = [x for x in current_program.instructions if type(x) != program.Skip]
        ins2 = [x for x in new_program.instructions if type(x) != program.Skip]
        equivalence = ins1 == ins2
        print("program equivalence", equivalence)

        # Save a copy of current_program BEFORE acceptance update
        saved_current_program = deepcopy(current_program)
        saved_current_cost = current_cost

        if changed and not equivalence:
            # Optimize new_program, which may modify it
            new_cost, new_program = optimize_program(
                new_program,
                expert_states,
                num_seeds,
                num_block,
                cem_N,
                cem_K,
                cem_iterations,
            )
            print("evaluated with cost", new_cost)
        else:
            new_cost = current_cost

        # Save a copy of new_program AFTER optimization
        saved_new_program = deepcopy(new_program)

        # Decide acceptance
        if changed and not equivalence:
            acceptance_ratio = math.exp(new_cost - current_cost)
            print("acceptance_ratio is", acceptance_ratio)
            if random.random() < acceptance_ratio:
                print("new program accepted")
                current_program = new_program
                current_cost = new_cost
            else:
                print("new program NOT accepted")

        # Save pickle files
        with open(os.path.join(iter_folder, "current_program.pkl"), "wb") as f:
            pickle.dump(saved_current_program, f)
        with open(os.path.join(iter_folder, "new_program.pkl"), "wb") as f:
            pickle.dump(saved_new_program, f)

        # Save text summary
        with open(os.path.join(iter_folder, "summary.txt"), "w") as f:
            f.write("=== Current Program ===\n")
            f.write(str(saved_current_program) + "\n")
            f.write(f"Current Cost: {saved_current_cost}\n\n")
            f.write("=== New Program ===\n")
            f.write(str(saved_new_program) + "\n")
            f.write(f"New Cost: {new_cost}\n")
            f.write(f"Program Equivalence: {equivalence}\n")
            f.write(f"Accepted: {current_program == new_program}\n")

        # --- Evaluate program and save images/video ---
        frames_dir = os.path.join(iter_folder, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        # Evaluate new_program and get images
        _, _, imgs = evaluate_program(
            saved_new_program, n=num_seeds, num_block=num_block, return_img=True
        )

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

        samples.append(deepcopy(new_program))
        costs.append(new_cost)

    return samples, costs


def optimize_program(
    p: program.Program,
    expert_states,
    num_seeds: int,
    num_block: int,
    cem_N: int,
    cem_K: int,
    cem_iterations: int,
) -> tuple[float, program.Program]:
    f = Runner(p, expert_states, num_seeds, num_block)
    initial_parameters = p.register_trainable_parameter()
    iterations = cem_iterations if initial_parameters else 0
    best_cost, best_parameter = cem.cem_optimize(
        f,
        len(initial_parameters),
        iterations,
        N=cem_N,
        K=cem_K,
        init_mu=initial_parameters,
    )
    p.update_trainable_parameter(best_parameter)
    return best_cost, p


def set_np_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)


def make_roboverify_stack_env(
    num_blocks: int = 4,
    render_mode: str = "rgb_array",
) -> GymToGymnasium:
    """RoboVerifyStack environment matching ``synthesis/entry/main.py``."""
    return GymToGymnasium(
        FetchPickAndPlaceConstruction(
            name="roboverify_stack_3",
            sparse=False,
            shaped_reward=False,
            num_blocks=num_blocks,
            reward_type="sparse",
            case="RoboVerifyStack",
            visualize_mocap=False,
            simple=True,
            base_block_id=0,
        ),
        render_mode=render_mode,
    )


def images_to_video(input_dir, output_video_path="output.mp4", framerate=30):
    """
    Convert PNG images in a directory to a video using ffmpeg.
    Assumes images are named as img0000.png, img0001.png, ...

    Parameters:
    - input_dir: directory containing the PNG images
    - output_video_path: output video file path
    - framerate: frames per second for the video
    """
    input_pattern = os.path.join(input_dir, "img%04d.png")

    try:
        (
            ffmpeg.input(input_pattern, framerate=framerate)
            .output(output_video_path, vcodec="libx264", pix_fmt="yuv420p")
            .overwrite_output()
            .run()
        )
        print(f"Video saved to {output_video_path}")
    except ffmpeg.Error as e:
        print("FFmpeg error:", e.stderr.decode())


def save_demo_trajectories(
    trajectories: list,
    demo_dir: str,
    *,
    seeds: Optional[list] = None,
    num_blocks: Optional[int] = None,
    expert_states: Optional[list] = None,
) -> None:
    """Persist per-demo and combined trajectory files under ``demo_dir``."""
    os.makedirs(demo_dir, exist_ok=True)
    if seeds is None:
        seeds = list(range(len(trajectories)))

    traj_arrays = []
    for i, traj in enumerate(trajectories):
        arr = np.array(traj)
        traj_arrays.append(arr)
        np.save(os.path.join(demo_dir, f"demo_{i:04d}.npy"), arr)
        with open(os.path.join(demo_dir, f"demo_{i:04d}.pkl"), "wb") as f:
            pickle.dump({"seed": seeds[i], "obs": list(traj)}, f)

    with open(os.path.join(demo_dir, "all_trajectories.pkl"), "wb") as f:
        pickle.dump(
            {
                "trajectories": traj_arrays,
                "seeds": seeds,
                "num_blocks": num_blocks,
            },
            f,
        )
    np.save(
        os.path.join(demo_dir, "all_trajectories.npy"),
        np.array(traj_arrays, dtype=object),
    )
    if expert_states is not None:
        np.save(os.path.join(demo_dir, "expert_states.npy"), np.array(expert_states))

    print(f"Saved {len(trajectories)} demo trajectories to {demo_dir}/")


def save_numpy_arrays_as_images(arrays, output_dir="images"):
    """
    Save a list of numpy arrays as PNG images with filenames like img0000.png, img0001.png, ...

    Parameters:
    - arrays: list of numpy arrays (each should represent an image)
    - output_dir: directory where images will be saved
    """
    os.makedirs(output_dir, exist_ok=True)

    for idx, arr in enumerate(arrays):
        filename = f"img{idx:04d}.png"
        filepath = os.path.join(output_dir, filename)
        imageio.imwrite(filepath, arr)


def roboverify_env_success(env, final_obs) -> bool:
    """Return whether ``final_obs`` satisfies RoboVerifyStack tower success."""
    if final_obs is None:
        return False
    inner = getattr(env, "env", env)
    if not hasattr(inner, "_is_success"):
        return False
    return bool(inner._is_success(final_obs))


def rollout_demos(
    p: program.Program,
    num_demo: int,
    *,
    num_blocks: int = 4,
    save_imgs: bool = False,
    verbose: bool = True,
) -> Tuple[list, list, list, list]:
    """Roll out ``p`` once per seed without saving to disk.

    Returns ``(individual_traj, flat_states, successes, imgs)``.
    """
    states = []
    imgs = []
    individual_traj = []
    successes = []
    for i in range(num_demo):
        set_np_seed(i)
        env = make_roboverify_stack_env(num_blocks=num_blocks)
        try:
            result = p.eval(env, return_img=save_imgs)
            if save_imgs:
                traj, traj_imgs = result
                imgs.extend(traj_imgs)
            else:
                traj = result
            success = roboverify_env_success(env, traj[-1] if traj else None)
            successes.append(success)
            if verbose:
                print(f"seed {i}: success={success}")
                print("initial layout")
                on.print_block_layout(traj[0], num_blocks)
                print("final layout")
                on.print_block_layout(traj[-1], num_blocks)

            copy_traj = [deepcopy(state) for state in traj]
            individual_traj.append(copy_traj)
            states.extend(copy_traj)
        finally:
            env.close()

    if verbose:
        n_success = sum(successes)
        print(
            f"demo success rate: {n_success}/{num_demo} "
            f"({100.0 * n_success / num_demo:.1f}%)"
        )
        print("number of expert states", len(states))
    return individual_traj, states, successes, imgs


def trajectories_match(
    traj_a: list,
    traj_b: list,
    *,
    atol: float = 1e-5,
) -> Tuple[bool, str]:
    """Return whether two demo trajectories contain the same states."""
    a = np.asarray(traj_a, dtype=np.float64)
    b = np.asarray(traj_b, dtype=np.float64)
    if a.shape != b.shape:
        return False, f"shape mismatch {a.shape} vs {b.shape}"
    if not np.allclose(a, b, atol=atol, rtol=0.0):
        max_diff = float(np.max(np.abs(a - b)))
        return False, f"max abs diff {max_diff:.3e}"
    return True, "states match"


def verify_demo_reproducibility(
    p: program.Program,
    num_demo: int,
    reference_trajs: list,
    *,
    num_blocks: int = 4,
    atol: float = 1e-5,
) -> bool:
    """Re-collect demos and check each seed reproduces ``reference_trajs``."""
    if len(reference_trajs) != num_demo:
        raise ValueError(
            f"reference_trajs has length {len(reference_trajs)}, expected {num_demo}"
        )

    print("=== verifying demo reproducibility (second collection) ===")
    replay_trajs, _, replay_successes, _ = rollout_demos(
        p,
        num_demo,
        num_blocks=num_blocks,
        save_imgs=False,
        verbose=False,
    )

    all_ok = True
    n_matched = 0
    for seed in range(num_demo):
        ok, msg = trajectories_match(
            reference_trajs[seed], replay_trajs[seed], atol=atol
        )
        if ok:
            n_matched += 1
        else:
            all_ok = False
        print(
            f"seed {seed}: reproduced={ok}, success={replay_successes[seed]}"
            + (f" ({msg})" if not ok else "")
        )

    print(
        f"reproducibility: {n_matched}/{num_demo} seeds matched "
        f"({100.0 * n_matched / num_demo:.1f}%)"
    )
    return all_ok


def collect_trajectories(
    p: program.Program,
    num_demo: int,
    *,
    num_blocks: int = 4,
    save_imgs: bool = False,
    demo_dir: Optional[str] = "demos",
    img_dir: str = "images",
    verify_reproducible: bool = False,
    repro_atol: float = 1e-5,
):
    """Roll out ``p`` in RoboVerifyStack ``num_demo`` times with fixed seeds.

    Seed ``i`` is used for demo ``i`` (via :func:`set_np_seed`), so rollouts are
    reproducible. Trajectories are written under ``demo_dir`` (default
    ``"demos"``) via :func:`save_demo_trajectories`.

    When ``verify_reproducible`` is True, runs a second collection and checks
    that every seed yields the same state trajectory.

    Returns flattened states and a list of per-demo trajectories.
    """
    individual_traj, states, _successes, imgs = rollout_demos(
        p,
        num_demo,
        num_blocks=num_blocks,
        save_imgs=save_imgs,
        verbose=True,
    )

    if demo_dir is not None:
        save_demo_trajectories(
            individual_traj,
            demo_dir,
            seeds=list(range(num_demo)),
            num_blocks=num_blocks,
            expert_states=states,
        )

    if save_imgs:
        save_numpy_arrays_as_images(imgs, img_dir)

    if verify_reproducible:
        verify_demo_reproducibility(
            p,
            num_demo,
            individual_traj,
            num_blocks=num_blocks,
            atol=repro_atol,
        )

    return states, individual_traj


def evaluate_program(p: program.Program, n: int, num_block: int, return_img=False):
    states = []
    imgs = []
    individual_traj = []
    for i in range(n):
        set_np_seed(i)
        env_name = f"pickmulti{num_block}"
        env = GymToGymnasium(
            FetchPickAndPlaceConstruction(
                name=env_name,
                sparse=False,
                shaped_reward=False,
                num_blocks=int(num_block),
                reward_type="sparse",
                case="Singletower",
                visualize_mocap=False,
                stack_only=True,
                simple=True,
            ),
        )
        traj = p.eval(env, return_img)
        if return_img:
            traj, traj_imgs = traj
            for image in traj_imgs:
                imgs.append(image)

        copy_traj = []
        for state in traj:
            states.append(state)
            copy_traj.append(deepcopy(state))
        individual_traj.append(copy_traj)

    print("number of policy states", len(states))
    if return_img:
        return states, individual_traj, imgs
    return states, individual_traj, []


class Runner:
    def __init__(
        self, program: program.Program, expert_states, num_seeds: int, num_block: int
    ):
        self.p = program
        self.expert_states = np.array(expert_states)
        self.slices: list[Any] = [slice(None)] * self.expert_states.ndim
        self.slices[1] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 22, 23, 24]
        self.tuple_slices = tuple(self.slices)
        self.expert_states = self.expert_states[self.tuple_slices]
        self.num_seeds = num_seeds
        self.num_block = num_block

    def __call__(self, new_parameters):
        p = deepcopy(self.p)
        p.update_trainable_parameter(new_parameters)
        policy_states, _individual_trajs, _imgs = evaluate_program(
            p, self.num_seeds, self.num_block
        )
        policy_states = np.array(policy_states)[self.tuple_slices]
        kl_value = cost_func.kl_divergence_kde(policy_states, self.expert_states)
        return -kl_value
