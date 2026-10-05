import contextlib
import math
import os
import pickle
import random
import tempfile
from copy import deepcopy
from typing import Any, Callable, Optional, Tuple

import ffmpeg
import imageio
import numpy as np

from synthesis.api import program
from synthesis.cfg.reset import restore
from synthesis.environment.cee_us_env.fpp_construction_env import (
    FetchPickAndPlaceConstruction,
)
from synthesis.environment.general_env import GymToGymnasium
from synthesis.mcmc import cem, cost_func, decision_tree
from synthesis.mcmc.search_core import acceptance_probability, imitation_objective
from synthesis.util import on
from synthesis.verification_lib.bmc_lib import (
    BMCTraceSymbols,
    ConstraintSpec,
    bmc_feasible,
    default_table_initial_state,
    goal_on_box_ids,
)

BMC_FAILED_COST = -1e6
DEFAULT_GOAL_FEATURE_REWARD_WEIGHT = 1.0


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


def executable_instructions(p: program.Program) -> list:
    """Non-Skip instructions that affect execution and BMC."""
    return [ins for ins in p.instructions if type(ins) != program.Skip]


def describe_instruction_slot_changes(
    old_program: program.Program, new_program: program.Program
) -> list[str]:
    """Human-readable per-slot diff between two programs."""
    lines: list[str] = []
    for i, (old_ins, new_ins) in enumerate(
        zip(old_program.instructions, new_program.instructions)
    ):
        if str(old_ins) != str(new_ins):
            lines.append(f"  slot {i}: {old_ins!s} -> {new_ins!s}")
    if not lines:
        lines.append("  (no per-slot instruction changes)")
    return lines


def format_mutation_report(
    mutation_info: dict,
    old_program: program.Program,
    new_program: program.Program,
    *,
    equivalence: bool,
) -> str:
    """Build a detailed mutation report for logging and summaries."""
    lines = [
        f"mutation type: {mutation_info['type']}",
        f"mutation changed program: {mutation_info['changed']}",
    ]
    lines.extend(f"  {detail}" for detail in mutation_info["details"])
    lines.append("per-slot instruction diff:")
    lines.extend(describe_instruction_slot_changes(old_program, new_program))
    lines.append(
        "executable instructions (non-Skip) before: "
        + repr([str(ins) for ins in executable_instructions(old_program)])
    )
    lines.append(
        "executable instructions (non-Skip) after: "
        + repr([str(ins) for ins in executable_instructions(new_program)])
    )
    lines.append(f"executable equivalence: {equivalence}")
    if equivalence and mutation_info["changed"]:
        lines.append(
            "note: mutation changed Skip-only slots or otherwise left "
            "executable instructions identical"
        )
    return "\n".join(lines)


def mutate_program(
    current_program: program.Program,
    available_operands: dict,
    available_instructions: list,
) -> tuple[program.Program, bool, dict]:
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
    mutation_info = {
        "type": mutate_type_name,
        "changed": False,
        "details": [f"sampled mutation operator: {mutate_type_name}"],
    }

    new_program = deepcopy(current_program)
    if sampled_mutation == 0:
        mutation_info["details"].append(
            "opcode mutation is not implemented yet (no-op)"
        )
    elif sampled_mutation == 1:
        index = random.randint(0, len(new_program.instructions) - 1)
        target = new_program.instructions[index]
        old_operands = target.get_operand()
        mutation_info["details"].append(
            f"operand mutation at slot {index} on {target!s}"
        )
        if old_operands:
            operand_index = random.randint(0, len(old_operands) - 1)
            operand_spec = old_operands[operand_index]
            proposed_operand = random.choice(available_operands[operand_spec["type"]])
            mutation_info["details"].append(
                f"operand index {operand_index} ({operand_spec['type']}): "
                f"old={operand_spec.get('val', operand_spec)!r} "
                f"proposed={proposed_operand!r}"
            )
            if operand_spec.get("val", operand_spec) == proposed_operand:
                mutation_info["details"].append(
                    "proposed operand equals current value; no change"
                )
                return new_program, False, mutation_info
            operand_spec["val"] = proposed_operand
            target.set_operand(old_operands)
            mutation_info["changed"] = True
        else:
            mutation_info["details"].append(
                f"instruction at slot {index} has no mutable operands"
            )
    elif sampled_mutation == 2:
        i = random.randint(0, len(new_program.instructions) - 1)
        j = random.randint(0, len(new_program.instructions) - 1)
        before_i, before_j = (
            new_program.instructions[i],
            new_program.instructions[j],
        )
        new_program.instructions[i], new_program.instructions[j] = (
            new_program.instructions[j],
            new_program.instructions[i],
        )
        mutation_info["details"].append(
            f"swap slots {i} and {j}: " f"{before_i!s} <-> {before_j!s}"
        )
        if i == j:
            mutation_info["details"].append("same slot selected twice; swap is a no-op")
        else:
            mutation_info["changed"] = True
    elif sampled_mutation == 3:
        index = random.randint(0, len(new_program.instructions) - 1)
        old_instruction = new_program.instructions[index]
        pu = 0.25  # probability the SKIP token is proposed
        propose_skip = random.random() < pu
        mutation_info["details"].append(
            f"instruction mutation at slot {index}; "
            f"old={old_instruction!s}; propose_skip={propose_skip}"
        )
        if propose_skip:
            if type(old_instruction) == program.Skip:
                mutation_info["details"].append("Skip -> Skip; no change")
                return new_program, False, mutation_info
            new_program.instructions[index] = program.Skip()
            mutation_info["details"].append(
                f"replaced with Skip (was {old_instruction!s})"
            )
            mutation_info["changed"] = True
        else:
            new_instruction = random.choice(available_instructions)()
            operands = new_instruction.get_operand()
            operand_choices = []
            for operand_i in range(len(operands)):
                chosen = random.choice(available_operands[operands[operand_i]["type"]])
                operands[operand_i]["val"] = chosen
                operand_choices.append(f"{operands[operand_i]['type']}={chosen!r}")
            new_instruction.set_operand(operands)
            new_program.instructions[index] = new_instruction
            mutation_info["details"].append(
                f"replaced with {new_instruction!s} "
                f"(operands: {', '.join(operand_choices)})"
            )
            mutation_info["changed"] = True
    else:
        assert False, "unknown mutation"

    return new_program, mutation_info["changed"], mutation_info


def program_instructions_for_bmc(p: program.Program) -> list:
    """Drop ``Skip`` tokens so BMC sees only executable instructions."""
    return [ins for ins in p.instructions if not isinstance(ins, program.Skip)]


def roboverify_bmc_initial_constraints() -> Callable[[BMCTraceSymbols], list]:
    """Scattered tabletop layout aligned with RoboVerifyStack separation."""

    def make_init(sym: BMCTraceSymbols) -> list:
        sep = 2.0 * on.BLOCK_LENGTH
        xy_positions = {b: (float(i) * sep, 0.0) for i, b in enumerate(sym.block_names)}
        return default_table_initial_state(sym, xy_positions=xy_positions)

    return make_init


def bmc_goal_from_on_feature(feature) -> Callable[[BMCTraceSymbols], Any]:
    """Build a BMC goal for ``ON(b1, b2)`` from a learned :class:`ON_feature`."""

    def goal(sym: BMCTraceSymbols):
        return goal_on_box_ids(sym, feature.b1, feature.b2)

    return goal


def goal_feature_reward_at_execution_end(
    feature,
    individual_trajs: list,
    *,
    seeds: Optional[list] = None,
) -> tuple[float, str]:
    """Dense reward for a learned ON feature at each rollout's final state.

    Uses :func:`on.on_reward`, which scores xy alignment and vertical gap on a
    continuous [0, 1] scale aligned with the ON predicate geometry.
    """
    if not individual_trajs:
        return 0.0, f"goal feature {feature}: no trajectories to evaluate"

    per_seed_rewards: list[tuple[Any, float]] = []
    for traj_idx, traj in enumerate(individual_trajs):
        seed_id = seeds[traj_idx] if seeds is not None else traj_idx
        if not traj:
            per_seed_rewards.append((seed_id, 0.0))
            continue
        per_seed_rewards.append((seed_id, float(feature.reward(traj[-1]))))

    reward = float(np.mean([seed_reward for _, seed_reward in per_seed_rewards]))
    seed_report = ", ".join(
        f"{seed_id}={seed_reward:.3f}" for seed_id, seed_reward in per_seed_rewards
    )
    report = (
        f"goal feature {feature}: dense reward={reward:.3f} "
        f"(mean over {len(individual_trajs)} seeds at final state; "
        f"per-seed: {seed_report})"
    )
    return reward, report


def print_goal_feature_reward(label: str, report: str) -> None:
    print(f"=== {label} goal-feature reward ===\n{report}")


def candidate_checks_passed(
    *,
    bmc_goal: Optional[Callable[[BMCTraceSymbols], Any]],
    bmc_passed: bool,
) -> bool:
    """Return whether configured BMC feasibility checks passed."""
    if bmc_goal is not None and not bmc_passed:
        return False
    return True


def check_bmc_candidate(
    p: program.Program,
    goal: Optional[Callable[[BMCTraceSymbols], Any]],
    *,
    initial_constraints: Optional[ConstraintSpec] = None,
) -> tuple[Optional[bool], str]:
    """Run BMC on ``p`` and return ``(feasible, report)``.

    When ``goal`` is ``None``, BMC is disabled and ``feasible`` is ``None``.
    """
    if goal is None:
        return None, "BMC: disabled (no bmc_goal configured)"

    instrs = program_instructions_for_bmc(p)
    if not instrs:
        return False, (
            "BMC: INFEASIBLE — no non-Skip instructions "
            f"({len(p.instructions)} program slots, 0 executable)"
        )

    instr_summary = "; ".join(str(ins) for ins in instrs)
    try:
        feasible = bmc_feasible(
            instrs,
            goal,
            initial_constraints=initial_constraints,
        )
    except KeyError as exc:
        return False, (
            "BMC: INFEASIBLE — goal references blocks not present in program "
            f"({exc}); instructions: [{instr_summary}]"
        )
    except (TypeError, ValueError) as exc:
        return False, (
            "BMC: INFEASIBLE — solver error "
            f"({exc}); instructions: [{instr_summary}]"
        )

    verdict = "FEASIBLE" if feasible else "INFEASIBLE"
    return feasible, (
        f"BMC: {verdict} — {len(instrs)} executable instruction(s): "
        f"[{instr_summary}]"
    )


def bmc_feasible_program(
    p: program.Program,
    goal: Callable[[BMCTraceSymbols], Any],
    *,
    initial_constraints: Optional[ConstraintSpec] = None,
) -> bool:
    """Check whether ``p`` can reach ``goal`` for some float offset assignment."""
    feasible, _report = check_bmc_candidate(
        p, goal, initial_constraints=initial_constraints
    )
    return bool(feasible)


def print_bmc_check(label: str, report: str) -> None:
    print(f"=== {label} BMC check ===\n{report}")


def score_candidate_program(
    p: program.Program,
    expert_states,
    num_seeds: int,
    num_block: int,
    cem_N: int,
    cem_K: int,
    cem_iterations: int,
    *,
    seeds: Optional[list] = None,
    bmc_goal: Optional[Callable[[BMCTraceSymbols], Any]] = None,
    bmc_initial_constraints: Optional[ConstraintSpec] = None,
    bmc_failed_cost: float = BMC_FAILED_COST,
    goal_feature=None,
    goal_feature_reward_weight: float = DEFAULT_GOAL_FEATURE_REWARD_WEIGHT,
    bmc_label: str = "candidate program",
    video_dir: Optional[str] = None,
    video_fps: int = 30,
    motion_penalty=None,
    motion_penalty_weight: float = 1.0,
    initial_snapshots: Optional[dict] = None,
) -> tuple[float, program.Program, bool, str, str]:
    """Optionally require BMC feasibility, then CEM-optimize ``p``."""
    bmc_passed, bmc_report = check_bmc_candidate(
        p,
        bmc_goal,
        initial_constraints=bmc_initial_constraints,
    )
    print_bmc_check(bmc_label, bmc_report)

    if bmc_goal is not None:
        if not bmc_passed:
            print(f"BMC feasibility failed; skipping CEM, cost -> {bmc_failed_cost}")
            return (
                bmc_failed_cost,
                p,
                False,
                bmc_report,
                "goal feature: skipped (BMC failed)",
            )
        print("BMC feasibility passed; running CEM")
    cost, p, _goal_reward, goal_feature_report = optimize_program(
        p,
        expert_states,
        num_seeds,
        num_block,
        cem_N,
        cem_K,
        cem_iterations,
        seeds=seeds,
        initial_snapshots=initial_snapshots,
        goal_feature=goal_feature,
        goal_feature_reward_weight=goal_feature_reward_weight,
        motion_penalty=motion_penalty,
        motion_penalty_weight=motion_penalty_weight,
    )
    print_goal_feature_reward(bmc_label, goal_feature_report)
    checks_passed = candidate_checks_passed(
        bmc_goal=bmc_goal,
        bmc_passed=bool(bmc_passed),
    )
    if video_dir is not None and checks_passed:
        save_program_seed_videos(
            p,
            num_seeds=num_seeds,
            num_block=num_block,
            video_dir=video_dir,
            seeds=seeds,
            initial_snapshots=initial_snapshots,
            video_fps=video_fps,
        )
    return (
        cost,
        p,
        checks_passed,
        bmc_report,
        goal_feature_report,
    )


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
    seeds: Optional[list] = None,
    bmc_goal: Optional[Callable[[BMCTraceSymbols], Any]] = None,
    bmc_initial_constraints: Optional[ConstraintSpec] = None,
    bmc_failed_cost: float = BMC_FAILED_COST,
    goal_feature=None,
    goal_feature_reward_weight: float = DEFAULT_GOAL_FEATURE_REWARD_WEIGHT,
    save_candidate_videos: bool = True,
    video_fps: int = 30,
    motion_penalty=None,
    motion_penalty_weight: float = 1.0,
    initial_snapshots: Optional[dict] = None,
):
    # Ensure save_dir exists
    os.makedirs(save_dir, exist_ok=True)

    print("=== initial program ===\n", current_program)
    initial_video_dir = (
        os.path.join(save_dir, "initial", "videos") if save_candidate_videos else None
    )
    (
        current_cost,
        current_program,
        current_checks_passed,
        current_bmc_report,
        current_goal_report,
    ) = score_candidate_program(
        current_program,
        expert_states,
        num_seeds,
        num_block,
        cem_N,
        cem_K,
        cem_iterations,
        seeds=seeds,
        initial_snapshots=initial_snapshots,
        bmc_goal=bmc_goal,
        bmc_initial_constraints=bmc_initial_constraints,
        bmc_failed_cost=bmc_failed_cost,
        goal_feature=goal_feature,
        goal_feature_reward_weight=goal_feature_reward_weight,
        bmc_label="initial program",
        video_dir=initial_video_dir,
        video_fps=video_fps,
        motion_penalty=motion_penalty,
        motion_penalty_weight=motion_penalty_weight,
    )
    print(
        "evaluated with cost",
        current_cost,
        f"(checks_passed={current_checks_passed})",
    )

    samples = [deepcopy(current_program)]
    costs = [current_cost]

    for i in range(iters):
        iter_folder = os.path.join(save_dir, f"iter{i}")
        os.makedirs(iter_folder, exist_ok=True)

        new_program, changed, mutation_info = mutate_program(
            current_program, available_operands, available_instructions
        )

        equivalence = executable_instructions(
            current_program
        ) == executable_instructions(new_program)
        mutation_report = format_mutation_report(
            mutation_info,
            current_program,
            new_program,
            equivalence=equivalence,
        )
        print(f"=== iter {i} mutation ===\n{mutation_report}")
        print(f"=== iter {i} new program ===\n{new_program}")

        # Save a copy of current_program BEFORE acceptance update
        saved_current_program = deepcopy(current_program)
        saved_current_cost = current_cost
        new_checks_passed = current_checks_passed
        new_bmc_report = current_bmc_report
        new_goal_report = current_goal_report

        candidate_video_dir: Optional[str] = None
        if changed and not equivalence:
            candidate_video_dir = (
                os.path.join(iter_folder, "videos") if save_candidate_videos else None
            )
            (
                new_cost,
                new_program,
                new_checks_passed,
                new_bmc_report,
                new_goal_report,
            ) = score_candidate_program(
                new_program,
                expert_states,
                num_seeds,
                num_block,
                cem_N,
                cem_K,
                cem_iterations,
                seeds=seeds,
                initial_snapshots=initial_snapshots,
                bmc_goal=bmc_goal,
                bmc_initial_constraints=bmc_initial_constraints,
                bmc_failed_cost=bmc_failed_cost,
                goal_feature=goal_feature,
                goal_feature_reward_weight=goal_feature_reward_weight,
                bmc_label=f"iter {i} candidate",
                video_dir=candidate_video_dir,
                video_fps=video_fps,
                motion_penalty=motion_penalty,
                motion_penalty_weight=motion_penalty_weight,
            )
            print(
                "evaluated with cost",
                new_cost,
                f"(checks_passed={new_checks_passed})",
            )
        else:
            new_cost = current_cost
            candidate_video_dir = None
            if equivalence:
                new_bmc_report = (
                    "BMC: skipped (executable equivalence); "
                    f"reusing prior result -> checks_passed={current_checks_passed}"
                )
                new_goal_report = (
                    "goal feature: skipped (executable equivalence); "
                    f"reusing prior result -> checks_passed={current_checks_passed}"
                )
                print(
                    "program equivalence: skipping cost evaluation "
                    "(reusing current cost)"
                )
            else:
                new_bmc_report = (
                    "BMC: skipped (mutation produced no change); "
                    f"reusing prior result -> checks_passed={current_checks_passed}"
                )
                new_goal_report = (
                    "goal feature: skipped (mutation produced no change); "
                    f"reusing prior result -> checks_passed={current_checks_passed}"
                )
                print("mutation produced no change: reusing current cost")
            print_bmc_check(f"iter {i} candidate", new_bmc_report)
            print_goal_feature_reward(f"iter {i} candidate", new_goal_report)

        # Save a copy of new_program AFTER optimization
        saved_new_program = deepcopy(new_program)

        # Decide acceptance
        if changed and not equivalence:
            acceptance_ratio = acceptance_probability(new_cost - current_cost)
            print("acceptance_ratio is", acceptance_ratio)
            if random.random() < acceptance_ratio:
                print("new program accepted")
                current_program = new_program
                current_cost = new_cost
                current_checks_passed = new_checks_passed
                current_bmc_report = new_bmc_report
                current_goal_report = new_goal_report
            else:
                print("new program NOT accepted")

        # Save pickle files
        with open(os.path.join(iter_folder, "current_program.pkl"), "wb") as f:
            pickle.dump(saved_current_program, f)
        with open(os.path.join(iter_folder, "new_program.pkl"), "wb") as f:
            pickle.dump(saved_new_program, f)

        eval_skipped = equivalence or not new_checks_passed
        rollout_skip_reasons: list[str] = []
        if equivalence:
            rollout_skip_reasons.append("executable equivalence")
        if not new_checks_passed:
            rollout_skip_reasons.append("candidate checks failed")
        # Save text summary
        with open(os.path.join(iter_folder, "summary.txt"), "w") as f:
            f.write("=== Mutation Report ===\n")
            f.write(mutation_report + "\n\n")
            f.write("=== Current Program ===\n")
            f.write(str(saved_current_program) + "\n")
            f.write(f"Current Cost: {saved_current_cost}\n")
            f.write(f"Current BMC: {current_bmc_report}\n")
            f.write(f"Current Goal Feature: {current_goal_report}\n\n")
            f.write("=== New Program ===\n")
            f.write(str(saved_new_program) + "\n")
            f.write(f"New Cost: {new_cost}\n")
            f.write(f"New BMC: {new_bmc_report}\n")
            f.write(f"New Goal Feature: {new_goal_report}\n")
            f.write(f"Checks Passed: {new_checks_passed}\n")
            f.write(f"Program Equivalence: {equivalence}\n")
            f.write(f"Accepted: {current_program == new_program}\n")
            if candidate_video_dir is not None:
                f.write(f"Videos: {candidate_video_dir}/seed_XXXX.mp4\n")
            elif eval_skipped:
                f.write(
                    "Evaluation/Video: skipped ("
                    + "; ".join(rollout_skip_reasons)
                    + ")\n"
                )

        if eval_skipped and save_candidate_videos:
            print(
                "skipping candidate rollout videos ("
                + "; ".join(rollout_skip_reasons)
                + ")"
            )

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
    seeds: Optional[list] = None,
    goal_feature=None,
    goal_feature_reward_weight: float = DEFAULT_GOAL_FEATURE_REWARD_WEIGHT,
    motion_penalty=None,
    motion_penalty_weight: float = 1.0,
    initial_snapshots: Optional[dict] = None,
) -> tuple[float, program.Program, float, str]:
    f = Runner(
        p,
        expert_states,
        num_seeds,
        num_block,
        seeds=seeds,
        initial_snapshots=initial_snapshots,
        goal_feature=goal_feature,
        goal_feature_reward_weight=goal_feature_reward_weight,
        motion_penalty=motion_penalty,
        motion_penalty_weight=motion_penalty_weight,
    )
    initial_parameters = p.register_trainable_parameter()
    iterations = cem_iterations if initial_parameters else 0
    best_cost, best_parameter = cem.cem_optimize(
        f,
        len(initial_parameters),
        iterations,
        N=cem_N,
        K=cem_K,
        init_mu=initial_parameters,
        num_workers=0 if motion_penalty is not None else None,
    )
    p.update_trainable_parameter(best_parameter)
    if goal_feature is not None:
        best_cost = f(best_parameter)
        return best_cost, p, f.last_goal_feature_reward, f.last_goal_feature_report
    return best_cost, p, 0.0, "goal feature: disabled"


def set_np_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)


@contextlib.contextmanager
def preserved_global_rng():
    """Restore the global ``numpy``/``random`` state when the block exits.

    Rollouts must reseed the global RNG per demo: the environment draws its
    initial layout from ``numpy.random``, and a policy rollout is only
    comparable to its demonstration if both start from the same state. That
    property is load-bearing -- the KL/MMD objective is meaningless without it.

    The problem is that the seeding also clobbers the *caller's* stream, and
    the caller is the optimizer. ``cem_optimize`` draws its perturbations with
    ``np.random.randn(N, dim)`` in the parent process and then calls ``f(mu)``
    in the parent at the end of every iteration; ``f`` runs rollouts, which
    reset the global RNG to a state fixed by the last demo seed. Every
    iteration therefore began from the same state and drew the *identical*
    N x dim perturbation matrix, so CEM re-explored one frozen set of
    directions for the whole optimization instead of sampling fresh ones.

    Wrapping the rollout loops keeps both properties: each rollout still gets
    its demo seed, and whatever stream the caller was drawing from is handed
    back untouched.
    """
    np_state = np.random.get_state()
    py_state = random.getstate()
    try:
        yield
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)


# Shared RoboVerifyStack settings for demo collection and MCMC evaluation.
ROBOVERIFY_STACK_ENV_KWARGS = {
    "sparse": False,
    "shaped_reward": False,
    "reward_type": "sparse",
    "case": "RoboVerifyStack",
    "visualize_mocap": False,
    "simple": True,
    "base_block_id": 0,
}

ROBOVERIFY_UNSTACK_ENV_KWARGS = {
    "sparse": False,
    "shaped_reward": False,
    "reward_type": "sparse",
    "case": "RoboVerifyUnstack",
    "visualize_mocap": False,
    "simple": True,
    "base_block_id": 0,
}

ROBOVERIFY_REVERSE_ENV_KWARGS = {
    "sparse": False,
    "shaped_reward": False,
    "reward_type": "sparse",
    "case": "RoboVerifyReverse",
    "visualize_mocap": False,
    "simple": True,
    "base_block_id": 0,
}

ROBOVERIFY_PARTIAL_STACK_ENV_KWARGS = {
    "sparse": False,
    "shaped_reward": False,
    "reward_type": "sparse",
    "case": "RoboVerifyPartialStack",
    "visualize_mocap": False,
    "simple": True,
    "base_block_id": 0,
}

ROBOVERIFY_GRID_ENV_KWARGS = {
    "sparse": False,
    "shaped_reward": False,
    "reward_type": "sparse",
    "case": "RoboVerifyGrid",
    "visualize_mocap": False,
    "simple": True,
    "visualize_target": True,
}

ROBOVERIFY_PYRAMID_ENV_KWARGS = {
    "sparse": False,
    "shaped_reward": False,
    "reward_type": "sparse",
    "case": "RoboVerifyPyramid",
    "visualize_mocap": False,
    "simple": True,
    "visualize_target": True,
}


def make_roboverify_stack_env(
    num_blocks: int = 4,
    render_mode: str = "rgb_array",
) -> GymToGymnasium:
    """Create the RoboVerifyStack env used for stack demo rollouts and MCMC eval."""
    return GymToGymnasium(
        FetchPickAndPlaceConstruction(
            name=f"roboverify_stack_{num_blocks}",
            num_blocks=num_blocks,
            **ROBOVERIFY_STACK_ENV_KWARGS,
        ),
        render_mode=render_mode,
    )


def make_roboverify_unstack_env(
    num_blocks: int = 4,
    render_mode: str = "rgb_array",
) -> GymToGymnasium:
    """Create the RoboVerifyUnstack env for unstack demo rollouts and MCMC eval."""
    return GymToGymnasium(
        FetchPickAndPlaceConstruction(
            name=f"roboverify_unstack_{num_blocks}",
            num_blocks=num_blocks,
            **ROBOVERIFY_UNSTACK_ENV_KWARGS,
        ),
        render_mode=render_mode,
    )


def make_roboverify_reverse_env(
    num_blocks: int = 4,
    render_mode: str = "rgb_array",
) -> GymToGymnasium:
    """Create the RoboVerifyReverse env for reverse demo rollouts and MCMC eval."""
    return GymToGymnasium(
        FetchPickAndPlaceConstruction(
            name=f"roboverify_reverse_{num_blocks}",
            num_blocks=num_blocks,
            **ROBOVERIFY_REVERSE_ENV_KWARGS,
        ),
        render_mode=render_mode,
    )


def make_roboverify_partial_stack_env(
    num_blocks: int = 4,
    render_mode: str = "rgb_array",
) -> GymToGymnasium:
    """Create the RoboVerifyPartialStack env for partial-stack rollouts and MCMC eval."""
    return GymToGymnasium(
        FetchPickAndPlaceConstruction(
            name=f"roboverify_partial_stack_{num_blocks}",
            num_blocks=num_blocks,
            **ROBOVERIFY_PARTIAL_STACK_ENV_KWARGS,
        ),
        render_mode=render_mode,
    )


def make_roboverify_grid_env(
    grid_rows: int = 3,
    grid_cols: int = 2,
    render_mode: str = "rgb_array",
) -> GymToGymnasium:
    """Create the RoboVerifyGrid env with ``grid_rows * grid_cols`` blocks."""
    return GymToGymnasium(
        FetchPickAndPlaceConstruction(
            name=f"roboverify_grid_{grid_rows}x{grid_cols}",
            num_blocks=grid_rows * grid_cols,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            **ROBOVERIFY_GRID_ENV_KWARGS,
        ),
        render_mode=render_mode,
    )


def make_roboverify_pyramid_env(
    render_mode: str = "rgb_array",
) -> GymToGymnasium:
    """Create the RoboVerifyPyramid env with six blocks in a 3+2+1 goal pyramid."""
    return GymToGymnasium(
        FetchPickAndPlaceConstruction(
            name="roboverify_pyramid_6",
            num_blocks=6,
            **ROBOVERIFY_PYRAMID_ENV_KWARGS,
        ),
        render_mode=render_mode,
    )


def make_roboverify_env(
    task: str,
    num_blocks: int = 4,
    render_mode: str = "rgb_array",
) -> GymToGymnasium:
    """Create a RoboVerify env for stack / unstack / reverse / partial_stack tasks."""
    if task == "stack":
        return make_roboverify_stack_env(num_blocks=num_blocks, render_mode=render_mode)
    if task == "unstack":
        return make_roboverify_unstack_env(
            num_blocks=num_blocks, render_mode=render_mode
        )
    if task == "reverse":
        return make_roboverify_reverse_env(
            num_blocks=num_blocks, render_mode=render_mode
        )
    if task in ("partial_stack", "partial"):
        return make_roboverify_partial_stack_env(
            num_blocks=num_blocks, render_mode=render_mode
        )
    if task in ("grid", "2dgrid"):
        raise ValueError(
            "use make_roboverify_grid_env(grid_rows=..., grid_cols=...) for grid tasks"
        )
    if task == "pyramid":
        raise ValueError("use make_roboverify_pyramid_env() for pyramid tasks")
    raise ValueError(f"unsupported RoboVerify task: {task!r}")


def save_frames_as_video(frames: list, output_path: str, *, fps: int = 30) -> None:
    """Write a list of RGB frames to an mp4 file."""
    if not frames:
        return
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        for idx, frame in enumerate(frames):
            imageio.imwrite(
                os.path.join(tmp_dir, f"img{idx:04d}.png"),
                np.asarray(frame, dtype=np.uint8),
            )
        input_pattern = os.path.join(tmp_dir, "img%04d.png")
        (
            ffmpeg.input(input_pattern, framerate=fps)
            .output(output_path, vcodec="libx264", pix_fmt="yuv420p")
            .overwrite_output()
            .run(quiet=True)
        )
    print(f"Video saved to {output_path}")


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


def group_trajectory_images(flat_imgs: list, trajectories: list) -> list[list]:
    """Group a flat frame list into one sequence per trajectory."""
    expected = sum(len(traj) for traj in trajectories)
    if len(flat_imgs) != expected:
        raise ValueError(
            f"expected {expected} frames for {len(trajectories)} trajectories, "
            f"got {len(flat_imgs)}"
        )
    grouped: list[list] = []
    offset = 0
    for traj in trajectories:
        length = len(traj)
        grouped.append(flat_imgs[offset : offset + length])
        offset += length
    return grouped


def load_demo_images_grouped(img_dir: str, trajectories: list) -> list[list]:
    """Load ``imgXXXX.png`` frames from ``img_dir`` and group them per demo."""
    expected = sum(len(traj) for traj in trajectories)
    flat_imgs: list = []
    for idx in range(expected):
        img_path = os.path.join(img_dir, f"img{idx:04d}.png")
        if not os.path.exists(img_path):
            raise FileNotFoundError(
                f"missing {img_path} (expected {expected} frames for "
                f"{len(trajectories)} demos)"
            )
        flat_imgs.append(imageio.imread(img_path))
    return group_trajectory_images(flat_imgs, trajectories)


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
    """Return whether ``final_obs`` satisfies the env's RoboVerify task success."""
    if final_obs is None:
        return False
    inner = getattr(env, "env", env)
    if not hasattr(inner, "_is_success"):
        return False
    return bool(inner._is_success(final_obs))


@preserved_global_rng()
def rollout_demos(
    p: program.Program,
    num_demo: int,
    *,
    num_blocks: int = 4,
    save_imgs: bool = False,
    verbose: bool = True,
    seeds: Optional[list] = None,
    task: str = "stack",
    initial_snapshots: Optional[dict] = None,
) -> Tuple[list, list, list, list]:
    """Roll out ``p`` once per seed without saving to disk.

    ``initial_snapshots`` maps each seed to its archived full simulator start.
    Supplied starts are restored directly, with no reset or additional settling.
    Returns ``(individual_traj, flat_states, successes, imgs)``.
    """
    rollout_seeds = list(seeds) if seeds is not None else list(range(num_demo))
    if len(rollout_seeds) != num_demo:
        raise ValueError(
            f"seeds length {len(rollout_seeds)} must match num_demo {num_demo}"
        )

    states = []
    imgs = []
    individual_traj = []
    successes = []
    if initial_snapshots is not None:
        missing = set(rollout_seeds) - initial_snapshots.keys()
        if missing:
            raise ValueError(
                f"Missing saved initial snapshots for seeds {sorted(missing)}"
            )
    for seed in rollout_seeds:
        set_np_seed(seed)
        env = make_roboverify_env(task, num_blocks=num_blocks)
        try:
            first = (
                None
                if initial_snapshots is None
                else restore(env, initial_snapshots[seed])
            )
            result = p.eval(env, return_img=save_imgs, initial_observation=first)
            if save_imgs:
                traj, traj_imgs = result
                imgs.extend(traj_imgs)
            else:
                traj = result
            success = roboverify_env_success(env, traj[-1] if traj else None)
            successes.append(success)
            if verbose:
                print(f"seed {seed}: success={success}")
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


@preserved_global_rng()
def save_program_seed_videos(
    p: program.Program,
    *,
    num_seeds: int,
    num_block: int,
    video_dir: str,
    seeds: Optional[list] = None,
    video_fps: int = 30,
    verbose: bool = True,
    task: str = "stack",
    initial_snapshots: Optional[dict] = None,
) -> list[str]:
    """Roll out ``p`` once per seed and write ``seed_XXXX.mp4`` under ``video_dir``."""
    os.makedirs(video_dir, exist_ok=True)
    rollout_seeds = list(seeds) if seeds is not None else list(range(num_seeds))
    if len(rollout_seeds) != num_seeds:
        raise ValueError(
            f"seeds length {len(rollout_seeds)} must match num_seeds {num_seeds}"
        )

    saved_paths: list[str] = []
    if initial_snapshots is not None:
        missing = set(rollout_seeds) - initial_snapshots.keys()
        if missing:
            raise ValueError(
                f"Missing saved initial snapshots for seeds {sorted(missing)}"
            )
    for seed in rollout_seeds:
        set_np_seed(seed)
        env = make_roboverify_env(task, num_blocks=num_block)
        try:
            first = (
                None
                if initial_snapshots is None
                else restore(env, initial_snapshots[seed])
            )
            traj, imgs = p.eval(env, return_img=True, initial_observation=first)
            success = roboverify_env_success(env, traj[-1] if traj else None)
            if verbose:
                print(
                    f"seed {seed}: success={success}, "
                    f"frames={len(imgs)}, saving video"
                )
            output_path = os.path.join(video_dir, f"seed_{seed:04d}.mp4")
            save_frames_as_video(imgs, output_path, fps=video_fps)
            saved_paths.append(output_path)
        finally:
            env.close()
    return saved_paths


@preserved_global_rng()
def rollout_demos_from_initial_states(
    p: program.Program,
    initial_states: list,
    *,
    num_blocks: int = 4,
    save_imgs: bool = False,
    video_dir: Optional[str] = None,
    demo_indices: Optional[list] = None,
    video_fps: int = 30,
    verbose: bool = True,
    task: str = "stack",
) -> Tuple[list, list, list, list]:
    """Roll out ``p`` from fixed initial observations (one per demo).

    Unlike :func:`rollout_demos`, the environment is not reset; each rollout
    starts from the corresponding entry in ``initial_states``. When ``video_dir``
    is set, saves ``random_program.mp4`` for each rollout under
    ``video_dir/demo_XXXX/``.
    """
    if not initial_states:
        return [], [], [], []

    states: list = []
    imgs: list = []
    individual_traj: list = []
    successes: list = []
    imgs_per_demo: list = []

    if demo_indices is not None and len(demo_indices) != len(initial_states):
        raise ValueError("demo_indices must have the same length as initial_states")

    for rollout_idx, initial_state in enumerate(initial_states):
        demo_idx = (
            demo_indices[rollout_idx] if demo_indices is not None else rollout_idx
        )
        set_np_seed(demo_idx)
        env = make_roboverify_env(task, num_blocks=num_blocks)
        try:
            result = p.eval_from_observation(
                env, initial_state, return_img=save_imgs or video_dir is not None
            )
            capture_imgs = save_imgs or video_dir is not None
            if capture_imgs:
                traj, traj_imgs = result
                if save_imgs:
                    imgs.extend(traj_imgs)
                imgs_per_demo.append(traj_imgs)
            else:
                traj = result
                imgs_per_demo.append([])

            success = roboverify_env_success(env, traj[-1] if traj else None)
            successes.append(success)
            if verbose:
                print(
                    f"demo {demo_idx} (from checkpoint): success={success}, "
                    f"{len(traj)} states"
                )

            copy_traj = [deepcopy(state) for state in traj]
            individual_traj.append(copy_traj)
            states.extend(copy_traj)
        finally:
            env.close()

    if video_dir is not None:
        os.makedirs(video_dir, exist_ok=True)
        for rollout_idx, traj_imgs in enumerate(imgs_per_demo):
            if not traj_imgs:
                continue
            demo_idx = (
                demo_indices[rollout_idx] if demo_indices is not None else rollout_idx
            )
            demo_video_dir = os.path.join(video_dir, f"demo_{demo_idx:04d}")
            save_frames_as_video(
                traj_imgs,
                os.path.join(demo_video_dir, "random_program.mp4"),
                fps=video_fps,
            )

    if verbose:
        n_success = sum(successes)
        print(
            f"checkpoint rollout success rate: {n_success}/{len(initial_states)} "
            f"({100.0 * n_success / len(initial_states):.1f}%)"
        )
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
    task: str = "stack",
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
        task=task,
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
    img_dir: str = "images",
    verify_reproducible: bool = False,
    repro_atol: float = 1e-5,
    task: str = "stack",
):
    """Roll out ``p`` in a RoboVerify env ``num_demo`` times with fixed seeds.

    ``task`` is ``"stack"`` (RoboVerifyStack) or ``"unstack"`` (RoboVerifyUnstack).

    Seed ``i`` is used for demo ``i`` (via :func:`set_np_seed`), so rollouts are
    reproducible. This helper returns observations in memory. Use
    synthesis.entry.collect_demos to persist validated resettable demonstrations.

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
        task=task,
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
            task=task,
        )

    return states, individual_traj


def evaluate_program(
    p: program.Program,
    n: int,
    num_block: int,
    return_img: bool = False,
    seeds: Optional[list] = None,
    task: str = "stack",
    initial_snapshots: Optional[dict] = None,
):
    """Evaluate ``p`` using the same RoboVerify env as demo collection."""
    individual_traj, states, _successes, imgs = rollout_demos(
        p,
        n,
        num_blocks=num_block,
        save_imgs=return_img,
        verbose=False,
        seeds=seeds,
        initial_snapshots=initial_snapshots,
        task=task,
    )
    print("number of policy states", len(states))
    if return_img:
        return states, individual_traj, imgs
    return states, individual_traj, []


class Runner:
    def __init__(
        self,
        program: program.Program,
        expert_states,
        num_seeds: int,
        num_block: int,
        seeds: Optional[list] = None,
        goal_feature=None,
        goal_feature_reward_weight: float = DEFAULT_GOAL_FEATURE_REWARD_WEIGHT,
        motion_penalty=None,
        motion_penalty_weight: float = 1.0,
        initial_snapshots: Optional[dict] = None,
    ):
        if not np.isfinite(motion_penalty_weight) or motion_penalty_weight < 0:
            raise ValueError("Motion penalty weight must be finite and nonnegative")
        self.motion_penalty = motion_penalty
        self.motion_penalty_weight = motion_penalty_weight
        self.last_motion_penalty = 0
        self.p = program
        self.expert_states = np.array(expert_states)
        self.slices: list[Any] = [slice(None)] * self.expert_states.ndim
        self.slices[1] = on.state_comparison_indices(num_block)
        self.tuple_slices = tuple(self.slices)
        self.expert_states = self.expert_states[self.tuple_slices]
        self.num_seeds = num_seeds
        self.num_block = num_block
        self.seeds = seeds
        self.initial_snapshots = initial_snapshots
        self.goal_feature = goal_feature
        self.goal_feature_reward_weight = goal_feature_reward_weight
        self.last_goal_feature_reward = 0.0
        self.last_goal_feature_report = "goal feature: disabled"

    def __call__(self, new_parameters):
        p = deepcopy(self.p)
        p.update_trainable_parameter(new_parameters)
        policy_states, individual_trajs, _imgs = evaluate_program(
            p,
            self.num_seeds,
            self.num_block,
            seeds=self.seeds,
            initial_snapshots=self.initial_snapshots,
        )
        policy_states = np.array(policy_states)[self.tuple_slices]
        mmd_value = cost_func.maximum_mean_discrepancy_rbf(
            policy_states, self.expert_states
        )
        score = imitation_objective(mmd_value)
        if self.goal_feature is not None:
            goal_reward, goal_feature_report = goal_feature_reward_at_execution_end(
                self.goal_feature,
                individual_trajs,
                seeds=self.seeds,
            )
            self.last_goal_feature_reward = goal_reward
            self.last_goal_feature_report = goal_feature_report
            score += self.goal_feature_reward_weight * goal_reward
        if self.motion_penalty is not None:
            self.last_motion_penalty = self.motion_penalty(p)
            score -= self.motion_penalty_weight * self.last_motion_penalty
        return score
