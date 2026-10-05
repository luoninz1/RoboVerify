"""Instrumented MCMC program search.

Same search as :func:`synthesis.mcmc.synthesis.MCMC` -- same mutation operators, same
acceptance rule, same RNG consumption -- with the per-iteration prints replaced by
structured records, and with metrics chosen so that a bad run can be diagnosed rather
than merely observed.

Three deliberate differences from the original, each of which shows up in
``config.json`` so a result is never silently attributed to the wrong behaviour:

``beta``
    The acceptance rule becomes ``exp(beta * cost_delta)``. ``beta=1.0`` is the
    original.
``video_policy``
    Videos are rendered for new bests only by default, instead of for every
    candidate, and they are rendered *after* the acceptance draw so that the video
    setting cannot perturb the Markov chain.
``task`` threading
    The objective is rolled out on the configured task. The original
    ``Runner.__call__`` calls ``evaluate_program`` without ``task``, so it always
    evaluated on the ``stack`` environment regardless of the task being searched.
"""

import math
import pickle
import random
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from synthesis.experiment.mcmc import cem as instrumented_cem
from synthesis.experiment.run_logger import RollingRate, RunLogger
from synthesis.mcmc import cost_func
from synthesis.mcmc.search_core import acceptance_probability, imitation_objective
from synthesis.mcmc.synthesis import (
    BMC_FAILED_COST,
    DEFAULT_GOAL_FEATURE_REWARD_WEIGHT,
    Runner,
    candidate_checks_passed,
    check_bmc_candidate,
    executable_instructions,
    format_mutation_report,
    goal_feature_reward_at_execution_end,
    mutate_program,
    rollout_demos,
    save_program_seed_videos,
)

# Short labels for the branches of ``check_bmc_candidate``, so that "the search
# never finds a feasible program" can be split by *why* in a report histogram.
BMC_REASON_PATTERNS = (
    ("disabled", "disabled"),
    ("no non-Skip instructions", "empty_program"),
    ("goal references blocks not present", "missing_block"),
    ("solver error", "solver_error"),
    # "INFEASIBLE" must be tested before "FEASIBLE", which is a substring of it.
    ("INFEASIBLE", "infeasible"),
    ("FEASIBLE", "feasible"),
)


def classify_bmc_report(report: str) -> str:
    """Map a ``check_bmc_candidate`` report onto a short reason label.

    This matches on the report text rather than restructuring
    ``check_bmc_candidate``, which keeps ``synthesis/mcmc/`` untouched. Pattern
    order is load-bearing twice over: the specific failure phrases come before the
    generic verdicts (a failure report also says "INFEASIBLE"), and "INFEASIBLE"
    comes before "FEASIBLE" because it contains it as a substring.
    """
    for needle, label in BMC_REASON_PATTERNS:
        if needle in report:
            return label
    return "unknown"


class InstrumentedRunner(Runner):
    """Objective that also reports its components and task success.

    The base :class:`~synthesis.mcmc.synthesis.Runner` routes through
    ``evaluate_program``, which computes per-seed task success in ``rollout_demos``
    and then discards it (``synthesis.py:1357`` binds it to ``_successes``). Task
    success is the only direct measure of whether a synthesized program actually
    performs the manipulation, so this subclass calls ``rollout_demos`` directly and
    keeps it, along with the MMD and goal-reward terms separately -- an objective
    whose two components are only ever observed summed cannot be debugged.
    """

    def __init__(self, *args: Any, task: str = "stack", **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.task = task
        self.last_mmd: Optional[float] = None
        self.last_success_rate: Optional[float] = None
        self.last_score: Optional[float] = None

    def __call__(self, new_parameters):
        candidate = deepcopy(self.p)
        candidate.update_trainable_parameter(new_parameters)

        individual_trajs, states, successes, _imgs = rollout_demos(
            candidate,
            self.num_seeds,
            num_blocks=self.num_block,
            save_imgs=False,
            verbose=False,
            seeds=self.seeds,
            initial_snapshots=self.initial_snapshots,
            task=self.task,
        )

        policy_states = np.array(states)[self.tuple_slices]
        mmd_value = cost_func.maximum_mean_discrepancy_rbf(
            policy_states, self.expert_states
        )
        score = imitation_objective(mmd_value)
        self.last_mmd = float(mmd_value)
        self.last_success_rate = float(np.mean(successes)) if len(successes) else None

        if self.goal_feature is not None:
            goal_reward, goal_report = goal_feature_reward_at_execution_end(
                self.goal_feature, individual_trajs, seeds=self.seeds
            )
            self.last_goal_feature_reward = goal_reward
            self.last_goal_feature_report = goal_report
            score += self.goal_feature_reward_weight * goal_reward

        if self.motion_penalty is not None:
            self.last_motion_penalty = self.motion_penalty(candidate)
            score -= self.motion_penalty_weight * self.last_motion_penalty
        self.last_score = float(score)
        return score


@dataclass
class CandidateResult:
    """Everything learned from evaluating one candidate program."""

    cost: float
    program: Any
    checks_passed: bool
    bmc_feasible: Optional[bool]
    bmc_reason: str
    bmc_report: str
    goal_report: str
    mmd: Optional[float] = None
    goal_reward: Optional[float] = None
    success_rate: Optional[float] = None
    cem: Optional[instrumented_cem.CEMStats] = None
    scored: bool = True

    def metrics(self) -> dict:
        fields = {
            "cost": round(float(self.cost), 6),
            "checks_passed": self.checks_passed,
            "bmc_feasible": self.bmc_feasible,
            "bmc_reason": self.bmc_reason,
            "mmd": None if self.mmd is None else round(self.mmd, 6),
            "goal_reward": (
                None if self.goal_reward is None else round(float(self.goal_reward), 6)
            ),
            "success_rate": self.success_rate,
            "scored": self.scored,
        }
        if self.cem is not None:
            fields.update(self.cem.summary())
        return fields


@dataclass
class MCMCResult:
    """Outcome of a search."""

    best_program: Any = None
    best_cost: float = -math.inf
    best_iter: Optional[int] = None
    costs: list = field(default_factory=list)
    recent_programs: list = field(default_factory=list)
    iters_completed: int = 0


def optimize_program(
    p,
    expert_states,
    num_seeds: int,
    num_block: int,
    cem_N: int,
    cem_K: int,
    cem_iterations: int,
    *,
    seeds: Optional[list] = None,
    task: str = "stack",
    goal_feature=None,
    goal_feature_reward_weight: float = DEFAULT_GOAL_FEATURE_REWARD_WEIGHT,
    cem_init_std: float = 0.1,
    refresh_best_metrics: bool = True,
    motion_penalty=None,
    motion_penalty_weight: float = 1.0,
    initial_snapshots: Optional[dict] = None,
) -> tuple:
    """CEM-optimize ``p``'s float offsets and report what happened.

    ``refresh_best_metrics`` re-evaluates the objective at the winning parameters in
    the parent process. This is necessary to observe MMD, goal reward and success
    rate for the chosen parameters at all, because CEM evaluates candidates in a
    ``multiprocessing`` pool and attribute updates inside a worker do not propagate
    back. The original does the same re-evaluation, but only when a goal feature is
    configured; turning it on unconditionally costs one extra rollout batch and
    changes the RNG stream, so it is a flag rather than a default in the parity
    test.
    """
    runner = InstrumentedRunner(
        p,
        expert_states,
        num_seeds,
        num_block,
        seeds=seeds,
        initial_snapshots=initial_snapshots,
        task=task,
        goal_feature=goal_feature,
        goal_feature_reward_weight=goal_feature_reward_weight,
        motion_penalty=motion_penalty,
        motion_penalty_weight=motion_penalty_weight,
    )
    initial_parameters = p.register_trainable_parameter()
    iterations = cem_iterations if initial_parameters else 0

    best_cost, best_parameter, cem_stats = instrumented_cem.cem_optimize(
        runner,
        len(initial_parameters),
        iterations,
        N=cem_N,
        K=cem_K,
        init_mu=initial_parameters,
        init_std=cem_init_std,
        num_workers=0 if motion_penalty is not None else None,
    )
    p.update_trainable_parameter(best_parameter)

    if goal_feature is not None or refresh_best_metrics:
        best_cost = runner(best_parameter)

    return best_cost, p, runner, cem_stats


def score_candidate_program(
    p,
    expert_states,
    num_seeds: int,
    num_block: int,
    cem_N: int,
    cem_K: int,
    cem_iterations: int,
    *,
    seeds: Optional[list] = None,
    task: str = "stack",
    bmc_goal: Optional[Callable] = None,
    bmc_initial_constraints=None,
    bmc_failed_cost: float = BMC_FAILED_COST,
    goal_feature=None,
    goal_feature_reward_weight: float = DEFAULT_GOAL_FEATURE_REWARD_WEIGHT,
    cem_init_std: float = 0.1,
    refresh_best_metrics: bool = True,
    motion_penalty=None,
    motion_penalty_weight: float = 1.0,
    initial_snapshots: Optional[dict] = None,
) -> CandidateResult:
    """Require BMC feasibility if configured, then optimize ``p``'s parameters."""
    bmc_feasible, bmc_report = check_bmc_candidate(
        p, bmc_goal, initial_constraints=bmc_initial_constraints
    )
    bmc_reason = classify_bmc_report(bmc_report)

    if bmc_goal is not None and not bmc_feasible:
        return CandidateResult(
            cost=bmc_failed_cost,
            program=p,
            checks_passed=False,
            bmc_feasible=False,
            bmc_reason=bmc_reason,
            bmc_report=bmc_report,
            goal_report="goal feature: skipped (BMC failed)",
        )

    cost, p, runner, cem_stats = optimize_program(
        p,
        expert_states,
        num_seeds,
        num_block,
        cem_N,
        cem_K,
        cem_iterations,
        seeds=seeds,
        initial_snapshots=initial_snapshots,
        task=task,
        goal_feature=goal_feature,
        goal_feature_reward_weight=goal_feature_reward_weight,
        cem_init_std=cem_init_std,
        refresh_best_metrics=refresh_best_metrics,
        motion_penalty=motion_penalty,
        motion_penalty_weight=motion_penalty_weight,
    )

    return CandidateResult(
        cost=cost,
        program=p,
        checks_passed=candidate_checks_passed(
            bmc_goal=bmc_goal, bmc_passed=bool(bmc_feasible)
        ),
        bmc_feasible=bmc_feasible,
        bmc_reason=bmc_reason,
        bmc_report=bmc_report,
        goal_report=runner.last_goal_feature_report,
        mmd=runner.last_mmd,
        goal_reward=(
            runner.last_goal_feature_reward if goal_feature is not None else None
        ),
        success_rate=runner.last_success_rate,
        cem=cem_stats,
    )


def MCMC(
    initial_program,
    available_operands: dict,
    available_instructions: list,
    config,
    expert_states,
    *,
    logger: RunLogger,
    seeds: Optional[list] = None,
    bmc_goal: Optional[Callable] = None,
    bmc_initial_constraints=None,
    goal_feature=None,
    refresh_best_metrics: bool = True,
    motion_penalty=None,
    motion_penalty_weight: float = 1.0,
    initial_snapshots: Optional[dict] = None,
) -> MCMCResult:
    """Run the instrumented search, writing records through ``logger``."""
    num_seeds = config.num_seeds if config.num_seeds is not None else len(seeds or [])
    result = MCMCResult()
    accept_rate = RollingRate(config.rolling_window)
    feasible_rate = RollingRate(config.rolling_window)
    floor_rate = RollingRate(config.rolling_window)
    first_feasible_iter: Optional[int] = None

    def score(candidate) -> CandidateResult:
        return score_candidate_program(
            candidate,
            expert_states,
            num_seeds,
            config.num_blocks,
            config.cem_N,
            config.cem_K,
            config.cem_iterations,
            seeds=seeds,
            initial_snapshots=initial_snapshots,
            task=config.task,
            bmc_goal=bmc_goal,
            bmc_initial_constraints=bmc_initial_constraints,
            bmc_failed_cost=config.bmc_failed_cost,
            goal_feature=goal_feature,
            goal_feature_reward_weight=config.goal_feature_reward_weight,
            cem_init_std=config.cem_init_std,
            refresh_best_metrics=refresh_best_metrics,
            motion_penalty=motion_penalty,
            motion_penalty_weight=motion_penalty_weight,
        )

    def save_checkpoint(candidate, subdir: str) -> None:
        directory = logger.artifact_dir(subdir)
        (directory / "program.txt").write_text(str(candidate))
        with open(directory / "program.pkl", "wb") as handle:
            pickle.dump(candidate, handle)

    logger.set_status(phase="initial")
    logger.write_artifact("initial/program.txt", str(initial_program))

    initial = score(initial_program)
    current_program = initial.program
    current_cost = initial.cost
    current_checks_passed = initial.checks_passed

    result.best_program = current_program
    result.best_cost = current_cost
    result.best_iter = -1
    result.costs.append(current_cost)

    logger.log_metrics(-1, phase="initial", **initial.metrics())
    logger.log_event(
        "initial_scored",
        f"initial program cost={current_cost:.6g} "
        f"(bmc={initial.bmc_reason}, checks_passed={current_checks_passed})",
        step=-1,
        force=True,
        cost=current_cost,
        bmc_reason=initial.bmc_reason,
    )
    if current_cost <= config.bmc_failed_cost:
        # The chain starting on the BMC floor means every candidate has a positive
        # cost delta and is accepted unconditionally until the first feasible
        # program appears, i.e. the early search is an unguided random walk.
        logger.log_event(
            "chain_starts_at_bmc_floor",
            "initial program sits at bmc_failed_cost; acceptance is unconditional "
            "until the first feasible candidate",
            step=-1,
            force=True,
            bmc_reason=initial.bmc_reason,
        )
    save_checkpoint(current_program, "best")
    logger.set_status(phase="mcmc")

    for i in range(config.iters):
        iter_start = time.time()

        new_program, changed, mutation_info = mutate_program(
            current_program, available_operands, available_instructions
        )
        equivalence = executable_instructions(
            current_program
        ) == executable_instructions(new_program)
        evaluated = bool(changed) and not equivalence

        if evaluated:
            candidate = score(new_program)
            new_cost = candidate.cost
            new_program = candidate.program
        else:
            candidate = CandidateResult(
                cost=current_cost,
                program=new_program,
                checks_passed=current_checks_passed,
                bmc_feasible=None,
                bmc_reason="skipped_equivalent" if equivalence else "skipped_unchanged",
                bmc_report="BMC: skipped",
                goal_report="goal feature: skipped",
                scored=False,
            )
            new_cost = current_cost

        accepted = False
        acceptance_ratio: Optional[float] = None
        if evaluated:
            acceptance_ratio = acceptance_probability(
                config.beta * (new_cost - current_cost)
            )
            # The acceptance draw must stay the only extra RNG consumption in the
            # loop, and must happen here, so the chain matches the original.
            if random.random() < acceptance_ratio:
                accepted = True
                current_program = new_program
                current_cost = new_cost
                current_checks_passed = candidate.checks_passed

        is_new_best = evaluated and new_cost > result.best_cost
        if is_new_best:
            result.best_cost = new_cost
            result.best_iter = i
            result.best_program = new_program

        if evaluated:
            accept_rate.add(accepted)
            feasible_rate.add(bool(candidate.bmc_feasible))
            floor_rate.add(new_cost <= config.bmc_failed_cost)
            if candidate.bmc_feasible and first_feasible_iter is None:
                first_feasible_iter = i
                logger.log_event(
                    "first_feasible",
                    f"first BMC-feasible candidate at iteration {i}",
                    step=i,
                    force=True,
                )

        result.costs.append(new_cost)
        result.recent_programs.append(new_program)
        if len(result.recent_programs) > config.keep_last:
            result.recent_programs.pop(0)
        result.iters_completed = i + 1

        logger.log_metrics(
            i,
            iter_s=round(time.time() - iter_start, 3),
            current_cost=round(float(current_cost), 6),
            best_cost=round(float(result.best_cost), 6),
            accepted=accepted,
            changed=bool(changed),
            equivalent=equivalence,
            evaluated=evaluated,
            mutation_kind=mutation_info["type"],
            acceptance_ratio=(
                None if acceptance_ratio is None else round(acceptance_ratio, 6)
            ),
            **candidate.metrics(),
        )
        logger.set_progress(
            i,
            config.iters,
            best_cost=round(float(result.best_cost), 6),
            best_iter=result.best_iter,
            current_cost=round(float(current_cost), 6),
            accept_rate_window=accept_rate.rate,
            bmc_feasible_rate_window=feasible_rate.rate,
            cost_at_floor_rate_window=floor_rate.rate,
            first_feasible_iter=first_feasible_iter,
        )

        if is_new_best:
            logger.log_event(
                "new_best",
                f"new best cost={new_cost:.6g} at iteration {i}",
                step=i,
                force=True,
                cost=new_cost,
                success_rate=candidate.success_rate,
                mutation_kind=mutation_info["type"],
            )
        if evaluated and not candidate.bmc_feasible and bmc_goal is not None:
            logger.log_event(
                "bmc_infeasible",
                f"candidate rejected by BMC ({candidate.bmc_reason})",
                step=i,
                bmc_reason=candidate.bmc_reason,
            )

        mutation_report = format_mutation_report(
            mutation_info, current_program, new_program, equivalence=equivalence
        )
        checkpoint_due = (
            config.checkpoint_every > 0 and i % config.checkpoint_every == 0
        )
        if checkpoint_due or is_new_best:
            logger.write_artifact(
                f"iter{i}/summary.txt",
                "\n".join(
                    [
                        "=== Mutation Report ===",
                        mutation_report,
                        "",
                        "=== Candidate Program ===",
                        str(new_program),
                        f"Cost: {new_cost}",
                        f"BMC: {candidate.bmc_report}",
                        f"Goal feature: {candidate.goal_report}",
                        f"Accepted: {accepted}",
                        f"New best: {is_new_best}",
                    ]
                ),
            )
            save_checkpoint(new_program, f"iter{i}")
        if is_new_best:
            save_checkpoint(new_program, "best")

        # Videos are rendered after the acceptance draw, so that the video policy
        # cannot perturb the Markov chain through the shared RNG (rollouts reseed
        # it via set_np_seed).
        wants_video = config.video_policy == "all" or (
            config.video_policy == "best" and is_new_best
        )
        if wants_video and candidate.checks_passed:
            target = (
                logger.artifact_dir("best", "videos")
                if is_new_best
                else logger.artifact_dir(f"iter{i}", "videos")
            )
            save_program_seed_videos(
                new_program,
                num_seeds=num_seeds,
                num_block=config.num_blocks,
                video_dir=str(target),
                seeds=seeds,
                initial_snapshots=initial_snapshots,
                video_fps=30,
                verbose=False,
                task=config.task,
            )

        if config.progress_every > 0 and i % config.progress_every == 0:
            logger.progress_line(
                f"[{logger.name}] iter {i}/{config.iters} "
                f"cost={new_cost:.4g} best={result.best_cost:.4g} "
                f"accept={accept_rate.rate} feasible={feasible_rate.rate}"
            )

    return result
