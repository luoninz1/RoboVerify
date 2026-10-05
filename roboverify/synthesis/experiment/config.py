"""Resolved configuration for an instrumented MCMC run.

Everything that can change a result must land in ``config.json``, because a run
whose configuration is not recorded cannot be compared against another run. Some
arguments to :func:`synthesis.mcmc.synthesis.MCMC` are not JSON-serializable -- the
real call site in ``synthesis/entry/main.py`` passes a closure for ``bmc_goal`` and a
learned feature object for ``goal_feature`` -- so those are recorded as descriptors
via :func:`describe_runtime` instead of being dropped.
"""

import argparse
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

VIDEO_POLICIES = ("none", "best", "all")


@dataclass
class MCMCConfig:
    """Every knob that can change the outcome of an MCMC search."""

    # Task / environment
    task: str = "stack"
    num_blocks: int = 4
    demos: str = ""

    # Search
    iters: int = 2000
    program_slots: int = 4
    rng_seed: int = 0
    beta: float = 1.0
    bmc_failed_cost: float = -1e6

    # Inner parameter optimization (CEM)
    cem_N: int = 16
    cem_K: int = 4
    cem_iterations: int = 10
    cem_init_std: float = 0.1

    # Objective
    goal_feature_reward_weight: float = 1.0

    # Rollouts
    num_seeds: Optional[int] = None
    seeds: Optional[list] = None

    # Logging / artifact policy
    run_root: str = "runs"
    run_name: str = "mcmc"
    slug: Optional[str] = None
    video_policy: str = "best"
    checkpoint_every: int = 25
    progress_every: int = 10
    keep_last: int = 5
    rolling_window: int = 100
    event_min_step_gap: int = 50
    capture_stdout: bool = True

    # Descriptors for runtime objects that cannot be serialized directly.
    runtime: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.video_policy not in VIDEO_POLICIES:
            raise ValueError(
                f"video_policy must be one of {VIDEO_POLICIES}, got {self.video_policy!r}"
            )
        if self.beta <= 0:
            raise ValueError(f"beta must be positive, got {self.beta}")
        if self.cem_K > self.cem_N:
            raise ValueError(f"cem_K ({self.cem_K}) cannot exceed cem_N ({self.cem_N})")

    @property
    def resolved_slug(self) -> str:
        return self.slug or f"{self.task}-nb{self.num_blocks}"

    def describe_runtime(
        self,
        *,
        bmc_goal: Any = None,
        goal_feature: Any = None,
        expert_states: Any = None,
        initial_program: Any = None,
        available_instructions: Any = None,
        available_operands: Any = None,
    ) -> None:
        """Record descriptors for objects that cannot go into JSON directly."""
        runtime: dict = {
            "bmc_goal": None if bmc_goal is None else repr(bmc_goal),
            "bmc_enabled": bmc_goal is not None,
            "goal_feature": None if goal_feature is None else str(goal_feature),
        }
        if expert_states is not None:
            try:
                import numpy as np

                arr = np.asarray(expert_states)
                runtime["expert_states"] = {
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "source_archive": self.demos,
                }
            except Exception:  # pragma: no cover - descriptor only
                runtime["expert_states"] = {"len": len(expert_states)}
        if initial_program is not None:
            runtime["initial_program"] = str(initial_program)
        if available_instructions is not None:
            runtime["available_instructions"] = [
                getattr(i, "__name__", str(i)) for i in available_instructions
            ]
        if available_operands is not None:
            runtime["available_operands"] = {
                str(k): list(v) for k, v in available_operands.items()
            }
        self.runtime = runtime

    def to_json_dict(self) -> dict:
        return asdict(self)


def add_cli_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add flags mirroring :class:`MCMCConfig` defaults."""
    defaults = MCMCConfig()

    task = parser.add_argument_group("task")
    task.add_argument("--task", default=defaults.task)
    task.add_argument("--num-blocks", type=int, default=defaults.num_blocks)
    task.add_argument(
        "--demos",
        default=defaults.demos,
        help="Validated full-state demonstration archive",
    )
    task.add_argument(
        "--num-seeds",
        type=int,
        default=None,
        help="Rollout seeds per candidate; defaults to the number of loaded demos.",
    )

    search = parser.add_argument_group("search")
    search.add_argument("--iters", type=int, default=defaults.iters)
    search.add_argument("--program-slots", type=int, default=defaults.program_slots)
    search.add_argument("--rng-seed", type=int, default=defaults.rng_seed)
    search.add_argument(
        "--beta",
        type=float,
        default=defaults.beta,
        help=(
            "Acceptance temperature: accept with probability exp(beta * cost_delta). "
            "beta=1.0 reproduces the original behaviour; raise it to sharpen "
            "selection when the acceptance rate is near 1."
        ),
    )
    search.add_argument(
        "--goal-feature-reward-weight",
        type=float,
        default=defaults.goal_feature_reward_weight,
    )

    cem = parser.add_argument_group("cem")
    cem.add_argument("--cem-N", type=int, default=defaults.cem_N)
    cem.add_argument("--cem-K", type=int, default=defaults.cem_K)
    cem.add_argument("--cem-iterations", type=int, default=defaults.cem_iterations)
    cem.add_argument("--cem-init-std", type=float, default=defaults.cem_init_std)

    logging_group = parser.add_argument_group("logging")
    logging_group.add_argument("--run-root", default=defaults.run_root)
    logging_group.add_argument("--run-name", default=defaults.run_name)
    logging_group.add_argument("--slug", default=None)
    logging_group.add_argument(
        "--video-policy",
        choices=VIDEO_POLICIES,
        default=defaults.video_policy,
        help=(
            "Render rollout videos for no candidate, only new bests, or every "
            "candidate. 'all' reproduces the original behaviour and costs one "
            "MuJoCo rollout plus an mp4 encode per seed per iteration."
        ),
    )
    logging_group.add_argument(
        "--checkpoint-every", type=int, default=defaults.checkpoint_every
    )
    logging_group.add_argument(
        "--progress-every", type=int, default=defaults.progress_every
    )
    logging_group.add_argument("--keep-last", type=int, default=defaults.keep_last)
    logging_group.add_argument(
        "--no-capture",
        action="store_true",
        help="Leave stdout on the console instead of capturing it to stdout.log.",
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Fast pathology check: 20 iters, 2 CEM iterations, 2 seeds, no videos. "
            "Overrides the corresponding flags."
        ),
    )
    return parser


def config_from_args(args: argparse.Namespace) -> MCMCConfig:
    """Build a config from parsed CLI arguments, applying ``--smoke`` last."""
    config = MCMCConfig(
        task=args.task,
        num_blocks=args.num_blocks,
        demos=args.demos,
        iters=args.iters,
        program_slots=args.program_slots,
        rng_seed=args.rng_seed,
        beta=args.beta,
        cem_N=args.cem_N,
        cem_K=args.cem_K,
        cem_iterations=args.cem_iterations,
        cem_init_std=args.cem_init_std,
        goal_feature_reward_weight=args.goal_feature_reward_weight,
        num_seeds=args.num_seeds,
        run_root=args.run_root,
        run_name=args.run_name,
        slug=args.slug,
        video_policy=args.video_policy,
        checkpoint_every=args.checkpoint_every,
        progress_every=args.progress_every,
        keep_last=args.keep_last,
        capture_stdout=not args.no_capture,
    )
    if getattr(args, "smoke", False):
        config.iters = 20
        config.cem_iterations = 2
        config.num_seeds = 2
        config.video_policy = "none"
        config.checkpoint_every = 10
        config.progress_every = 1
        config.slug = config.slug or f"smoke-{config.task}"
    return config
