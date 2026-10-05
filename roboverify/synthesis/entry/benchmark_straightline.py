"""Phase F timing gate on a deterministic simulator demonstration segment."""

import json
from time import perf_counter

from synthesis.api.instructions import PickByName
from synthesis.api.program import Program
from synthesis.cfg.demos import DemoSegment, DemoTrace
from synthesis.cfg.reset import collect_recording
from synthesis.cfg.straightline import (
    SearchBudget,
    _features,
    segment_rollout,
    straight_line_synthesize,
)
from synthesis.mcmc.distance import KDEDistance
from synthesis.mcmc.synthesis import (
    make_roboverify_stack_env,
    preserved_global_rng,
    set_np_seed,
)
from synthesis.predicates.term import boolean


def main():
    with preserved_global_rng():
        set_np_seed(0)
        env = make_roboverify_stack_env(num_blocks=2)
        candidate = Program(1, [PickByName("b0")])
        try:
            states, record = collect_recording(candidate, env)
        finally:
            env.close()
    trace = DemoTrace(
        states,
        tuple(record.snapshots),
        (tuple(record.actions), tuple(record.action_indices)),
        num_blocks=2,
    )
    segment = DemoSegment(0, 0, len(states) - 1, trace, {"b0": 0})
    rollout = lambda p, s: segment_rollout(
        p, s, lambda t: make_roboverify_stack_env(num_blocks=t.num_blocks), "reset"
    )
    distance = KDEDistance(_features(states, 2))
    start = perf_counter()
    value = distance(_features(rollout(candidate, segment), 2), samples=128)
    objective_seconds = perf_counter() - start
    result = straight_line_synthesize(
        [segment],
        boolean(True),
        candidate,
        lambda p, r: p,
        rollout=rollout,
        budget=SearchBudget(iterations=0, inner_samples=32, final_samples=128),
    )
    print(
        json.dumps(
            {
                "objective_seconds": objective_seconds,
                "straightline_seconds": result.elapsed,
                "postscore_seconds": result.postscore_seconds,
                "distance": value,
                "post_score": result.post_score,
                "status": result.status,
                "pool_limit": 10,
            }
        )
    )


if __name__ == "__main__":
    main()
