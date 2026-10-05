"""Bounded Phase D solver spike and a repeatable MotionVerify timing fixture."""

import json
from time import perf_counter

import z3

from synthesis.entry.verify_stack_with_learned_invariant import build_stack_programs
from synthesis.util.symbols import fresh_const
from synthesis.verification_lib.bmc_lib import NoiseSpec, bounded_noise
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext
from synthesis.verification_lib.lowlevel_verification_lib import LowLevelContext
from synthesis.verification_lib.motion_verification import (
    MotionContract,
    verify_motion_block,
)


def benchmark():
    context = LowLevelContext(default_L=0.05)
    a = context.get_consts("obstacle")
    eta, bounds = bounded_noise(0.005, "spike_endpoint")
    end = tuple(z3.RealVal(v) + e for v, e in zip((0, 0, 0.2), eta))
    solver = z3.Solver()
    solver.set(timeout=5000)
    solver.add(
        context.L == 0.05,
        *bounds,
        context.X(a) == 0.15,
        context.Y(a) == 0,
        context.Z(a) == 0.2
    )
    solver.add(context.encode_collision(a, (0.3, 0, 0.2), end))
    start = perf_counter()
    answer = solver.check()
    collision = {
        "status": str(answer),
        "seconds": perf_counter() - start,
        "epsilon": 0.005,
        "encoding": "exact swept cube, bilinear",
    }
    high = HighLevelContext(mode="declare")
    _, program = build_stack_programs(high)
    root, target = high.get_consts("b0"), high.get_consts("b")
    member = fresh_const(high.BoxSort, prefix="benchmark_member")
    # Numeric positions alone do not establish the symbolic root criterion.
    conditions = [
        target == root,
        z3.ForAll(
            [member],
            z3.Implies(
                high.ON_star(root, member),
                member == root,
            ),
        ),
    ]
    body = program.instructions[1].body
    scene = {"b0": [0, 0, 0], "b": [0, 0, 0], "b_prime": [0.3, 0, 0], "sym": [2, 2, 0]}
    calls = []
    for noise in (None, NoiseSpec(0.005, 0.005, 0.005)):
        result = verify_motion_block(
            conditions,
            body,
            ["b0", "b", "b_prime"],
            MotionContract("b_prime", "b"),
            context=context,
            initial_positions=scene,
            noise=noise,
        )
        calls.append(
            {
                "mode": result.mode,
                "seconds": result.elapsed_seconds,
                "ok": result.ok,
                "checks": len(result.checks),
                "failures": [
                    {"obligation": c.obligation, "status": c.status}
                    for c in result.checks
                    if c.status != "valid"
                ],
            }
        )
    return {"collision_spike": collision, "motion_calls": calls}


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2))
