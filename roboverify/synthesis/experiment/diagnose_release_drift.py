"""Replay Release from saved states, inspecting physics substeps and interventions.

Explicitly reconstructs the earlier head-contact, 10 mm Pick, Z-only Release
baseline. Interventions isolate causes; these results are not a motion proof.
"""

import argparse
import json
from collections import defaultdict
from dataclasses import replace
from unittest.mock import patch

import mujoco_py
import numpy as np

from synthesis.api.control import DEFAULT_CONTROL, PrimitiveController
from synthesis.cfg.collection import record_execution, validate_trace
from synthesis.cfg.program_source import load_program
from synthesis.cfg.recordings import save_traces
from synthesis.cfg.reset import inner_env, restore
from synthesis.experiment.controller_baseline import (
    previous_release,
    previous_release_control,
    stack_env_with_head_contacts,
)
from synthesis.experiment.run_logger import RunLogger
from synthesis.util.actions import bound_delta_action
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext


def replay(trace, start, end, variant):
    env = stack_env_with_head_contacts(num_blocks=trace.num_blocks)
    inner = inner_env(env)
    sim, samples, phases, commands = inner.sim, [], [], []
    phase = None
    contacts = defaultdict(
        lambda: dict(samples=0, maximum_normal_force=0.0, deepest_penetration_m=0.0)
    )
    try:
        restore(env, trace.snapshots[start])
        if variant == "no_robot_contacts":
            for i in range(sim.model.ngeom):
                name = sim.model.body_id2name(sim.model.geom_bodyid[i])
                if name.startswith("robot0:"):
                    sim.model.geom_contype[i] = sim.model.geom_conaffinity[i] = 0
        elif variant in ("no_head_contacts", "no_head_hold_xy"):
            for i in range(sim.model.ngeom):
                if sim.model.body_id2name(sim.model.geom_bodyid[i]).startswith(
                    "robot0:head"
                ):
                    sim.model.geom_contype[i] = sim.model.geom_conaffinity[i] = 0
        elif variant == "zero_velocity":
            sim.data.qvel[:] = 0
            sim.data.qacc_warmstart[:] = 0
        sim.forward()
        if variant == "baseline":
            # Match restore's warm-start policy after the diagnostic forward.
            sim.data.qacc_warmstart[:] = trace.snapshots[start].arrays["qacc_warmstart"]

        def sample():
            for i in range(sim.data.ncon):
                contact = sim.data.contact[i]
                bodies = [
                    sim.model.body_id2name(sim.model.geom_bodyid[g])
                    for g in (contact.geom1, contact.geom2)
                ]
                if not any(b.startswith("robot0:") for b in bodies):
                    continue
                force = np.zeros(6)
                mujoco_py.functions.mj_contactForce(sim.model, sim.data, i, force)
                key = " / ".join(sorted(bodies))
                info = contacts[key]
                info["samples"] += 1
                info["maximum_normal_force"] = max(
                    info["maximum_normal_force"], abs(float(force[0]))
                )
                info["deepest_penetration_m"] = min(
                    info["deepest_penetration_m"], float(contact.dist)
                )
            margins = {}
            for i, name in enumerate(sim.model.joint_names):
                if name.startswith("robot0:") and sim.model.jnt_limited[i]:
                    q = float(sim.data.get_joint_qpos(name))
                    lo, hi = sim.model.jnt_range[i]
                    margins[name] = float(min(q - lo, hi - q))
            samples.append(
                dict(
                    phase=phase,
                    time=float(sim.data.time),
                    grip=sim.data.get_site_xpos("robot0:grip").tolist(),
                    grip_velocity=sim.data.get_site_xvelp("robot0:grip").tolist(),
                    body=sim.data.get_body_xpos("robot0:gripper_link").tolist(),
                    body_quaternion=sim.data.get_body_xquat(
                        "robot0:gripper_link"
                    ).tolist(),
                    mocap=sim.data.mocap_pos[0].tolist(),
                    joint_margins=margins,
                )
            )

        def step(action):
            commands.append(np.asarray(action).tolist())
            inner._set_action(bound_delta_action(action))
            # MjSim.step does this same loop. Reading fields adds no physics steps.
            for _ in range(sim.nsubsteps):
                sample()
                mujoco_py.functions.mj_step(sim.model, sim.data)
            inner._step_callback()
            sample()

        env.step = step
        original_until = PrimitiveController._until

        def until(controller, label, *args, **kwargs):
            nonlocal phase
            phase = label
            first = len(samples)
            sample()
            if label == "retreat" and variant == "zero_retreat_velocity":
                sim.data.qvel[:] = 0
                sim.data.qacc_warmstart[:] = 0
                sim.forward()
            result = original_until(controller, label, *args, **kwargs)
            phases.append(
                dict(
                    phase=label,
                    start_sample=first,
                    end_sample=len(samples) - 1,
                    converged=bool(result),
                )
            )
            return result

        original_move = PrimitiveController.move

        def move(controller, target, **kwargs):
            if variant in ("hold_xy", "no_head_hold_xy") and kwargs.get(
                "vertical_only"
            ):
                kwargs["vertical_only"] = False
            return original_move(controller, target, **kwargs)

        controller = PrimitiveController(
            env,
            [],
            control=(
                replace(DEFAULT_CONTROL, gain=5)
                if variant == "slow_z"
                else DEFAULT_CONTROL
            ),
        )
        box_id = trace.snapshots[start].bindings["b_prime"]
        with patch.object(PrimitiveController, "_until", until), patch.object(
            PrimitiveController, "move", move
        ):
            success = previous_release(controller, box_id, 0.15)
        if variant == "baseline":
            np.testing.assert_allclose(
                commands, trace.actions[0][start:end], rtol=0, atol=1e-8
            )
            np.testing.assert_allclose(
                controller.observation, trace.states[end], rtol=0, atol=1e-8
            )
        summary = dict(
            variant=variant,
            converged=bool(success),
            steps=controller.steps,
            contacts=dict(contacts),
            phases=[],
        )
        for p in phases:
            s = samples[p["start_sample"] : p["end_sample"] + 1]
            grip, body = np.array([v["grip"] for v in s]), np.array(
                [v["body"] for v in s]
            )
            offset = grip - body
            summary["phases"].append(
                dict(
                    phase=p["phase"],
                    grip_xy_drift_mm=float(
                        np.max(np.linalg.norm(grip[:, :2] - grip[0, :2], axis=1)) * 1000
                    ),
                    body_xy_drift_mm=float(
                        np.max(np.linalg.norm(body[:, :2] - body[0, :2], axis=1)) * 1000
                    ),
                    site_offset_xy_change_mm=float(
                        np.max(np.linalg.norm(offset[:, :2] - offset[0, :2], axis=1))
                        * 1000
                    ),
                    final_grip_delta_mm=((grip[-1] - grip[0]) * 1000).tolist(),
                    joint_min_margins={
                        name: min(v["joint_margins"][name] for v in s)
                        for name in s[0]["joint_margins"]
                    },
                )
            )
        return summary, samples
    finally:
        env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 9, 52, 62, 98])
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=[
            "baseline",
            "hold_xy",
            "no_robot_contacts",
            "no_head_contacts",
            "no_head_hold_xy",
            "zero_velocity",
            "zero_retreat_velocity",
            "slow_z",
        ],
        default=[
            "baseline",
            "hold_xy",
            "no_head_contacts",
            "no_head_hold_xy",
            "slow_z",
        ],
    )
    args = parser.parse_args(argv)
    results = []
    with RunLogger("runs", "release-drift", vars(args), slug="substeps") as logger:
        for seed in args.seeds:
            definition = load_program(
                "synthesis.examples.stack:build_program", HighLevelContext(), 4
            )
            pick = definition.program.instructions[1].body[0]
            pick.control = replace(pick.control, position_tolerance=0.01)
            with previous_release_control():
                trace = record_execution(
                    definition,
                    seed=seed,
                    num_blocks=4,
                    max_loop_iterations=3,
                    env_factory=stack_env_with_head_contacts,
                )
            assert validate_trace(trace), trace.metadata
            starts = [
                e["index"]
                for e in trace.events
                if e["kind"] == "instruction_start" and e["path"] == "1.4"
            ]
            ends = [
                e["index"]
                for e in trace.events
                if e["kind"] == "instruction_end" and e["path"] == "1.4"
            ]
            save_traces(logger.run_dir / "artifacts" / f"seed-{seed}.npz", [trace])
            for placement, (start, end) in enumerate(zip(starts, ends), 1):
                for variant in args.variants:
                    summary, samples = replay(trace, start, end, variant)
                    summary.update(seed=seed, placement=placement)
                    results.append(summary)
                    logger.write_artifact(
                        f"seed-{seed}-placement-{placement}-{variant}.json",
                        json.dumps(samples),
                    )
                    logger.log_metrics(
                        len(results),
                        seed=seed,
                        placement=placement,
                        variant=variant,
                        success_rate=int(summary["converged"]),
                    )
                    logger.set_status(
                        phase="replay", seed=seed, placement=placement, variant=variant
                    )
            logger.progress_line(f"Diagnosed seed {seed}")
        logger.write_artifact("summary.json", json.dumps(results, indent=2))
        logger.finish(
            "completed", releases=len(results), report="artifacts/summary.json"
        )
    print(f"Report: {logger.run_dir}/artifacts/summary.json")


if __name__ == "__main__":
    main()
