"""Paired Stack executions comparing Pick stopping tolerances under uniform scaling."""

import argparse
import json
from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from synthesis.api.control import PrimitiveController
from synthesis.cfg.collection import record_execution, validate_trace
from synthesis.cfg.program_source import load_program
from synthesis.cfg.recordings import save_traces
from synthesis.experiment.compare_stack_paths import cross_track
from synthesis.experiment.run_logger import RunLogger
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext


@contextmanager
def measure(phases, picks):
    """Capture phase and grasp geometry without changing commands or stopping tests."""
    original_move = PrimitiveController.move
    original_relative = PrimitiveController.move_relative
    original_pick = PrimitiveController.pick
    original_step = PrimitiveController._step
    active = None
    placement = -1
    move_number = 0

    def pick(controller, box_id):
        nonlocal placement
        placement += 1
        success = original_pick(controller, box_id)
        offset = controller.observation[:3] - controller.box_position(box_id)
        picks.append(
            dict(
                placement=placement + 1,
                steps=controller.steps,
                converged=bool(success),
                grasp_xy_mm=float(np.linalg.norm(offset[:2]) * 1000),
                grasp_z_mm=float(offset[2] * 1000),
            )
        )
        return success

    def relative(controller, reference_ids, offsets):
        nonlocal move_number
        controller._measurement_phase = ("lift", "transfer", "lower")[move_number % 3]
        move_number += 1
        return original_relative(controller, reference_ids, offsets)

    def move(controller, target, **kwargs):
        nonlocal active
        target = np.asarray(target, dtype=float).copy()
        if kwargs.get("vertical_only"):
            target[:2] = controller.observation[:2]
        label = kwargs.get("phase", getattr(controller, "_measurement_phase", "move"))
        row = dict(
            phase=label,
            placement=placement + 1,
            target=target.tolist(),
            positions=[controller.observation[:3].tolist()],
        )
        active = row
        try:
            success = original_move(controller, target, **kwargs)
        finally:
            active = None
            end = np.asarray(row["positions"][-1]) - target
            row.update(
                steps=len(row["positions"]) - 1,
                endpoint_mm=float(np.linalg.norm(end) * 1000),
                path_max_mm=float(cross_track(row["positions"], target).max() * 1000),
            )
            phases.append(row)
        if label == "lift":
            box_id = controller.env.unwrapped.symbolic_name_to_box_id["b_prime"]
            offset = controller.observation[:3] - controller.box_position(box_id)
            row.update(
                payload_xy_mm=float(np.linalg.norm(offset[:2]) * 1000),
                payload_z_mm=float(offset[2] * 1000),
            )
        row["converged"] = bool(success)
        return success

    def step(controller, action):
        if np.max(np.abs(action[:3])) > 1 + 1e-12:
            raise AssertionError("Primitive action escaped the uniform XYZ bound")
        result = original_step(controller, action)
        if active is not None:
            active["positions"].append(controller.observation[:3].tolist())
        return result

    with patch.object(PrimitiveController, "pick", pick), patch.object(
        PrimitiveController, "move_relative", relative
    ), patch.object(PrimitiveController, "move", move), patch.object(
        PrimitiveController, "_step", step
    ):
        yield


def distribution(values):
    if not values:
        return None
    a = np.asarray(values)
    return dict(
        n=len(a),
        median=float(np.median(a)),
        p95=float(np.percentile(a, 95)),
        maximum=float(a.max()),
        mean=float(a.mean()),
    )


def summarize(records):
    result = {}
    for tolerance in sorted({r["tolerance_mm"] for r in records}, reverse=True):
        runs = [r for r in records if r["tolerance_mm"] == tolerance]
        phases = [p for r in runs for p in r["phases"]]
        picks = [p for r in runs for p in r["picks"]]
        result[tolerance] = dict(
            executions=len(runs),
            valid=sum(r["valid"] for r in runs),
            failed_seeds=[r["seed"] for r in runs if not r["valid"]],
            failed_controls=[
                dict(seed=r["seed"], **e) for r in runs for e in r["failed_controls"]
            ],
            program_actions=distribution([r["actions"] for r in runs]),
            pick_steps=distribution([p["steps"] for p in picks]),
            grasp_xy_mm=distribution([p["grasp_xy_mm"] for p in picks]),
            lifted_payload_xy_mm=distribution(
                [p["payload_xy_mm"] for p in phases if p["phase"] == "lift"]
            ),
            final_max_stack_xy_mm=distribution(
                [r["final_max_stack_xy_mm"] for r in runs if r["valid"]]
            ),
            phases={
                p: {
                    metric: distribution([r[metric] for r in phases if r["phase"] == p])
                    for metric in ("endpoint_mm", "path_max_mm", "steps")
                }
                for p in ("approach", "descend", "lift", "transfer", "lower", "retreat")
            },
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-seeds", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument(
        "--tolerances-mm", type=float, nargs="+", default=[10, 5, 3, 2, 1, 0.5]
    )
    parser.add_argument("--run-name", default="paired-grid")
    args = parser.parse_args(argv)
    if (
        args.num_seeds < 1
        or args.num_blocks < 2
        or any(not np.isfinite(v) or v <= 0 for v in args.tolerances_mm)
    ):
        parser.error("Require positive seeds/tolerances and at least two blocks")
    records = []
    with RunLogger("runs", "pick-tolerance", vars(args), slug=args.run_name) as logger:
        for seed in range(args.seed_start, args.seed_start + args.num_seeds):
            snapshot = None
            for tolerance in args.tolerances_mm:
                definition = load_program(
                    "synthesis.examples.stack:build_program",
                    HighLevelContext(),
                    args.num_blocks,
                )
                instruction = definition.program.instructions[1].body[0]
                instruction.control = replace(
                    instruction.control, position_tolerance=tolerance / 1000
                )
                phases, picks = [], []
                with measure(phases, picks):
                    trace = record_execution(
                        definition,
                        seed=seed,
                        num_blocks=args.num_blocks,
                        initial_snapshot=snapshot,
                        max_loop_iterations=args.num_blocks - 1,
                    )
                valid = validate_trace(trace)
                if not trace.snapshots:
                    raise RuntimeError(trace.metadata)
                if snapshot is None:
                    snapshot = trace.snapshots[0]
                else:
                    np.testing.assert_array_equal(
                        trace.snapshots[0].gt_state, snapshot.gt_state
                    )
                    for name, values in snapshot.arrays.items():
                        np.testing.assert_array_equal(
                            trace.snapshots[0].arrays[name], values
                        )
                positions = np.array(
                    [
                        trace.states[-1][10 + 12 * i : 13 + 12 * i]
                        for i in range(args.num_blocks)
                    ]
                )
                row = dict(
                    seed=seed,
                    tolerance_mm=tolerance,
                    valid=valid,
                    status=trace.metadata["status"],
                    reason=trace.metadata["reason"],
                    actions=len(trace.actions[0]),
                    picks=picks,
                    phases=phases,
                    final_max_stack_xy_mm=float(
                        np.max(
                            np.linalg.norm(positions[1:, :2] - positions[0, :2], axis=1)
                        )
                        * 1000
                    ),
                    failed_controls=[
                        e
                        for e in trace.events
                        if "control" in e and not e["control"]["converged"]
                    ],
                )
                records.append(row)
                if seed == args.seed_start or (
                    not valid
                    and sum(
                        not r["valid"] and r["tolerance_mm"] == tolerance
                        for r in records
                    )
                    <= 2
                ):
                    trace.metadata.update(
                        validation_status=trace.metadata["status"], status="diagnostic"
                    )
                    save_traces(
                        logger.run_dir
                        / "artifacts"
                        / f"seed-{seed}-pick-{tolerance:g}mm.npz",
                        [trace],
                    )
                logger.log_metrics(
                    len(records),
                    seed=seed,
                    tolerance_mm=tolerance,
                    success_rate=int(valid),
                    actions=row["actions"],
                )
                logger.set_status(
                    phase="measure",
                    seed=seed,
                    tolerance_mm=tolerance,
                    executions=len(records),
                )
            if (seed - args.seed_start + 1) % 10 == 0:
                logger.write_artifact(
                    "summary.json", json.dumps(summarize(records), indent=2)
                )
                logger.progress_line(
                    f"Completed {seed - args.seed_start + 1}/{args.num_seeds} paired seeds"
                )
        summary = summarize(records)
        logger.write_artifact("executions.json", json.dumps(records))
        logger.write_artifact("summary.json", json.dumps(summary, indent=2))
        lines = [
            "# Pick tolerance comparison",
            "",
            "Uniform XYZ scaling; 50-step Pick budget, fixed-XY Release and head contacts disabled. Each seed uses identical full settled starts across tolerances. Failed seeds are retained. Endpoint and payload measurements are at control-step boundaries.",
            "",
            "| Pick tolerance (mm) | Valid runs | Pick steps mean / max | Approach error P95 / max (mm) | Descent error P95 / max (mm) | Lifted payload XY P95 / max (mm) | Final tower XY P95 / max (mm) |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for tolerance, s in summary.items():

            def pair(d):
                return "n/a" if d is None else f"{d['p95']:.3f} / {d['maximum']:.3f}"

            lines.append(
                f"| {tolerance:g} | {s['valid']}/{s['executions']} | {s['pick_steps']['mean']:.2f} / {s['pick_steps']['maximum']:.0f} | {pair(s['phases']['approach']['endpoint_mm'])} | {pair(s['phases']['descend']['endpoint_mm'])} | {pair(s['lifted_payload_xy_mm'])} | {pair(s['final_max_stack_xy_mm'])} |"
            )
        logger.write_artifact("summary.md", "\n".join(lines) + "\n")
        logger.finish(
            "completed",
            executions=len(records),
            valid=sum(r["valid"] for r in records),
            report="artifacts/summary.md",
        )
    print(f"Report: {logger.run_dir}/artifacts/summary.md")


if __name__ == "__main__":
    main()
