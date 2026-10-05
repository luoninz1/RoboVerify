"""Compare complete Stack runs before/after the combined controller changes.

Records measurements only, never demonstration archives. Both modes use uniform
scaling. Each updated execution restores its baseline's exact settled snapshot.
"""

import argparse
import base64
import html
import json
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from synthesis.api.control import PrimitiveController
from synthesis.cfg.collection import record_execution, validate_trace
from synthesis.cfg.program_source import load_program
from synthesis.experiment.controller_baseline import (
    previous_release_control,
    stack_env_with_head_contacts,
)
from synthesis.experiment.run_logger import RunLogger
from synthesis.experiment.tune_pick_tolerance import distribution
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext

PHASES = ("approach", "descend", "lift", "transfer", "lower", "retreat")


def segment_distance(points, start, target):
    points, start, target = map(np.asarray, (points, start, target))
    vector = target - start
    length2 = vector @ vector
    if length2 == 0:
        return np.linalg.norm(points - start, axis=1)
    progress = np.clip((points - start) @ vector / length2, 0, 1)
    return np.linalg.norm(points - start - progress[:, None] * vector, axis=1)


@contextmanager
def measure(rows):
    original_move, original_step = PrimitiveController.move, PrimitiveController._step
    original_pick, original_relative = (
        PrimitiveController.pick,
        PrimitiveController.move_relative,
    )
    active, placement, relative_number = None, 0, 0

    def pick(controller, box_id):
        nonlocal placement
        placement += 1
        return original_pick(controller, box_id)

    def relative(controller, references, offsets):
        nonlocal relative_number
        controller._measured_phase = ("lift", "transfer", "lower")[relative_number % 3]
        relative_number += 1
        return original_relative(controller, references, offsets)

    def move(controller, target, **kwargs):
        nonlocal active
        label = kwargs.get("phase", getattr(controller, "_measured_phase", "move"))
        start, target = controller.observation[:3].copy(), np.asarray(target).copy()
        if label == "retreat":
            # The desired vertical line is anchored before finger opening.
            start[:2] = target[:2]
        box_id = None
        if label in ("lift", "transfer", "lower"):
            box_id = controller.env.unwrapped.symbolic_name_to_box_id["b_prime"]
        row = dict(
            phase=label,
            placement=placement,
            target=target.tolist(),
            reference_start=start.tolist(),
            positions=[controller.observation[:3].tolist()],
            payload=[],
            box_id=box_id,
        )
        if box_id is not None:
            row["payload"].append(controller.box_position(box_id).tolist())
        active = row
        try:
            success = original_move(controller, target, **kwargs)
            row["converged"] = bool(success)
            return success
        finally:
            active = None
            positions = np.asarray(row["positions"])
            row.update(
                steps=len(positions) - 1,
                endpoint_mm=float(np.linalg.norm(positions[-1] - target) * 1000),
                path_mm=float(segment_distance(positions, start, target).max() * 1000),
            )
            if label == "retreat":
                row["xy_max_mm"] = float(
                    np.linalg.norm(positions[:, :2] - target[:2], axis=1).max() * 1000
                )
                row["xy_end_mm"] = float(
                    np.linalg.norm(positions[-1, :2] - target[:2]) * 1000
                )
            if box_id is not None:
                payload = np.asarray(row["payload"])
                offsets = payload - positions
                row.update(
                    payload_path_mm=float(
                        segment_distance(payload, payload[0], target + offsets[0]).max()
                        * 1000
                    ),
                    attachment_change_mm=float(
                        np.linalg.norm(offsets - offsets[0], axis=1).max() * 1000
                    ),
                    payload_xy_offset_mm=float(
                        np.linalg.norm(offsets[:, :2], axis=1).max() * 1000
                    ),
                )
            rows.append(row)

    def step(controller, action):
        result = original_step(controller, action)
        if active is not None:
            active["positions"].append(controller.observation[:3].tolist())
            if active["box_id"] is not None:
                active["payload"].append(
                    controller.box_position(active["box_id"]).tolist()
                )
        return result

    with patch.object(PrimitiveController, "pick", pick), patch.object(
        PrimitiveController, "move_relative", relative
    ), patch.object(PrimitiveController, "move", move), patch.object(
        PrimitiveController, "_step", step
    ):
        yield


def summarize(records):
    summary = {}
    for mode in ("previous", "updated"):
        runs = [r for r in records if r["mode"] == mode]
        phases = [p for r in runs for p in r["phases"]]
        summary[mode] = dict(
            executions=len(runs),
            valid=sum(r["valid"] for r in runs),
            failed_runs=[
                {k: r[k] for k in ("seed", "status", "reason", "failed_controls")}
                for r in runs
                if not r["valid"]
            ],
            actions=distribution([r["actions"] for r in runs]),
            instruction_steps=distribution(
                [v for r in runs for v in r["instruction_steps"]]
            ),
            final_tower_xy_mm=distribution([r["final_tower_xy_mm"] for r in runs]),
            run_max_path_mm=distribution(
                [max(p["path_mm"] for p in r["phases"]) for r in runs if r["phases"]]
            ),
            phases={
                name: {
                    metric: distribution(
                        [
                            p[metric]
                            for p in phases
                            if p["phase"] == name and metric in p
                        ]
                    )
                    for metric in (
                        "path_mm",
                        "endpoint_mm",
                        "steps",
                        "xy_max_mm",
                        "xy_end_mm",
                        "payload_path_mm",
                        "attachment_change_mm",
                        "payload_xy_offset_mm",
                    )
                }
                for name in PHASES
            },
        )
    return summary


def write_report(logger, records, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for mode, color in (("previous", "#bf4935"), ("updated", "#176ea1")):
        rows = [
            p
            for r in records
            if r["mode"] == mode
            for p in r["phases"]
            if p["phase"] == "retreat"
        ]
        for ax, metric in zip(axes, ("xy_max_mm", "xy_end_mm")):
            values = np.sort([p[metric] for p in rows])
            ax.step(
                values,
                np.arange(1, len(values) + 1) / len(values) * 100,
                label=mode,
                color=color,
            )
            ax.set(ylabel="Release phases at or below value (%)", ylim=(0, 101))
            ax.grid(alpha=0.2)
            ax.legend()
    axes[0].set(
        title="Release path deviation",
        xlabel="Maximum XY distance from original vertical line (mm)",
    )
    axes[1].set(
        title="Release endpoint", xlabel="Final XY error from original line (mm)"
    )
    path = logger.run_dir / "artifacts" / "release.png"
    fig.savefig(path, dpi=170)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)
    image = base64.b64encode(path.read_bytes()).decode()

    def table(headers, rows):
        return (
            "<div class='table'><table><tr>"
            + "".join(f"<th>{html.escape(str(x))}</th>" for x in headers)
            + "</tr>"
            + "".join(
                "<tr>"
                + "".join(f"<td>{html.escape(str(x))}</td>" for x in row)
                + "</tr>"
                for row in rows
            )
            + "</table></div>"
        )

    def stats(value):
        return (
            "—"
            if value is None
            else f"{value['median']:.3f} / {value['p95']:.3f} / {value['maximum']:.3f}"
        )

    phase_rows = [
        [p, *[stats(summary[m]["phases"][p]["path_mm"]) for m in summary]]
        for p in PHASES
    ]
    endpoint_rows = [
        [p, *[stats(summary[m]["phases"][p]["endpoint_mm"]) for m in summary]]
        for p in PHASES
    ]
    payload_rows = [
        [
            p,
            *[stats(summary[m]["phases"][p]["payload_path_mm"]) for m in summary],
            stats(summary["updated"]["phases"][p]["attachment_change_mm"]),
            stats(summary["updated"]["phases"][p]["payload_xy_offset_mm"]),
        ]
        for p in ("lift", "transfer", "lower")
    ]
    overview = table(
        [
            "Configuration",
            "Valid programs",
            "Mean actions",
            "Maximum primitive steps",
            "Final tower XY: median / P95 / max (mm)",
        ],
        [
            [
                m,
                f"{s['valid']}/{s['executions']}",
                f"{s['actions']['mean']:.2f}",
                s["instruction_steps"]["maximum"],
                stats(s["final_tower_xy_mm"]),
            ]
            for m, s in summary.items()
        ],
    )
    previous_cases = {1, 9, 21, 34, 48, 62, 63, 78, 84, 98}
    cases = table(
        [
            "Seed",
            "Previous max XY (mm)",
            "Updated max XY (mm)",
            "Updated final XY (mm)",
            "Updated valid",
        ],
        [
            [
                seed,
                *[
                    f"{p['xy_max_mm']:.3f}"
                    for mode in summary
                    for r in records
                    if r["seed"] == seed and r["mode"] == mode
                    for p in r["phases"]
                    if p["phase"] == "retreat" and p["placement"] == 3
                ],
                *[
                    f"{p['xy_end_mm']:.3f}"
                    for r in records
                    if r["seed"] == seed and r["mode"] == "updated"
                    for p in r["phases"]
                    if p["phase"] == "retreat" and p["placement"] == 3
                ],
                next(
                    r["valid"]
                    for r in records
                    if r["seed"] == seed and r["mode"] == "updated"
                ),
            ]
            for seed in sorted(previous_cases & {r["seed"] for r in records})
        ],
    )
    failures = html.escape(
        json.dumps({m: s["failed_runs"] for m, s in summary.items()}, indent=2)
    )
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stack: combined controller validation</title><style>body{{font:16px/1.6 system-ui;color:#172536;background:#f4f7f9;margin:0}}main{{max-width:1180px;margin:auto;padding:32px}}h1{{line-height:1.2}}h2{{margin-top:36px}}.table{{overflow:auto}}table{{border-collapse:collapse;background:white;width:100%;font-size:14px}}th,td{{padding:10px;border-bottom:1px solid #d7e0e6;text-align:right}}th{{background:#e7eef3}}th:first-child,td:first-child{{text-align:left}}img{{width:100%}}pre{{white-space:pre-wrap}}small{{color:#54677b}}</style><main>
<h1>Combined controller changes: complete Stack executions</h1>
<p><strong>Updated production behavior:</strong> head collisions disabled, fixed XY reference during Release with 2 mm full-3D stopping, Pick tolerance 2 mm. Both configurations use uniform XYZ scaling and 50-step instruction budgets.</p>
<p><strong>Previous comparison:</strong> head collisions enabled, Z-only Release, Pick tolerance 10 mm. Each updated program restores its previous counterpart's exact full settled initial state; all initial arrays are checked for equality. Every seed is retained, including failures.</p>{overview}
<img src="data:image/png;base64,{image}" alt="Release maximum sideways deviation and final error distributions">
<h2>Gripper path deviation (median / P95 / max, mm)</h2>
<p>Maximum distance to the finite start-to-target segment, per motion phase. Release's vertical reference uses the fixed pre-opening XY; other phases start at their actual initial gripper position. Sampling is every 40 ms, so these are observed values rather than continuous-time bounds.</p>
{table(['Phase','Previous','Updated'],phase_rows)}
<h2>Gripper endpoint error (median / P95 / max, mm)</h2>{table(['Phase','Previous','Updated'],endpoint_rows)}
<h2>Held-block measurements (median / P95 / max, mm)</h2>
<p>Payload path deviation uses a segment translated by the measured initial block–gripper offset. Attachment change measures change in that relative vector during the phase. Absolute payload XY offset also exposes initial grasp misalignment. These quantities do not assume the block is perfectly centered on the gripper.</p>
{table(['Phase','Previous payload path','Updated payload path','Updated attachment change','Updated absolute XY offset'],payload_rows)}
<h2>Previously identified third-Release outliers</h2>{cases}
<h2>Failures and scope</h2><pre>{failures}</pre>
<p>Program validity requires complete execution, all primitive convergence checks, and the shared Stack pre/postconditions. Final tower XY is the largest block-center distance from the base block in XY at program end. Opening/closing motion is not included as a separate path phase, but Release error retains the original pre-opening XY reference.</p>
<p>These four-block trials establish empirical behavior, not physical refinement or certified drift bounds. The geometric motion model is unchanged. Head contacts are intentionally outside the simulated collision model. No demonstration archives are written and no additional execution-version mechanism is added.</p>
<p><small>Source files: executions.json and summary.json beside this HTML. The plot is embedded and works offline. Reproduce with python -m synthesis.experiment.compare_stack_control.</small></p></main></html>"""
    logger.write_artifact("report.html", page)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-seeds", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--run-name", default="combined-200")
    args = parser.parse_args(argv)
    if args.num_seeds < 1 or args.seed_start < 0:
        parser.error("Require a positive count and nonnegative starting seed")
    records = []
    config = dict(
        vars(args),
        num_blocks=4,
        sampling_seconds=0.04,
        previous=dict(pick_mm=10, head_contacts=True, release="Z only"),
        updated=dict(
            pick_mm=2, head_contacts=False, release="fixed XY, full 3D stopping"
        ),
    )
    with RunLogger("runs", "stack-control", config, slug=args.run_name) as logger:
        for seed in range(args.seed_start, args.seed_start + args.num_seeds):
            snapshot = None
            for mode in ("previous", "updated"):
                definition = load_program(
                    "synthesis.examples.stack:build_program", HighLevelContext(), 4
                )
                if mode == "previous":
                    pick = definition.program.instructions[1].body[0]
                    pick.control = replace(pick.control, position_tolerance=0.01)
                rows = []
                with (
                    previous_release_control() if mode == "previous" else nullcontext()
                ):
                    with measure(rows):
                        trace = record_execution(
                            definition,
                            seed=seed,
                            num_blocks=4,
                            env_factory=(
                                stack_env_with_head_contacts
                                if mode == "previous"
                                else None
                            ),
                            initial_snapshot=snapshot,
                            max_loop_iterations=3,
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
                    assert trace.snapshots[0].bindings == snapshot.bindings
                    for key, value in snapshot.arrays.items():
                        np.testing.assert_array_equal(
                            trace.snapshots[0].arrays[key], value
                        )
                positions = np.array(
                    [trace.states[-1][10 + 12 * i : 13 + 12 * i] for i in range(4)]
                )
                records.append(
                    dict(
                        seed=seed,
                        mode=mode,
                        valid=valid,
                        status=trace.metadata["status"],
                        reason=trace.metadata["reason"],
                        actions=len(trace.actions[0]),
                        phases=rows,
                        instruction_steps=[
                            e["control"]["steps"]
                            for e in trace.events
                            if "control" in e
                        ],
                        failed_controls=[
                            e
                            for e in trace.events
                            if "control" in e and not e["control"]["converged"]
                        ],
                        final_tower_xy_mm=float(
                            np.linalg.norm(
                                positions[1:, :2] - positions[0, :2], axis=1
                            ).max()
                            * 1000
                        ),
                    )
                )
                logger.log_metrics(
                    len(records),
                    seed=seed,
                    mode=mode,
                    success_rate=int(valid),
                    actions=len(trace.actions[0]),
                )
                logger.set_status(phase="measure", seed=seed, mode=mode)
            if (seed - args.seed_start + 1) % 10 == 0:
                logger.write_artifact(
                    "summary.json", json.dumps(summarize(records), indent=2)
                )
                logger.progress_line(
                    f"Completed {seed-args.seed_start+1}/{args.num_seeds} paired seeds"
                )
        summary = summarize(records)
        logger.write_artifact("executions.json", json.dumps(records))
        logger.write_artifact("summary.json", json.dumps(summary, indent=2))
        write_report(logger, records, summary)
        logger.finish(
            "completed",
            executions=len(records),
            valid=sum(r["valid"] for r in records),
            report="artifacts/report.html",
        )
    print(f"Report: {logger.run_dir}/artifacts/report.html")


if __name__ == "__main__":
    main()
