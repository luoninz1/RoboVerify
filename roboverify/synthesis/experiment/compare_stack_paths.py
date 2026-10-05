"""Compare historical clipped Stack gripper paths with current uniform scaling.

The trial patches only this serial diagnostic process, never the controller source.
Each variant uses a fresh environment; paired trials restore the baseline's settled
full initial snapshot. Opening/closing and initial settling are not motion phases.
"""

import argparse
import json
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np

from synthesis.api import control as control_module
from synthesis.api.control import PrimitiveController
from synthesis.experiment.run_logger import RunLogger

PHASES = ("approach", "descend", "lift", "transfer", "lower", "retreat")


def direction_angles(raw, sent):
    """Angular change in degrees; zero commands have no direction change."""
    raw, sent = np.asarray(raw), np.asarray(sent)
    scale = np.linalg.norm(raw, axis=1) * np.linalg.norm(sent, axis=1)
    cosine = np.ones(len(raw))
    np.divide(np.sum(raw * sent, axis=1), scale, out=cosine, where=scale > 0)
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def cross_track(points, target):
    """Perpendicular distances to the infinite line from phase start to target."""
    points, target = np.asarray(points), np.asarray(target)
    delta = points - points[0]
    direction = target - points[0]
    length = np.linalg.norm(direction)
    if length == 0:
        return np.linalg.norm(delta, axis=1)
    unit = direction / length
    return np.linalg.norm(delta - np.outer(delta @ unit, unit), axis=1)


def phase_metrics(row):
    points = np.asarray(row["positions"])
    raw = np.asarray(row["raw_actions"]).reshape(-1, 4)[:, :3]
    sent = np.asarray(row["sent_actions"]).reshape(-1, 4)[:, :3]
    clipped = np.clip(sent, -1, 1)
    angles = direction_angles(raw, clipped)
    return dict(
        steps=len(raw),
        raw_saturated_steps=int(np.any(np.abs(raw) > 1, axis=1).sum()),
        direction_changed_steps=int((angles > 1).sum()),
        max_direction_change_deg=float(angles.max(initial=0)),
        max_cross_track_mm=float(cross_track(points, row["target"]).max() * 1000),
        max_chord_deviation_mm=float(cross_track(points, points[-1]).max() * 1000),
        endpoint_error_mm=float(np.linalg.norm(points[-1] - row["target"]) * 1000),
    )


@contextmanager
def capture_phases(rows, *, uniform=False):
    """Compare historical component clipping with current uniform scaling.

    Raw command generation is patched only for this serial diagnostic so the
    historical baseline stays reproducible after uniform scaling becomes default.
    """
    original_move, original_step = PrimitiveController.move, PrimitiveController._step
    active = None

    def raw_action(observation, target_position, *, gain=20.0, close_gripper=False):
        command = gain * (np.asarray(target_position) - np.asarray(observation)[:3])
        return np.r_[command, -0.2 if close_gripper else 0.0]

    def move(controller, target, **kwargs):
        nonlocal active
        start = controller.observation[:3].copy()
        target = np.asarray(target, dtype=float).copy()
        if kwargs.get("vertical_only", False):
            target[:2] = start[:2]
        phase = PHASES[len(rows) % len(PHASES)]
        expected = phase if phase in ("approach", "descend", "retreat") else "move"
        if kwargs.get("phase", "move") != expected:
            raise ValueError("Expected the supplied Stack program's six motion phases")
        row = dict(
            phase=phase,
            iteration=len(rows) // len(PHASES),
            target=target.tolist(),
            positions=[start.tolist()],
            raw_actions=[],
            sent_actions=[],
        )
        active = row
        try:
            row["converged"] = bool(original_move(controller, target, **kwargs))
        finally:
            active = None
            row.update(phase_metrics(row))
            rows.append(row)
        return row["converged"]

    def step(controller, action):
        command = np.asarray(action).copy()
        if active is not None:
            active["raw_actions"].append(command.tolist())
            if uniform:
                command[:3] /= max(1.0, float(np.max(np.abs(command[:3]))))
            else:
                # Explicitly reproduce the historical backend, now removed.
                command[:3] = np.clip(command[:3], -1, 1)
            active["sent_actions"].append(command.tolist())
        result = original_step(controller, command)
        if active is not None:
            active["positions"].append(controller.observation[:3].tolist())
        return result

    with patch.object(control_module, "get_move_action", raw_action), patch.object(
        PrimitiveController, "move", move
    ), patch.object(PrimitiveController, "_step", step):
        yield


def summarize(rows):
    def stats(values):
        return dict(
            median=float(np.median(values)),
            p95=float(np.percentile(values, 95)),
            maximum=float(max(values)),
        )

    return dict(
        phases=len(rows),
        steps=sum(r["steps"] for r in rows),
        raw_saturated_steps=sum(r["raw_saturated_steps"] for r in rows),
        direction_changed_steps=sum(r["direction_changed_steps"] for r in rows),
        phases_direction_changed=sum(r["direction_changed_steps"] > 0 for r in rows),
        phases_over_2mm=sum(r["max_cross_track_mm"] > 2 for r in rows),
        phases_over_5mm=sum(r["max_cross_track_mm"] > 5 for r in rows),
        phases_over_10mm=sum(r["max_cross_track_mm"] > 10 for r in rows),
        max_cross_track_mm=stats([r["max_cross_track_mm"] for r in rows]),
        max_chord_deviation_mm=stats([r["max_chord_deviation_mm"] for r in rows]),
        max_direction_change_deg=stats([r["max_direction_change_deg"] for r in rows]),
    )


def markdown_report(report):
    lines = [
        "# Stack controller path experiment",
        "",
        "Fresh resets with 50 settling steps; the supplied Stack program and default controller settings. "
        "Paired uniform-scaling executions restore the baseline's exact settled start. "
        "No seeds are filtered or replaced. All task validations use the current pre/postconditions.",
        "",
        "Bend = maximum perpendicular gripper distance from the phase's original start-to-target line, "
        "sampled at control-step boundaries. Counts above 5 mm are descriptive, not safety thresholds. "
        "A direction change means more than 1 degree between the raw proportional command and the "
        "command after scaling/clipping. Saturation alone need not change direction. "
        "Opening, closing, and intentional turns between phases are excluded. Substep excursions, "
        "finger/arm geometry, and held-block trajectories are not measured.",
        "",
    ]
    for mode, result in report.items():
        lines.extend(
            [
                f"## {mode}",
                "",
                f"Executions: {result['executions']}; validated: {result['valid']}; "
                f"executions with direction changes: {result['executions_direction_changed']}; "
                f"executions with a phase over 5 mm: {result['executions_over_5mm']}.",
                "",
                "| Phase | Count | Direction changed | Bend >5 mm | Median / p95 / max bend (mm) |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for phase, value in result["by_phase"].items():
            bend = value["max_cross_track_mm"]
            lines.append(
                f"| {phase} | {value['phases']} | {value['phases_direction_changed']} | "
                f"{value['phases_over_5mm']} | {bend['median']:.2f} / {bend['p95']:.2f} / {bend['maximum']:.2f} |"
            )
        lines.append("")
    return "\n".join(lines)


def plot_paths(rows, destination):
    """Save an example and empirical distribution, using only measured positions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = list(dict.fromkeys(r["mode"] for r in rows))
    colors = ("#bd432f", "#176ea1")
    labels = ("Historical: component clipping", "Current: uniform scaling")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    example = next(r for r in rows if r["phase"] == "approach")
    origin = np.array(example["positions"][0])
    target = (np.array(example["target"]) - origin) * 1000
    axes[0].plot(
        [0, target[0]], [0, target[1]], "--", color="0.5", label="Start-to-target line"
    )
    for mode, color, label in zip(modes, colors, labels):
        row = next(
            r
            for r in rows
            if r["mode"] == mode
            and r["seed"] == example["seed"]
            and r["phase"] == "approach"
        )
        points = (np.array(row["positions"]) - origin) * 1000
        axes[0].plot(
            points[:, 0], points[:, 1], "o-", color=color, label=label, markersize=4
        )
        values = sorted(
            r["max_cross_track_mm"]
            for r in rows
            if r["mode"] == mode and r["phase"] in ("approach", "transfer")
        )
        axes[1].step(
            values,
            np.arange(1, len(values) + 1) / len(values) * 100,
            color=color,
            label=label,
        )
    axes[0].set(
        title=f"First Pick approach, seed {example['seed']} (top view)",
        xlabel="X from start (mm)",
        ylabel="Y from start (mm)",
    )
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].legend(fontsize=8)
    axes[1].set(
        title="All Pick approaches and loaded transfers",
        xlabel="Maximum distance from original 3D line (mm)",
        ylabel="Phases at or below this deviation (%)",
        ylim=(0, 101),
    )
    axes[1].axvline(5, color="0.5", linestyle="--", linewidth=1)
    axes[1].legend(fontsize=8, loc="lower right")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-seeds", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--compare-uniform", action="store_true")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--run-name", default="stack-paths")
    args = parser.parse_args(argv)
    if args.num_seeds < 1 or args.num_blocks < 2:
        parser.error("Require positive seeds and at least two blocks")

    from synthesis.cfg.collection import record_execution, validate_trace
    from synthesis.cfg.program_source import load_program
    from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext

    context = HighLevelContext()
    definition = load_program(
        "synthesis.examples.stack:build_program", context, args.num_blocks
    )
    variants = (
        ["component_clip", "uniform_scale"]
        if args.compare_uniform
        else ["component_clip"]
    )
    executions, phases = [], []
    config = dict(
        vars(args),
        program=definition.metadata,
        bend_threshold_mm=5,
        direction_threshold_deg=1,
        measurement="control-step gripper site positions",
    )
    with RunLogger(
        args.output_dir, "controller-paths", config, slug=args.run_name
    ) as logger:
        for seed in range(args.seed_start, args.seed_start + args.num_seeds):
            snapshot = None
            for mode in variants:
                rows = []
                with capture_phases(rows, uniform=mode == "uniform_scale"):
                    trace = record_execution(
                        definition,
                        seed=seed,
                        num_blocks=args.num_blocks,
                        initial_snapshot=snapshot,
                        max_loop_iterations=args.num_blocks - 1,
                    )
                valid = validate_trace(trace)
                if mode == "component_clip":
                    if not trace.snapshots:
                        raise RuntimeError(
                            f"Seed {seed} has no start: {trace.metadata}"
                        )
                    snapshot = trace.snapshots[0]
                for row in rows:
                    row.update(seed=seed, mode=mode)
                phases.extend(rows)
                executions.append(
                    dict(
                        seed=seed,
                        mode=mode,
                        valid=valid,
                        status=trace.metadata["status"],
                        reason=trace.metadata["reason"],
                        actions=len(trace.actions[0]),
                        phases=len(rows),
                    )
                )
                logger.log_metrics(
                    len(executions),
                    seed=seed,
                    mode=mode,
                    valid=valid,
                    actions=len(trace.actions[0]),
                )
                logger.set_status(
                    phase="measure", seed=seed, mode=mode, executions=len(executions)
                )
            if (seed - args.seed_start + 1) % 10 == 0:
                logger.progress_line(
                    f"Measured {seed - args.seed_start + 1}/{args.num_seeds} seeds"
                )

        report = {}
        for mode in variants:
            selected = [r for r in phases if r["mode"] == mode]
            runs = [r for r in executions if r["mode"] == mode]
            report[mode] = dict(
                executions=len(runs),
                valid=sum(r["valid"] for r in runs),
                executions_direction_changed=len(
                    {r["seed"] for r in selected if r["direction_changed_steps"]}
                ),
                executions_over_5mm=len(
                    {r["seed"] for r in selected if r["max_cross_track_mm"] > 5}
                ),
                overall=summarize(selected),
                by_phase={
                    p: summarize([r for r in selected if r["phase"] == p])
                    for p in PHASES
                },
                failures=[r for r in runs if not r["valid"]],
            )
        logger.write_artifact("phases.json", json.dumps(phases))
        logger.write_artifact("executions.json", json.dumps(executions, indent=2))
        logger.write_artifact("comparison.json", json.dumps(report, indent=2))
        logger.write_artifact("comparison.md", markdown_report(report))
        plot_paths(phases, logger.run_dir / "artifacts" / "paths.png")
        logger.finish(
            "completed",
            executions=len(executions),
            valid=sum(r["valid"] for r in executions),
            report="artifacts/comparison.md",
        )
    print(f"Report: {logger.run_dir}/artifacts/comparison.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
