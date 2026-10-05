"""Render paired Stack controllers with measured path overlays and slow motion.

Rendering restores recorded snapshots and never advances the physics. Complete
videos use a shared execution clock; detail clips use a shared phase-local clock.
Frames are repeated for slow motion, without inventing intermediate positions.
"""

import argparse
import json
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from synthesis.cfg.collection import record_execution, validate_trace
from synthesis.cfg.program_source import load_program
from synthesis.cfg.recordings import save_traces
from synthesis.cfg.reset import inner_env, restore
from synthesis.experiment.compare_stack_paths import capture_phases, cross_track
from synthesis.experiment.run_logger import RunLogger
from synthesis.mcmc.synthesis import make_roboverify_stack_env
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext

MODES = ("component_clip", "uniform_scale")
COLORS = ("#bf4935", "#176ea1")
TITLES = ("CLIPPING COMPARISON", "UNIFORM XYZ SCALING")
WIDTH, HEIGHT, FPS = 1280, 900, 25
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
PHASE_LABELS = {
    "approach": "Pick: horizontal approach",
    "descend": "Pick: descend to block",
    "lift": "Lift the held block",
    "transfer": "Transfer the held block",
    "lower": "Lower onto the tower",
    "retreat": "Release: vertical retreat",
}


@lru_cache(maxsize=8)
def font(size):
    return ImageFont.truetype(FONT, size)


def align_phases(trace, rows):
    """Locate whole measured phase paths in the original action-aligned recording."""
    states = np.asarray(trace.states)[:, :3]
    assert trace.actions[1] == tuple(range(len(states)))
    assert len(states) == len(trace.actions[0]) + 1
    cursor = 0
    for row in rows:
        points = np.asarray(row["positions"])
        candidates = (
            np.flatnonzero(
                np.all(
                    np.isclose(states[cursor:], points[0], rtol=0, atol=1e-8), axis=1
                )
            )
            + cursor
        )
        matches = [
            i
            for i in candidates
            if i + len(points) <= len(states)
            and np.allclose(states[i : i + len(points)], points, rtol=0, atol=1e-8)
        ]
        if not matches:
            raise ValueError("Phase positions do not match the recorded snapshots")
        row["start_index"] = int(matches[0])
        row["end_index"] = int(matches[0] + len(points) - 1)
        cursor = row["end_index"]


class Video:
    def __init__(self, path):
        self.path, self.frames = path, 0
        self.process = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-n",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{WIDTH}x{HEIGHT}",
                "-r",
                str(FPS),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "19",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )

    def append(self, frame, repeat=1):
        data = np.asarray(frame, dtype=np.uint8).tobytes()
        for _ in range(repeat):
            self.process.stdin.write(data)
            self.frames += 1

    def close(self):
        self.process.stdin.close()
        if self.process.wait(timeout=60):
            raise RuntimeError(f"Encoding failed: {self.path}")
        result = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-count_frames",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,avg_frame_rate,nb_read_frames,duration",
                    "-of",
                    "json",
                    str(self.path),
                ],
                text=True,
            )
        )["streams"][0]
        assert int(result["nb_read_frames"]) == self.frames
        assert (result["width"], result["height"]) == (WIDTH, HEIGHT)
        assert result["avg_frame_rate"] == f"{FPS}/1"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-xerror",
                "-i",
                str(self.path),
                "-f",
                "null",
                "-",
            ],
            check=True,
        )
        return dict(path=self.path.name, frames=self.frames, **result)


def dashed(draw, start, end, fill):
    start, end = np.array(start), np.array(end)
    count = max(1, int(np.linalg.norm(end - start) / 10))
    for i in range(0, count, 2):
        a, b = (
            start + (end - start) * i / count,
            start + (end - start) * min(i + 1, count) / count,
        )
        draw.line([tuple(a), tuple(b)], fill=fill, width=2)


class Renderer:
    def __init__(self, pairs):
        self.pairs = pairs
        self.env = make_roboverify_stack_env(num_blocks=4)
        self.inner = inner_env(self.env)
        self.dt = self.inner.dt
        assert abs(self.dt - 1 / FPS) < 1e-9
        self.viewer = self.inner._get_viewer("rgb_array")

    @lru_cache(maxsize=300)
    def scene(self, seed, mode, index, close):
        trace, _ = self.pairs[seed][mode]
        restore(self.env, trace.snapshots[index])
        cam = self.viewer.cam
        cam.lookat[:] = [1.30, 0.75, 0.52]
        cam.distance, cam.elevation, cam.azimuth = (
            (0.95, -43, 135) if close else (1.25, -32, 135)
        )
        return Image.fromarray(
            self.inner.render(mode="rgb_array", width=640, height=370)
        )

    def frame(self, seed, indices, title, subtitle, *, detail=None):
        frame = Image.new("RGB", (WIDTH, HEIGHT), "#f1f3f6")
        draw = ImageDraw.Draw(frame)
        draw.text((22, 14), title, fill="#172133", font=font(28))
        draw.text((22, 54), subtitle, fill="#495468", font=font(18))
        for side, (mode, index) in enumerate(zip(MODES, indices)):
            x, color = side * 640, COLORS[side]
            trace, rows = self.pairs[seed][mode]
            draw.rounded_rectangle((x + 10, 88, x + 630, 132), radius=9, fill=color)
            draw.text((x + 24, 96), TITLES[side], fill="white", font=font(23))
            frame.paste(self.scene(seed, mode, index, detail is not None), (x, 142))
            row = (
                rows[detail]
                if detail is not None
                else next(
                    (r for r in reversed(rows) if r["start_index"] <= index), rows[0]
                )
            )
            row_index = rows.index(row)
            other = self.pairs[seed][MODES[1 - side]][1][row_index]
            local_step = min(max(index - row["start_index"], 0), row["steps"])
            points = np.array(row["positions"][: local_step + 1])
            bend = cross_track(points, row["target"]).max() * 1000
            complete = index == len(trace.states) - 1
            hold = index > row["end_index"]
            label = (
                "Stack complete (final frame held)"
                if complete
                else PHASE_LABELS[row["phase"]]
            )
            if hold and not complete:
                label = "Gripper action / phase transition"
            draw.text(
                (x + 22, 522),
                f"Placement {row['iteration'] + 1}/3  |  {label}",
                fill="#172133",
                font=font(18),
            )
            draw.text(
                (x + 22, 552),
                f"Maximum 3D line deviation so far: {bend:.2f} mm",
                fill=color,
                font=font(20),
            )
            self.path_chart(draw, x, row, other, points, color)
            draw.text(
                (x + 22, 823),
                "Solid: measured gripper path   Dashed: start-to-target line",
                fill="#495468",
                font=font(16),
            )
        draw.text(
            (22, 866),
            "Actual MuJoCo snapshots. Slow motion repeats frames; no interpolated robot motion. Both runs passed Stack validation.",
            fill="#495468",
            font=font(16),
        )
        return frame

    @staticmethod
    def path_chart(draw, x, row, other, points, color):
        box = (x + 22, 588, x + 618, 807)
        draw.rounded_rectangle(box, radius=8, fill="white")
        all_points = np.array(
            row["positions"] + other["positions"] + [row["target"], other["target"]]
        )[:, :2]
        lower, upper = all_points.min(axis=0), all_points.max(axis=0)
        center = (upper + lower) / 2
        extent = np.maximum(upper - lower, 0.06)
        scale = min(460 / extent[0], 170 / extent[1])

        def screen(point):
            p = (np.asarray(point)[:2] - center) * scale
            return float(x + 320 + p[0]), float(699 - p[1])

        start, target = screen(row["positions"][0]), screen(row["target"])
        dashed(draw, start, target, "#777f8a")
        curve = [screen(p) for p in points]
        if len(curve) > 1:
            draw.line(curve, fill=color, width=4)
        for p in curve:
            draw.ellipse((p[0] - 3, p[1] - 3, p[0] + 3, p[1] + 3), fill=color)
        draw.ellipse(
            (start[0] - 5, start[1] - 5, start[0] + 5, start[1] + 5),
            outline="#495468",
            width=2,
        )
        draw.line(
            (target[0] - 6, target[1], target[0] + 6, target[1]),
            fill="#495468",
            width=2,
        )
        draw.line(
            (target[0], target[1] - 6, target[0], target[1] + 6),
            fill="#495468",
            width=2,
        )
        draw.text((x + 36, 597), "TOP VIEW (XY)", fill="#495468", font=font(14))
        bar = min(0.05, 80 / scale)
        draw.line((x + 42, 780, x + 42 + bar * scale, 780), fill="#495468", width=3)
        draw.text((x + 42, 755), f"{bar * 1000:.0f} mm", fill="#495468", font=font(13))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--measurements",
        type=Path,
        required=True,
        help="phases.json from a four-block compare_stack_paths run with both modes",
    )
    parser.add_argument("--output-dir", default="runs")
    args = parser.parse_args(argv)
    source = args.measurements
    previous = json.loads(source.read_text())
    worst = max(
        (r for r in previous if r["mode"] == "component_clip"),
        key=lambda r: r["max_cross_track_mm"],
    )
    seeds = list(dict.fromkeys((min(r["seed"] for r in previous), worst["seed"])))
    config = dict(
        seeds=seeds,
        num_blocks=4,
        fps=FPS,
        source=str(source),
        full_speed=0.5,
        detail_speed=0.1,
        selection="first seed and largest phase deviation in the supplied measured baseline",
        production_controller_changed=False,
    )
    with RunLogger(
        args.output_dir, "controller-videos", config, slug="side-by-side"
    ) as logger:
        out = logger.run_dir / "artifacts"
        definition = load_program(
            "synthesis.examples.stack:build_program", HighLevelContext(), 4
        )
        pairs, all_traces, metadata = {}, [], []
        for seed in seeds:
            pairs[seed], snapshot = {}, None
            for mode in MODES:
                rows = []
                with capture_phases(rows, uniform=mode == "uniform_scale"):
                    trace = record_execution(
                        definition,
                        seed=seed,
                        num_blocks=4,
                        initial_snapshot=snapshot,
                        max_loop_iterations=3,
                    )
                assert validate_trace(trace), trace.metadata
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
                align_phases(trace, rows)
                for row in rows:
                    reference = next(
                        r
                        for r in previous
                        if r["seed"] == seed
                        and r["mode"] == mode
                        and r["iteration"] == row["iteration"]
                        and r["phase"] == row["phase"]
                    )
                    np.testing.assert_allclose(
                        row["positions"], reference["positions"], rtol=0, atol=1e-8
                    )
                # Trial archives are explicitly diagnostic, not pipeline demonstrations.
                trace.metadata.update(
                    status="diagnostic",
                    validation_status="valid",
                    controller_trial=mode,
                )
                pairs[seed][mode] = (trace, rows)
                all_traces.append(trace)
                metadata.append(
                    dict(seed=seed, mode=mode, phases=rows, task_valid=True)
                )
            logger.progress_line(
                f"Seed {seed}: paired starts and original measured paths reproduced"
            )
        save_traces(out / "diagnostic_rollouts.npz", all_traces)
        logger.write_artifact("phases.json", json.dumps(metadata, indent=2))
        renderer, videos = Renderer(pairs), []
        try:
            for seed in seeds:
                video = Video(out / f"stack_seed_{seed:03d}_side_by_side.mp4")
                ends = [len(pairs[seed][mode][0].states) - 1 for mode in MODES]
                for step in range(max(ends) + 1):
                    frame = renderer.frame(
                        seed,
                        [min(step, end) for end in ends],
                        f"Stack controller comparison | seed {seed}",
                        f"Same saved start | shared simulation clock: {step * renderer.dt:.2f} s | half speed (0.5x)",
                    )
                    if step == 0:
                        video.append(frame, FPS * 2)
                    video.append(frame, 2)
                    if step == max(ends):
                        video.append(frame, FPS * 3)
                videos.append(video.close())
                logger.set_status(phase="render", seed=seed, videos=len(videos))
                logger.progress_line(
                    f"Rendered complete side-by-side execution for seed {seed}"
                )

            seed, iteration = worst["seed"], worst["iteration"]
            video = Video(out / "horizontal_motion_explained.mp4")
            for detail in (iteration * 6, iteration * 6 + 3):
                rows = [pairs[seed][mode][1][detail] for mode in MODES]
                for step in range(max(row["steps"] for row in rows) + 1):
                    indices = [
                        row["start_index"] + min(step, row["steps"]) for row in rows
                    ]
                    frame = renderer.frame(
                        seed,
                        indices,
                        f"Why paths bend | seed {seed}, placement {iteration + 1}: {PHASE_LABELS[rows[0]['phase']]}",
                        f"Phase starts synchronized | local time: {step * renderer.dt:.2f} s | slow motion (0.1x) | largest-bend example",
                        detail=detail,
                    )
                    if step == 0:
                        video.append(frame, FPS * 2)
                    video.append(frame, 10)
                    if step == 5:
                        frame.save(out / f"detail_{rows[0]['phase']}.png")
                    if step == max(row["steps"] for row in rows):
                        video.append(frame, FPS * 3)
                logger.progress_line(f"Rendered slow-motion {rows[0]['phase']}")
            videos.append(video.close())
        finally:
            renderer.env.close()
        logger.write_artifact("videos.json", json.dumps(videos, indent=2))
        logger.write_artifact(
            "README.md",
            "# Stack controller videos\n\n"
            "Left: historical component clipping. Right: current uniform XYZ scaling; gripper command unchanged.\n\n"
            f"Both four-block programs use identical full settled initial snapshots. Seeds {seeds} reproduce the prior measured paths within 1e-8 m. "
            f"Seed {worst['seed']} is deliberately the largest-bend example from the supplied sample, not a typical-case estimate.\n\n"
            "Complete videos share simulation time and run at half speed. The detailed approach/transfer clip synchronizes each phase's start and runs at one-tenth speed; "
            "later phases can begin from slightly different physical states because earlier controller choices differ. "
            "Opening title and final-frame holds are pauses. Shorter executions/phases hold their last frame.\n\n"
            "The lower diagrams show actual recorded gripper XY paths and intended straight lines; displayed deviation is measured in 3D from each phase's own start-to-target line. "
            "They are sampled at control boundaries (0.04 s), with no motion interpolation. Both variants passed task validation and primitive convergence. "
            "Uniform scaling is now the production behavior. diagnostic_rollouts.npz is marked diagnostic because this report includes historical controller replays.\n\n"
            + "\n".join(
                f"- [{v['path']}]({v['path']}): {float(v['duration']):.2f} s, {WIDTH}x{HEIGHT}, {FPS} FPS."
                for v in videos
            )
            + "\n",
        )
        logger.finish(
            "completed",
            videos=len(videos),
            validated_executions=len(all_traces),
            report="artifacts/README.md",
        )
    print(f"Videos: {out}")


if __name__ == "__main__":
    main()
