"""Bounded collection shared by the CLI and candidate tracing."""

import contextlib
import shutil
import signal
import subprocess
import time
from pathlib import Path

import numpy as np

from synthesis.api.control import get_move_action
from synthesis.cfg.demo_validation import validate_demonstrations
from synthesis.cfg.demos import DemoSegment, DemoTrace
from synthesis.cfg.reset import (
    Recording,
    collect_recording,
    inner_env,
    restore,
)
from synthesis.cfg.tasks import task_spec
from synthesis.util.on import get_higher_tolerance

STACK_SETTLING_STEPS = 50


@contextlib.contextmanager
def execution_deadline(seconds):
    """Preserve a shorter enclosing deadline (including the Unstack process cap)."""
    previous = signal.getitimer(signal.ITIMER_REAL)
    if previous[0] and previous[0] <= seconds:
        yield
        return
    handler = signal.getsignal(signal.SIGALRM)
    started = time.monotonic()

    def expired(signum, frame):
        raise TimeoutError(f"Trajectory exceeded {seconds:g} seconds")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, handler)
        if previous[0]:
            signal.setitimer(
                signal.ITIMER_REAL,
                max(0.000001, previous[0] - (time.monotonic() - started)),
                previous[1],
            )


class VideoRecorder:
    """Stream RGB frames from this execution to one MP4 at fixed 20 FPS."""

    fps = 20

    @staticmethod
    def check_available():
        if shutil.which("ffmpeg") is None:
            raise ValueError("--save-video requires the ffmpeg executable")

    def __init__(self, path):
        self.check_available()
        self.path = Path(path)
        self.process = None
        self.frames = 0
        self.error = None
        self._log = None

    def append(self, frame):
        if self.error:
            return
        try:
            frame = np.ascontiguousarray(frame, dtype=np.uint8)
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError("Renderer did not return an RGB image")
            if self.process is None:
                height, width = frame.shape[:2]
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._log = self.path.with_suffix(".encoder.log").open("wb")
                self.process = subprocess.Popen(
                    [
                        "ffmpeg",
                        "-nostdin",
                        "-v",
                        "error",
                        "-n",
                        "-f",
                        "rawvideo",
                        "-pix_fmt",
                        "rgb24",
                        "-s",
                        f"{width}x{height}",
                        "-r",
                        "20",
                        "-i",
                        "pipe:0",
                        "-an",
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        "-vf",
                        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                        "-movflags",
                        "+faststart",
                        str(self.path),
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=self._log,
                )
            self.process.stdin.write(frame.tobytes())
            self.frames += 1
        except TimeoutError:
            raise
        except Exception as exc:
            self.error = str(exc)

    def close(self):
        try:
            if self.process is not None:
                try:
                    self.process.stdin.close()
                except (BrokenPipeError, OSError):
                    pass
                if self.process.wait(timeout=15):
                    self.error = self.error or "ffmpeg failed; inspect the encoder log"
            elif not self.error:
                self.error = "No video frames were captured"
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
            self.error = "Video encoder did not finish"
        finally:
            if self._log is not None:
                self._log.close()
        if not self.error:
            self.path.with_suffix(".encoder.log").unlink(missing_ok=True)
        return dict(
            path=str(self.path),
            fps=self.fps,
            frames=self.frames,
            status="failed" if self.error else "saved",
            reason=self.error,
        )


def record_execution(
    definition,
    *,
    seed,
    num_blocks,
    task="stack",
    env_factory=None,
    initial_snapshot=None,
    max_loop_iterations=100,
    timeout_seconds=60,
    video_path=None,
    render=False,
    guard_choices=None,
):
    """Record a program from a settled Stack reset or an exact saved snapshot.

    Fresh Stack collection holds the reset gripper position for 50 steps before
    recording. A supplied snapshot is already the requested start: never reset
    or settle it again. Settling is outside demo actions, events, and video.
    Optional guard_choices replay an enabled binding prefix for counterexamples;
    after that prefix, execution uses its ordinary guard-selection policy.
    """
    from synthesis.api.guard_eval import replay_guard_choices
    from synthesis.api.instructions import LoopBudgetExceeded
    from synthesis.mcmc.synthesis import (
        make_roboverify_env,
        preserved_global_rng,
        set_np_seed,
    )

    recording = Recording()
    metadata = dict(definition.metadata)
    metadata["higher_tolerance"] = get_higher_tolerance()
    if guard_choices is not None:
        metadata["guard_choices"] = guard_choices
    metadata["initialization"] = dict(
        source="reset" if initial_snapshot is None else "snapshot", settling_steps=0
    )
    video = VideoRecorder(video_path) if video_path is not None else None
    env = None
    render_error = None
    status, reason = "completed", ""
    started = time.monotonic()

    def frame(current_env):
        nonlocal render_error
        inner = inner_env(current_env)
        if video is not None and not video.error:
            try:
                video.append(inner.render(mode="rgb_array"))
            except TimeoutError:
                raise
            except Exception as exc:
                video.error = str(exc)
        if render and render_error is None:
            try:
                inner.render(mode="human")
            except TimeoutError:
                raise
            except Exception as exc:
                render_error = str(exc)

    try:
        with preserved_global_rng(), execution_deadline(
            timeout_seconds
        ), replay_guard_choices(guard_choices):
            set_np_seed(seed)
            env = (
                env_factory
                or (lambda: make_roboverify_env(task, num_blocks=num_blocks))
            )()
            if initial_snapshot is None:
                first = env.reset()[0]
                if task == "stack":
                    target = first[:3].copy()
                    inner = inner_env(env)
                    for _ in range(STACK_SETTLING_STEPS):
                        env.step(get_move_action(first, target, close_gripper=False))
                        metadata["initialization"]["settling_steps"] += 1
                        first = inner.flatten_observation(inner._get_obs())
            else:
                first = restore(env, initial_snapshot)
            inner_env(env).symbolic_name_to_box_id.update(definition.initial_bindings)
            collect_recording(
                definition.program,
                env,
                recording=recording,
                initial_observation=first,
                max_loop_iterations=max_loop_iterations,
                on_frame=frame if video is not None or render else None,
            )
    except (TimeoutError, LoopBudgetExceeded) as exc:
        status, reason = "incomplete", str(exc)
    except Exception as exc:
        status, reason = "failed", f"{type(exc).__name__}: {exc}"
    finally:
        if video is not None:
            metadata["video"] = video.close()
        if env is not None:
            env.close()
    metadata.update(
        status=status,
        reason=reason,
        render_error=render_error,
        elapsed_seconds=time.monotonic() - started,
        max_loop_iterations=max_loop_iterations,
        timeout_seconds=timeout_seconds,
    )
    return DemoTrace(
        tuple(recording.states),
        tuple(recording.snapshots),
        (tuple(recording.actions), tuple(recording.action_indices)),
        seed,
        task,
        num_blocks,
        tuple(recording.events),
        metadata,
    )


def validate_trace(trace):
    """A valid demonstration completes and satisfies the initial/final task predicates."""
    if trace.metadata.get("status") != "completed" or not trace.states:
        return False
    failed_controls = [
        e
        for e in trace.events
        if e["kind"] == "instruction_end"
        and not e.get("control", {}).get("converged", True)
    ]
    if failed_controls:
        event = failed_controls[0]
        trace.metadata.update(
            status="incomplete",
            reason=f"Controller step limit at {event['path']}: {event['control']}",
        )
        return False
    pre, post = task_spec(trace.task)
    segment = DemoSegment(
        0, 0, len(trace.states) - 1, trace, trace.metadata["initial_bindings"]
    )
    result = validate_demonstrations([segment], pre, post)
    trace.metadata["validation"] = result.as_dict()
    trace.metadata["status"] = "valid" if result else "invalid"
    if not result:
        trace.metadata["reason"] = str(result.issues)
    return bool(result)
