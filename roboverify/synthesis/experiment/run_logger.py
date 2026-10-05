"""Structured run logging with a bounded, agent-readable on-disk footprint.

The design rule is that reading a run must cost the same whether it is at iteration
10 or 10,000: ``status.json`` is *overwritten* rather than appended, ``metrics.jsonl``
holds only flat numeric records meant to be aggregated, and anything large (program
text, pickles, videos, tracebacks) lives under ``artifacts/`` and is referenced by
path.
"""

import atexit
import json
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def utc_now() -> str:
    """ISO-8601 UTC timestamp, second resolution."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_stamp() -> str:
    """Compact UTC stamp suitable for a directory name."""
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def git_info(repo_dir: Optional[str] = None) -> dict:
    """Best-effort git provenance for the run.

    A run whose git state is unknown or dirty is still worth recording; the flags let
    a later reader decide how much to trust the attribution of a result to code.
    """

    def run(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(
                args,
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if out.returncode != 0:
            return None
        return out.stdout.strip()

    sha = run("git", "rev-parse", "--short", "HEAD")
    status = run("git", "status", "--porcelain")
    return {
        "sha": sha or "nogit",
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def package_versions() -> dict:
    """Versions of the libraries that can silently change numeric results."""
    versions = {"python": sys.version.split()[0]}
    for module_name, attr in (
        ("numpy", "__version__"),
        ("torch", "__version__"),
        ("z3", "get_version_string"),
    ):
        try:
            module = __import__(module_name)
            value = getattr(module, attr)
            versions[module_name] = value() if callable(value) else str(value)
        except Exception:  # pragma: no cover - provenance is best effort
            versions[module_name] = None
    return versions


class RollingRate:
    """Rate of true values over a fixed-size sliding window."""

    def __init__(self, window: int = 100):
        self.window = window
        self._values: deque = deque(maxlen=window)

    def add(self, value: bool) -> None:
        self._values.append(bool(value))

    @property
    def rate(self) -> Optional[float]:
        if not self._values:
            return None
        return round(sum(self._values) / len(self._values), 4)


class RunLogger:
    """Owns one run directory and the files inside it.

    Parameters
    ----------
    root:
        Directory holding all runs, e.g. ``runs``.
    name:
        Experiment family, e.g. ``mcmc``. Runs land in ``<root>/<name>/<run_id>``
        and ``<root>/<name>/latest`` is repointed at the newest one.
    config:
        JSON-serializable resolved configuration; written verbatim to
        ``config.json`` alongside provenance.
    slug:
        Short human tag appended to the run id, e.g. ``stack-nb4``.
    capture_stdout:
        Redirect file descriptors 1 and 2 into ``stdout.log``. This catches prints
        from imported code and C-level output from MuJoCo/OpenGL, and is inherited
        by ``multiprocessing`` fork children, so no call site needs to be edited.
    mirrors:
        Optional objects with ``log(step, dict)`` and ``close()``; kept so a
        Weights & Biases mirror can be added later without changing call sites.
        The JSONL files always remain the source of truth.
    """

    def __init__(
        self,
        root: str,
        name: str,
        config: dict,
        *,
        slug: Optional[str] = None,
        capture_stdout: bool = True,
        event_min_step_gap: int = 50,
        mirrors: tuple = (),
        repo_dir: Optional[str] = None,
    ):
        self.git = git_info(repo_dir)
        parts = [utc_stamp(), self.git["sha"]]
        if slug:
            parts.append(slug)
        self.run_id = "-".join(parts)
        self.name = name
        self.run_dir = Path(root) / name / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "artifacts").mkdir(exist_ok=True)
        self._point_latest_symlink(Path(root) / name)

        self._t0 = time.time()
        self._mirrors = tuple(mirrors)
        self._event_min_step_gap = event_min_step_gap
        self._event_last_step: dict = {}
        self._finished = False
        self._status: dict = {
            "run_id": self.run_id,
            "name": name,
            "phase": "init",
            "alive": True,
            "pid": os.getpid(),
            "started_at": utc_now(),
        }

        self._write_json(
            self.run_dir / "config.json",
            {
                "run_id": self.run_id,
                "name": name,
                "slug": slug,
                "started_at": utc_now(),
                "git": self.git,
                "argv": sys.argv,
                "hostname": socket.gethostname(),
                "cwd": os.getcwd(),
                "versions": package_versions(),
                "config": config,
            },
        )

        self._metrics_fh = open(self.run_dir / "metrics.jsonl", "a", buffering=1)
        self._events_fh = open(self.run_dir / "events.jsonl", "a", buffering=1)

        self.terminal = sys.stdout
        self._saved_fds: Optional[tuple] = None
        self._stdout_log = None
        if capture_stdout:
            self._start_stdout_capture()

        self.set_status(phase="init")
        self.log_event("run_start", f"run {self.run_id} started", git=self.git)

        self._install_exit_hooks()

    # ---------------------------------------------------------------- plumbing

    def _point_latest_symlink(self, family_dir: Path) -> None:
        """Repoint ``<family>/latest`` at this run.

        A stable path is what makes monitoring cheap: a reader never has to list and
        sort directories to find the current run.
        """
        link = family_dir / "latest"
        try:
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(self.run_id)
        except OSError:
            pass  # filesystems without symlink support are not fatal

    def _start_stdout_capture(self) -> None:
        sys.stdout.flush()
        sys.stderr.flush()
        self._saved_fds = (os.dup(1), os.dup(2))
        # The terminal stream keeps writing to the real console so a human still
        # sees progress lines while the firehose goes to disk.
        self.terminal = os.fdopen(os.dup(self._saved_fds[0]), "w", buffering=1)
        self._stdout_log = open(self.run_dir / "stdout.log", "ab", buffering=0)
        os.dup2(self._stdout_log.fileno(), 1)
        os.dup2(self._stdout_log.fileno(), 2)
        # Redirecting onto a regular file makes Python pick block buffering, which
        # loses output from ``multiprocessing`` pool workers when they exit (CEM runs
        # its objective in a pool) and can interleave partial lines from concurrent
        # writers. Forcing line buffering here also applies to fork children.
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(line_buffering=True)
            except (AttributeError, ValueError):  # pragma: no cover
                pass

    def _stop_stdout_capture(self) -> None:
        if self._saved_fds is None:
            return
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except (ValueError, OSError):
            pass
        saved_out, saved_err = self._saved_fds
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        for fd in self._saved_fds:
            os.close(fd)
        self._saved_fds = None
        if self._stdout_log is not None:
            self._stdout_log.close()
            self._stdout_log = None

    def _install_exit_hooks(self) -> None:
        atexit.register(self._atexit_finish)
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                previous = signal.getsignal(sig)
                signal.signal(sig, self._make_signal_handler(sig, previous))
            except (ValueError, OSError):  # pragma: no cover - non-main thread
                pass

    def _make_signal_handler(self, sig, previous):
        def handler(signum, frame):
            self.finish("interrupted", exit_reason=f"signal {signum}")
            signal.signal(sig, previous)
            os.kill(os.getpid(), signum)

        return handler

    def _atexit_finish(self) -> None:
        if not self._finished:
            self.finish("unknown", exit_reason="process exited without finish()")

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        """Write JSON atomically so a concurrent reader never sees a partial file."""
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
        os.replace(tmp, path)

    # ------------------------------------------------------------------- status

    def set_status(self, **kv: Any) -> None:
        """Merge ``kv`` into the status object and rewrite ``status.json``."""
        self._status.update(kv)
        self._status["elapsed_s"] = round(time.time() - self._t0, 1)
        self._status["heartbeat"] = utc_now()
        self._write_json(self.run_dir / "status.json", self._status)

    def set_progress(self, step: int, total: Optional[int] = None, **kv: Any) -> None:
        """Update status with derived progress fields.

        ``heartbeat`` combined with ``alive`` is what lets a reader distinguish a
        finished run from one that is wedged: a hung run keeps ``alive`` true while
        its heartbeat goes stale.
        """
        elapsed = max(time.time() - self._t0, 1e-9)
        done = step + 1
        fields = {"iter": step, "total": total}
        fields["iters_per_min"] = round(60.0 * done / elapsed, 2)
        if total:
            fields["pct"] = round(100.0 * done / total, 1)
            remaining = max(total - done, 0)
            fields["eta_s"] = round(remaining * elapsed / done, 1)
        fields.update(kv)
        self.set_status(**fields)

    def progress_line(self, text: str) -> None:
        """Write one short line to the real console, bypassing ``stdout.log``."""
        try:
            self.terminal.write(text.rstrip() + "\n")
            self.terminal.flush()
        except (ValueError, OSError):
            pass

    # ------------------------------------------------------------------ records

    def log_metrics(self, step: int, **kv: Any) -> None:
        """Append one flat record to ``metrics.jsonl``.

        Values must be numbers, booleans, ``None`` or short strings. Anything large
        belongs in ``artifacts/``.
        """
        record = {"iter": step, "t": round(time.time() - self._t0, 3)}
        record.update(kv)
        self._metrics_fh.write(json.dumps(record, default=str) + "\n")
        for mirror in self._mirrors:
            try:
                mirror.log(step, kv)
            except Exception:  # pragma: no cover - a mirror must never break a run
                pass

    def log_event(
        self,
        kind: str,
        message: str,
        *,
        step: Optional[int] = None,
        force: bool = False,
        **kv: Any,
    ) -> bool:
        """Append a rare, meaningful event.

        Events are rate-limited per ``kind`` so that a condition holding for
        thousands of iterations (a long run of BMC-infeasible candidates, say)
        cannot flood the file. Returns whether the event was written.
        """
        if not force and step is not None:
            last = self._event_last_step.get(kind)
            if last is not None and step - last < self._event_min_step_gap:
                return False
            self._event_last_step[kind] = step

        record = {"ts": utc_now(), "kind": kind, "message": message}
        if step is not None:
            record["iter"] = step
        record.update(kv)
        self._events_fh.write(json.dumps(record, default=str) + "\n")
        self._status["last_event"] = f"{kind}@{step}" if step is not None else kind
        return True

    def log_exception(self, exc: BaseException, *, step: Optional[int] = None) -> Path:
        """Persist a traceback under ``artifacts/`` and record a pointer to it."""
        name = f"traceback_iter{step}.txt" if step is not None else "traceback.txt"
        path = self.run_dir / "artifacts" / name
        path.write_text("".join(traceback.format_exception(exc)))
        self.log_event(
            "exception",
            f"{type(exc).__name__}: {exc}",
            step=step,
            force=True,
            traceback_path=str(path),
        )
        return path

    # ---------------------------------------------------------------- artifacts

    def artifact_dir(self, *parts: str) -> Path:
        """Create and return ``artifacts/<parts...>``."""
        path = (
            self.run_dir / "artifacts" / Path(*parts)
            if parts
            else self.run_dir / "artifacts"
        )
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_artifact(self, relative_path: str, text: str) -> Path:
        path = self.run_dir / "artifacts" / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    # ------------------------------------------------------------------- finish

    def finish(self, status: str, **kv: Any) -> None:
        """Write ``result.json``, mark the run not alive and release resources."""
        if self._finished:
            return
        self._finished = True

        result = {
            "run_id": self.run_id,
            "status": status,
            "finished_at": utc_now(),
            "elapsed_s": round(time.time() - self._t0, 1),
            "git": self.git,
        }
        result.update(kv)
        self._write_json(self.run_dir / "result.json", result)

        self.log_event("run_end", f"run finished: {status}", force=True, **kv)
        self.set_status(alive=False, phase="finished", result_status=status, **kv)

        for handle in (self._metrics_fh, self._events_fh):
            try:
                handle.close()
            except (ValueError, OSError):
                pass
        for mirror in self._mirrors:
            try:
                mirror.close()
            except Exception:  # pragma: no cover
                pass
        self._stop_stdout_capture()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._finished:
            return False
        if exc is not None:
            self.log_exception(exc)
            self.finish("failed", exit_reason=f"{type(exc).__name__}: {exc}")
        else:
            self.finish("completed")
        return False
