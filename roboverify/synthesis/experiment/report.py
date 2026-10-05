"""Bounded reporting over run directories.

This is the intended way to read a run. Output is capped at ``--max-lines`` so that
inspecting a run costs a known, constant amount regardless of how many iterations it
ran for, and the shape is stable so two reports of the same run diff cleanly.

Do not read ``metrics.jsonl`` or ``stdout.log`` directly; that is what this exists to
avoid.

Usage::

    python -m synthesis.experiment.report --run runs/mcmc/latest
    python -m synthesis.experiment.report --run runs/mcmc/latest --since 300
    python -m synthesis.experiment.report --glob 'runs/mcmc/*' --table
"""

import argparse
import glob as globlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Optional

DEFAULT_MAX_LINES = 60
CURVE_POINTS = 16
STDOUT_TAIL_LINES = 15
HEALTHY_STATUSES = {"completed", "verified_model", "verified_symbolic"}
FLOOR_THRESHOLD = -1e5  # costs at or below this are the BMC-failure sentinel


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def iter_jsonl(path: Path) -> Iterator[dict]:
    """Yield records, tolerating a truncated final line from a killed run."""
    try:
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except OSError:
        return


def fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def fmt_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:  # NaN
        return "nan"
    if abs(number) >= 1e5:
        return f"{number:.3g}"
    return f"{number:.{digits}f}"


def fmt_rate(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.2f}"


class RunSummary:
    """Aggregates one run directory into a small fixed set of numbers."""

    def __init__(self, run_dir: Path, since: Optional[int] = None):
        self.dir = run_dir
        self.config = read_json(run_dir / "config.json")
        self.status = read_json(run_dir / "status.json")
        self.result = read_json(run_dir / "result.json")
        self.events = list(iter_jsonl(run_dir / "events.jsonl"))

        self.curve: list = []
        self.reasons: Counter = Counter()
        self.mutations: Counter = Counter()
        self.accepted = 0
        self.evaluated = 0
        self.rows = 0
        self.success_values: list = []
        self.best_success: tuple = (None, None)
        self.cem_deltas: list = []
        self.cem_sigmas: list = []
        self.cem_zero_delta = 0
        self.cegis_progress = []
        self.cfg_progress = []

        for record in iter_jsonl(run_dir / "metrics.jsonl"):
            step = record.get("iter")
            if since is not None and (step is None or step < since):
                continue
            self.rows += 1
            if record.get("phase") == "straightline":
                self.cfg_progress.append(
                    {
                        key: record.get(key)
                        for key in ("iter", "block_id", "distance", "pool_size")
                    }
                )
                self.cfg_progress = self.cfg_progress[-6:]
            if record.get("phase") in (
                "symbolic_initial",
                "symbolic",
                "motion",
                "invariant_learning",
            ):
                self.cegis_progress.append(
                    {
                        key: record.get(key)
                        for key in (
                            "iter",
                            "phase",
                            "n_states",
                            "n_clauses",
                            "n_penalties",
                            "verified",
                            "invariant_sexpr_path",
                        )
                    }
                )
                self.cegis_progress = self.cegis_progress[-8:]
            self.curve.append((step, record.get("cost"), record.get("best_cost")))
            reason = record.get("bmc_reason")
            if reason:
                self.reasons[reason] += 1
            kind = record.get("mutation_kind")
            if kind:
                self.mutations[kind] += 1
            if record.get("evaluated"):
                self.evaluated += 1
                if record.get("accepted"):
                    self.accepted += 1
            success = record.get("success_rate")
            if success is not None:
                self.success_values.append(float(success))
                if self.best_success[0] is None or success > self.best_success[0]:
                    self.best_success = (float(success), step)
            delta = record.get("cem_delta")
            if delta is not None:
                self.cem_deltas.append(float(delta))
                if float(delta) == 0.0:
                    self.cem_zero_delta += 1
            sigma = record.get("cem_sigma_norm")
            if sigma is not None:
                self.cem_sigmas.append(float(sigma))

    # ------------------------------------------------------------------ helpers

    @property
    def run_id(self) -> str:
        return self.config.get("run_id") or self.dir.resolve().name

    @property
    def overall_status(self) -> str:
        if self.result:
            return str(self.result.get("status", "?"))
        if self.status.get("alive"):
            return "running"
        return "unknown"

    @property
    def looks_unhealthy(self) -> bool:
        return bool(self.result) and self.overall_status not in HEALTHY_STATUSES

    def config_value(self, key: str, default: Any = None) -> Any:
        return (self.config.get("config") or {}).get(key, default)

    def mean(self, values: list) -> Optional[float]:
        return sum(values) / len(values) if values else None

    def stdout_tail(self, lines: int = STDOUT_TAIL_LINES) -> list:
        path = self.dir / "stdout.log"
        try:
            with open(path, errors="replace") as handle:
                return [line.rstrip() for line in handle.readlines()[-lines:]]
        except OSError:
            return []

    # ------------------------------------------------------------------- render

    def header_lines(self) -> list:
        git = self.config.get("git") or {}
        dirty = " (dirty)" if git.get("dirty") else ""
        runtime = self.config_value("runtime") or {}
        lines = [
            f"run    {self.run_id}   [{self.overall_status}]",
            f"git    {git.get('sha', '?')} on {git.get('branch', '?')}{dirty}"
            f"   started {self.config.get('started_at', '?')}",
            "cfg    "
            + " ".join(
                [
                    f"task={self.config_value('task')}",
                    f"nb={self.config_value('num_blocks')}",
                    f"iters={self.config_value('iters')}",
                    f"beta={self.config_value('beta')}",
                    f"cem={self.config_value('cem_N')}/{self.config_value('cem_K')}"
                    f"x{self.config_value('cem_iterations')}",
                    f"video={self.config_value('video_policy')}",
                    f"seeds={self.config_value('num_seeds')}",
                ]
            ),
        ]
        goal = runtime.get("goal_feature")
        if goal or runtime.get("bmc_enabled") is not None:
            lines.append(
                f"obj    goal_feature={goal or 'none'}"
                f"   bmc={'on' if runtime.get('bmc_enabled') else 'off'}"
                f"   weight={self.config_value('goal_feature_reward_weight')}"
            )
        return lines

    def status_lines(self) -> list:
        status = self.status
        alive = "ALIVE" if status.get("alive") else "STOPPED"
        total = status.get("total")
        pct = f" ({status.get('pct')}%)" if status.get("pct") is not None else ""
        lines = [
            "",
            f"status {alive}  iter {status.get('iter')}/{total}{pct}"
            f"  elapsed {fmt_duration(status.get('elapsed_s'))}"
            f"  eta {fmt_duration(status.get('eta_s'))}"
            f"  {status.get('iters_per_min', '-')} it/min",
            f"       best {fmt_number(status.get('best_cost'))} @{status.get('best_iter')}"
            f"   current {fmt_number(status.get('current_cost'))}"
            f"   heartbeat {status.get('heartbeat', '?')}",
            f"rates  accept {fmt_rate(status.get('accept_rate_window'))}"
            f"  bmc_feasible {fmt_rate(status.get('bmc_feasible_rate_window'))}"
            f"  at_floor {fmt_rate(status.get('cost_at_floor_rate_window'))}"
            f"  first_feasible {status.get('first_feasible_iter', '-')}",
        ]
        if self.evaluated:
            lines.append(
                f"       overall accept {self.accepted}/{self.evaluated}"
                f" ({self.accepted / self.evaluated:.2f})"
                f"   scored {self.evaluated}/{self.rows} iters"
            )
        return lines

    def diagnostics_lines(self) -> list:
        lines = ["", "diagnostics"]
        if self.result.get("formal_verification"):
            lines.append(f"  verification {self.result['formal_verification']}")
            lines.append(f"  symbolic     {self.result.get('symbolic', 'not run')}")
            lines.append(f"  motion       {self.result.get('motion', 'not run')}")
        if "verification_attempts" in self.result:
            lines.append(f"  symbolic     {self.result.get('symbolic_status')}")
            lines.append(f"  motion       {self.result.get('motion_status')}")
            lines.append(
                f"  invariant experiment: {self.result['verification_attempts']} proof attempts, "
                f"{self.result.get('counterexample_executions', 0)} accepted executions, "
                f"{self.result.get('learner_updates', 0)} learner updates"
            )
            lines.append("  progression: artifacts/summary.md")
        if self.cfg_progress:
            lines.append(
                f"  CFG result: {self.result.get('status','running')}  formal verification: {self.result.get('formal_verification','not established')}"
            )
            for row in self.cfg_progress:
                lines.append(
                    f"  block {row['block_id']} iter {row['iter']}: distance={fmt_number(row['distance'])} pool={row['pool_size']}"
                )
            for event in [e for e in self.events if e.get("kind") == "block_result"][
                -4:
            ]:
                lines.append(
                    f"  {event.get('block_id')}: PostScore={event.get('post_score')} elapsed={fmt_duration(event.get('elapsed'))} ranking={fmt_number(event.get('postscore_seconds'))}s"
                )
        if self.cegis_progress:
            lines.append(
                f"  CEGIS result: {self.result.get('status', 'running')}  scope: {self.result.get('proof_scope', 'not established')}"
            )
            for row in self.cegis_progress:
                lines.append(
                    f"  {row['phase']} {row['iter']}: states={row['n_states']} clauses={row['n_clauses']} penalties={row['n_penalties']} verified={row['verified']}"
                )
            latest = next(
                (
                    r
                    for r in reversed(self.cegis_progress)
                    if r.get("invariant_sexpr_path")
                ),
                None,
            )
            if latest:
                lines.append(f"  invariant: {latest['invariant_sexpr_path']}")
            if self.result.get("failed_vc"):
                lines.append(
                    f"  failed VC: {self.result['failed_vc']} at {self.result.get('num_blocks')} blocks"
                )
            if self.result.get("reason"):
                lines.append(f"  reason: {self.result['reason']}")
        if self.reasons:
            top = "  ".join(
                f"{name} {count}" for name, count in self.reasons.most_common(6)
            )
            lines.append(f"  bmc_reason   {top}")
        if self.mutations:
            top = "  ".join(
                f"{name} {count}" for name, count in self.mutations.most_common(4)
            )
            lines.append(f"  mutations    {top}")
        if self.success_values:
            best_value, best_step = self.best_success
            lines.append(
                f"  success      mean {fmt_number(self.mean(self.success_values), 3)}"
                f"  max {fmt_number(best_value, 3)} @{best_step}"
                f"  n={len(self.success_values)}"
            )
        else:
            lines.append("  success      no success_rate recorded")
        if self.cem_deltas:
            lines.append(
                f"  cem          delta mean {fmt_number(self.mean(self.cem_deltas), 5)}"
                f"  sigma mean {fmt_number(self.mean(self.cem_sigmas), 5)}"
                f"  zero-delta {self.cem_zero_delta}/{len(self.cem_deltas)}"
            )
        return lines

    def curve_lines(self, points: int) -> list:
        rows = [row for row in self.curve if row[2] is not None]
        if not rows or points <= 0:
            return []
        step = max(1, len(rows) // points)
        sampled = rows[::step][-points:]
        # The BMC-failure sentinel (-1e6) is many orders of magnitude below any real
        # cost, so scaling the bars over it would render every other point as full.
        # Scale over real costs only and mark sentinel rows explicitly.
        real = [row[2] for row in sampled if row[2] > FLOOR_THRESHOLD]
        low, high = (min(real), max(real)) if real else (0.0, 1.0)
        span = (high - low) or 1.0
        lines = ["", f"best-so-far cost  ({len(rows)} iters -> {len(sampled)} pts)"]
        for iteration, _cost, best in sampled:
            if best <= FLOOR_THRESHOLD:
                bar = "<bmc floor>"
            else:
                filled = int(round(12 * (best - low) / span))
                bar = "#" * filled + "." * (12 - filled)
            lines.append(f"  {str(iteration):>6}  {fmt_number(best):>13}  {bar}")
        return lines

    def event_lines(self, budget: int) -> list:
        if not self.events:
            return []
        lines = ["", f"events ({len(self.events)})"]

        # A long run can legitimately record hundreds of new bests, which would
        # otherwise push every other event kind out of the report. Summarize each
        # kind first, then show the newest few events in full.
        kinds: Counter = Counter(str(e.get("kind", "?")) for e in self.events)
        by_kind = []
        for kind, count in kinds.most_common():
            steps = [e.get("iter") for e in self.events if e.get("kind") == kind]
            steps = [s for s in steps if s is not None]
            span = f" latest@{steps[-1]}" if steps else ""
            by_kind.append(f"{kind}x{count}{span}")
        lines.append("  " + "  ".join(by_kind)[:200])

        shown = self.events
        detail_budget = max(budget - 2, 1)
        if len(shown) > detail_budget:
            shown = shown[-detail_budget:]
            lines.append(f"  ... showing the last {len(shown)}")
        for event in shown:
            timestamp = str(event.get("ts", ""))[11:19]
            step = event.get("iter")
            where = f"@{step}" if step is not None else ""
            lines.append(
                f"  {timestamp} {event.get('kind', '?')}{where}: {event.get('message', '')}"[
                    :150
                ]
            )
        return lines

    def failure_lines(self) -> list:
        if not self.looks_unhealthy:
            return []
        lines = ["", f"FAILED ({self.overall_status})"]
        reason = self.result.get("exit_reason")
        if reason:
            lines.append(f"  exit_reason: {reason}")
        tail = self.stdout_tail()
        if tail:
            lines.append(f"  stdout.log tail ({len(tail)} lines):")
            lines.extend(f"    {line}"[:160] for line in tail)
        return lines

    def render(
        self, max_lines: int = DEFAULT_MAX_LINES, force_tail: bool = False
    ) -> str:
        fixed = self.header_lines() + self.status_lines() + self.diagnostics_lines()
        failure = self.failure_lines() if (self.looks_unhealthy or force_tail) else []
        # Curve and events share whatever budget the fixed sections leave, so the
        # total stays under the cap without dropping the parts that always matter.
        remaining = max(max_lines - len(fixed) - len(failure), 6)
        curve_budget = min(CURVE_POINTS, max(remaining // 2 - 2, 0))
        curve = self.curve_lines(curve_budget)
        events = self.event_lines(max(remaining - len(curve) - 2, 1))
        lines = fixed + curve + events + failure
        if len(lines) > max_lines:
            lines = lines[: max_lines - 1] + [
                f"  ... output truncated at {max_lines} lines"
            ]
        return "\n".join(lines)

    def table_row(self) -> str:
        best = self.status.get("best_cost")
        success = self.best_success[0]
        accept = self.accepted / self.evaluated if self.evaluated else None
        return "  ".join(
            [
                f"{self.run_id[:44]:<44}",
                f"{self.overall_status[:9]:<9}",
                f"{str(self.status.get('iter', '-')):>6}",
                f"{fmt_number(best):>13}",
                f"{fmt_number(success, 2):>5}",
                f"{fmt_rate(accept):>6}",
                f"{fmt_duration(self.status.get('elapsed_s')):>7}",
                f"{str(self.config_value('task'))[:8]:<8}",
                f"{str(self.config_value('beta')):>5}",
            ]
        )


TABLE_HEADER = "  ".join(
    [
        f"{'run_id':<44}",
        f"{'status':<9}",
        f"{'iter':>6}",
        f"{'best_cost':>13}",
        f"{'succ':>5}",
        f"{'accept':>6}",
        f"{'elapsed':>7}",
        f"{'task':<8}",
        f"{'beta':>5}",
    ]
)


def resolve_run_dirs(pattern: str) -> list:
    """Expand a glob to run directories, skipping ``latest`` symlink duplicates."""
    matches = sorted(Path(p) for p in globlib.glob(pattern))
    seen: set = set()
    run_dirs = []
    for path in matches:
        if not (path / "config.json").exists():
            continue
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        run_dirs.append(path)
    return run_dirs


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print a bounded report for one run, or a table across many."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", help="A single run directory, e.g. runs/mcmc/latest")
    source.add_argument("--glob", help="Glob over run directories, e.g. 'runs/mcmc/*'")
    parser.add_argument(
        "--table",
        action="store_true",
        help="One row per run instead of a full report (implied by --glob).",
    )
    parser.add_argument("--since", type=int, default=None, help="Ignore earlier iters.")
    parser.add_argument("--max-lines", type=int, default=DEFAULT_MAX_LINES)
    parser.add_argument(
        "--tail",
        action="store_true",
        help="Always show the stdout.log tail, not only for failed runs.",
    )
    args = parser.parse_args(argv)

    if args.run:
        run_dir = Path(args.run)
        if not (run_dir / "config.json").exists():
            parser.error(f"{run_dir} does not look like a run directory")
        summary = RunSummary(run_dir, since=args.since)
        if args.table:
            print(TABLE_HEADER)
            print(summary.table_row())
        else:
            print(summary.render(max_lines=args.max_lines, force_tail=args.tail))
        return 0

    run_dirs = resolve_run_dirs(args.glob)
    if not run_dirs:
        print(f"no run directories matched {args.glob!r}")
        return 1
    print(TABLE_HEADER)
    for run_dir in run_dirs:
        print(RunSummary(run_dir, since=args.since).table_row())
    print(f"\n{len(run_dirs)} run(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
