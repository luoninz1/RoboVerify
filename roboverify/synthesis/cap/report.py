"""Bounded task comparison from compact CaP artifacts, never raw logs."""

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

METHODS = ("cap_only", "cap_cem", "cap_roboverify")


def failure_kind(execution):
    status, error = execution["status"], execution.get("error") or ""
    if status != "execution_error":
        return status
    if "Unconverged" in error or any(
        term in error
        for term in ("Grasp failed", "Dropped", "Placement failed", "Gripper failed")
    ):
        return "controller_failure"
    if any(
        term in error
        for term in (
            "Unsupported",
            "Unavailable",
            "Undefined",
            "Cannot replace",
            "Cannot shadow",
        )
    ):
        return "language_or_api_violation"
    return "primitive_or_predicate_contract"


def read_run(path):
    path = Path(path)
    config = json.loads((path / "config.json").read_text())["config"]
    summary_path = path / "artifacts/summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    failures = Counter()
    index_path = path / "artifacts/episode_index.json"
    if index_path.exists():
        for row in json.loads(index_path.read_text()):
            if row["phase"] == "held_out" and not row["success"]:
                execution = json.loads(
                    (path / "artifacts" / row["artifact"]).read_text()
                )
                failures[(row["method"], failure_kind(execution))] += 1
    budget_path = path / "artifacts/budget.json"
    budget = json.loads(budget_path.read_text()) if budget_path.exists() else {}
    return config, summary, failures, budget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run")
    group.add_argument("--glob")
    parser.add_argument("--max-lines", type=int, default=60)
    args = parser.parse_args()
    lines = []
    paths = (
        [args.run]
        if args.run
        else sorted(
            path for path in glob.glob(args.glob) if not Path(path).is_symlink()
        )
    )
    for path in paths:
        config, summary, failures, budget = read_run(path)
        lines += [
            f"run {Path(path).name}",
            f"model {config['model']['model']} @ {config['model'].get('revision', '?')}",
            "task               CaP        CaP+CEM    CaP+RoboVerify   training model check",
        ]
        for task, row in summary.items():
            counts = [
                f"{row[method]['successes']}/{row[method]['cases']}"
                for method in METHODS
            ]
            lines.append(
                f"{task:18} {counts[0]:10} {counts[1]:10} {counts[2]:15} {row['verification_status']}"
            )
        if summary:
            totals = [
                f"{sum(row[m]['successes'] for row in summary.values())}/{sum(row[m]['cases'] for row in summary.values())}"
                for m in METHODS
            ]
            lines.append("TOTAL              " + "   ".join(totals))
        lines.append(f"budget {budget}")
        for method in METHODS:
            kinds = {
                kind: count
                for (name, kind), count in failures.items()
                if name == method
            }
            lines.append(f"{method} held-out failures: {kinds}")
    limit = max(1, args.max_lines)
    for line in lines[:limit]:
        print(line)
    if len(lines) > limit:
        print(f"... {len(lines) - limit} more lines omitted")


if __name__ == "__main__":
    main()
