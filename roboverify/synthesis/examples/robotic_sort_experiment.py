"""Reproducible real-simulator corpus and advisor-facing inference evidence."""

from __future__ import annotations

import contextlib
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
import io
from itertools import permutations
import json
from pathlib import Path
from time import perf_counter

from synthesis.examples.robotic_sort_traces import RoboticSortSnapshot, run_physical_case
from synthesis.examples.robotic_sort_inference import (
    infer_loop, validate_learned, adapt_snapshot, evaluate_formula,
)


def experiment_cases():
    """Disjoint whole-run split, with ties/negative keys beyond the 24 permutations."""
    cases = []
    for index, sequence in enumerate(permutations((1, 2, 3, 4))):
        cases.append({"name": f"permutation_{index:02d}", "split": "train" if index % 2 == 0 else "held_out",
                      "keys": sequence, "seed": 100 + index})
    for split, sequences in (
        ("train", ((2, 1, 2, 1), (2, 2, 2, 2), (1, 3, 2, 2), (3, 2, 2, 1), (0, -1, 2, -1))),
        ("held_out", ((-2, 4, -2, 1), (2, 1, 1, 2), (3, 3, 1, 2))),
    ):
        for index, sequence in enumerate(sequences):
            cases.append({"name": f"{split}_ties_{index}", "split": split, "keys": sequence, "seed": 200 + len(cases)})
    # A genuine symbolic preservation counterexample from refinement round 0:
    # after buffering key 0 at i=1, the untouched suffix contains equal keys.
    cases.append({"name": "refinement_01_equal_suffix", "split": "train", "keys": (1, 0, 1, 1),
                  "seed": 901, "origin": "outer_to_inner_after_buffering counterexample from round 0"})
    return tuple(cases)


def collect_corpus(env, slots, buffer_position, baseline, path: Path, *, resume=False) -> dict:
    cases = experiment_cases()
    path.parent.mkdir(parents=True, exist_ok=True)
    document = json.loads(path.read_text()) if resume and path.exists() else {}
    records = document.get("cases", [])
    source_hashes = {name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                     for name in ("robotic_insertion_sort.py", "robotic_sort_traces.py")}
    if records:
        if document.get("trace_source_sha256") != source_hashes:
            raise ValueError("Cannot resume traces produced by different controller/trace code; collect a fresh corpus.")
        manifest = {case["name"]: case for case in cases}
        for record in records:
            expected = manifest.get(record["name"])
            if expected is None or any(record[key] != expected[key] for key in ("seed", "split")) or tuple(record["keys"]) != tuple(expected["keys"]):
                raise ValueError(f"Resume case configuration changed: {record['name']}")
            if record["status"] == "passed":
                state = record["run"]["snapshots"][0]
                if state["slots"] != [list(map(float, slot)) for slot in slots] or state["buffer_position"] != list(map(float, buffer_position)):
                    raise ValueError("Resume scene geometry differs from the saved physical traces.")
    started = perf_counter()
    for index, case in enumerate(cases):
        if any(record["name"] == case["name"] for record in records):
            continue
        print(f"Physical case {index + 1}/{len(cases)}: {case['name']} {case['split']} {case['keys']}", flush=True)
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                run = run_physical_case(env, slots, buffer_position, baseline=baseline,
                                       key_sequence=case["keys"], seed=case["seed"], case_name=case["name"])
            record = {**case, "status": "passed", "run": asdict(run)}
        except Exception as exc:
            record = {**case, "status": "failed", "error": repr(exc), "controller_log": output.getvalue()}
        records.append(record)
        # Preserve failures and partial progress even if a subsequent run is interrupted.
        document = {"schema_version": 1, "source": "actual MuJoCo execution of insertion_sort_blocks",
                    "generated_utc": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds": perf_counter() - started, "cases": records,
                    "trace_source_sha256": source_hashes}
        path.write_text(json.dumps(document, indent=2) + "\n")
        print(f"  {record['status']}" + (f" ({run.steps} steps, {len(run.snapshots)} snapshots)" if record['status'] == "passed" else f": {record['error']}"), flush=True)
    return document


def corpus_snapshots(corpus: dict, split: str):
    return tuple(RoboticSortSnapshot.from_dict(snapshot)
                 for record in corpus["cases"] if record["split"] == split and record["status"] == "passed"
                 for snapshot in record["run"]["snapshots"])


def structural_violations(state: RoboticSortSnapshot) -> list[str]:
    """Explicit supplied structural/safety checks, separate from learned clauses."""
    problems = []
    n = len(state.objects)
    names = [name for name in state.order if name is not None]
    locations = state.locations
    if len(set(locations.values())) != n:
        problems.append("physical occupancy is not unique")
    if len(set(names)) != len(names) or set(state.initial_order) != set(state.objects):
        problems.append("block identity conservation")
    if state.held is not None:
        problems.append("gripper not empty")
    if any(rotation[2][2] <= .98 for rotation in state.rotation_map.values()):
        problems.append("block not upright")
    if state.program_point == "inner_head":
        if state.j is None or not (-1 <= state.j < state.i < n):
            problems.append("inner index bounds")
        if set(names) != set(state.objects) - {state.selected}:
            problems.append("inner identity conservation")
        if state.j is None or [index for index, name in enumerate(state.order) if name is None] != [state.j + 1]:
            problems.append("vacancy must be j + 1")
        if state.order[state.i + 1:] != state.initial_order[state.i + 1:]:
            problems.append("untouched suffix changed")
    else:
        if not (1 <= state.i <= n) or set(names) != set(state.objects) or None in state.order:
            problems.append("outer row permutation/index bounds")
        if state.order[state.i:] != state.initial_order[state.i:]:
            problems.append("untouched suffix changed")
    return problems


def fault_detection_checks(learned, states):
    """Fault injection validates the detector; altered snapshots are NOT training data."""
    outer = next(state for state in states if state.program_point == "outer_exit"
                 and len(set(state.key_map.values())) == len(state.objects))
    inner = next(state for state in states if state.program_point == "inner_head")
    a, b = outer.order[:2]
    positions = outer.position_map
    positions[a], positions[b] = positions[b], positions[a]
    displaced = replace(outer, positions=tuple(positions.items()))
    swapped_order = (b, a, *outer.order[2:])
    unsorted = replace(displaced, order=swapped_order)
    wrong_selected = replace(inner, selected=next(name for name in inner.objects if name != inner.selected))
    fault_cases = [("physical/logical mismatch", "outer", displaced),
                   ("physically realized but unsorted row", "outer", unsorted),
                   ("incorrect selected-block identity", "inner", wrong_selected)]
    result = []
    for name, phase, state in fault_cases:
        loop = learned[phase]
        holds = evaluate_formula(loop.invariant, loop.domain,
                                 adapt_snapshot(state, selected_constant=phase == "inner"))
        result.append({"fault": name, "detected": not holds, "layer": "actual learned formula"})
    dropped_positions = outer.position_map
    p = dropped_positions[a]
    dropped_positions[a] = (p[0] + .5, p[1], p[2])
    dropped = replace(outer, positions=tuple(dropped_positions.items()))
    try:
        dropped.locations
        detected = False
    except ValueError:
        detected = True
    result.append({"fault": "block outside every valid location", "detected": detected,
                   "layer": "geometric abstraction validity"})
    wrong_hole = replace(inner, j=-1 if inner.j != -1 else 0)
    result.append({"fault": "wrong vacancy index", "detected": bool(structural_violations(wrong_hole)),
                   "layer": "explicit structural check"})
    return result


def infer_and_validate(corpus: dict, report_path: Path):
    """Infer only on training runs, then evaluate frozen formulas on held-out runs."""
    train = corpus_snapshots(corpus, "train")
    held_out = corpus_snapshots(corpus, "held_out")
    learned = {}
    report = {"schema_version": 1, "training_snapshots": len(train), "held_out_snapshots": len(held_out),
              "physical_cases": len(corpus["cases"]),
              "failed_cases": [record["name"] for record in corpus["cases"] if record["status"] != "passed"],
              "loops": {}}
    report["structural_validation"] = {
        "snapshots_checked": len(train) + len(held_out),
        "violations": [{"case": state.case_name, "step": state.step, "problems": problems}
                       for state in (*train, *held_out) if (problems := structural_violations(state))],
        "status": "supplied checks, not learned invariant clauses",
    }
    for phase in ("outer", "inner"):
        loop = infer_loop(train, phase)
        learned[phase] = loop
        report["loops"][phase] = {**loop.summary(), "training": validate_learned(loop, train),
                                               "held_out": validate_learned(loop, held_out)}
        print(f"{phase}: " + json.dumps(report["loops"][phase]["held_out"]), flush=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    from synthesis.examples.robotic_sort_verification import verify_sort_invariants
    report["abstract_verification"] = verify_sort_invariants(
        learned["outer"].domain, learned["outer"].invariant,
        learned["inner"].domain, learned["inner"].invariant, n=4,
    )
    report["fault_detection"] = fault_detection_checks(learned, held_out)
    report["all_passed"] = (not report["failed_cases"] and
        not report["structural_validation"]["violations"] and
        all(item["detected"] for item in report["fault_detection"]) and
        all(check[split]["all_passed"] for check in report["loops"].values() for split in ("training", "held_out"))
        and report["abstract_verification"]["all_passed"])
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return learned, report


def main():
    import argparse
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("notebooks"))
    parser.add_argument("--reuse-traces", action="store_true", help="Explicitly reuse saved physical traces when iterating on inference")
    parser.add_argument("--resume-corpus", action="store_true", help="Keep existing runs and physically execute newly added training cases")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    traces = args.output_dir / "robotic_insertion_sort_traces.json"
    report_path = args.output_dir / "robotic_insertion_sort_invariants.json"
    if args.reuse_traces:
        corpus = json.loads(traces.read_text())
    else:
        from synthesis.environment.cee_us_env.runtime import configure_local_mujoco
        configure_local_mujoco()
        from synthesis.environment.cee_us_env.fpp_construction_env import FetchPickAndPlaceConstruction
        from synthesis.examples.robotic_sort_traces import capture_simulator_state
        np.random.seed(42)
        env = FetchPickAndPlaceConstruction(name="robotic_sort_inference", sparse=False,
            shaped_reward=False, num_blocks=4, reward_type="sparse", case="PickAndPlace",
            simple=True, visualize_target=False, visualize_mocap=False)
        try:
            env.reset()
            slots = np.column_stack((np.full(4, env.initial_gripper_xpos[0] + .10),
                                     env.initial_gripper_xpos[1] + (np.arange(4) - 1.5) * .11,
                                     np.full(4, env.height_offset)))
            buffer_position = np.array([slots[0, 0] - .18, env.initial_gripper_xpos[1], env.height_offset])
            corpus = collect_corpus(env, slots, buffer_position, capture_simulator_state(env), traces, resume=args.resume_corpus)
        finally:
            env.close()
    _, report = infer_and_validate(corpus, report_path)
    print(json.dumps({"all_passed": report["all_passed"], "abstract_verification": report["abstract_verification"]}, indent=2))
    if not report["all_passed"]:
        raise SystemExit("Invariant experiment has failing checks; see report")


if __name__ == "__main__":
    main()
