"""Reproducible physical corpus and invariant comparison for the new sorter."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from itertools import permutations
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import z3

from synthesis.examples.relational_sort_inference import (
    infer_loop, compare_with_theory, validate_snapshots, target_formulas,
)


PREFIX = "relational_robotic_insertion_sort"


def experiment_cases():
    cases = [{"name": f"permutation_{i:02d}", "split": "train" if i % 2 == 0 else "held_out",
              "keys": keys, "seed": 300 + i} for i, keys in enumerate(permutations((1, 2, 3, 4)))]
    for split, sequences in (
        ("train", ((2, 1, 2, 1), (2, 2, 2, 2), (1, 3, 2, 2), (3, 2, 2, 1),
                   (0, -1, 2, -1), (1, 0, 1, 1))),
        ("held_out", ((-2, 4, -2, 1), (2, 1, 1, 2), (3, 3, 1, 2))),
    ):
        for i, keys in enumerate(sequences):
            cases.append({"name": f"{split}_ties_{i}", "split": split,
                          "keys": keys, "seed": 500 + len(cases)})
    return cases


def source_hashes():
    return {name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("relational_robotic_sort.py", "relational_sort_controller.py")}


def validate_corpus_metadata(corpus, *, require_complete=True):
    """Reject obsolete/partial manifests on every reuse path, not just resume."""
    if corpus.get("source_sha256") != source_hashes():
        raise ValueError("Physical corpus program/controller provenance mismatch")
    manifest = {case["name"]: case for case in experiment_cases()}
    seen = set()
    for case in corpus.get("cases", []):
        name = case["name"]
        if name not in manifest or name in seen:
            raise ValueError(f"Unexpected or duplicate physical case: {name}")
        seen.add(name)
        expected = manifest[name]
        if (tuple(case["keys"]) != tuple(expected["keys"]) or case["seed"] != expected["seed"]
                or case["split"] != expected["split"]):
            raise ValueError(f"Saved corpus manifest mismatch: {name}")
    if require_complete and seen != set(manifest):
        raise ValueError("Saved physical corpus is incomplete")


def collect_corpus(env, slots, buffer, baseline, path: Path, *, resume=False):
    from synthesis.examples.relational_robotic_sort import run_physical_case
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cases = experiment_cases()
    hashes = source_hashes()
    document = json.loads(path.read_text()) if resume and path.exists() else {}
    records = document.get("cases", [])
    if records:
        validate_corpus_metadata(document, require_complete=False)
    start = perf_counter()
    for i, case in enumerate(cases):
        if any(old["name"] == case["name"] for old in records):
            continue
        print(f"Physical run {i + 1}/{len(cases)}: {case['name']} {case['keys']}", flush=True)
        initial = tuple(str(name) for name in np.random.default_rng(case["seed"]).permutation(env.object_names))
        keys = dict(zip(initial, case["keys"]))
        try:
            run = run_physical_case(env, slots, buffer, baseline=baseline,
                initial_order=initial, keys=keys, case_name=case["name"])
            serial = {key: ([asdict(s) for s in value] if key == "snapshots" else value)
                      for key, value in run.items()}
            record = {**case, "status": "passed", "run": serial}
            print(f"  passed: {run['steps']} physics steps; {len(run['snapshots'])} snapshots", flush=True)
        except Exception as exc:
            record = {**case, "status": "failed", "error": repr(exc)}
            print(f"  failed: {exc}", flush=True)
        records.append(record)
        document = {"schema_version": 1, "source": "actual Fetch/MuJoCo pick/move/release",
                    "generated_utc": datetime.now(timezone.utc).isoformat(),
                    "source_sha256": hashes, "elapsed_seconds": perf_counter() - start,
                    "cases": records}
        path.write_text(json.dumps(document, indent=2) + "\n")
    return document


def corpus_snapshots(corpus, split):
    from synthesis.examples.relational_robotic_sort import RelationalSortSnapshot
    return tuple(RelationalSortSnapshot.from_dict(s)
        for case in corpus["cases"] if case["split"] == split and case["status"] == "passed"
        for s in case["run"]["snapshots"])


def infer_and_validate(corpus, report_path: Path):
    from synthesis.examples.relational_sort_verification import verify_sort_invariants
    validate_corpus_metadata(corpus)
    train, held_out = (corpus_snapshots(corpus, split) for split in ("train", "held_out"))
    report = {"schema_version": 1, "source_sha256": corpus["source_sha256"],
              "canonical_corpus_sha256": sha256(json.dumps(corpus, sort_keys=True,
                  separators=(",", ":")).encode()).hexdigest(),
              "analysis_source_sha256": {str(path.relative_to(Path(__file__).parents[2])):
                  sha256(path.read_bytes()).hexdigest() for path in (
                      Path(__file__), Path(__file__).with_name("relational_sort_inference.py"),
                      Path(__file__).with_name("relational_sort_verification.py"),
                      Path(__file__).parents[1] / "inference_lib" / "relational.py",
                      Path(__file__).parents[1] / "inference_lib" / "inference.py")},
              "physical_runs": len(corpus["cases"]),
              "training_runs": sum(case["split"] == "train" for case in corpus["cases"]),
              "held_out_runs": sum(case["split"] == "held_out" for case in corpus["cases"]),
              "failed_cases": [case for case in corpus["cases"] if case["status"] != "passed"],
              "training_snapshots": len(train), "held_out_snapshots": len(held_out), "loops": {}}
    learned = {}
    for phase in ("outer", "inner"):
        learned[phase] = loop = infer_loop(train, phase)
        report["loops"][phase] = {**loop.summary(),
            "training": validate_snapshots(loop, train),
            "held_out": validate_snapshots(loop, held_out),
            "comparison": compare_with_theory(loop)}
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        solver = z3.Solver()
        loop.domain.add_axioms(solver)
        solver.add(loop.invariant)
        report_path.with_name(f"{PREFIX}_{phase}.smt2").write_text(solver.to_smt2())
        print(f"{phase} comparison: entails={report['loops'][phase]['comparison']['entails_theory']}; "
              f"equivalent={report['loops'][phase]['comparison']['equivalent']}", flush=True)
    report["abstract_verification"] = verify_sort_invariants(
        learned["outer"].domain, learned["outer"].invariant,
        learned["inner"].domain, learned["inner"].invariant, n=4)
    report["all_passed"] = (not report["failed_cases"] and
        all(item["training"]["all_passed"] and item["held_out"]["all_passed"]
            and item["comparison"]["consistent"] and item["comparison"]["entails_theory"]
            for item in report["loops"].values())
        and report["abstract_verification"]["all_passed"])
    report["scope"] = (
        "Training and held-out evidence comes from actual simulation. Logical implication/equivalence "
        "uses the explicit domain axioms. Inductiveness uses a fixed four-node model with arbitrary "
        "integer keys and successful, frame-preserving transfers. This is not automatic Python or "
        "continuous-controller verification, nor a physical deployment guarantee.")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return learned, report


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("notebooks"))
    parser.add_argument("--reuse-traces", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args.output_dir / f"{PREFIX}_traces.json"
    if args.reuse_traces:
        corpus = json.loads(trace_path.read_text())
        validate_corpus_metadata(corpus)
    else:
        from synthesis.examples.relational_robotic_sort import create_environment, capture_simulator_state
        env, slots, buffer = create_environment()
        try:
            corpus = collect_corpus(env, slots, buffer, capture_simulator_state(env), trace_path, resume=args.resume)
        finally:
            env.close()
    _, report = infer_and_validate(corpus, args.output_dir / f"{PREFIX}_invariants.json")
    print(json.dumps({"all_passed": report["all_passed"], "physical_runs": report["physical_runs"]}), flush=True)
    if not report["all_passed"]:
        raise SystemExit("At least one empirical, implication, or inductiveness check failed; inspect report.")


if __name__ == "__main__":
    main()
