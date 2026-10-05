"""Compare saved Stack loop states with ideal placements, without rerunning MuJoCo.

The oracle replays the recorded b/b_prime placement sequence on 50 mm levels.
It does not derive the expected ordering by rounding measured block heights.
Only the supplied single-tower Stack program's flat-loop convention is supported.
"""

import argparse
import itertools
import json
from copy import deepcopy
from pathlib import Path

from synthesis.cfg.recordings import load_traces
from synthesis.entry.predicate_options import add_predicate_options
from synthesis.experiment.run_logger import RunLogger
from synthesis.predicates.scene import Scene, evaluate
from synthesis.predicates.term import atom, ref
from synthesis.util import on

RELATIONS = ("Higher", "ON_star", "ON_star_zero", "Scattered", "eq")


def ideal_stack_states(trace):
    """Yield measured/oracle scenes and integer levels at heads and normal exits."""
    if trace.task != "stack" or trace.num_blocks < 2:
        raise ValueError(
            "The comparison requires a Stack trace with at least two blocks"
        )
    events = [e for e in trace.events if e["kind"] in ("loop_head", "loop_exit")]
    if not events or len({(e["path"], e.get("invocation")) for e in events}) != 1:
        raise ValueError("Expected one flat Stack loop invocation")
    if events[-1]["kind"] != "loop_exit" or any(
        e["kind"] == "loop_exit" for e in events[:-1]
    ):
        raise ValueError("Expected one normal exit after the continuing heads")
    ids = range(trace.num_blocks)
    initial = {i: list(map(float, on.get_block_pos(trace.states[0], i))) for i in ids}
    if not all(
        on.scattered_implementation(initial[a], initial[b])
        for a, b in itertools.combinations(ids, 2)
    ):
        raise ValueError("Expected initially scattered singleton blocks")
    root = events[0]["bindings"]["b0"]
    if root not in initial:
        raise ValueError("Invalid b0 binding")
    # Common table at 0.4 m; blocks have half-height L/2.
    ideal_entry = {
        i: [p[0], p[1], 0.4 + on.BLOCK_LENGTH / 2] for i, p in initial.items()
    }
    ideal = deepcopy(ideal_entry)
    levels = dict.fromkeys(ids, 0)
    tower = [root]
    for event in events:
        bindings = event["bindings"]
        if (
            bindings["b0"] != root
            or bindings["b"] != tower[-1]
            or event["iteration"] != len(tower) - 1
        ):
            raise ValueError(
                "Trace does not follow the supplied Stack placement convention"
            )
        observed = {
            i: list(map(float, on.get_block_pos(trace.states[event["index"]], i)))
            for i in ids
        }
        yield event, Scene(observed, {}, initial), Scene(
            deepcopy(ideal), {}, ideal_entry
        ), dict(levels)
        if event["kind"] == "loop_head":
            source = bindings["b_prime"]
            if source not in initial or source in tower:
                raise ValueError(
                    "Expected a fresh singleton placement at each iteration"
                )
            levels[source] = len(tower)
            ideal[source] = [
                ideal[root][0],
                ideal[root][1],
                ideal[root][2] + len(tower) * on.BLOCK_LENGTH,
            ]
            tower.append(source)
    if len(tower) != trace.num_blocks:
        raise ValueError("The trace exits before every block joins the tower")


def compare_traces(traces, *, tolerance=on.DEFAULT_HIGHER_TOLERANCE):
    stats = dict(
        trajectories=0,
        continuing_heads=0,
        normal_exits=0,
        strict_higher_mismatches=0,
        strict_higher_mismatch_states=0,
        tolerant_higher_mismatch_states=0,
        higher_order_violation_states=0,
        max_same_level_height_spread_m=0.0,
        min_distinct_level_height_gap_m=None,
        max_height_error_from_ideal_m=0.0,
        relations={
            name: dict(
                evaluations=0, mismatches=0, mismatch_states=0, mismatch_seeds=[]
            )
            for name in RELATIONS
        },
        mismatch_examples=[],
    )
    terms = {name: atom(name, ref("lhs"), ref("rhs")) for name in RELATIONS}
    with on.using_higher_tolerance(tolerance):
        for trace in traces:
            stats["trajectories"] += 1
            for event, observed, ideal, levels in ideal_stack_states(trace):
                kind = (
                    "continuing_heads"
                    if event["kind"] == "loop_head"
                    else "normal_exits"
                )
                stats[kind] += 1
                ids = list(observed.positions)
                higher = {}
                strict_bad = tolerant_bad = False
                bad_relations = set()
                for a, b in itertools.product(ids, repeat=2):
                    observed.bindings = ideal.bindings = {"lhs": a, "rhs": b}
                    expected_higher = levels[a] >= levels[b]
                    strict = on.higher_implementation(
                        observed.positions[a], observed.positions[b], tolerance=0
                    )
                    strict_bad |= strict != expected_higher
                    stats["strict_higher_mismatches"] += int(strict != expected_higher)
                    dz = abs(observed.positions[a][2] - observed.positions[b][2])
                    if levels[a] == levels[b]:
                        stats["max_same_level_height_spread_m"] = max(
                            stats["max_same_level_height_spread_m"], dz
                        )
                    else:
                        current = stats["min_distinct_level_height_gap_m"]
                        stats["min_distinct_level_height_gap_m"] = (
                            dz if current is None else min(current, dz)
                        )
                    for relation, term in terms.items():
                        actual = evaluate(term, observed)
                        expected = (
                            expected_higher
                            if relation == "Higher"
                            else evaluate(term, ideal)
                        )
                        counts = stats["relations"][relation]
                        counts["evaluations"] += 1
                        counts["mismatches"] += int(actual != expected)
                        if actual != expected:
                            bad_relations.add(relation)
                            if trace.seed not in counts["mismatch_seeds"]:
                                counts["mismatch_seeds"].append(trace.seed)
                        if relation == "Higher":
                            higher[a, b] = actual
                            tolerant_bad |= actual != expected
                        if actual != expected and len(stats["mismatch_examples"]) < 20:
                            stats["mismatch_examples"].append(
                                dict(
                                    seed=trace.seed,
                                    kind=event["kind"],
                                    index=event["index"],
                                    relation=relation,
                                    lhs=a,
                                    rhs=b,
                                    actual=actual,
                                    expected=expected,
                                    observed_positions=[
                                        observed.positions[a],
                                        observed.positions[b],
                                    ],
                                    ideal_positions=[
                                        ideal.positions[a],
                                        ideal.positions[b],
                                    ],
                                )
                            )
                for relation in bad_relations:
                    stats["relations"][relation]["mismatch_states"] += 1
                stats["strict_higher_mismatch_states"] += int(strict_bad)
                stats["tolerant_higher_mismatch_states"] += int(tolerant_bad)
                ordered = (
                    all(higher[a, a] for a in ids)
                    and all(
                        higher[a, b] or higher[b, a]
                        for a, b in itertools.product(ids, repeat=2)
                    )
                    and all(
                        not (higher[a, b] and higher[b, c]) or higher[a, c]
                        for a, b, c in itertools.product(ids, repeat=3)
                    )
                )
                stats["higher_order_violation_states"] += int(not ordered)
                stats["max_height_error_from_ideal_m"] = max(
                    stats["max_height_error_from_ideal_m"],
                    *(
                        abs(observed.positions[i][2] - ideal.positions[i][2])
                        for i in ids
                    ),
                )
    stats["higher_matches_ideal"] = stats["relations"]["Higher"]["mismatches"] == 0
    stats["matches_ideal"] = (
        not any(row["mismatches"] for row in stats["relations"].values())
        and not stats["higher_order_violation_states"]
    )
    return stats


def markdown_report(report):
    comparisons = [source["comparison"] for source in report["sources"]]
    checked = sum(s["relations"]["Higher"]["evaluations"] for s in comparisons)
    before = sum(s["strict_higher_mismatches"] for s in comparisons)
    after = sum(s["relations"]["Higher"]["mismatches"] for s in comparisons)
    same = max(s["max_same_level_height_spread_m"] for s in comparisons)
    distinct = min(s["min_distinct_level_height_gap_m"] for s in comparisons)
    lines = [
        "# Stack predicate comparison",
        "",
        f"Higher tolerance: {report['higher_tolerance_m']} m.",
        "",
        f"Higher: {checked} ordered-pair evaluations; exact mismatches {before}, tolerant mismatches {after}.",
        f"Largest same-level spread: {same * 1000:.6f} mm. Smallest different-level gap: {distinct * 1000:.6f} mm.",
        "",
        "Expected heights come from replaying recorded placements at exact 50 mm levels above a 0.4 m table, not from rounding measured heights. Initial XY coordinates are retained; ideal placements align with b0. All ordered physical-block pairs, including self-pairs, are checked at continuing heads and normal exits.",
        "",
        "| Archive | Trajectories | Heads / exits | Strict Higher mismatches | Tolerant Higher mismatches | Other predicate mismatches |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for source in report["sources"]:
        s = source["comparison"]
        others = sum(
            v["mismatches"] for k, v in s["relations"].items() if k != "Higher"
        )
        lines.append(
            f"| [{Path(source['archive']).parent.name}/{Path(source['archive']).name}]({source['archive']}) | {s['trajectories']} | {s['continuing_heads']} / {s['normal_exits']} | {s['strict_higher_mismatches']} | {s['relations']['Higher']['mismatches']} | {others} |"
        )
    lines += [
        "",
        "## Predicate totals",
        "",
        "| Predicate | Evaluations | Mismatches |",
        "| --- | ---: | ---: |",
    ]
    for relation in RELATIONS:
        total = sum(s["relations"][relation]["evaluations"] for s in comparisons)
        wrong = sum(s["relations"][relation]["mismatches"] for s in comparisons)
        lines.append(f"| {relation} | {total} | {wrong} |")
    for source in report["sources"]:
        scattered = source["comparison"]["relations"]["Scattered"]
        if scattered["mismatches"]:
            lines += [
                "",
                f"Scattered differs in {scattered['mismatch_states']} states across {len(scattered['mismatch_seeds'])} seeds in {Path(source['archive']).parent.name}. These are separate horizontal-separation comparisons; the height tolerance does not affect Scattered.",
            ]
    lines += [
        "",
        "Detailed counts, height spreads, ordering checks and bounded mismatch examples are in comparison.json. This checks recorded states only; it does not establish invariant inductiveness or universal simulator/model equivalence. Tolerant comparison is not transitive on arbitrary continuous heights.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demos",
        nargs="+",
        required=True,
        help="Archives of the supplied single-tower Stack program; candidate trace archives are also accepted",
    )
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--run-name", default="stack-higher")
    add_predicate_options(parser)
    args = parser.parse_args(argv)
    report = dict(
        higher_tolerance_m=args.higher_tolerance,
        block_length_m=on.BLOCK_LENGTH,
        sources=[],
    )
    with RunLogger(
        args.output_dir, "height-comparison", vars(args), slug=args.run_name
    ) as logger:
        try:
            for archive in args.demos:
                comparison = compare_traces(
                    load_traces(archive), tolerance=args.higher_tolerance
                )
                report["sources"].append(
                    dict(archive=str(Path(archive).resolve()), comparison=comparison)
                )
                logger.progress_line(
                    f"{archive}: {comparison['trajectories']} trajectories; Higher mismatches {comparison['strict_higher_mismatches']} -> {comparison['relations']['Higher']['mismatches']}"
                )
            report["higher_matches_ideal"] = all(
                s["comparison"]["higher_matches_ideal"] for s in report["sources"]
            )
            report["matches_ideal"] = all(
                s["comparison"]["matches_ideal"] for s in report["sources"]
            )
            logger.write_artifact(
                "comparison.json", json.dumps(report, indent=2) + "\n"
            )
            logger.write_artifact("comparison.md", markdown_report(report))
            logger.finish(
                "matched_ideal" if report["matches_ideal"] else "predicate_mismatch",
                comparison=report,
            )
        except Exception as exc:
            logger.log_exception(exc)
            logger.finish("failed", reason=str(exc))
            raise
    print(f"Comparison: {logger.run_dir}/artifacts/comparison.md")
    return 0 if report["matches_ideal"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
