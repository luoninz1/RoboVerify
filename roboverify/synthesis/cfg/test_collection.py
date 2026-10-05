"""Collection contracts: no failed-seed substitution or observation-only archives."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import z3

from synthesis.api.instructions import LoopBudgetExceeded, PickPlaceByName, Skip, While
from synthesis.api.program import Program
from synthesis.cfg.collection import VideoRecorder, validate_trace
from synthesis.cfg.demos import DemoTrace
from synthesis.cfg.program_source import (
    load_program,
    prepare_program,
    program_fingerprint,
)
from synthesis.cfg.recordings import load_traces, save_traces
from synthesis.cfg.reset import Snapshot
from synthesis.cfg.tasks import task_spec
from synthesis.entry.collect_demos import (
    build_parser,
    main,
    output_directory,
    resolve_seeds,
)
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext


def example_trace(seed=0):
    start = np.zeros(43)
    start[10:13] = [0, 0, 0.425]
    start[22:25] = [0.2, 0, 0.425]
    end = start.copy()
    end[22:25] = [0, 0, 0.475]
    snapshots = tuple(
        Snapshot(s.copy(), {"ctrl": np.zeros(2)}, {"b0": 0}) for s in (start, end)
    )
    return DemoTrace(
        (start, end),
        snapshots,
        ((), (0, 0)),
        seed,
        "stack",
        2,
        metadata={"status": "completed", "reason": "", "initial_bindings": {"b0": 0}},
    )


class CollectionTests(unittest.TestCase):
    def test_seeds_count_and_conflicts(self):
        def parse(*options):
            return resolve_seeds(
                build_parser().parse_args(["--program", "example:build", *options])
            )

        self.assertEqual(parse(), [0, 1, 2, 3, 4])
        self.assertEqual(
            parse("--num-trajectories", "3", "--seed-start", "10"), [10, 11, 12]
        )
        self.assertEqual(parse("--seeds", "7", "9"), [7, 9])
        for options in (
            ("--seeds", "1", "1"),
            ("--num-trajectories", "0"),
            ("--seeds", "1", "--seed-start", "0"),
            ("--seeds", "1", "--num-trajectories", "2"),
        ):
            with self.assertRaises(ValueError):
                parse(*options)

    def test_final_transition_not_transient_success(self):
        good = example_trace()
        self.assertTrue(validate_trace(good))
        transient = example_trace()
        transient.states = (*transient.states, transient.states[0])
        self.assertFalse(validate_trace(transient))
        self.assertEqual(transient.metadata["validation"]["post_reached"], 1)
        invalid_start = example_trace()
        invalid_start.states = (invalid_start.states[1], invalid_start.states[1])
        self.assertFalse(validate_trace(invalid_start))
        for status in ("incomplete", "failed"):
            trace = example_trace()
            trace.metadata["status"] = status
            self.assertFalse(validate_trace(trace))

    def test_unequal_initial_heights_are_rejected_even_when_stacking_succeeds(self):
        for offset in (-0.002, 0.002):
            with self.subTest(offset=offset):
                trace = example_trace()
                trace.states[0][24] += offset
                self.assertFalse(validate_trace(trace))
                validation = trace.metadata["validation"]
                self.assertEqual(validation["pre_holds"], 0)
                self.assertEqual(validation["post_at_end"], 1)
                self.assertEqual(
                    validation["issues"][0]["reason"], "Initial task condition is false"
                )

    def test_initial_contact_drift_uses_configured_tolerance(self):
        from synthesis.util.on import using_higher_tolerance

        for tolerance, expected in ((0, False), (0.0001, False), (0.001, True)):
            with self.subTest(tolerance=tolerance), using_higher_tolerance(tolerance):
                trace = example_trace()
                trace.states[0][24] += 0.00025
                self.assertEqual(validate_trace(trace), expected)

    def test_archive_roundtrip_and_rejection(self):
        trace = example_trace()
        validate_trace(trace)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demonstrations.npz"
            save_traces(path, [trace])
            loaded = load_traces(path, require_valid=True)[0]
            np.testing.assert_array_equal(loaded.states, trace.states)
            np.testing.assert_array_equal(
                loaded.snapshots[0].gt_state, trace.snapshots[0].gt_state
            )
            trace.metadata["status"] = "incomplete"
            save_traces(path, [trace])
            with self.assertRaisesRegex(ValueError, "unvalidated"):
                load_traces(path, require_valid=True)
            np.savez(path, metadata=json.dumps({"version": 1, "traces": []}))
            with self.assertRaisesRegex(ValueError, "recollect"):
                load_traces(path)
            with self.assertRaisesRegex(ValueError, "recollect"):
                save_traces(path, [DemoTrace(trace.states)])

    def test_failed_seed_prevents_archive_but_runs_remaining_seeds(self):
        definition = SimpleNamespace(metadata={"fingerprint": "same"})

        def execution(definition, **kwargs):
            trace = example_trace(kwargs["seed"])
            if trace.seed == 1:
                trace.metadata.update(status="incomplete", reason="budget exhausted")
            return trace

        with tempfile.TemporaryDirectory() as directory, patch(
            "synthesis.entry.collect_demos.load_program", return_value=definition
        ), patch(
            "synthesis.entry.collect_demos.record_execution", side_effect=execution
        ) as record:
            output = Path(directory) / "collection"
            code = main(
                [
                    "--program",
                    "example:build",
                    "--num-blocks",
                    "2",
                    "--num-trajectories",
                    "3",
                    "--output-dir",
                    str(output),
                ]
            )
            self.assertEqual(code, 2)
            self.assertEqual(
                [c.kwargs["seed"] for c in record.call_args_list], [0, 1, 2]
            )
            self.assertFalse((output / "demonstrations.npz").exists())
            self.assertTrue((output / "diagnostics/seed_0001.npz").exists())
            self.assertTrue((output / "diagnostics/seed_0000.npz").exists())
            self.assertTrue((output / "diagnostics/seed_0002.npz").exists())
            self.assertEqual(
                len(
                    json.loads((output / "collection.json").read_text())["trajectories"]
                ),
                3,
            )

    def test_explicit_output_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output_dir=directory)
            with self.assertRaises(FileExistsError):
                output_directory(args, 5)

    def test_program_loader_identity_and_outdated_macro_rejection(self):
        context = HighLevelContext()
        definition = load_program("synthesis.examples.stack:build_program", context, 3)
        changed = copy.deepcopy(definition.program)
        # Explicit constructor change avoids depending on Parameter internals.
        from synthesis.api.instructions import MoveByName

        changed.instructions[1].body[1] = MoveByName(
            "b", "b", "b", target_offset=[0.1, 0, 0.2]
        )
        self.assertNotEqual(
            program_fingerprint(definition.program), program_fingerprint(changed)
        )
        macro = PickPlaceByName(
            grab_box_name="b0",
            target_box_name_x="b0",
            target_box_name_y="b0",
            target_box_name_z="b0",
            target_offset=[0, 0, 0.1],
        )
        with self.assertRaisesRegex(ValueError, "explicit"):
            prepare_program(Program(1, [macro]), 3)

    def test_loop_events_include_zero_exit_but_not_budget_exit(self):
        obs = example_trace().states[0]
        env = SimpleNamespace(num_blocks=2, symbolic_name_to_box_id={"b0": 0})
        events = []
        loop = While(z3.BoolVal(False), [], [Skip(0)], None)
        loop.eval(env, [obs], on_event=events.append)
        self.assertEqual([e["kind"] for e in events], ["loop_enter", "loop_exit"])
        events = []
        loop = While(z3.BoolVal(True), [], [Skip(0)], None)
        with self.assertRaises(LoopBudgetExceeded):
            loop.eval(env, [obs], on_event=events.append, max_loop_iterations=1)
        self.assertNotIn("loop_exit", [e["kind"] for e in events])

    def test_program_file_factory_and_shadowed_guard_roundtrip(self):
        from synthesis.predicates.term import free_names, from_z3, to_z3

        context = HighLevelContext()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "my_program.py"
            path.write_text("from synthesis.examples.stack import build_program\n")
            supplied = load_program(f"{path}:build_program", context, 3)
            module = load_program("synthesis.examples.stack:build_program", context, 3)
            self.assertEqual(
                supplied.metadata["fingerprint"], module.metadata["fingerprint"]
            )
        x, y, free = [context.get_consts(name) for name in ("x", "y", "guard_var")]
        formula = z3.ForAll(
            [x, y],
            z3.And(
                context.ON_star(x, free),
                z3.Exists([x], z3.And(context.Higher(x, y), x != free)),
            ),
        )
        term = from_z3(formula)
        self.assertEqual(free_names(term), frozenset(["guard_var"]))
        solver = z3.Solver()
        solver.add(formula != to_z3(term, context))
        self.assertEqual(solver.check(), z3.unsat)

    def test_candidate_traces_never_replace_expert_repair_targets(self):
        from synthesis.cfg.candidate_traces import prepare_candidate
        from synthesis.cfg.demos import DemoSegment
        from synthesis.cfg.program_adapter import program_to_cfg

        context = HighLevelContext()
        definition = prepare_program(Program(1, [Skip(0)]), 2)
        expert = example_trace()
        expert.events = (
            {
                "kind": "instruction_start",
                "path": "0",
                "index": 0,
                "bindings": {"b0": 0},
            },
            {"kind": "instruction_end", "path": "0", "index": 1, "bindings": {"b0": 0}},
        )
        cfg = program_to_cfg(
            definition, [DemoSegment(0, 0, 1, expert)], *task_spec("stack")
        )
        candidate = copy.deepcopy(expert)
        with patch(
            "synthesis.cfg.candidate_traces.record_execution", return_value=candidate
        ) as execute:
            prepare_candidate(cfg, context)
            self.assertIs(
                execute.call_args.kwargs["initial_snapshot"], expert.snapshots[0]
            )
            self.assertIs(cfg.demos.for_node("v0")[0].trace, expert)
            self.assertIs(cfg._candidate_demos["v0"][0].trace, candidate)
        revised = copy.deepcopy(candidate)
        with patch(
            "synthesis.cfg.candidate_traces.record_execution", return_value=revised
        ) as execute:
            prepare_candidate(cfg, context, revision=1)
            self.assertIs(
                execute.call_args.kwargs["initial_snapshot"], expert.snapshots[0]
            )
            self.assertIs(cfg.demos.for_node("v0")[0].trace, expert)
            self.assertIs(cfg._candidate_demos["v0"][0].trace, revised)

    def test_symbolic_and_physical_loop_paths_map_to_same_cfg_region(self):
        from synthesis.cfg.lower import lower_with_locations
        from synthesis.cfg.program_adapter import program_to_cfg

        context = HighLevelContext()
        definition = prepare_program(
            Program(
                3, [Skip(0), Skip(0), While(z3.BoolVal(False), [], [Skip(0)], None)]
            ),
            2,
        )
        cfg = program_to_cfg(definition, [], *task_spec("stack"))
        cfg.nodes["v0"].region.symbolic = (Skip(0),)
        loop = cfg.nodes["v1"].region
        loop.invariant = z3.BoolVal(True)
        loop.body_cfg.nodes["v0"].region.symbolic = (Skip(0),)
        _, physical = lower_with_locations(cfg, context)
        _, symbolic = lower_with_locations(cfg, context, physical=False)
        self.assertEqual(set(physical["loops"]), {"2"})
        self.assertEqual(set(symbolic["loops"]), {"1"})
        self.assertIs(physical["loops"]["2"][1], symbolic["loops"]["1"][1])

    def test_supplied_cfg_lowering_preserves_executable(self):
        from synthesis.cfg.lower import lower_with_locations
        from synthesis.cfg.program_adapter import program_to_cfg

        context = HighLevelContext()
        definition = load_program("synthesis.examples.stack:build_program", context, 3)
        cfg = program_to_cfg(definition, [], *task_spec("stack"))
        lowered, mapping = lower_with_locations(cfg, context)
        self.assertEqual(
            program_fingerprint(lowered), program_fingerprint(definition.program)
        )
        self.assertEqual(set(mapping["loops"]), {"1"})
        self.assertEqual(len(mapping["blocks"]), 3)


class VideoTests(unittest.TestCase):
    def test_video_is_playable_at_twenty_fps_including_last_frame(self):
        import shutil
        import subprocess

        if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
            self.skipTest("Video tools are not installed")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seed_0000.mp4"
            video = VideoRecorder(path)
            video.append(np.zeros((32, 32, 3), dtype=np.uint8))
            video.append(np.full((32, 32, 3), 255, dtype=np.uint8))
            status = video.close()
            self.assertEqual(status["status"], "saved", status)
            payload = json.loads(
                subprocess.check_output(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=r_frame_rate,nb_frames",
                        "-of",
                        "json",
                        str(path),
                    ]
                )
            )
            self.assertEqual(payload["streams"][0]["r_frame_rate"], "20/1")
            self.assertEqual(payload["streams"][0]["nb_frames"], "2")
            frames = subprocess.check_output(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-i",
                    str(path),
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "pipe:1",
                ]
            )
            pixels = np.frombuffer(frames, np.uint8).reshape(2, 32, 32, 3)
            self.assertLess(pixels[0].mean(), 5)
            self.assertGreater(pixels[-1].mean(), 250)

    def test_encoder_does_not_swallow_trajectory_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = VideoRecorder(Path(directory) / "timeout.mp4")
            with patch(
                "synthesis.cfg.collection.subprocess.Popen",
                side_effect=TimeoutError("deadline"),
            ):
                with self.assertRaisesRegex(TimeoutError, "deadline"):
                    writer.append(np.zeros((32, 32, 3), dtype=np.uint8))
            writer.close()

    def test_encoder_failure_retains_clear_status(self):
        with patch("synthesis.cfg.collection.shutil.which", return_value=None):
            with self.assertRaisesRegex(ValueError, "ffmpeg"):
                VideoRecorder.check_available()
        with tempfile.TemporaryDirectory() as directory:
            writer = VideoRecorder(Path(directory) / "empty.mp4")
            writer.append(np.zeros((1,), dtype=np.uint8))
            result = writer.close()
            self.assertEqual(result["status"], "failed")
            self.assertIn("RGB", result["reason"])
