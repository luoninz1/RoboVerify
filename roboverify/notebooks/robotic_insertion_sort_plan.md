# Robotic insertion sort

- [x] Inspect Fetch action API and choose a temporary table position for insertion sort's saved block.
- [x] Assign stable block keys and initialize a reproducibly shuffled row.
- [x] Execute insertion sort with physical pick, move, and release actions.
- [x] Verify grasping, placements, final physical order, and object preservation.
- [x] Execute the notebook, inspect the MP4, and record validation evidence.

## Verified results (2026-09-08)

The executed notebook and `robotic_insertion_sort.mp4` show seed 42 sorting
`[4, 3, 2, 1]` to `[1, 2, 3, 4]`. Twelve transfers use 1,975 physics steps
(79 simulated seconds). Final maximum 3D slot error is 2.49 mm.
The four block identities are preserved, each occupies the expected row slot,
each remains upright and at tabletop height, and the buffer and gripper are empty.
Transfer destinations and measured results are in `robotic_insertion_sort_result.json`.

The MP4 is H.264, 960×720, 12.5 fps, 1,050 frames, 84 seconds including pauses.
Full ffmpeg decoding passed without errors. Six intermediate frames and the final
image were visually checked: the robot grasps/lifts/carries/releases the blocks,
the buffer is used, keys match the block colors, and the final row is sorted.
The video is also embedded in the executed notebook.

Additional headless MuJoCo checks in `robotic_insertion_sort_validation.json`:

| Seed | Initial keys | Final keys | Transfers | Steps | Maximum XY slot error |
| --- | --- | --- | --- | --- | --- |
| 7 | 1, 3, 2, 4 | 1, 2, 3, 4 | 3 | 495 | 2.11 mm |
| 21 | 1, 2, 4, 3 | 1, 2, 3, 4 | 3 | 502 | 1.69 mm |

Four algorithm tests passed, including all 24 unique-key permutations, all 24
identity permutations with duplicate keys (stable ordering), physical occupancy
constraints, and empty/singleton/already-sorted cases. These use a fake occupancy
model and do not establish MuJoCo success for all permutations.

```bash
unset LD_PRELOAD
uv run python3 -m unittest synthesis.examples.test_robotic_insertion_sort -v
uv run python3 -m jupyter nbconvert --to notebook --execute notebooks/robotic_insertion_sort.ipynb --inplace --ExecutePreprocessor.timeout=600
```

For the moving robot, slot spacing was increased from 9 to 11 cm after a live
9 cm test detected a neighboring block displaced by an open gripper. Blocks
remain 5 cm cubes. Scene restoration now includes actuator inputs, mocap targets,
applied forces, and solver warm-start values in addition to legacy GT state.

Scope: four simulated Fetch table cubes, unique keys 1–4. The simulator's measured
poses provide feedback. The simulation demonstration does not establish physical
robot deployment or formal correctness of the controller.

## Type annotations (2026-09-09)

Annotated all five inputs and the return value of `insertion_sort_blocks`, including
a structural robot protocol and documented NumPy array shapes. The input list can
temporarily contain a vacant slot; successful return contains only block names and
preserves the original list object. Existing four algorithm tests and `git diff
--check` passed. Runtime behavior is unchanged; the simulation video was not rerun.

## Robotic invariant inference and verification (2026-09-09)

- [x] Instrument the actual sorter with read-only observations at coherent outer/inner loop heads and outer exit.
- [x] Capture immutable program variables, measured poses, initial identities, and physical occupancy.
- [x] Define sorting predicates and row-guarded order axioms; reuse the existing RoboVerify learner unchanged.
- [x] Collect a disjoint physical training/held-out corpus, including all 24 distinct-key permutations and duplicate/negative keys.
- [x] Infer the actual outer/inner formulas using small vocabulary projections; check concrete axioms and semantic entailment.
- [x] Check symbolic initiation, buffer entry, shift preservation, insertion, skip, transfer preconditions, and final physical ordering.
- [x] Execute a genuine solver counterexample in MuJoCo and reinfer until all obligations pass.
- [x] Add fault detection, regression tests, an executed report notebook, an HTML export, and measured loop-state visualization.

The first round used 32 successful physical runs (142 training and 140 held-out
snapshots). Its learned formulas passed trace evaluation, but buffer-entry
preservation failed. The solver exposed row keys `[1, 0, 1, 1]` at `i=1`: the
candidate incorrectly excluded equal-key pairs in the untouched suffix. The
archived report is `robotic_insertion_sort_refinement_0.json`. A real physical
run of that counterexample added six training snapshots; no clause was manually
deleted and no verification assumption was weakened.

The final corpus contains **33 physical runs**: 18 training runs / 148 snapshots
and 15 held-out runs / 140 snapshots. All physical runs passed. The actual learned
formulas contain 15 retained clauses across five projections (including repeated
clauses across projections), with zero training/held-out violations. All seven
target-entailment checks are UNSAT for the negated target; axioms and candidates
are satisfiable. All nine abstract verification obligations have SAT premises
and UNSAT counterexample queries. All 288 snapshots pass separately supplied
structural checks. Five injected faults are detected by the learned formula,
geometric validity checks, or structural checks, with the responsible layer named.

The reorganized notebook has 13 numbered report sections, 33 cells, and 18 code
cells executed in order from a fresh kernel with no errors. That execution reran
the video and the entire 33-case physical corpus. The figure
`robotic_insertion_sort_invariant_trace.png` shows all 13 measured boundaries of
the reverse-order video. The MP4 remains H.264, 960x720, 12.5 fps, 1,050 frames,
84 seconds; complete decoding passed and a midpoint frame and the trace figure
were visually inspected. An HTML report is saved as `robotic_insertion_sort.html`.

Validation: **47 tests passed** across sorting, tracing, inference, abstract
verification, existing relational inference, and existing BMC. Tests include a
regression for the discovered equal-suffix overfit and negative controls for
identity/geometry/index errors. Exported outer and inner SMT files are separate
because loop invariants hold at different program points.

Guarantee: the learned formulas, conjoined with explicit structural conditions,
are inductive for the supplied four-block symbolic model with arbitrary integer
keys, conditional on successful transfers and preservation of other occupancies.
This does not prove the Python-to-model translation, continuous controller/grasp
success, physical deployment, arbitrary block counts, or equal-key stability.
Physical final-state checks do test stable outputs on the collected runs.

```bash
unset LD_PRELOAD
uv run python3 -m synthesis.examples.robotic_sort_experiment
uv run python3 -m unittest synthesis.examples.test_robotic_insertion_sort synthesis.examples.test_robotic_sort_traces synthesis.examples.test_robotic_sort_inference synthesis.examples.test_robotic_sort_verification synthesis.inference_lib.test_relational_inference synthesis.verification_lib.test_bmc_lib -v
uv run python3 -m jupyter nbconvert --to notebook --execute notebooks/robotic_insertion_sort.ipynb --inplace --ExecutePreprocessor.timeout=600
uv run python3 -m jupyter nbconvert --to html notebooks/robotic_insertion_sort.ipynb
```
