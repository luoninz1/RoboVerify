# Collecting demonstrations and learning loop invariants

Use full-state NPZ archives for collection, synthesis, and inference. The pipeline
requires archives matching the current task specification; observation-only and
loop-head JSON demonstration inputs are unsupported. Generated archives are not
bundled in the repository. The four local demonstration archives were deleted
at the user's request. A separate fresh four-block archive was collected for the
[2026-10-01 PyEDA verification recheck](../cfg/VERIFICATION.md#provided-stack-verification).
The commands below describe collection when the requested archive is absent.
Preserved historical reports and videos describe their original controller configurations.

## Collect Stack demonstrations

Run from `roboverify/`:

```bash
unset LD_PRELOAD
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia"
uv run python -m synthesis.entry.collect_demos \
  --program synthesis.examples.stack:build_program \
  --num-blocks 3 --num-trajectories 5 --save-video
```

The editable factory in `synthesis/examples/stack.py` uses explicit Pick, Move,
and Release primitives. Its transfer and placement waypoints use the base block
`b0` for X/Y alignment and the current tower top `b` for Z. A custom
`--program module:factory` or
`--program path/to/program.py:factory` must return a `Program` from
`factory(context, *, num_blocks)`. Named and numeric physical operands, Assign,
Get, Skip, and flat While loops are supported. PickPlace and nested loops are
rejected. Numeric operands get fixed aliases; their identities are not learned.

Collection runs exactly N distinct seeds, defaulting to five seeds starting at
zero. Use `--seed-start 10` for another consecutive range or `--seeds 3 8 12`
for explicit seeds. An explicit trajectory count must match the explicit list.
Failed seeds are retained as failures, never replaced by easier seeds.

Before running the DSL, collection holds the reset gripper position for exactly
**50 control steps** with the gripper open. The full state after those steps
(S50, the 51st state counting reset) becomes demonstration **state zero**. The
first 50 states and preparation actions are excluded from the archive, loop
traces, and video. The trajectory timeout includes this preparation. Every
trace records `initialization.source` and `initialization.settling_steps`.

Synthesis, candidate execution for invariant inference/verification, and
standalone MCMC restore this archived full state when restarting a demo. They
do not reconstruct it by calling reset with the seed, and do not settle it
again. Seeds identify trajectories; snapshots define their actual starting
states. Later segments use their saved snapshot or replay only the recorded
program actions from state zero. Archives replay their own stored states.
Recollect archives made before the settling policy to obtain settled starts, and
archives made before the equal-height precondition to match the current task
identity. A current file format alone does not establish task compatibility.

Default output:

```text
demos/stack/3-blocks-5-trajectories/
  demonstrations.npz
  collection.json
  videos/seed_0000.mp4
  videos/seed_0001.mp4
  ...
```

Repeated default collections use numbered suffixes. `--output-dir demos/my-stack-demonstrations` chooses a new destination; an existing explicit
directory is rejected. Use the actual archive path printed by collection in
subsequent commands; examples below assume the first default collection. There
is no collection run-name flag.

Every accepted trajectory must finish normally, start with unstacked,
pairwise-scattered blocks at one height level, and end with all blocks in the tower
rooted at b0. The entry premise is `forall x,y. Higher(x,y)`; since Higher means
at least as high within the configured tolerance, both ordered pairs enforce
one initial height level (at most 1 mm spread by default). This is an entry
condition, not a requirement on later loop states or the final tower.
Transient success is insufficient. The accepted archive is published only if
all requested trajectories pass. Diagnostic archives and the per-seed report
retain failures. `--max-loop-iterations` defaults to 100 and
`--trajectory-timeout-seconds` to 60; exhaustion is incomplete execution.

`--save-video` records the same execution, headlessly, at fixed **20 FPS**. Each
seed gets its own MP4, including partial failed executions where possible.
`--render` independently displays a live window. The ffmpeg executable is
required for videos. Encoding failures are reported separately and produce a
nonzero command result without discarding valid trajectory data. Frames are
streamed rather than retained in memory. Rendering adds no simulator steps;
recording starts after the 50-step preparation.

## Higher height tolerance

`Higher(x,y)` evaluates `z(x) >= z(y) - tolerance`. The default tolerance is
**0.001 m (1 mm)**, compared with a **0.05 m** block side. This treats small
contact-induced height differences as the same level while retaining the order
of separated block levels. `Higher(x,x)` remains true, and the table remains an
isolated logical marker.

Both `collect_demos` and `synthesize_cfg` accept `--higher-tolerance METRES`.
Use `--higher-tolerance 0` for exact comparison. Values must be finite,
nonnegative and below half a block length (0.025 m). Collection metadata and
pipeline configuration record the setting. Archived coordinates are never
rewritten; existing observations can be re-evaluated at another tolerance.
New candidate executions record their active setting as well.

For direct Python calls, use a scoped setting:

```python
from synthesis.inference_lib.demo_store import InvInference
from synthesis.util.on import using_higher_tolerance

with using_higher_tolerance(0.001):
    # Runtime guards, predicate search, invariant learning and verification
    # invoked here share this setting. It is restored when the scope exits.
    invariant = InvInference(store, loop_id, vocabulary, context)
```

The numeric comparison, low-level Z3 translation, and geometric counterexample
realization share the formula. A `LowLevelContext` captures the active setting
when constructed and also accepts an explicit `higher_tolerance` argument.
Abstract Higher axioms and placement WP rules remain unchanged. Tolerant
comparison is not transitive for arbitrary continuous heights; interpreting it
as an order requires a suitable separated-level domain. The threshold is not
a controller tolerance and does not establish physical/model equivalence.

Compare already recorded Stack loop heads and normal exits with ideal placements.
Use your own collection path; this example assumes a local 500-trajectory archive:

```bash
uv run python -m synthesis.experiment.compare_stack_heights \
  --demos demos/stack/4-blocks-500-trajectories/demonstrations.npz \
  --higher-tolerance 0.001
```

The script supports the supplied single-tower Stack program, reconstructing
50 mm levels from recorded `b`/`b_prime` placements instead of rounding observed
heights. It checks every ordered physical-block pair, reports exact and tolerant
Higher results, and separately checks ON*, frozen ON*, Scattered and equality.
It also checks Higher reflexivity, totality and transitivity on each saved state.
Reports go under `runs/height-comparison/`; `--output-dir` and `--run-name` change
the destination. A difference in any checked predicate produces a nonzero exit;
a Higher match must not hide an independent Scattered mismatch. This analysis can
read saved states from an earlier task specification without making that archive
compatible with the current synthesis/verification pipeline.

## Stack reset workspace

New Stack resets sample block centers relative to the robot base: X is
0.54–0.70 m forward, Y is within ±0.20 m, and horizontal distance from the base
is at most **0.70 m**. Every layout retains at least
0.10 m separation in X or Y between blocks and 0.10 m horizontal clearance from
the initial gripper. Blocks start at the resting table height; b0 is unchanged.
Sampling restarts a crowded layout with bounded retries and reports an error if
it cannot fit the requested count; it never expands the region as a fallback.

The bound applies to every newly sampled layout, independently of seed. It is
an XY workspace restriction, not a proof of reachability at every height or for
every tower size. Archives collected with earlier reset bounds keep their saved
layouts; recollect them to use this region.

## Primitive controller settings

`Pick`/`Move`/`Release` and their ByName variants share the controllers in
`api/control.py`. A named instruction only resolves its operands before running
the same controller. Every primitive keeps a default **50-step total budget**;
Pick shares it across approach, opening, descent, and closing.

| Setting | Pick / PickByName | Move / MoveByName | Release / ReleaseByName |
| --- | --- | --- | --- |
| Position tolerance | 0.002 m | 0.002 m | 0.002 m |
| Proportional gain | 20 | 20 | 20 |
| Step limit | 50 | 50 | 50 |

All three primitives use 3D gripper-position error. Release freezes X/Y before
opening, then corrects sideways drift while retreating to the target Z.
These are controller tolerances, not bounds on final block placement.
Gripper opening uses the summed finger positions with threshold 0.052 m and
margin 0.001 m. Open/closed tests are complementary.

Customize one instruction or share an immutable configuration:

```python
from synthesis.api.control import ControlConfig
from synthesis.api.instructions import MoveByName

control = ControlConfig(position_tolerance=0.002, gain=20.0)
move = MoveByName(
    "b0", "b0", "b",
    target_offset=[0, 0, 0.05],
    limit=50,
    control=control,
)
```

The action helper `get_move_action` computes the proportional XYZ command and
divides all three coordinates by `max(1, max(abs(XYZ)))`. This preserves direction
when bounding the command to [-1, 1]. The active CEE-US Fetch backend uses the
same operation for direct actions; it no longer clips Cartesian axes separately.
The independent gripper command is bounded separately. The helper has no
tolerance argument: the controller applies the configured tolerance to its
stopping test. Release's fixed XY target survives every retreat step; it is not
reanchored to the drifting gripper position. The active CEE-US Fetch model disables
both collision masks on the head geometries, retaining their appearance and mass.

Controller settings and the action-scaling version are preserved during
ID-to-name conversion and included in program descriptions/fingerprints. Archives
from the earlier component-clipping controller must be recollected for matching
supplied-program verification. The later head-contact/Release update adds no new
version mechanism. Existing local demonstration archives have been deleted;
no replacement collection was requested. These are fixed settings, separate from the
waypoint offsets optimized by CEM. Scaling improves direction fidelity; it does
not establish physical refinement of the motion proof (review entry 36).

Each instruction retains `last_control_result` (convergence, steps, final phase,
and position error when applicable), also saved in its `instruction_end` event.
A step limit stops that instruction without claiming convergence. Collection
rejects a trace containing an unconverged primitive, even if its final task
predicate happens to hold. Convergence alone does not certify a grasp or goal.

The Stack example uses a 0.10 m transfer height above the current top, then
lowers to 0.05 m. For four-block, three-placement demonstrations, collect with
a three-iteration cap:

```bash
uv run python -m synthesis.entry.collect_demos \
  --program synthesis.examples.stack:build_program \
  --num-blocks 4 --num-trajectories 5 --max-loop-iterations 3 --save-video \
  --output-dir demos/stack/4-blocks-5-trajectories-precise
```

Move controls the gripper site, so controller convergence does not guarantee
exact centering of a held block. The task's ON relation allows 25 mm per axis;
its tolerance is distinct from the controller's stopping tolerance. The 50-step
preparation removes the initial robot transient before demonstration recording.
See [review entry 24](../../../PAPER-DISCREPANCIES.md#24-the-first-stack-placement-inherits-a-transient-robot-state-and-a-grasp-offset)
for the saved-state policy. Recollect after changing controller settings;
program fingerprints identify the executable used by each archive.

### Measuring physical path straightness

The backend uniformly scales XYZ, preserving the requested direction. The
previous component-clipping backend could change direction above 50 mm axis
error at gain 20. Endpoint convergence still does not guarantee a straight
physical path. The historical Stack discrepancy and scope are recorded in
[entry 36](../../../PAPER-DISCREPANCIES.md#36-component-wise-saturation-bends-the-physical-gripper-path).

Compare the previous uniform-scaling controller (10 mm Pick, head contacts,
Z-only Release) with the combined production changes (2 mm Pick, no head contacts,
fixed-XY Release) using complete programs from identical full settled starts:

```bash
uv run python -m synthesis.experiment.compare_stack_control --num-seeds 200
uv run python -m synthesis.experiment.report --run runs/stack-control/latest
```

This writes path/payload metrics and a standalone HTML report, without saving
demonstration archives. See [measurement definitions](../experiment/CONTROLLER-PATHS.md).

To isolate component clipping versus uniform XYZ scaling under the **current**
Pick/Release/collision defaults, configure the simulator environment
as above, then run from `roboverify/`:

```bash
uv run python -m synthesis.experiment.compare_stack_paths \
  --num-blocks 4 --num-seeds 100 --compare-uniform
uv run python -m synthesis.experiment.report --run runs/controller-paths/latest
```

This serial diagnostic observes each fixed-target motion phase and temporarily
patches raw action generation and clipping/scaling inside the diagnostic. It does not change the
production controller. Both variants are checked against the current task's
pre/postconditions; failed seeds remain in the results. Artifacts include a
phase table, raw commands and gripper paths, execution outcomes and `paths.png`.
Bend measures distance from the original phase line at control-step boundaries;
it excludes intentional waypoint turns and does not bound substep or held-block
motion. Opening/closing and the initial settling period are not motion phases.
These commands now use 2 mm Pick and fixed-XY Release with head contacts disabled;
their results should not be labeled as the older entry-36 controller cohort.

For a newly generated four-block paired measurement, generate annotated
side-by-side videos using its `artifacts/phases.json` (replace the example run path
with your own). Earlier measurements must be rendered with their original code
revision; the renderer checks replay equality and refuses mismatched paths:

```bash
uv run python -m synthesis.experiment.render_stack_paths \
  --measurements runs/controller-paths/latest/artifacts/phases.json
uv run python -m synthesis.experiment.report --run runs/controller-videos/latest
```

This reproduces the first seed and the measured largest-bend seed, validates both
controller executions, and checks that their paths match the measurements before
rendering saved snapshots. Complete videos share simulation time at half speed;
the detail clip aligns phase starts at one-tenth speed. Both include measured XY
paths and 3D line-deviation readouts. MP4 frame counts and full decoding are checked.
The saved rollout archive is marked diagnostic, including the uniform-scaling
trials, rather than published as pipeline demonstrations.

## Run either pipeline mode

These commands show the two interfaces. For the checked supplied Stack result,
use the [complete configuration and motion premises](../cfg/VERIFICATION.md#provided-stack-verification).

```bash
uv run python -m synthesis.entry.synthesize_cfg \
  --mode full --task stack --num-blocks 3 --quotient \
  --demos demos/stack/3-blocks-5-trajectories/demonstrations.npz
uv run python -m synthesis.entry.synthesize_cfg \
  --mode verify --task stack --num-blocks 3 \
  --program synthesis.examples.stack:build_program \
  --demos demos/stack/3-blocks-5-trajectories/demonstrations.npz
uv run python -m synthesis.experiment.report --run runs/cfg/latest
```

Full mode synthesizes from the archive. Verify mode starts from the supplied
program, whose executable fingerprint must match the collected source. Both
execute the current candidate from the saved initial simulator states, infer
invariants, and perform the same symbolic and motion verification with feedback.
All symbolic inference calls `InvInference` → `inference.loop_inference`, the
intended partition-based algorithm. There is no learner-selection flag or fallback.
Verification-only mode may enter resynthesis later; this is recorded explicitly.
See [the integrated workflow](../cfg/VERIFICATION.md).

For inference alone from collected runtime loop events:

```bash
uv run python -m synthesis.entry.inference \
  --demos demos/stack/3-blocks-5-trajectories/demonstrations.npz \
  --task stack --loop-id 1
```

This standalone inference CLI uses the default 1 mm Higher tolerance and has no
`--higher-tolerance` option. For another threshold, call the Python inference API
inside `using_higher_tolerance(...)` as shown above.

## Truth-table minimization

Choose `--invariant-minimizer sympy` (default) or `--invariant-minimizer pyeda`
in `synthesize_cfg` (full or verify), `learn_invariant`, the standalone
`inference` CLI, or the tower verification CLIs. Instrumented runs record
`invariant_minimizer` in their resolved configuration. PyEDA 0.29.0 is included
in the project dependencies; `uv sync` installs it. Building its C extension
requires a C compiler and Python development headers if no wheel is available.

For example, append `--invariant-minimizer pyeda` to the
[complete Stack verification command](../cfg/VERIFICATION.md#provided-stack-verification),
or run inference alone on your current archive:

```bash
uv run python -m synthesis.entry.inference \
  --demos demos/stack/3-blocks-5-trajectories/demonstrations.npz \
  --task stack --loop-id 1 --invariant-minimizer pyeda
```

The intended partition learner, selected predicates, and truth-table completion
policies stay the same. Phi accepts every row outside U; phi-prime accepts only
S. Unobserved rows are **not don't-cares**. PyEDA's
[Espresso minimizer](https://pyeda.readthedocs.io/en/latest/2llm.html) uses a
heuristic search for a smaller equivalent Boolean expression; a globally minimum
formula is not guaranteed. It can reduce minimization time, but truth-table
enumeration, predicate selection and Z3 verification still contribute to runtime.
End-to-end speedups require measuring the workload.

SymPy retains its existing POS policy: above five selected predicates, it builds
unminimized CNF to avoid expensive minimization. PyEDA minimizes at those sizes
too. For CNF it minimizes the DNF of rejected rows and complements the result
using De Morgan's laws, preserving the clause form required by inference.
Expressions are converted to SymPy Boolean nodes for the existing Z3 translator,
without invoking SymPy minimization on the PyEDA path. Equivalent expressions
can have different clause layouts and redundancy-check costs; coverage and formal
verification remain required.

For a single Python call:

```python
invariant, clauses = InvInference(store, loop_id, vocabulary, context, minimizer="pyeda")
```

To select a backend for nested inference calls throughout CFG verification,
standalone CEGIS, or direct `inference.loop_inference` calls:

```python
from synthesis.inference_lib.minimization import using_invariant_minimizer

with using_invariant_minimizer("pyeda"):
    invariant, clauses = InvInference(store, loop_id, vocabulary, context)
```

The scoped setting is restored on exit, including exceptions. An explicit
`InvInference` option overrides the enclosing scope for that call. The standalone
experiment's Python API accepts `ExperimentConfig(invariant_minimizer="pyeda")`.

## Recording and invariant data

Archives retain full simulator snapshots, controls, mocap and solver arrays,
actions, observation/action indices, aliases, instruction boundaries, and loop
events. `cfg.recordings.save_traces/load_traces` handle this one current format.
`--reset-mode replay` remains the pipeline default; `reset` restores a segment
snapshot directly. At the beginning of a newly collected Stack demo, both modes
restore the settled state. For later segments, `reset` restores that segment
snapshot and `replay` restores state zero then replays the recorded prefix.
Neither mode repeats the discarded 50-step preparation.
Observations alone cannot restore a segment.

Runtime events identify each loop path, invocation, iteration, continuing head,
and normal guard-false exit, including zero-iteration loops. Frozen geometry
belongs to that loop invocation. Budget failures are never normal exits.
Candidates use their own execution traces; the expert recordings remain separate
for imitation and resynthesis. Persistent in-scope aliases become invariant
constants; selected guard witnesses are retained as metadata, not assumed to
remain defined at loop exit. Symbolic preservation covers every matching witness.

`DemoStore` remains the in-memory inference adapter. `from_archive` extracts
heads and exits, and `save_diagnostic` writes solver/debugging samples only.
The `on_loop_head` callback records successful guard bindings before
bodies; the richer `on_event` interface also reports normal exits and instruction
boundaries. Physical candidate execution does not require an invariant.

Tests use generated scenes and recordings. Golden literal fixtures remain
historical learner regressions, not accepted demonstrations or correctness targets.
