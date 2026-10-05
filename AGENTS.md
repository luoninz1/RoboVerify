# Agent guide and architecture

`AGENTS.md` is the single, tool-neutral guide for coding agents in this repository.
Read the environment, project-status and convention sections before making changes;
use the architecture and workflow sections for the affected subsystem.
[README.md](README.md) indexes the remaining project documentation.

- [Environment](#environment)
- [Project status and decisions](#project-status-and-decisions--read-before-starting)
- [Conventions](#conventions)
- [Project overview](#what-this-project-is)
- [Commands and workflows](#commands-and-workflows)
- [Architecture](#architecture)
- [Experiment monitoring](#monitoring-runs)

## Environment

Configure the working directory and both environment variables before importing
or running simulator code:

```bash
cd roboverify
unset LD_PRELOAD
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia"
```

Without `LD_LIBRARY_PATH`, `import mujoco_py` raises `Missing path to your environment
variable` and every simulator-backed test fails at import.

Run everything as a module from `roboverify/` — the `synthesis` package uses relative
imports and files under `synthesis/entry/` are not runnable as bare scripts:

```bash
uv run python -m synthesis.entry.collect_demos --program synthesis.examples.stack:build_program --save-video
uv run python -m synthesis.entry.synthesize_cfg --mode verify --program synthesis.examples.stack:build_program --demos demos/stack/3-blocks-5-trajectories/demonstrations.npz
uv run python -m synthesis.entry.synthesize_cfg --mode full --quotient --demos demos/stack/3-blocks-5-trajectories/demonstrations.npz
uv run python -m unittest synthesis.verification_lib.test_bmc_lib -v
uv run python -m unittest synthesis.experiment.test_run_logger -v     # fast, no simulator
uv run python -m unittest synthesis.experiment.test_mcmc_parity -v    # drives MuJoCo
```

Tests are `unittest`, not pytest. No linter is configured. The four local
`demonstrations.npz` archives were deleted at the user's request; reports and
videos remain. A fresh five-trajectory, four-block archive was collected for the
2026-10-01 PyEDA proof; use the path in the verification configuration below.
Other pipeline examples require collection when their archive is absent. If collection
prints a numbered output directory, use that archive path in subsequent commands.
The commands above illustrate the interfaces. For the supplied Stack proof, use
the [complete verification configuration](roboverify/synthesis/cfg/VERIFICATION.md#provided-stack-verification),
including its motion premises.

## Project status and decisions — read before starting

Read [README.md](README.md#project-status) for current project status and
[PAPER-DISCREPANCIES.md](PAPER-DISCREPANCIES.md) for numbered findings, settled
reasoning and remaining actions. Collection and supplied-program verification
are implemented; full synthesis acceptance is resolved. Stack reset bounds initial
blocks to 0.70 m XY from the robot base.
Stack collection holds the initial gripper position for 50 steps before
recording; the settled full snapshot becomes state zero. Synthesis, candidate
verification, and standalone MCMC restore archived states without repeating
settling or recreating a layout from its seed. The shared Stack precondition
includes `forall x,y. Higher(x,y)`, requiring one height level at entry only.
Higher uses a shared configurable 1 mm tolerance (`--higher-tolerance`); zero
restores exact comparison. Numeric evaluation, geometric verification and
counterexample realization must use the same setting.
All symbolic inference uses the
intended partition-based algorithm, `InvInference` → `inference.loop_inference`;
there is no alternate learner flag or callback. See the
[verification command](roboverify/synthesis/cfg/VERIFICATION.md#provided-stack-verification).
`--invariant-minimizer sympy|pyeda` selects only Boolean minimization inside this
algorithm (default SymPy); PyEDA uses Espresso with the same fully specified
truth tables. Python callers can use `InvInference(..., minimizer="pyeda")` or
`using_invariant_minimizer` for an entire verification/refinement scope.
Supplied Stack verification passes with `ON_star Higher Scattered equality`,
the 1 mm Higher tolerance and the documented supported-tower motion model;
the supplied-program bootstrap currently passes without invariant refinement.
This supplied-program result does not establish search or loop recovery.
The 2026-10-01 PyEDA recheck used five fresh four-block demonstrations with the
current controllers and passed all 12 symbolic checks (sizes 2–4 and unbounded)
and 63 noiseless motion checks. The bootstrap needed no refinement or repair;
the earlier SymPy proof predates the combined controller update.
The standalone [Section 6.2 experiment](roboverify/synthesis/experiment/invariant_learning/README.md)
starts from no demos and False, directly minimizes reachable VC failures with
bounded unordered execution queries, and learns only from complete validated
MuJoCo executions that reproduce the selected failure. Solver guard choices are
replayed and checked; ordinary execution retains lowest-ID selection.
`learn_invariant --verification-level symbolic|both` keeps the supplied program
fixed; Stack is the initial adapter. Its proof is unbounded and separate from
the bounded search for executable counterexamples.
Update the relevant status or entry when it changes, rather than maintaining a
separate implementation-plan history.

- **The paper is an artifact under test, not a specification.** Neither paper nor
  code automatically wins a disagreement. Do not change code solely because the
  paper says so, and do not use its experimental numbers as regression targets.
- **Discrepancies get logged, not silently fixed.** Preserve entry IDs and the
  distinction between implementation defects, paper corrections and model limits.
- Keep Unstack end-to-end invocations under the user's **60-second wall-clock
  limit**. A timeout or exhausted budget is not successful verification.

### Settled implementation decisions

These seven original decisions are settled, not a new task list. Related proofs
and paper corrections are recorded in the numbered review entries.

| # | Decision |
| --- | --- |
| 1 | Higher WP rules introduce quantifiers; do not force the paper's original AFR-closure claim by changing valid code. The corrected rules and theorem issue are entries 16 and 1. |
| 2 | Use KL where convergence/epsilon thresholds depend on its scale; retain MMD as an option, cache demo density and bound sampling costs. MMD and KL thresholds are not interchangeable. |
| 3 | Straight-line search optimizes imitation distance, then ranks the near-best pool by PostScore. Retain legacy weighted-goal scoring for comparison. The default pool limit is 10; rank at convergence. |
| 4 | Keep premise consistency checks and distinguish valid, invalid, vacuous and unknown. An unsat core is a shortcut only when it establishes inconsistent premises. |
| 5 | Top is removed from the predicate vocabulary. |
| 6 | ON_star_zero is frozen entry geometry. Tasks using it equate it with current ON* in the precondition, never through a global link axiom (entry 5). |
| 7 | Optimize all three Move coordinates. Reassess CEM budgets when dimensionality changes; a single smoke run does not justify new defaults. |

The experiment assumes reliable primitive skills. In the active CEE-US Fetch
model, head geometry has both collision masks disabled; appearance and mass are
retained. This intentionally excludes head contacts from the simulated task.
Simulator skill failures must still be diagnosed,
and accepted demonstrations must actually satisfy their pre/postconditions.

## Conventions

- All real work lives under `roboverify/`; the repository root holds project
  guidance, the paper and review decisions.
- Commit in meaningful increments, one coherent change per commit, rather than one large
  commit at the end.
- Work on a topic branch; do not commit directly to `main`.
- When staging, use explicit paths. Generated `roboverify/demos/` collections and
  unrelated untracked files (`plot.py`, `create_env_figure.py`) must not be swept
  into commits with `git add -A`.
- If a tool needs its own rules configuration, point it to `AGENTS.md` rather than
  duplicating these instructions in a separate agent-specific Markdown file.

## What this project is

RoboVerify synthesizes and *formally verifies* robot manipulation programs (block
stacking/unstacking/reversing on a Fetch pick-and-place rig). A program is a small DSL
(`Pick`/`Move`/`Release`/`PickPlace`, `While`, `Put`/`Assign` for the logical/verification
view) that can be: executed in a MuJoCo simulator, mutated/optimized via MCMC + CEM against
expert demonstrations, and verified two ways — high-level (quantified Z3 reasoning over an
abstract `ON`/`ON_star`/`Higher`/`Scattered` block algebra, with loop invariants *learned*
from example traces) and low-level (bounded model checking / geometric reasoning over actual
box coordinates).

Implementation lives under `roboverify/`; the repository root holds onboarding,
project status, the paper and its review/decision record.

## Commands and workflows

Use the environment and module commands above; the full-suite command is below.

- [Trace workflow](roboverify/synthesis/inference_lib/README.md): collection and inference.
- [Motion API](roboverify/synthesis/verification_lib/README.md): geometric checks and noise.
- [Standalone CEGIS](roboverify/synthesis/verification_lib/CEGIS.md): existing-program refinement.
- [Counterexample learning](roboverify/synthesis/experiment/invariant_learning/README.md): fixed-program Section 6.2 experiment, without initial demos.
- [CFG workflow](roboverify/synthesis/cfg/VERIFICATION.md): integrated synthesis and verification.

Instrumented MCMC entry point: `uv run python -m synthesis.experiment.mcmc.run`.
Use `--smoke --task stack --num-blocks 3
--demos demos/stack/3-blocks-5-trajectories/demonstrations.npz` (on one line)
for a bounded search with the collection above, or iteration options for
longer runs. Read the resulting directory through the report tool described below.

### Validation

From `roboverify/`, with the simulator environment above configured:

```bash
uv run python - <<'PYTEST'
from pathlib import Path
import unittest
modules = sorted('.'.join(p.with_suffix('').parts)
                 for p in Path('synthesis').rglob('test_*.py'))
result = unittest.TextTestRunner(verbosity=2).run(
    unittest.defaultTestLoader.loadTestsFromNames(modules))
raise SystemExit(not result.wasSuccessful())
PYTEST
```

Run focused tests for changed behavior and the appropriate broader checks. Scope
formatting to changed files (`uvx isort --profile black`, then `uvx black`) and run
`git diff --check`. Documentation-only edits need reference checks, not simulator
runs. Tests use synthetic scenes or generated traces, not saved demonstration files.

## Architecture

Read the relevant package notes before non-trivial changes. This section describes
the DSL, verification backends, inference, search and integrated CFG pipeline.

- **`synthesis/api/`** — the program representation.
  - `control.py`: shared bounded controllers for explicit Pick/Move/Release.
    ID/ByName pairs share execution after operand lookup. Immutable `ControlConfig`
    uses 2 mm Pick/Move/Release tolerances, gain 20, uniform XYZ scaling,
    and the unchanged 50-step instruction budget. Release freezes XY before
    opening, corrects all three axes during retreat, and stops on 3D position
    error. `last_control_result` and runtime events report
    convergence/step exhaustion; collection rejects unconverged primitives.
    Control settings and action-scaling semantics enter executable fingerprints;
    the existing fingerprint mechanism is unchanged by the combined update. Legacy
    PickPlace macros retain separate controllers and are not collection inputs.
  - `instructions.py`: `Instruction` subclasses. Physical instructions (`Pick`, `Move`,
    `Release`, `PickPlace`, and their `...ByName` variants that resolve symbolic box names via
    `env.symbolic_name_to_box_id`) implement `eval()` to drive a MuJoCo env, and
    `register_trainable_parameter`/`update_trainable_parameter` to expose float offsets
    (`Parameter`) as a flat vector for MCMC/CEM optimization. Verification-only instructions
    (`Put`, `GoalAssign`, `MarkGoal`, `MoveRight`, `MoveDown`) raise on `eval()` and
    must be lowered to physical instructions before execution. `Assign` updates
    runtime aliases, and `Get` finds an object satisfying its binding condition.
    `While` evaluates a restricted subset of Z3 formulas against concrete block
    positions to find/bind existential guard variables each loop iteration.
  - `program.py`: `Program` (holds a list of instructions plus trainable parameters), the
    weakest-precondition machinery (`wp`, `VC_aux`) and the `rewrite_for_put_for_*`/
    `rewrite_for_put_on_tbl_for_*` family that specializes VC generation for each predicate
    (`ON_star`, `Higher`, `Scattered`) across a `Put`. `Program.highlevel_verification(...)`
    and `Program.lowlevel_verification(...)` are the two verification entry points a program
    exposes.

- **`synthesis/verification_lib/`** — the two verification backends.
  - `highlevel_verification_lib.py`: `HighLevelContext` sets up the Z3 sort for boxes in one
    of two modes — `"declare"` (an uninterpreted `DeclareSort`, used for generic
    inference and unbounded verification) or `"enum"` (a finite `EnumSort` with a
    concrete `num_blocks`,
    used to *check* a learned invariant is sound for a specific finite instance, optionally
    rendering a scene). Defines the `ON_star`/`ON_star_zero`/`Higher`/`Scattered` predicates as Z3
    functions.
  - `bmc_lib.py`: bounded model checking. Encodes a fixed-length sequence of `Pick`/`Move`/
    `Release` instructions as Z3 constraints over per-timestep box positions
    (`BMCTraceSymbols`, `encode_step`), then `bmc_feasible`/`bmc_solve`/`bmc_verify` check
    reachability, solve for unknown offsets, or verify a fully-instantiated low-level program
    against a goal.
  - `symbolic_verify.py`: labeled VC results, exact vacuity detection, increasing-size
    finite counterexample search and optional unbounded proof. `counterexamples.py`
    realizes relation tables as geometry or explicitly refuses the model.
  - `cegis.py`: bounded symbolic/motion refinement using the intended
    `InvInference` algorithm, explicit coverage/progress checks,
    `NeedsResynthesis` for entry failures, and counterexample penalties.
    See [standalone CEGIS workflow](roboverify/synthesis/verification_lib/CEGIS.md).
  - `motion_verification.py`: explicit placement contracts, frame preservation, and
    swept-cube checks for lowered loop bodies. Results retain proof mode, failed
    obligations, counterexamples, and timings. `NoiseSpec` is opt-in, off by default;
    all tower verification CLIs accept `--motion-noise GRASP MOVE RELEASE`.
    Missing physical programs (Reverse/Partial) and uncovered instructions fail closed.
    See [motion verification semantics](roboverify/synthesis/verification_lib/README.md).
  - `lowlevel_verification_lib.py`: geometric low-level context, box-corner/cube drawing
    helpers used to visualize/verify concrete 3D placements.

- **`synthesis/inference_lib/demo_store.py`** — in-memory loop-head data and
  adaptation to invariant inference. `Program.eval(..., on_loop_head=store.add)`
  records successful guard bindings before the body; `on_event` additionally
  records normal exits (including zero iterations) and instruction boundaries.
  Every row copies all physical blocks and its invocation's frozen geometry for
  `ON_star_zero`. Loop IDs are instruction paths (`"1"` after an assignment).
  `DemoStore.from_archive` reads the current full-state NPZ archive;
  `save_diagnostic` exports JSON for inspection only. Old demo formats are removed.
  `InvInference(store, loop_id, vocab, context)` calls the intended partition learner.
  See [the trace workflow](roboverify/synthesis/inference_lib/README.md).

- **`synthesis/inference_lib/inference.py`** — invariant learning. Builds a
  boolean-formula vocabulary over the block predicates and partitions supplied
  loop states into truth-table rows. The integrated workflow supplies current
  candidate executions, including normal exits; classifier positives/negatives
  are a separate input to CFG refinement. The learner extracts a quantified
  invariant (`loop_inference`, `forall_exists_loop_inference`,
  `learn_from_partition`) that is later instantiated into an `EnumSort` context for finite
  checking (`instantiate_invariant`/`serialize_invariant` round-trip an invariant between the
  inference context and a verification context).

- **`synthesis/mcmc/`** — program synthesis by search, not primarily verification.
  - `synthesis.py`: the orchestration hub — collects/replays expert trajectories against a
    MuJoCo env, mutates programs (`mutate_program`), scores candidates by trajectory
    divergence from demos (via `cost_func`) combined with BMC feasibility checks
    (`check_bmc_candidate`/`score_candidate_program`), and drives the outer `MCMC(...)` search
    loop. Also has the `make_roboverify_*_env` factories (stack/unstack/reverse/partial/grid/
    pyramid) and video/frame saving utilities.
  - `decision_tree.py`: a compatibility alias only. `ON_feature` now resolves to
    `predicates.atoms.GroundON`; the shallow-tree feature learner it used to hold was
    superseded by the bounded predicate enumerator in `synthesis/predicates/`, which can
    express quantified separators rather than a single ground `ON(b1, b2)`.
  - `search_core.py`: the acceptance rule, annealing schedule, imitation objective and
    epsilon candidate pool used by CFG straight-line synthesis. Original and
    instrumented MCMC share its acceptance and objective helpers; keep the
    Metropolis ratio here rather than writing it out a second time.
  - `distance.py`: cached KL/MMD trajectory distances for the candidate pool.
  - `cem.py` / `cost_func.py`: cross-entropy-method parameter optimizer, and KL/MMD-based
    trajectory-distribution distance metrics used as the optimization objective.

- **`synthesis/util/symbols.py`** — shared symbol allocation and quantifier hygiene.
  Use `fresh_const(sort, prefix, avoid=(...))` for auxiliary solver variables;
  include the surrounding formulas and operands in `avoid` when adding a binder.
  Use `rewrite_quantifier` to transform an existing quantified body, or
  `open_quantifier` when moving binders: these alpha-rename and substitute de
  Bruijn indices without capturing free names or merging nested shadowed binders.
  Use `fresh_name(preferred, occupied)` for generated program-level names; it
  reserves each result in the supplied set. Keep `get_consts`/named constructors
  for program identities, fixed vocabulary, and intentionally shared BMC state
  symbols. Never rebuild an auxiliary by guessing or reusing its printed name.

- **`synthesis/util/on.py`** — shared geometric interpretations of the block algebra
  (`on_star_implementation`, `higher_implementation`, `scattered_implementation`)
  on block coordinates. Runtime guards, features, inference and geometric
  verification share these definitions. `using_higher_tolerance(value)` scopes
  Higher to `z1 >= z2 - value`, default 0.001 m; values must be finite and in
  `[0, 0.025)` m. Construct low-level contexts within that scope. Abstract ordering
  axioms are unchanged; tolerant comparisons require a suitable separated-level
  domain to satisfy them. See the [tolerance guide](roboverify/synthesis/inference_lib/README.md#higher-height-tolerance).

- **`synthesis/entry/`** — runnable pipelines.
  `collect_demos.py` loads `--program module:factory` or `path.py:factory`, runs
  distinct seeds, validates pre/post transitions, and writes current archives to
  `demos/stack/<blocks>-blocks-<count>-trajectories/`. Optional MP4s run at 20 FPS.
  It requires primitive DSL programs; PickPlace and nested loops are rejected.
  `synthesize_cfg.py` requires `--demos`. `--mode full` searches first;
  `--mode verify --program ...` starts with the matching supplied program.
  Both execute the actual candidate for inference, then perform symbolic and
  motion verification with shared feedback. Changed executables get fresh traces.
  Experiment results stay in `runs/`; `--output-dir` changes that root and
  `--run-name` adds a label. `main.py` forwards to this driver.
  `verified_synthesis.py` and the Stack verifier CLI forward to verify mode.
  Unstack/Reverse/Partial standalone verifiers consume `--demos` via the in-memory
  inference adapter; Reverse/Partial still report unsupported motion because they
  lack lowered physical programs. The 2D entry point remains outside tower scope.

- **`synthesis/environment/stack_reset.py`** — Stack-only bounded reset sampling.
  Initial block centers have base-relative X in [0.54, 0.70] m, Y in [-0.20,
  0.20] m, and XY radius at most 0.70 m. Retain Scattered separation and 0.10 m
  initial-gripper clearance. Bounded retries fail explicitly without expanding
  the workspace; archived initial states still restore exactly.

- **`synthesis/environment/`** — MuJoCo/Gymnasium environments (Fetch pick-and-place block
  construction, ant maze, etc.), largely vendored/adapted from CEE-US and
  `fetch_block_construction`. Observations pack agent state first, then per-block state in
  fixed-width slots; the `10 + 12*i : 10 + 12*i + 3` box-position slice convention recurs
  across `instructions.py`, `on.py`, and the `While` guard evaluator — keep it in sync if the
  observation layout ever changes. The four tower tasks expose
  `table_surface_height` separately (the world z of the simulator's table plane);
  `height_offset` is the resting block-center height, and relational `tbl` has no
  coordinates.

- **`synthesis/experiment/invariant_learning/`** — a fixed-program experiment runner
  and environment adapter protocol. `witness.py` builds finite initial-state
  preimages to selected VC failures with shared placement WP and fresh symbolic
  guard witnesses, without ordering or separate finite inductiveness checks;
  `tasks.py` installs solver-generated Stack scenes, settles for 50 steps,
  validates the saved start, replays enabled solver choices, and checks the
  selected failure at its predicted iteration in a complete physical execution.
  `runner.py` starts from False and no data, accumulates heads/exits through
  `InvInference`, checks coverage/enlargement, and requires an unbounded proof.
  Optional motion verification checks the unchanged program afterward.
  The CLI is `synthesis.entry.learn_invariant`; artifacts live under
  `runs/invariant-learning/`. Induction countermodels and executable initial
  witnesses are distinct. Do not inject abstract successors into this dataset.

- **`synthesis/experiment/`** — run logging and reporting, plus an instrumented copy of
  the MCMC search. `run_logger.py` owns the run-directory contract; `report.py` is the
  bounded reader; `config.py` resolves standalone MCMC settings for `config.json`.
  Other entry points pass their own resolved configuration to the logger.
  `compare_stack_heights.py` compares saved Stack heads and exits with ideal
  placement levels and reports every predicate mismatch, including Scattered.
  Under `mcmc/`,
  `search.py`, `cem.py` and `run.py` reimplement only the four functions that need to
  emit records (`MCMC`, `score_candidate_program`, `optimize_program`, `cem_optimize`)
  and import the remaining implementation from `synthesis.mcmc`. Both copies share
  acceptance and objective
  helpers, and `test_mcmc_parity.py` checks matching cost sequences for both
  seed-only and saved-snapshot starts.

- **`synthesis/predicates/`** — the predicate language and its bounded search.
  `term.py` holds canonical interned first-order terms, `scene.py` the concrete
  observation semantics, `enumerate.py` the bottom-up prenex enumerator with explicit
  depth/binder/candidate/timeout bounds, and `classifier.py`/`guard.py` the two call
  shapes (`LearnClassifier`, loop-guard synthesis). Search reports `found`, `no_separator`
  or `budget_exhausted` — an approximate separator is never returned as an exact one.

- **`synthesis/cfg/`** — the relational CFG and the synthesis algorithms over it:
  `graph.py`/`region.py` (IR), `lower.py` (CFG to `Program`, emitting the scoped `Get`
  a refinement's existential prefix requires), `refine.py` (Algorithm 3),
  `straightline.py` (Algorithm 5), `synthesize.py` (the Algorithm 2 recursive driver),
  `quotient.py`/`kleene.py` (Algorithm 4, flat case only), plus demo recording,
  segment reset/replay and split validation. `--synthesis-approach relational`
  retains early scoped binding and is the default. `id-first` keeps the initial
  MCMC and CFG refinement numeric, then runs relational quotienting after concrete
  search completes. Folded loop bodies are searched again with ByName operands
  over all extracted iterations, followed by execution checks of the loop and
  repartitioned continuation. Physical shape matching supplies an optional seed;
  different instruction counts or classes do not prevent relational folding.
  `id_first.py` converts residual IDs to fixed entry aliases before the named
  phase; inference and both verifiers receive only named instructions and
  predicates. Optional seed alignment compares coordinate operands independently,
  preserving fixed-base XY versus carried-top Z. Neither conversion nor body
  search establishes a relational summary; the shared verification stages check it. Learned guards
  accept demonstrated witnesses and reject all bindings at demonstrated exits;
  unselected continuing-state bindings are unlabeled. Runtime chooses the first
  matching witness in ascending physical ID order; symbolic preservation verification covers every matching choice.
  Multiple witnesses are permitted, without a separate uniqueness requirement.
  Extracted iterations share their invocation's frozen entry geometry.
  Symbolic checks use sizes 2–4 for `relational`, or `num_blocks` through
  `max(4, num_blocks)` for `id-first`, followed by an unbounded proof. The CLI
  applies this schedule in both full and verify mode; see [proof scope](roboverify/synthesis/cfg/VERIFICATION.md#finite-checks-and-unbounded-proof).

  `recordings.py` owns the current full-state NPZ format, `collection.py` executes
  bounded recorded programs, and `program_source.py` loads/fingerprints factories.
  Fresh Stack collection holds the initial gripper position for 50 steps before
  recording. Supplied snapshots bypass reset and settling; all recorded indices,
  frozen entry geometry, and video start at the saved settled state. Standalone
  MCMC threads each seed's archived snapshot through scoring, CEM, and video.
  `program_adapter.py` adapts supplied primitive programs to the CFG;
  `candidate_traces.py` maps runtime events to explicit CFG locations for invariant
  inference. Expert recordings remain imitation/resynthesis targets across repairs.

  Run `uv run python -m synthesis.entry.synthesize_cfg --task stack --num-blocks 3
  --mode full --demos demos/stack/3-blocks-5-trajectories/demonstrations.npz
  --smoke --quotient` (on one line) for a bounded integration smoke.
  The driver requires validated current archives and has no historical-oracle
  fallback. `--reset-mode replay` is the default; Unstack retains its 60-second
  process alarm. Success is `verified_model` in the documented scope.
  Supplied Stack verification passes with the intended learner, the equal-height
  task precondition and `ON_star Higher Scattered equality`; full synthesis
  acceptance is resolved. `--supported-towers` adds explicit height
  premises and checks arm-clearance, column-alignment and height loop invariants.
  Finite SAT witnesses accelerate consistency only; motion safety stays unbounded.
  `cfg/artifacts.py` records CFG structure and segment indices without expanding
  trajectory arrays; the arrays remain in their NPZ archives.

## Monitoring runs

Instrumented runs write `runs/<name>/<utc>-<sha>-<slug>/`, with `runs/<name>/latest`
symlinked to the newest one. The contract:

| file | pattern | notes |
|---|---|---|
| `config.json` | written once | resolved config, git sha + dirty flag, argv, library versions |
| `status.json` | **overwritten** each update | one small object forever; `alive` plus a stale `heartbeat` means the run is wedged, not finished |
| `metrics.jsonl` | one flat record per iteration | aggregate it; never read it line by line |
| `events.jsonl` | rare, rate-limited per kind | new bests, first feasible candidate, exceptions |
| `stdout.log` | fd-level capture | the firehose, including MuJoCo/OpenGL output; grep only |
| `result.json` | written once at exit | final verdict |
| `artifacts/` | as needed | programs, pickles, videos, tracebacks (referenced by path, never inlined) |

**Read a run with the report tool, not by opening the files:**

```bash
uv run python -m synthesis.experiment.report --run runs/mcmc/latest
uv run python -m synthesis.experiment.report --glob 'runs/mcmc/*' --table
```

Output is capped at `--max-lines` (default 60) so inspecting a run costs the same
whether it is at iteration 10 or 10,000, and the shape is stable so two reports diff
cleanly. **Never `cat` `metrics.jsonl` or `stdout.log`**; if you must grep the log,
bound it (`grep -m 20`). Avoiding those two reads is the entire point of the run
directory.

### Diagnosing a bad run

The metrics are chosen so each symptom points at a specific lever:

| symptom in the report | likely cause | lever |
|---|---|---|
| `bmc_feasible` near 0, `bmc_reason` dominated by one label | goal spec, operand pool, or program length | `--program-slots`, `--goal-feature`, `--num-blocks` |
| `accept` near 1.0 | acceptance rule is nearly unselective (a cost delta of 0.0065 gives ratio 0.993) | `--beta` |
| `at_floor` near 1.0, `first_feasible` unset | chain pinned at `bmc_failed_cost`, so acceptance is unconditional | seed program, BMC penalty shape |
| `cem` delta mean ~0 or many zero-delta iters | inner parameter optimization not improving anything | `--cem-N`, `--cem-K`, `--cem-iterations`, `--cem-init-std` |
| `success` mean ~0 while `best_cost` improves | objective is not tracking task success | reward weights, objective design |

For standalone MCMC, `--smoke` uses 20 iterations, 2 CEM iterations, 2 seeds,
and no videos. The separate CFG CLI uses 2 search iterations and 1 CEM iteration
in smoke mode. These are diagnostic budgets, not acceptance criteria.
