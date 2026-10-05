# Synthesis and verification

`synthesis.entry.synthesize_cfg` runs the structured CFG synthesizer and both
verification stages on the same candidate and task conditions.

## Collection and pipeline modes

First collect a supplied primitive DSL program using
`synthesis.entry.collect_demos`. The [collection guide](../inference_lib/README.md)
describes seeds, full-state archives, and optional 20 FPS videos. Demonstrations
belong under `demos/`; experiment results belong under `runs/`. Generated
archives are not bundled. The four local archives were deleted by request and
were not regenerated for the combined controller update. A fresh four-block
archive was subsequently collected for the [PyEDA recheck](#provided-stack-verification).
The general examples below first create a collection; if its
output directory gets a numbered suffix, use the printed archive path.

Stack collection holds the reset gripper position for 50 control steps, then
records the resulting full simulator state as demonstration state zero. The
preparation is excluded from program actions, loop events, and video. Both
pipeline modes restore archived starts for candidates and repairs; standalone
MCMC uses the same saved starts for CEM/scoring and candidate videos. Seeds
identify recordings rather than reconstructing their initial state. Restoration
never repeats settling. Recollect demonstrations made before this preparation
policy to obtain settled starts.

The shared Stack precondition requires unstacked, pairwise-scattered blocks
at the same height level (`forall x,y. Higher(x,y)`, using the configured
Higher tolerance for observed geometry);
the postcondition requires every block to be ON* b0. Every supplied demonstration
must complete and satisfy these initial/final conditions. The driver requires
`--demos`; collection is a separate step. These commands illustrate the two
interfaces. The [supplied Stack configuration](#provided-stack-verification) below
includes the explicit motion premises used by the successful proof.

```bash
uv run python -m synthesis.entry.collect_demos \
  --program synthesis.examples.stack:build_program \
  --num-blocks 3 --num-trajectories 5 --save-video
uv run python -m synthesis.entry.synthesize_cfg --task stack --num-blocks 3 \
  --mode full --quotient --demos demos/stack/3-blocks-5-trajectories/demonstrations.npz
uv run python -m synthesis.entry.synthesize_cfg --task stack --num-blocks 3 \
  --mode verify --program synthesis.examples.stack:build_program \
  --demos demos/stack/3-blocks-5-trajectories/demonstrations.npz
uv run python -m synthesis.experiment.report --run runs/cfg/latest
```

`--mode full` performs search and optional flat-loop recovery (`--quotient`).
`--mode verify` adapts the supplied executable into the same CFG and skips initial
search. Its fingerprint must match the demonstration source. PickPlace and nested
loops are unsupported in this workflow. `--output-dir` changes the experiment
results root; `--run-name` adds an optional readable label. `--smoke` is a small
search budget, not an acceptance criterion.

The collector and pipeline accept `--higher-tolerance METRES` (default 0.001,
zero for exact ordering). Runtime guards, inference and geometric verification
share this setting; see the [height comparison guide](../inference_lib/README.md#higher-height-tolerance).

## Provided Stack verification

The supplied program has an unchanged symbolic body
`Put(b_prime, b); Assign(b, b_prime)`. Its physical placement uses b0 for XY and b
for Z. Use the following explicit configuration with the settled archive:

```bash
uv run python -m synthesis.entry.synthesize_cfg \
  --mode verify --task stack --num-blocks 4 \
  --program synthesis.examples.stack:build_program \
  --demos demos/stack/4-blocks-5-trajectories-pyeda-verification/demonstrations.npz \
  --max-loop-iterations 3 --motion-iterations 0 \
  --invariant-relations ON_star Higher Scattered equality \
  --invariant-minimizer pyeda \
  --supported-towers --table-surface-height 0.4 \
  --initial-arm 1.3446426 0.74911606 0.5314612 \
  --motion-timeout-ms 10000 --verification-timeout-ms 60000 \
  --higher-tolerance 0.001 --run-name stack-pyeda-verification
```

If the archive is absent, collect it with `collect_demos --num-blocks 4
--num-trajectories 5 --program synthesis.examples.stack:build_program
--seed-start 0 --max-loop-iterations 3 --higher-tolerance 0.001
--output-dir demos/stack/4-blocks-5-trajectories-pyeda-verification` first.
Use the collector's printed archive path. The initial-arm values above
are the nominal settled reset pose in this environment; they are explicit formal
entry conditions, not automatic extraction of the complete simulator state into Z3.
Only candidate execution restores full simulator snapshots. Archives collected
before the equal-height premise was added record the earlier task specification;
recollect them before running this pipeline. Also recollect archives from before
uniform XYZ action scaling: action semantics now participate in the executable
fingerprint, so these older demonstrations intentionally fail source matching.
The archive format is unchanged.

Both modes use `InvInference` → `inference.loop_inference`, the intended
partition-based algorithm. Candidate execution supplies continuing loop heads and
normal exits; preservation feedback uses the same algorithm. There is no
`--learner` option and no automatic fallback to a different learner.
Use `--invariant-minimizer sympy` (default) or `--invariant-minimizer pyeda`
to select Boolean minimization within that algorithm. Both preserve the same
truth-table values. See [minimizer settings](../inference_lib/README.md#truth-table-minimization)
for Espresso's CNF conversion, Python APIs and runtime tradeoffs.

**Both verification stages passed with PyEDA on 2026-10-01:** the result was
`verified_model` in the noiseless, supported-tower model. The shared equal-height
precondition establishes the initial height facts. With the 1 mm Higher tolerance,
the bootstrap invariant passes all 12 symbolic checks (three obligations at sizes
2–4 and in the unbounded context), followed by all 63 motion obligations. No counterexample
refinement, program change or motion repair is needed for this configuration.

Run: `runs/cfg/20261001-174348-f563869-stack-pyeda-verification/`, about 89 seconds
excluding collection. All five fresh four-block demonstrations (seeds 0–4) passed
pre/post validation with the current controllers. Candidate executions supplied
15 continuing heads and five normal exits. The run retains the learned invariant,
candidate archives, and per-obligation verdicts under `artifacts/`; read its
bounded report with `uv run python -m synthesis.experiment.report --run
runs/cfg/20261001-174348-f563869-stack-pyeda-verification`.

Historical recheck after uniform XYZ scaling with five four-block demonstrations:
`runs/cfg/20260925-023502-3dc4cf3-stack-uniform-verification-60s/` returned
`verified_model`. An earlier run with a 10-second symbolic-query budget returned
`unknown` (unbounded exit check canceled); the command above gives that solver
60 seconds. An unknown result remains inconclusive, not successful verification.
This run used the earlier 10 mm Pick, head contacts and Z-only Release. The
current code uses 2 mm Pick, disabled head contacts and fixed-XY/full-3D Release;
the supplied symbolic program and geometric motion encoding are unchanged.
The controller update itself used complete-program comparisons and regression
tests without recollection. The original archive referenced by that historical
run remains deleted; the PyEDA recheck above uses a separate fresh archive.

Write `O(x,y)` for ON*, `H(x,y)` for Higher, and `S(x,y)` for Scattered. The learned
invariant has these six universally quantified clauses:

```text
H(x,y)                       => S(x,y) or O(x,y)
O(x,b)                       => x = b
O(b,x)                       => O(x,b0)
H(b0,y)                      => H(x,y)
H(x,y)                       => O(x,y) or H(b0,y)
H(b,x)
```

These constrain the tower's clear top and root, relative heights and separation.
All clauses in this configuration are learned from candidate executions;
none is substituted by a handwritten invariant. The explicit progress checks and
nonvacuity checks remain in force. See the [height-precondition decision](../../../PAPER-DISCREPANCIES.md#31-stack-resets-equal-height-assumption-belongs-in-the-task-precondition)
and [Higher interpretation](../../../PAPER-DISCREPANCIES.md#32-higher-tolerance-for-contact-induced-height-differences).

The narrower `ON_star equality` vocabulary can pass symbolic verification but
omits separation facts needed for motion. The `ON_star Scattered equality` vocabulary can
permit another tower and expose the attachment/replay mismatch described in entry
30. Those remain configuration/model limitations, not failures of this checked run.

Finite symbolic checks are followed by an unbounded proof; motion checks also
quantify over unnamed objects. The demonstration count supplies inference data,
not a bound on the proof's block count. Full MCMC/CFG synthesis acceptance is
resolved; it is separate from this supplied-program proof.

### Three-block demonstrations

In the earlier checked configuration, five three-block demonstrations also
sufficed for this supplied-program workflow.
Candidate execution supplies 10 continuing heads and 5 normal exits; the intended
learner recovers an invariant logically equivalent to the six clauses above.
It passes the same 2–4-block and unbounded symbolic checks and all 63 noiseless
motion obligations without counterexample refinement or program repair.

Collect with `--num-blocks 3 --num-trajectories 5 --seed-start 0
--max-loop-iterations 2 --higher-tolerance 0.001`, using the same Stack program
factory. In the verification command above, use `--num-blocks 3`,
`--max-loop-iterations 2`, and the resulting three-block archive (normally
`demos/stack/3-blocks-5-trajectories/demonstrations.npz`). Keep the vocabulary,
Higher tolerance, supported-tower premises, initial arm, table height and solver
budgets unchanged. The three-block demonstration size does not restrict the proof
to three blocks; no four-block learning states are required for this result.

## Counterexample learning without initial demonstrations

The separate [Section 6.2 experiment](../experiment/invariant_learning/README.md)
uses `synthesis.entry.learn_invariant --program ...` with an empty dataset and
invariant False. It generates valid initial environments, executes the fixed
program, and accumulates loop heads and exits. `--verification-level symbolic`
is the default; `both` also checks motion. It does not change the two pipeline
modes above, which still require demonstration archives.

## Synthesis approaches

`--synthesis-approach relational` is the default and retains the existing
behavior: bind numeric seed operands to names before local MCMC, search over
those names, and allow existential classifiers to introduce scoped bindings.
`--quotient` enables its existing interleaved flat-loop recovery.

`--synthesis-approach id-first` selects **ID-first synthesis**:

1. Initial MCMC and CEM search only numeric `Pick`, `Move`, and `Release` primitives
   (plus `Skip`), over IDs `0 .. num_blocks-1`. There is no early `Get(True)`
   conversion. Move still optimizes all three coordinate offsets.
2. CFG refinement learns ground predicates over those IDs and existing fixed
   task names, such as `ON(1, b0)` followed by `ON(2, 1)`. It introduces no new
   existential object variables. A missing separator or exhausted budget remains
   an unsuccessful search; there is no fallback to quantified refinement.
3. After the concrete CFG is synthesized, flat-loop quotienting generalizes
   repeated fragments to roles such as `b` and `b_prime`, starting from `b0`.
   ID-first enables quotienting automatically. Matching is relational; physical
   instruction counts and classes need not agree. Matching physical sequences
   can seed the next search, preserving fixed-base XY versus carried-top Z;
   incompatible sequences leave the new body unimplemented until search.
   A final completed placement may supply a concrete repetition letter when the
   outgoing task goal is quantified; the task postcondition itself is retained.
4. After a successful fold, the shared MCMC/CEM search realizes each loop body
   with **ByName operands** over all extracted iterations. An unimplemented body
   starts directly from in-scope names, without inventing bindings for the IDs
   of a particular iteration. Body refinement uses the relational policy in this
   phase. The driver also executes the resulting loop and checks any repartitioned
   continuation. Search failure, exhausted execution budgets, or a failed loop
   rollout prevents synthesis success. Supplied physical seeds are searched and
   checked too; structural quotienting alone never certifies an executable.
5. **Before synthesis returns, every physical instruction is ByName.** Any
   remaining concrete IDs become fixed entry aliases before the named phase,
   with their identity map and equality/distinctness facts preserved. They are not arbitrary `Get`
   witnesses. This also covers straight-line candidates when no loop is found.
   Inference, symbolic verification, and motion verification receive the same
   named representation as before, and candidate execution records fresh states
   after generalization. Later motion repair uses that named representation.

Using the three-block collection from the first example:

```bash
uv run python -m synthesis.entry.synthesize_cfg --task stack --num-blocks 3 \
  --mode full --synthesis-approach id-first \
  --demos demos/stack/3-blocks-5-trajectories/demonstrations.npz \
  --run-name stack-id-first
```

The flag is independent of `--mode`: verify mode still skips initial synthesis;
if verification requests resynthesis with additional demonstrations, it uses the
selected approach. Configuration and `artifacts/cfg.json` record the approach;
the latter also records any fixed ID bindings. The standalone
`synthesis.experiment.mcmc.run` already searches numeric primitives and does not
perform CFG refinement or quotienting; this switch belongs to the integrated CLI.

ID-first expects the same physical ID universe across demonstrations. Its
[finite checks](#finite-checks-and-unbounded-proof) start at the demonstrated block
count so smaller universes cannot contradict fixed-alias distinctness premises.
Runtime witness selection remains deterministic (first match in ascending physical
ID order). A carried role must initialize from an available alias, and extracted
demonstration boundaries must pass CFG validation. Physical shape compatibility
is only a warm-start opportunity, not a quotient condition. A successful numeric
search or later named body search does not prove the generalized loop; the shared
inference and verification pipeline still has to accept that returned candidate.
The selected ID-first policy is retained for any later whole-task resynthesis.

### Manual continuation experiment

To replace MCMC with supplied placement candidates while retaining real simulator
execution, refinement, and quotienting, use recordings from the current Stack
example. Recollect after changing controller settings or waypoints; the diagnostic
checks the source fingerprint. For example:

```bash
uv run python -m synthesis.experiment.id_first_continuation \
  --demos demos/stack/3-blocks-5-trajectories/demonstrations.npz
uv run python -m synthesis.experiment.report --run runs/id-first-continuation/latest
```

Use `--num-blocks 4` with four-block recordings to repeat the experiment with
an additional intermediate placement:

```bash
uv run python -m synthesis.entry.collect_demos \
  --program synthesis.examples.stack:build_program --task stack \
  --num-blocks 4 --num-trajectories 5 --seed-start 0 --save-video \
  --max-loop-iterations 3 \
  --output-dir demos/stack/4-blocks-5-trajectories
uv run python -m synthesis.experiment.id_first_continuation \
  --num-blocks 4 \
  --demos demos/stack/4-blocks-5-trajectories/demonstrations.npz
```

This diagnostic requires three- or four-block recordings from the current Stack
example, with `b0` bound to physical ID 0. It saves `artifacts/summary.json`, named
programs, and complete replay archives.
The script distinguishes automatic continuation from isolated calls to quotient
and from replaying rejected candidate loops; none is a formal verification result.
If folding succeeds, it also supplies a candidate for the new named body-search
call and records its actual per-iteration rollouts in `post_quotient_oracle_calls`.
That diagnostic replaces MCMC in both phases; the integrated CLI runs real MCMC.

## Finite checks and unbounded proof

The integrated CLI checks the same candidate and learned invariant at increasing
finite block counts before requesting an unbounded proof:

| Synthesis approach | Finite block counts | Final check |
| --- | --- | --- |
| `relational` (default) | 2, 3, 4 | Uninterpreted domain, with no fixed block count |
| `id-first` | `num_blocks` through `max(4, num_blocks)` | Uninterpreted domain, with no fixed block count |

Thus ID-first with four demonstration blocks checks size 4, then the unbounded
case; with five it checks size 5, then the unbounded case. The selected approach
controls this schedule even in `--mode verify`, where initial search is skipped.
There is no new learning step for each size. An invalid, vacuous or unknown check
stops the attempt; a preservation refinement triggers a new attempt with the
updated invariant. Failure of the unbounded query does not automatically extend
the finite range.

Finite success alone is not `verified_model`. The underlying Python
`symbolic_verify(..., prove_unbounded=False)` API can return an explicitly bounded
result, but the integrated CLI requires the unbounded proof. These block counts
are separate from runtime loop-iteration limits and BMC instruction horizons.

## Verification workflow

1. Acquire the synthesized or supplied CFG and propose checked placement summaries.
2. Execute that exact physical candidate from every recorded initial simulator
   state. Record instruction boundaries, continuing loop heads, and normal exits;
   learn initial invariants from these runtime states. The False invariant is
   logged before bootstrap. A loop may execute before its invariant is learned.
3. Search finite universes for symbolic counterexamples, then request the unbounded
   proof. Unknown, inconsistent, and finite-only outcomes cannot become verified.
4. Refine preservation failures with uncovered successors obtained by executing the
   abstract body, enumerating Get witnesses. This feedback is abstract contract
   replay, not a new MuJoCo trajectory. The learner must cover all positive states
   and prove `I_old => I_new`; an uncovered successor witnesses strict enlargement.
   Failed coverage, implication, realization or progress checks stop refinement.
5. Invalid establishment/body/exit checks can export `resynthesis_request.json`
   while refinement budget remains. Validated `--additional-demos` archives are
   added to complete expert recordings and synthesis is rerun. Without requested
   recordings, return `needs_demonstrations`. Unknown/vacuous queries and exhausted
   budgets stop with their own unsuccessful statuses. Verify mode records
   explicitly when it enters resynthesis.
6. Run motion verification and bounded structural repair using accumulated
   counterexamples. The abstract program, guards and bindings must be preserved.
   Re-execute changed physical candidates, collect new traces, infer and recheck
   invariants, and rerun both verification stages.

Candidate traces never replace expert demonstrations as resynthesis inputs.
Physical and symbolic instruction paths map explicitly to the same CFG regions;
loop traversal order or equal instruction counts are not assumed. Runtime traces and program identities are saved by candidate
revision; traces from earlier executables are not reused as later executions.
`--max-loop-iterations` (100) and `--trajectory-timeout-seconds` (60) bound candidate
execution. Incomplete executions return an unsuccessful result.

Bootstrap and counterexample refinement both use `InvInference`, implemented by
`inference.loop_inference`. The standalone symbolic CEGIS API uses the same
algorithm. This invariant refinement adds abstract successor states to learning
data; it differs from CFG refinement, which splits demonstration segments with
new predicates during synthesis. Positive-state coverage and explicit progress
checks remain mandatory; an unsuccessful inference attempt never switches learners.

Each verification attempt records structured obligations under
`artifacts/verification/`: symbolic files retain VC kinds, formulas, proof scope,
statuses and countermodels; motion files retain every obligation and its geometric
counterexample. Use the run report first, then these artifacts to diagnose the
specific failed check. Candidate programs and inferred invariants remain under
`artifacts/candidates/`.

## Scope and model

A successful result is named **`verified_model`**: partial correctness in the
explicit geometric primitive model. It does not assert total loop termination,
MuJoCo controller refinement, settling, or hardware safety. `--initial-arm X Y Z`
adds an explicit initial-arm condition; without it, all arm positions are checked.
`--motion-noise` and the associated bounds are available through the shared
motion options. Table placements require `--table-surface-height`.

The tower-task scope covers Stack, Unstack, Reverse and ReStack/Partial; the
integrated CLI currently exposes Stack and Unstack. Branch synthesis,
nested/starred quotient, Grid/Pyramid, total-termination proofs and controller
refinement are outside scope. Existing code for other tasks is not evidence of
verified physical execution.

Supported towers have uniform upright blocks of height L, a common flat table,
exact support and complete layers. The tight XY input invariant and placement
VCs are described below. `--supported-towers` explicitly adds height consequences
for all blocks, including unnamed objects: roots rest at table-center height,
blocks stay above that height, and distinct ON*-related members differ by at least
L vertically. These are weaker than complete support chains: they admit gaps but
suffice for this Stack proof. Without a numeric table height the API uses one
shared symbolic height. The two-height premise belongs only to a regression
fixture. Missing premises can cause rejection of otherwise valid motion;
failed/unknown checks are never silently accepted.

With this flag, every loop also checks entry and preservation of the height
consequences, empty-arm clearance (at least L/2 above every center), and exact XY
alignment of ON*-related blocks. The latter two are checked loop invariants, not
new input assumptions. In Stack, pairwise scattering establishes column alignment
at entry and ideal root-aligned placement preserves it. A slightly offset placement
can fail this check even if it satisfies the older L/4 bound. This matters for
Scattered at its sharp 2L threshold. Noiseless proof does not certify nonzero error.

Nonvacuity checks may use an explicitly closed finite domain to obtain a SAT
witness. All Box quantifiers are grounded there; finite UNSAT/UNKNOWN falls back
to the original solver. This shortcut never establishes a safety property or
restricts a violation query: those remain unbounded.

The primitive model follows §5.5's held-object and composed-position state.
Blocks share their current geometry, arm position, and held object. Loop bodies
start from fresh invariant/guard states; continuation uses a fresh
invariant/guard-false state. Frozen ON_star_zero geometry remains separate.
Empty-gripper paths use a point against block cubes; carried cubes use the swept
cube model. Pick checks its horizontal approach at the current arm height and
then its vertical descent, matching the primitive's waypoint sequence. Intentional Pick/Release contact with the selected object is exempt,
explicitly resolving the paper's Pick self-collision contradiction. Release
requires physical support; a missing support causes a failed obligation and an
arbitrary falling position, never an assumed stable placement.

Motion retains the shared-t straight-segment collision query. An enclosing
endpoint box is not an equivalent replacement (entry 6). Noise is opt-in and off
by default. BMC verifies bounded goals, not collision freedom; solve/feasibility
modes are existential even with noise, and do not prove robustness.

## Placement effects and alignment

Placement summaries are proposed from the outgoing ON relation or the final
transport reference, then checked against all ON*/Higher/Scattered WP effects.
Root discovery follows §5.5: enumerate in-scope physical names `r` and prove
`forall u. ON*(target,u) => ON*(u,r)`, including unnamed objects. The integrated
path uses `P = wp(remaining symbolic body, postcondition)` under the established
entry/invariant/guard context transported through prior symbolic instructions.
It also proves that context establishes P; a desired invariant alone cannot
manufacture a root. Standalone motion checking uses its declared entry conditions.
No named root, an inconsistent context, or solver unknown prevents certification;
`b0`, name order, concrete coordinates, and `frame_base` hints are not evidence.

**Input assumption:** existing towers satisfy tight root-relative alignment in
both horizontal coordinates, `abs(F(member)-F(root)) < L/4`. This is explicit
quantified geometry, not a consequence inferred from ON*'s looser `L/2` bound.
Contradictory concrete input scenes fail consistency. Fresh loop contexts carry
this additional geometric invariant alongside the learned relational invariant.
Before each placement, `alignment_entry` checks the destination tower's bound;
after motion, `alignment` checks the placed block against the proved root for
all allowed noise. The input assumption is never inserted on a placement's
output. The frame VC preserves every non-manipulated object, including the root;
support checks reject removing a root with blocks above it. A separated table
placement creates a singleton. Together these preserve the tight invariant.
Changing references requires proving the new root and its entry alignment.
The triangle inequality gives strict pairwise distance `< L/2`; separate
all-pairs placement checks are unnecessary. See
[entry 12](../../../PAPER-DISCREPANCIES.md#12-root-discovery-and-tight-alignment-premises--implemented-with-an-explicit-input-assumption)
for the proof and user decision. Complete ON*/Higher/Scattered effect checks remain in force.

A transfer may span adjacent blocks. A Get/assignment/control boundary inside an
unfinished transfer, multiple placements in one unsplit block, unknown primitive,
or unsupported summary produces an explicit unsupported result.

## Demonstration and loop semantics

Demo segments use absolute inclusive indices and share their cut state. Current
archives contain full snapshots and recorded actions. Direct restoration and
action replay reproduce segment starts. Replay is the default: it restores
archive state zero, then replays only recorded program actions up to the segment.
Direct reset restores the requested segment snapshot. Newly collected Stack
archives start at the settled state. Archives from earlier collection policies
retain their saved starts, but must still pass current task validation to enter the
pipeline. Observation-only and older archive formats are unsupported. Inference
uses candidate runtime loop heads and normal terminal heads with frozen
invocation-entry geometry.

A split replaces `P -> v0 -> Q` with `P -> v1 --C--> v2 -> Q`. Validate
`first(C) <= last(Q)` on the original unsplit segment, with P at the start, C's
first occurrence strictly interior and Q at the end. The boundary checks imply
the comparison. Entry and later blocks use the same rule; P and C need not persist
until the next condition. Whole-CFG boundary, binding and adjacency checks remain
atomic. See paper-review entry 17 for the notation correction.

Learned loop guards may have multiple witnesses. Demonstrated bindings are
positive examples; unselected bindings at continuing heads are unlabeled, while
all bindings at demonstrated exits are negative. Runtime selects the first match.
The symbolic preservation obligation covers every guard-satisfying witness, so
an unsafe alternative can refute verification even if it was never demonstrated.
No match exits the loop; standalone Get still requires witness existence.

Generated loops have no demonstration-derived execution cap. An explicit budget
raises `LoopBudgetExceeded` if a guard witness remains, reporting incomplete
execution instead of a normal loop exit.

Tests use synthetic scenes and scripted realization proposals to exercise the
real verification and feedback code. They do not depend on saved demos or on
the paper's experimental numbers. Real learning success still requires valid,
representative task demonstrations.
