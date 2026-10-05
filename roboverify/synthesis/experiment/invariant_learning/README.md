# Learning invariants from counterexample executions

This standalone experiment implements the fixed-program workflow of paper
Section 6.2. It starts with **an empty dataset and invariant False**, generates
initial environments with an SMT query, executes the supplied DSL program in
MuJoCo, and learns from the resulting loop heads and normal exits. It uses the
intended partition learner (`InvInference`), without synthesis or program repair.

Stack is the first supported task. The program must contain one non-nested loop,
use `b0=0` as its only fixed entry alias, and preserve block geometry before the
loop. The existing Stack example meets these requirements. The executable
fingerprint and entry bindings must remain identical at every block count.

Physical witness executions use the shared 2 mm Pick/Move/Release defaults,
uniform XYZ scaling and fixed-XY/full-3D Release feedback. The active CEE-US
Fetch model disables head contacts while retaining head appearance and mass.
See the [controller measurements](../CONTROLLER-PATHS.md) for empirical validation;
those measurements do not certify the physical refinement of symbolic witnesses.
Deletion of the local demonstration archives does not affect this workflow,
which starts from no demonstrations.

## Running the experiment

From `roboverify/`, configure the simulator environment as in
[AGENTS.md](../../../../AGENTS.md#environment). No demonstration collection is
needed before these commands.

Symbolic verification only (the default):

```bash
uv run python -m synthesis.entry.learn_invariant \
  --task stack --program synthesis.examples.stack:build_program \
  --verification-level symbolic --save-video \
  --run-name stack-counterexamples-symbolic
```

Both symbolic and motion verification, with the supported-tower premises used
by the supplied Stack proof:

```bash
uv run python -m synthesis.entry.learn_invariant \
  --task stack --program synthesis.examples.stack:build_program \
  --verification-level both \
  --supported-towers --table-surface-height 0.4 \
  --initial-arm 1.3446426 0.74911606 0.5314612 \
  --save-video --run-name stack-counterexamples-both
```

Motion verification runs only after symbolic verification succeeds. It uses the
learned invariant and unchanged physical program. These arm/table values are
explicit formal premises; full snapshot restoration is used for physical
executions. See the [motion model](../../verification_lib/README.md).
`--motion-noise GRASP MOVE RELEASE` opts into bounded errors; the default is
noiseless. A failed motion check produces diagnostics, without repairs or extra
training data.

| Option | Default / meaning |
| --- | --- |
| `--max-counterexample-blocks` | 4; search sizes 1 through this bound, starting again at 1 after each update. |
| `--max-rounds` | 10 accepted learner updates at most; up to 11 verification attempts including the final check. |
| `--max-loop-iterations` | Stack uses `num_blocks - 1`; override for a different bounded execution horizon. |
| `--verification-timeout-ms` | 10000 per symbolic/initial-domain/reachability solver query; does not bound Python formula construction or learning. |
| `--trajectory-timeout-seconds` | 60 for generated-scene preparation and execution together. |
| `--invariant-relations` | `ON_star Higher Scattered equality`. |
| `--invariant-variables` | 2 quantified learner variables. |
| `--invariant-minimizer` | `sympy` (default) or `pyeda` (Espresso); selects Boolean minimization within the same partition learner. |
| `--higher-tolerance` | 0.001 metres, shared by simulation predicates, inference, and geometric checking. |
| `--seed` | 0; execution seeds increase by update number. Seeds do not replace solver-generated layouts. |
| `--motion-timeout-ms` | 10000 per motion query. |
| `--save-video` | Optional MP4 for each generated execution, at 20 FPS. Preparation is excluded. |
| `--output-dir` | `runs`; experiment results root. |
| `--run-name` | Optional readable run label. |

## What counts as a counterexample

The unbounded verifier checks establishment, preservation, and exit using the
current invariant. A preservation countermodel can be an unreachable intermediate
state; it is **not** automatically a valid simulator reset. The experiment saves
verification countermodels separately from generated initial-state witnesses.

After an establishment or preservation failure, search block counts 1, 2, 3,
... directly for a **reachable failure of a selected failed VC**:

```text
valid_initial_environment(s0) and task_precondition(s0)
and an enabled execution of the supplied symbolic program from s0
    reproduces the selected VC failure within the configured bound
```

There is no separate finite-domain inductiveness check or minimization of
unreachable countermodels. Each size checks initial-domain consistency and then
the combined reachability query. UNSAT at smaller sizes establishes minimality
**within the adapter's initial domain and execution bounds**; UNKNOWN stops the
search. The first unbounded countermodel's relation table is not fixed: any
reachable countermodel of the selected VC is eligible.

For establishment, the target is a first loop head outside the invariant after
executing the program prefix. For preservation, the target is a head satisfying
`I` and the guard, whose actual symbolic body execution violates `I`. The query
also retains the negated preservation VC. All earlier iterations are unrolled
explicitly; the candidate invariant is not assumed along the path.

Preimages use the existing placement WP rules. Each unrolled guard gets fresh
symbolic block witnesses; every enabled binding is allowed, without an ID-order
constraint. Assignments propagate witnesses normally: `b := w_i` makes the next
iteration use `w_i` for `b`. `Get` choices are existential too. The loop itself
is never replaced by its unproven invariant during reachability search.

The bound counts total loop bodies, including the failing preservation body.
At bound K, preservation targets heads after 0 through K-1 preceding iterations;
establishment can fail with zero iterations. A normal guard-false head is still
an invariant state, including the one-block zero-iteration exit.

The solver returns a replay plan containing the chosen bindings and target head
index. During physical execution, the collector follows these choices and checks
each guard on the actual scene. A disabled, mismatched, or unused choice rejects
the execution. After the prescribed prefix, ordinary lowest-ID selection resumes
so the supplied program completes. Ordinary collection and synthesis retain their
existing selection policy. The executable instructions and parameters are unchanged.

Stack's solver domain uses the existing reset workspace, gripper clearance,
Scattered separation, and equal tabletop heights. The adapter installs the
requested block poses into a full simulator state, settles for 50 steps, and
rechecks the task precondition and selected path condition on the settled
geometry. That full snapshot is state zero and is restored without further settling. Failed
settling checks are retained as diagnostic archives.

An accepted execution must complete with converged primitives and satisfy the
actual task precondition and postcondition. It must reproduce the selected
failure at the predicted head: establishment has `not I` at the first head;
preservation must refute the selected symbolic VC on the observed head, with
`I` before the selected body and `not I` at its actual successor. A different
uncovered state alone is insufficient (`counterexample_not_reproduced`).
All observed heads/exits then enter the cumulative
dataset, with each trajectory's own block universe and frozen entry geometry.
The next invariant must cover all accumulated states and satisfy the existing
strict-enlargement checks. No abstract successor or handwritten clause enters
this dataset.

An exit failure needs a stronger invariant; enlargement cannot remove its bad
state, so this experiment reports `needs_stronger_invariant`. A failed induction
proof with no reachable selected VC failure in the searched bounds reports
`no_reachable_counterexample`. Neither case is silently repaired. The abstract
WP limitations and deferred Scattered mismatch remain documented in
[PAPER-DISCREPANCIES.md](../../../../PAPER-DISCREPANCIES.md).

## Results and artifacts

```bash
uv run python -m synthesis.experiment.report --run runs/invariant-learning/latest
```

Use a distinct `--run-name` for simultaneous experiments. Each run records:

- `artifacts/summary.md` and `progression.json`: checked invariants, failed
  obligations, generated sizes, state counts, phase timings, and outcomes.
- `artifacts/invariants/`: readable formulas, SMT expressions, and implication
  queries for accepted enlargement checks. Revision 0 is False.
- `artifacts/queries/`: executable SMT-LIB queries and outcomes at each size.
- `artifacts/verification/`: unbounded symbolic checks and countermodels;
  requested motion checks include their geometric counterexamples.
- `artifacts/counterexamples/`: generated coordinates, replay plans (zero-based
  physical IDs and target head iterations), and physical reproduction metadata.
- `artifacts/trajectories/`: current full-state NPZ archives, including rejected
  executions when states were recorded. These remain experiment artifacts.
- `artifacts/learning-states.json`: the accumulated learning dataset;
  `artifacts/videos/`: optional videos of those physical executions.

Exit code 0 means the requested stages passed: `verified_symbolic` for symbolic
only, or `verified_model` for both. A motion failure retains the symbolic result
separately. Exit code 2 means failure or an inconclusive/budget-limited experiment.
A finite search bound never substitutes for the unbounded proof. The motion
result concerns the configured geometric model, not universal MuJoCo dynamics.

Count verification attempts, accepted counterexample executions, and learner
updates separately. The initial False check and final successful check are
verification attempts. Do not assert the paper's table counts or exact printed
formulas as regression targets.

## Adding an environment

Implement `ExperimentTask` in `tasks.py` and register its factory in `TASKS`.
The CLI obtains its task choices from this registry. The adapter supplies:

1. Context, vocabulary, loop ID, fixed-program/task description, and invariant
   installation into that program's verification representation.
2. Unbounded symbolic verification and `search_query(size, timeout_ms, failures)`,
   returning a `WitnessQuery` with initial-domain constraints, the selected VC
   reachability query, execution bound, scene decoder, and replayable paths.
   The shared flat-loop preimage helper is optional.
3. Physical execution from the decoded scene and replay plan, initialization/
   settling, `validate(trace, search)` checking both the complete pre/post transition
   and reproduction of the selected failure, and learning states with frozen geometry.
4. Motion verification for `both` mode, or an explicit unsupported result.

`run_experiment` owns dataset accumulation, the fixed `InvInference` call,
progress checks, result statuses, and artifacts. A test adapter with a binding-only
program checks that this orchestration does not depend on Stack names or geometry.
New predicate languages, nested/multiple loops, and missing physical or symbolic
backends need their own implementation; registering an environment does not
implicitly provide those capabilities. Preserve the 60-second total limit when
adding Unstack.
