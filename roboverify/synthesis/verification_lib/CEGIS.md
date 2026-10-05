# Counterexample-guided verification

This guide covers the standalone APIs for refining an existing single-loop
program. The [integrated CFG workflow](../cfg/VERIFICATION.md)
connects synthesis, verification, demonstration requests and structural repair;
its capabilities are broader than the standalone offset-repair API below.

Physical executions use the current shared controller: uniform XYZ scaling,
2 mm Pick/Move/Release tolerances and fixed-XY Release feedback. Head contacts
are disabled in the active CEE-US Fetch model. These are simulator choices,
separate from the geometric proof semantics; see the
[controller measurements](../experiment/CONTROLLER-PATHS.md).
Local demonstration archives were deleted. The collection commands below are
future workflow instructions, not archives regenerated during this update.

For Section 6.2's experiment starting from **no demonstrations and False**, use
[`synthesis.entry.learn_invariant`](../experiment/invariant_learning/README.md).
It generates valid initial environments and executes the supplied physical
program, with `--verification-level symbolic|both`. The standalone APIs below
retain their existing demonstration/bootstrap and abstract-successor semantics.

`symbolic_verify.py` labels each VC as establishment,
preservation, exit, or straight-line body, and returns structured
valid/invalid/vacuous/unknown checks. A result is truthy only when all checks pass.
`Program.highlevel_verification` returns this object while remaining compatible
with Boolean callers. Vacuous means the premise and domain axioms are inconsistent.
Unsat cores provide a one-sided shortcut; a core containing the negated conclusion
requires a separate premise satisfiability check. Solver timeouts remain failures.

`SymbolicVerify(build_problem, ...)` searches finite sizes in increasing order,
then requests an unbounded proof unless disabled. `build_problem(n)` returns
`(program, precondition, postcondition, context)`; `n=None` requests DeclareSort.
Finite-only success (`prove_unbounded=False`) is labeled with its checked range
and does not establish correctness for arbitrary block counts. Unknown/vacuous
smaller instances stop the search rather than making a later counterexample appear
smallest. The integrated CLI requires an unbounded proof; its [finite schedule](../cfg/VERIFICATION.md#finite-checks-and-unbounded-proof)
depends on the selected synthesis approach.

`run_symbolic_cegis` records `False`, calls `InvInference` on the supplied
`DemoStore`, checks strict enlargement, verifies, and adds preservation-failure
successors. `InvInference` delegates to the intended partition-based algorithm in
`inference.loop_inference`; the integrated CFG bootstrap and refinement use it too.
There is no learner callback or alternate implementation to select. The learner
is not assumed monotone: updates must pass the explicit coverage and implication
checks, and nonmonotone updates are rejected. An exit failure requests a stronger
invariant; a vacuous/unknown query or unrealizable geometric model stops with its
own status. An establishment/body failure saves the model and raises
`NeedsResynthesis`, whose `.s0` is the concrete scene when realization succeeds.

Counterexample scene construction preserves ON*, frozen ON*, Higher, Scattered,
aliases, and table identity. Abstract tables not realizable by geometry are refused.
Symbolic replay supports straight-line `Put`, `Assign`, and `Skip`; it executes the
placement abstraction, not the physical controller. The successor may be the final
loop head where the guard is false, which is still required for exit verification.
This standalone driver supports one non-nested loop. The integrated workflow
propagates state across supported structured CFGs.

`run_motion_cegis` takes explicit `MotionBlockSpec`s covering every loop, accumulates
`PenStore` environments, and calls the supplied resynthesis callback. Entry
conditions must equal the program's invariant and guard. Full motion verification
runs after every repair. Constants, conditions, placement contracts, relational
summary, symbolic assignments, guards, and motion operand/release structure must
remain unchanged. This built-in repair searches offsets only. The integrated
CFG workflow supports instruction-structure repair while preserving the entire
symbolic program. Unknown, inconsistent, and unsupported checks
cannot become successes or training counterexamples.

`MotionPenalty` rechecks candidates in each saved environment, fixing current and
frozen positions and aliases while retaining the configured universal noise bounds.
Multiple failing obligations in one environment count once. The maximizing score
is `-MMD - weight * failed_environments` when no optional legacy goal reward is used.
The original and instrumented Runner/optimizer/MCMC APIs accept `motion_penalty` and
`motion_penalty_weight`; the penalty is off by default. Penalty-bearing CEM
runs serially because live Z3 objects cannot be sent through the multiprocessing
pickle queue. `optimize_motion_parameters` also supports a supplied demonstration
score; without one it is explicitly penalty-only repair. A zero training penalty
is never substituted for the full verification query.

Run the shared supplied-program workflow from `roboverify/`, with the simulator
environment variables from `AGENTS.md` set. Generated demo archives are not
bundled; use the actual output path printed by the collection command. This
example shows the interfaces; the [complete Stack proof configuration](../cfg/VERIFICATION.md#provided-stack-verification)
also supplies the explicit motion premises:

```bash
uv run python -m synthesis.entry.collect_demos \
  --program synthesis.examples.stack:build_program --save-video
uv run python -m synthesis.entry.synthesize_cfg --mode verify \
  --program synthesis.examples.stack:build_program \
  --demos demos/stack/3-blocks-5-trajectories/demonstrations.npz \
  --run-name stack-verification
uv run python -m synthesis.experiment.report --run runs/cfg/latest
```

`--mode full` searches first; verify mode starts from the supplied program and
enters the same inference and verification stages. Both rerun the actual physical
candidate from recorded initial states to obtain loop heads and normal exits.
New Stack collections save their state after 50 settling steps; candidate
execution restores it directly without repeating that preparation. Motion repairs
trigger fresh execution, inference and both verification checks.
The supplied program's executable fingerprint must match the archive. Expert
recordings remain separate for imitation scoring and later resynthesis.
The integrated CLI takes `--demos`, with `--output-dir` and `--run-name` for
experiment output. The separate `synthesis.experiment.mcmc.run` CLI uses
`--run-root` and `--slug`. `synthesis.entry.verified_synthesis` forwards to verify
mode.

`--motion-noise GRASP MOVE RELEASE` opts into bounded errors. The integrated CLI
also accepts `--higher-tolerance METRES`, default 0.001. Direct Python callers
configure Higher through `using_higher_tolerance(...)`; see the [tolerance guide](../inference_lib/README.md#higher-height-tolerance).
The standalone library APIs use the same fixed symbolic inference algorithm. There
is no learner-selection flag. No Unstack run may exceed the standing 60-second cap.

`inference_lib/observed_patterns.py` retains `MonotoneInvariantLearner` as an
independent observed-pattern utility for other analyses. Symbolic verification
workflows do not import or select it.

RunLogger stores candidate programs, actual execution traces and bootstrap
invariants under `artifacts/candidates/<revision>/`, counterexamples and later
invariant progression as referenced artifacts, and unresolved demonstration
requests in `artifacts/resynthesis_request.json`. Use the report tool to inspect
bounded summaries. Exit code 0 means both verification stages passed with result
`verified_model`; 2 means an explicit unsuccessful result. Budget exhaustion never
means verification succeeded.

Archived supplied Stack verification passed with the intended learner, the shared
equal-height entry premise, the full default vocabulary, the 1 mm Higher
tolerance and the documented supported-tower motion model. The six-clause
bootstrap invariant passed both proof stages without counterexample refinement.
That archived supplied-program run predates the combined controller update; current
full-program controller validation is reported separately without regenerating demos.
Full synthesis acceptance is resolved. See
[the integrated workflow](../cfg/VERIFICATION.md) for scope and
[the collection guide](../inference_lib/README.md) for seeds and video.
