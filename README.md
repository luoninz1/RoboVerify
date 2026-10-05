# RoboVerify

RoboVerify synthesizes block-manipulation programs from demonstrations and checks
symbolic correctness and geometric motion obligations. Programs execute in a
Fetch/MuJoCo environment. The pipeline uses relational CFGs, flat-loop recovery,
learned invariants and counterexample-guided refinement.

## Project status

The Stack workflow has a standalone multi-seed DSL demonstration collector,
full-state archives, optional 20 FPS videos, and shared `full` / `verify` pipeline
modes. Collection holds the initial gripper position for **50 settling steps**,
then saves that full simulator state as demonstration state zero. Settling is
excluded from recorded actions and video. Both modes and standalone MCMC restore
the archived start without resetting from the seed or settling again. Both modes
learn invariants from executions of the actual candidate and perform symbolic
and motion verification with feedback. See the
[collection guide](roboverify/synthesis/inference_lib/README.md).

Synthesis offers `--synthesis-approach relational` (the default) and
`--synthesis-approach id-first` (initial numeric MCMC/refinement, followed by
relational quotienting and named loop-body search).
Both return named programs to inference and verification. See the
[approach guide](roboverify/synthesis/cfg/VERIFICATION.md#synthesis-approaches).

Stack reset scatters blocks within **0.70 m horizontally of the robot base**,
retaining block separation and initial gripper clearance. The shared Stack
precondition also requires one initial block-height level, expressed as
`forall x,y. Higher(x,y)`. The shared Higher predicate has a configurable
**1 mm tolerance** (`--higher-tolerance`, with 0 restoring exact comparison).
Primitive ID/ByName instructions share configurable controllers and retain their
50-step budgets.
See [controller settings](roboverify/synthesis/inference_lib/README.md#primitive-controller-settings).
Cartesian delta commands use uniform XYZ scaling in both the primitive
controller and active Fetch backend. Pick, Move and Release default to 2 mm
position tolerance. Release freezes XY before opening and corrects XYZ during
retreat, stopping on full 3D error. Head collision geometry is disabled in the
active CEE-US Fetch model; its appearance and mass are retained.
The four historical demonstration archives were deleted at the user's request;
reports and videos remain. The PyEDA verification recheck collected a fresh
five-trajectory, four-block archive with the current controllers; its path is in
the [verification command](roboverify/synthesis/cfg/VERIFICATION.md#provided-stack-verification).
The [Stack controller measurements](roboverify/synthesis/experiment/CONTROLLER-PATHS.md)
diagnose Release drift and compare Pick stopping tolerances.

**Symbolic invariant inference uses the partition-based algorithm in
`inference.py`, through `InvInference`.** It is the intended algorithm for both
pipeline modes and standalone symbolic CEGIS; there is no learner-selection flag.
The truth-table minimizer is selectable with `--invariant-minimizer sympy|pyeda`
(SymPy by default). This preserves the partition algorithm and truth-table
semantics; see [minimizer settings](roboverify/synthesis/inference_lib/README.md#truth-table-minimization).

**Supplied Stack verification has passed with the intended learner and the vocabulary
`ON_star Higher Scattered equality`.** With the equal-height entry premise and
1 mm Higher tolerance, the bootstrap invariant passes without refinement.
The candidate passes unbounded symbolic verification and the documented
noiseless motion checks with explicit supported-tower geometry. Full synthesis
acceptance is resolved.
The **2026-10-01 PyEDA recheck passed on fresh demonstrations with the current
controllers**: all 12 symbolic checks (sizes 2–4 and unbounded) and all 63 noiseless
motion checks were valid, without invariant refinement or program repair.
The run returned `verified_model`; the earlier SymPy proof predates the combined
head-contact/Release/Pick update.
See the [verification command](roboverify/synthesis/cfg/VERIFICATION.md#provided-stack-verification)
and the decisions on [initial heights](PAPER-DISCREPANCIES.md#31-stack-resets-equal-height-assumption-belongs-in-the-task-precondition)
and [Higher tolerance](PAPER-DISCREPANCIES.md#32-higher-tolerance-for-contact-induced-height-differences).
Archives must contain full simulator states and match the current task specification;
collections from before the equal-height precondition require recollection for
pipeline use. Collect demonstrations when an example's archive is absent.

For paper Section 6.2, `synthesis.entry.learn_invariant` starts with an empty
dataset and invariant False, searches increasing block counts directly for valid
initial environments reaching a failed VC, and runs the fixed program in MuJoCo.
Symbolic guard witnesses have no ID-order restriction; physical replay checks
those choices and confirms the selected failure at the predicted iteration.
It supports symbolic-only verification (default) or both symbolic and motion
verification. Stack is
implemented with an adapter interface for future environments. See the
[counterexample-learning experiment](roboverify/synthesis/experiment/invariant_learning/README.md).

The supported scope is structured chains and flat loops for the tower tasks;
the integrated synthesis CLI exposes Stack and Unstack. A successful
`verified_model` result establishes partial correctness in the documented
geometric model, not total termination or physical-controller refinement.

## Documentation

| Document | Purpose |
| --- | --- |
| [AGENTS.md](AGENTS.md) | Development rules, environment, commands, architecture and experiment reporting. |
| [PAPER-DISCREPANCIES.md](PAPER-DISCREPANCIES.md) | The single paper-review record: stable entries, decisions/proofs, status and remaining actions. |
| [CFG verification](roboverify/synthesis/cfg/VERIFICATION.md) | Integrated synthesis/verification workflow and model assumptions. |
| [Motion API](roboverify/synthesis/verification_lib/README.md) | Motion contracts, collision/support checks, bounded noise and BMC distinctions. |
| [Standalone CEGIS](roboverify/synthesis/verification_lib/CEGIS.md) | APIs and commands for refining existing programs. |
| [Counterexample learning](roboverify/synthesis/experiment/invariant_learning/README.md) | Section 6.2 experiment from a fixed program and no demonstrations; symbolic or both proof stages. |
| [Trace inference](roboverify/synthesis/inference_lib/README.md) | Collecting loop-head states and learning invariants with DemoStore. |

Implementation lives in `roboverify/`. Configure MuJoCo through AGENTS.md and run
entry points as modules from that directory. Tests use `unittest`. The reviewed
paper is [POPL2027.pdf](POPL2027.pdf). Completed implementation plans and audit
history are retained in Git rather than maintained as active task documents.
