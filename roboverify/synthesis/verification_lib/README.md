# Motion verification and bounded errors

`Program.lowlevel_verification` requires an explicit `MotionContract` for each
loop. The contract identifies the manipulated source and placement target.
`frame_base` is a compatibility hint; root discovery must prove the reference.
The frame obligation protects all non-manipulated objects. Contract names refer
to bindings at block entry, before loop-carried `Assign` updates. The
[integrated CFG verifier](../cfg/VERIFICATION.md) supplies block contexts and
threads motion state across supported blocks and loop boundaries.

```python
from synthesis.verification_lib.bmc_lib import NoiseSpec
from synthesis.verification_lib.motion_verification import MotionContract

result = physical_program.lowlevel_verification(
    constants=["b0", "b", "b_prime"],
    contracts={"1": MotionContract("b_prime", "b", frame_base="b0")},
    noise=None,                 # default: no actuator perturbations
    timeout_ms=5000,            # per solver query
)
print(result.ok, result.mode, result.checked_blocks)
for counterexample in result.counterexamples:
    print(counterexample.block_v, counterexample.obligation, counterexample.mu_k)
```

`verify_motion_block` (also exported as `MotionVerify`) is the corresponding
basic-block API. It accepts high-level entry conditions and either a lowered
`PickPlaceByName` chain or named Pick/Move/Release primitives. `initial_positions` can additionally bind a concrete scene
for testing; the optional `sym` entry is the arbitrary other block. `sym` is
reserved and must not appear in the program's constants list.

The checks cover consistency, collision sweeps, release/support, direct placement,
frame preservation, complete ON*/Higher/Scattered effects and root alignment.
Aliases follow the current bindings,
and relative waypoint targets use updated geometry. `ON_star_zero` uses separate
frozen entry coordinates. An explicitly manipulated source is excluded from the
frame condition; all other physical objects, including the proved root, are protected.
Moving a support may displace blocks above it. Their positions are left
unconstrained, and a support obligation refuses to certify that manipulation.
BMC uses the same conservative disturbance policy rather than freezing them.

Results are truthy only when every required check is valid and at least one block
was examined. Checks distinguish `refuted`, `unknown`, `inconsistent`, and
`unsupported`. Only satisfiable violation queries produce counterexamples.
Counterexamples retain entry-of-block positions (`mu_k`), positions at the failed
query (`final_positions`), frozen loop-entry geometry (`entry_positions`), alias
bindings, and the chosen bounded errors. These are witnesses in the geometric
abstraction, not evidence of simulator reachability. The arbitrary collision
witness is constrained by the instantiated axioms/invariants; finite instantiation
alone is not a reason to call it spurious. Supplied geometric premises may still
admit scenes outside the agreed supported-height model (discrepancy 16). The CFG
verifier's opt-in `supported_tower_model=True` (`--supported-towers`) adds explicit
height consequences and checks height, arm-clearance and exact-column loop
invariants at entry and after the body. Exact columns and arm clearance are not
assumed at program entry. Only consistency queries may use a finite SAT witness;
safety queries stay unbounded. See the
[Stack configuration](../cfg/VERIFICATION.md#provided-stack-verification). Each
physical object's canonical name is given by `bindings`;
multiple symbolic names may share a position and refer to the same object.

The model is **idealized waypoint motion**, with no modeled settling, grasp-failure,
gripper-shape, or rigid/falling-stack dynamics. Collision checks cover blocks,
not the robot arm or the physical table plane. A table-placement contract requires
the actual `table_surface_height` and checks the resting center height at release;
it does not assign coordinates to relational `tbl`. The CFG verifier supplies
entry conditions for straight-line blocks; uncovered motion fails closed. The
handwritten Reverse and Partial verification entry points have no lowered physical
program and explicitly report unsupported motion verification.

Named Pick/Release primitives explicitly permit contact with their selected
object. Empty-gripper sweeps use a point; carried payloads use a cube. Primitive
Release proves support and treats unsupported positions as arbitrary falls.
These semantics differ from the legacy BMC Release formula described below.

The physical controller now uses uniform XYZ scaling and 2 mm stopping tolerances
for Pick/Move/Release. Release holds the XY reference captured before opening and
checks full 3D endpoint error. Head geometry is non-colliding in the active
CEE-US Fetch simulator; its appearance and mass remain. These changes improve
measured agreement with the waypoint model but do not change the motion proof's
formulas or supply certified tracking/attachment bounds. See the
[combined controller measurements](../experiment/CONTROLLER-PATHS.md).

## Running the shared workflow

The Stack command `synthesis.entry.synthesize_cfg --mode verify --program
synthesis.examples.stack:build_program --demos <demonstrations.npz>` starts with a
provided primitive program. `--mode full` synthesizes first. Both collect runtime
loop heads and normal exits from the current candidate, infer invariants, and run
symbolic verification before motion verification. A symbolic failure can stop the
pipeline before motion is attempted. Physical repairs require new runtime traces,
renewed inference, and both checks again. The expert archive remains the imitation
target; candidate traces cannot silently replace it.

Local demonstration archives have been deleted by request. The combined
controller update did not regenerate them or rerun this archive-consuming
pipeline. The subsequent [PyEDA recheck](../cfg/VERIFICATION.md#provided-stack-verification)
collected five fresh four-block demonstrations with the current controllers and
passed both symbolic verification and all 63 noiseless motion checks.

Collect full-state demonstrations with `synthesis.entry.collect_demos`. Stack
collection performs 50 holding steps before recording; the resulting full state
becomes archive state zero. Candidate runs restore it without further settling.
This simulator preparation does not add settling dynamics to the formal model.
Use `--save-video` for fixed 20 FPS MP4s starting at that saved state. Old observation-only
and loop-head JSON demonstration formats are removed. See the
[collection guide](../inference_lib/README.md) and
[integrated workflow](../cfg/VERIFICATION.md) for complete commands. Only
`verified_model` reports successful symbolic and motion verification within the
model described above.

## Opt-in noise

`NoiseSpec(eps_grasp, eps_move, eps_release)` gives independent per-axis bounds in
metres. The solver searches for a violating error assignment, so successful
verification covers **every** error within those bounds in this abstraction.
Noise propagates through the selected instruction encoding. In the named
primitive model, Pick perturbs the gripper endpoint, Move perturbs its endpoint
and updates a held payload, and Release perturbs the empty-gripper retreat;
supported block positions remain fixed. The legacy PickPlace/BMC encodings have
their own release displacement convention. Noise never becomes an optimized
instruction offset.

The integrated CFG CLI and all four standalone tower entry points accept:

```bash
--motion-noise 0.005 0.005 0.005 --motion-timeout-ms 5000
```

Omitting `--motion-noise` keeps noise off. The standalone Unstack verifier
requires `--table-surface-height`; integrated Unstack motion checking also needs
that value for table-placement contracts. Read it from the environment; the bundled tower environments
use `0.4`. Keep Unstack end-to-end invocations under `timeout 60s`.

The BMC APIs accept the same optional `noise`. `bmc_verify` returns a
`BMCVerificationResult` with boolean truthiness, `status`, `mode`, `model`, and
`symbols`. It checks the **goal**, not collision freedom. `bmc_verify_solve` keeps
its legacy tuple API. `bmc_solve` and `bmc_feasible` are existential searches,
including noise when supplied, and do not establish robustness. Fresh BMC solvers
have a 10-second timeout; caller-supplied solver settings remain in effect for both
consistency and counterexample queries.

BMC Move leaves blocks supported by a manipulated object unconstrained. Nominal
BMC Release changes only end-effector z, leaving block positions fixed; release
noise perturbs that nominal block location. Neither BMC nor the waypoint verifier
is a complete model of the physical release controller; this limitation is logged in
`PAPER-DISCREPANCIES.md`.

## Validation and timing

From `roboverify/`, with the environment variables in `AGENTS.md` configured:

```bash
uv run python -m unittest synthesis.verification_lib.test_bmc_lib \
  synthesis.verification_lib.test_bmc_noise \
  synthesis.verification_lib.test_motion_verification \
  synthesis.verification_lib.test_primitive_motion -v
uv run python -m synthesis.entry.benchmark_motion_verification
```

The benchmark uses the three-`PickPlaceByName` Stack fixture from
`build_stack_programs`, not the current primitive DSL demo program in
`synthesis/examples/stack.py`.
Its synthetic entry conditions explicitly establish the destination tower root;
numeric coordinates alone do not substitute for symbolic root discovery.
The benchmark checks placement contracts and noiseless/noisy swept paths using
the exact bilinear swept-cube encoding. An endpoint bounding-box fallback would
be a conservative overapproximation for diagonal paths, not an equivalent
rewrite; it is not implemented. Run the benchmark to measure current outcomes
and performance.

The [standalone refinement APIs](CEGIS.md) provide typed symbolic results,
partition-based invariant inference with checked progress, and fixed-environment
motion penalties.

Motion placement verification proves a scoped root with the §5.5 quantified
criterion. Existing towers are assumed tightly aligned (`< L/4` in X and Y);
new placements must prove the same bound against that root, including noise.
A name hint cannot bypass root discovery. See
[verification scope](../cfg/VERIFICATION.md) for the input assumption, loop
invariant, and failure behavior.

## Higher observation tolerance

Higher uses the shared comparison `z1 >= z2 - tolerance`, defaulting to 1 mm.
The numerical evaluator, low-level Z3 interpretation and geometric counterexample
realizer use the same setting. `LowLevelContext(higher_tolerance=0)` explicitly
selects exact Higher comparison; otherwise a context captures the active setting
when constructed. The integrated pipeline and collector expose `--higher-tolerance`;
see the [configuration and saved-state comparison](../inference_lib/README.md#higher-height-tolerance).

The abstract ordering axioms are unchanged. A pairwise tolerance need not be
transitive on arbitrary continuous heights; the intended interpretation uses
well-separated block levels. Increasing the threshold is not a proof that all
physical executions satisfy the abstract model. Motion/controller tolerances
and `--motion-noise` are separate parameters.
