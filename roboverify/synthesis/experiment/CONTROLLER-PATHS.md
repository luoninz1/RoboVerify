# Stack controller measurements

Production primitive actions and the active CEE-US Fetch backend now use uniform
XYZ scaling: `a[:3] /= max(1, max(abs(a[:3])))`. Finger control is independent.
Pick, Move and Release default to 2 mm 3D stopping tolerance. Release freezes
XY before opening, then corrects XYZ toward that target during retreat. The
head geometries in the active CEE-US Fetch XML have both collision masks zero;
appearance and explicit inertial properties remain. Gain 20 and 50-step budgets
are unchanged. These choices improve agreement but do not certify physical
refinement of the motion model.

The four local demonstration archives were deleted by request; reports and videos
remain. No additional versioning or replacement collection is included. The
combined validation below keeps full execution traces in memory and writes only
diagnostic paths, metrics and plots under `runs/`. See review entries
[36](../../../PAPER-DISCREPANCIES.md#36-component-wise-saturation-bends-the-physical-gripper-path)
and [37](../../../PAPER-DISCREPANCIES.md#37-release-drift-and-pick-stopping-tolerance-after-uniform-scaling).

## Current combined validation

Seeds 0–199 were executed in pairs: previous uniform-scaling behavior (10 mm Pick,
head contacts enabled, Z-only Release) versus the updated production behavior.
Each updated program restored its counterpart's exact full settled initial state;
all snapshot arrays and bindings were checked for equality. No seed was dropped
or replaced. Both modes validated **200/200 complete four-block Stack programs**,
with 600 Releases per mode. All primitives converged; the updated maximum was
24 of 50 steps. Mean program actions changed from 116.70 to 123.08.

| Measurement (mm) | Previous median / P95 / max | Updated median / P95 / max |
| --- | ---: | ---: |
| Release maximum XY deviation | 2.104 / 2.859 / 11.816 | 1.046 / 1.336 / 1.463 |
| Release final XY error | 2.079 / 2.859 / 11.816 | 0.177 / 0.270 / 0.305 |
| Maximum gripper path deviation per program | 2.483 / 4.628 / 11.816 | 1.594 / 2.235 / 2.500 |
| Pick approach endpoint error | 4.274 / 9.325 / 9.939 | 0.936 / 1.787 / 1.995 |
| Pick descent endpoint error | 2.801 / 4.775 / 5.356 | 0.424 / 1.130 / 1.327 |
| Final tower XY alignment error | 4.873 / 6.440 / 7.243 | 4.895 / 6.696 / 7.633 |

These are 40 ms control-boundary measurements. Path deviation uses the finite
start-to-target segment; Release uses the vertical line through its pre-opening
XY. This differs slightly from the older retreat-start reference below, explaining
11.82 mm versus the earlier 11.78 mm peak. Initial settling and separate gripper
opening/closing phases are excluded from path statistics. Release corrections
still reference the position before opening.

The updated gripper's largest sampled path deviation is 2.50 mm (approach), while
its largest Release deviation is 1.46 mm. Payload tracking is a separate issue:
the largest carried-block path deviation was 3.97 mm, measured relative to the
segment translated by its initial block–gripper offset. The relative attachment
vector changed by up to 4.34 mm during transfer; absolute block–gripper XY offset
reached 6.77 mm during lowering. Final tower alignment did not materially improve.
The HTML includes all phase metrics and all ten earlier >3 mm Release seeds.

The full **350-test regression suite passed**, including new tests for Release
correction after opening, rejection of residual XY error, retained head mass and
arm contacts, numeric/named defaults, and diagnostic execution parity. Tests also
cover collection/replay, inference, symbolic/motion verification and MCMC parity.
No fresh archive-driven verification run was performed; the geometric model is
unchanged and the previously recorded proof remains a model-level result.

Artifacts:
[current standalone HTML](../../runs/stack-control/20260925-031704-7bdffaa-combined-200/artifacts/report.html),
`summary.json`, `executions.json`, and embedded/exportable plots in the same
directory. The run's source metadata records the pre-commit SHA with dirty changes;
production behavior was subsequently committed as `69a440a`.

```bash
uv run python -m synthesis.experiment.compare_stack_control --num-seeds 200
uv run python -m synthesis.experiment.report --run runs/stack-control/latest
```

## Historical diagnosis of Release drift

The former Release opened the fingers, then requested `(0, 0, dz)` until its Z error was at most
2 mm. The Gym mocap update resets its target to the gripper body's **current**
pose before adding each delta. Zero XY therefore means no corrective XY
displacement: sideways error becomes the next step's starting XY. Move, including
lift and lowering, feeds back all three position errors. Uniform scaling cannot
change the direction of a command with only a Z component.

The gripper is driven through a soft mocap weld, not an exact position assignment.
Finite dynamic/constraint tracking errors produce lateral motion during vertical
travel. In seed 0's third Release, the gripper moved 2.42 mm sideways; gripper-body
translation accounted for it, with only 0.029 mm change in the site-to-body XY
offset. Diagnostic XY feedback reduced maximum drift to 1.18 mm and final lateral
displacement to 0.22 mm. Finger-opening drift was only 0.049 mm.

Large third-placement outliers have an additional cause: the upper arm contacts
`robot0:head_pan_link` / `robot0:head_tilt_link` during the high retreat. Replay
measurements at every 2 ms physics step give:

| Seed, third Release | Original XY drift (mm) | XY feedback (mm) | Head contacts disabled (mm) | Both interventions (mm) |
| --- | ---: | ---: | ---: | ---: |
| 0 | 2.421 | 1.179 | 2.421 | 1.179 |
| 1 | 6.673 | 5.307 | 2.382 | 1.140 |
| 9 | 11.781 | 8.131 | 2.496 | 1.211 |
| 52 | 2.933 | 1.458 | 2.933 | 1.458 |
| 62 | 8.466 | 5.966 | 2.494 | 1.209 |
| 63 | 5.600 | 4.635 | 2.473 | 1.202 |
| 98 | 6.620 | 4.795* | 2.474 | 1.201 |

Drift is maximum XY distance from the actual retreat start. These seeds include
all five >5 mm Release outliers from the earlier 100-seed sample, plus two
comparison cases; they are not a random frequency estimate. In the original
sample, 5/300 retreats exceeded 5 mm, all after the third placement. This replay
covers three placements, five variants, and seven seeds (105 releases). Every
baseline replay matched saved actions and final observations within 1e-8.

*XY feedback exhausted the 50-step instruction budget on seed 98, with about
4.42 mm final lateral displacement. Lowering gain to 5 also did not solve the
collision: seed 9's maximum increased to 13.81 mm. Disabling head contacts is a
causal diagnostic in this earlier experiment. The user subsequently chose to
disable head contacts in production alongside XY feedback. Head-contact behavior
is intentionally outside the resulting simulator model; this does not establish
head/arm collision clearance for a physical robot.

## Historical Pick tolerance sweep

Adopted choice: **2 mm** for tighter waypoint stopping with modest execution
cost. This is an empirical operating choice, not a bound on path deviation or
held-block position. The sweep below used the former Z-only Release and enabled
head contacts; the combined production results are reported above.

Six tolerances were compared at seeds 0–99: 600 executions, all valid, with 300
Picks per tolerance. Endpoint error is the 3D distance to the phase's controller
target at the end of approach or descent, before closing.

| Tolerance (mm) | Mean Pick steps | Approach error P95 (mm) | Descent error P95 (mm) | Held-block XY offset P95 after lift (mm) | Final tower XY error P95 (mm) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10 | 14.65 | 9.265 | 4.739 | 2.597 | 6.364 |
| 5 | 15.06 | 4.556 | 4.694 | 2.597 | 6.505 |
| 3 | 15.85 | 2.813 | 2.866 | 2.531 | 6.898 |
| 2 | 16.79 | 1.854 | 1.142 | 2.538 | 6.723 |
| 1 | 17.43 | 0.953 | 0.967 | 2.517 | 6.698 |
| 0.5 | 20.49 | 0.499 | 0.472 | 2.541 | 6.752 |

At 2 mm, worst observed approach/descent errors were 1.983/1.322 mm; every Pick
used at most 24 of 50 steps. Relative to 10 mm, it adds 2.14 steps per Pick
(0.086 simulated seconds), or 6.42 steps per four-block program. The 1 mm option
buys tighter stopping for another 0.64 steps per Pick. The 0.5 mm option costs
appreciably more, without corresponding improvement in payload alignment.

Held-out seeds 100–199 at 2 mm also validated 100/100 executions:
approach/descent maxima 1.978/1.327 mm and at most 23 steps per Pick. The recommended
tolerance thus passed 200 distinct starts and 600 Picks without seed replacement.
This is not a certified success probability or error bound.

Tighter stopping did **not** materially improve carried-block centering or final
tower alignment. Held-block XY offset is block center relative to gripper site
after the first lift. Final tower error is the largest block-center XY distance
from the base block at program end. These include grasp/contact dynamics and
must not be equated with the stopping tolerance. All 1,800 default-tolerance
phase position sequences exactly matched the earlier uniform experiment,
checking that the instrumentation did not alter those executions.

## Historical diagnostic commands and artifacts

From `roboverify/`, with the simulator environment in AGENTS.md. The tolerance
tool now uses the current Release/collision defaults; reproducing the historical
tolerance sweep exactly requires its original revision (`7bdffaa`). The Release
diagnosis tool explicitly reconstructs the old 10 mm/head-contact/Z-only baseline:

```bash
uv run python -m synthesis.experiment.tune_pick_tolerance --num-seeds 100
uv run python -m synthesis.experiment.tune_pick_tolerance \
  --seed-start 100 --num-seeds 100 --tolerances-mm 2 --run-name holdout-2mm
uv run python -m synthesis.experiment.diagnose_release_drift \
  --seeds 0 1 9 52 62 63 98
```

Use `synthesis.experiment.report --run <directory>` for status. Measurements are
JSON and Markdown in each run's `artifacts/`:

- Sweep: `runs/pick-tolerance/20260925-022037-067bf9a-paired-grid/`.
- Held-out check: `runs/pick-tolerance/20260925-023004-067bf9a-holdout-2mm/`.
- Release replay: `runs/release-drift/20260925-022743-067bf9a-substeps/`.
- Historical standalone HTML with embedded plots:
  [report.html](../../runs/controller-investigation/20260925-024148-3dc4cf3-uniform-release-pick/artifacts/report.html).

Historical validation before the combined update: 81 controller, collection/replay, MCMC parity, reset and
motion-verification tests passed. Five fresh default-controller demonstrations
validated, and the supplied Stack pipeline returned `verified_model` in
`runs/cfg/20260925-023502-3dc4cf3-stack-uniform-verification-60s/`. A preceding
10-second symbolic-query budget returned `unknown`; the successful run used
60 seconds. See the complete configuration in `cfg/VERIFICATION.md`.

Release replays sample every 2 ms and measure opening and retreat separately;
Pick/path statistics sample every 40 ms and exclude opening/closing. Neither
covers arbitrary programs or proves collision clearance. Remaining proof gaps
include tracking, payload attachment, contact and release dynamics.
