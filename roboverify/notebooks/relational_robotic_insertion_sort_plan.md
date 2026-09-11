# Relational robotic insertion sort

## Requested program

Implement the two-loop, existential-witness insertion sorter discussed in the
conversation. Its only algorithm variables are initialized/updated as
`b = head`, `r = x`, inner-tail `r = b`, and outer-tail `b = r`.
Use `next*`, strict `keyless`, equality, and quantified guards, with three
explicit pick/move/release sequences in the inner body. No `SwapAdjacent`
function, indexing-based sorting, conditional branches, or returned head.

## Work plan

- [x] Create an independent controller/perception adapter and actual two-loop program.
- [x] Create a new explanatory notebook with a rendered Fetch demonstration.
- [x] Record measured loop heads and exits from disjoint training/held-out runs.
- [x] Infer relational clauses using RoboVerify's existing learner without target leakage.
- [x] Compare actual learned clauses with every theoretical invariant conjunct.
- [x] Check fixed-object symbolic initiation, preservation, exit, and transfer obligations.
- [x] Run the notebook from a fresh kernel and inspect video/figures and exported reports.
- [x] Record verification results and isolate the new task files for delivery.

## Existing work to preserve

The original `robotic_insertion_sort.ipynb`, its HTML export, and
`robotic_insertion_sort.py` already have user changes. The Python file contains
an unfinished function declaration, so the new notebook uses an independent
copy of the reusable controller primitives and does not modify those files.

## Evidence boundaries

Measured Fetch/MuJoCo executions are simulation evidence. Learned clauses are
not assumed inductive. Consistency, target implication, reverse implication,
held-out satisfaction, and symbolic transition checks will be reported separately.
No result implies successful physical robot deployment or continuous-controller
verification.

## Verified results (2026-09-11)

The final notebook has 22 cells, including 11 code cells executed consecutively
from a fresh kernel, with zero error outputs. HTML includes the complete video.
An initial display-only Pygments lexer error was corrected and the whole
notebook rerun. Jupyter execution needed permission for its local kernel socket.

All 33 actual Fetch/MuJoCo corpus runs passed: 18 training runs with 170
snapshots and 15 held-out runs with 155 snapshots. The corpus includes all 24
distinct-key permutations and 9 duplicate/negative-key cases. The worst final
3D row-slot error was 2.748 mm (rounded upward). Every run preserved the block
identities and produced stable sorted order with upright blocks and empty buffer.

RoboVerify inferred 2 outer and 17 inner retained clauses across the small
vocabulary projections. All 325 snapshots satisfy the actual inferred formulas
and handwritten targets. Under the stated mathematical and feature-definition
axioms, both the outer and inner learned formulas are **logically equivalent**
to the handwritten invariants: both implication directions are UNSAT for the
negated conclusion, and candidate/theory conjunctions are SAT. None of the
handwritten target conjuncts follows from the supplied axioms alone.

All 13 fixed-four-node symbolic induction, transfer, and termination obligations
passed with SAT premises and UNSAT counterexamples, for arbitrary integer keys.
The proof explicitly composes x-to-buffer, p-to-old-x-slot, x-to-old-p-slot.
An independent measured-snapshot audit found zero discrepancies in 99 inner
initializations, 94 adjacent exchanges, 99 inner exits, and 282 transfers.
Insertions cover 37 zero-swap, 37 one-swap, and 25 multiple-swap cases.

The reverse-order demonstration sorts [4, 3, 2, 1] using 6 exchanges / 18
transfers / 2,906 physics steps. The video is H.264, 960x720, 12.5 fps,
1,477 frames, 118.16 seconds. Full decoding passed. Intermediate/final frames
and the measured-loop figure were visually inspected; figure colors match
the physical block colors.

The combined focused and existing inference/BMC suite passes **42 tests**.
Regressions cover restricted assignments, permutations and ties, invariant
counterexamples, incorrect boundary updates, reversed comparisons, vacuity,
and stale/incomplete/duplicated corpus manifests. Original user-edited files
retain their preexisting changes, including the unfinished declaration.

Saved evidence uses the `relational_robotic_insertion_sort` prefix:
`.ipynb`, `.html`, `.mp4`, `_demo.json`, `_traces.json`, `_invariants.json`,
`_outer.smt2`, `_inner.smt2`, `_transition_audit.json`, `_validation.json`,
`_initial.png`, `_final.png`, and `_trace.png`.
The report records the canonical corpus digest and analysis/backend source
hashes; corpus reuse checks the full case manifest and controller/program hashes.

Scope remains simulation evidence and conditional abstract-model verification.
The successful-transfer contract is assumed in symbolic checks; physical
deployment, continuous collision/grasp verification, arbitrary block counts,
and full automatic Python-to-model verification are not established.
