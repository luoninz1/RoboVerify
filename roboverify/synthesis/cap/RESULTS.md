# CaP baseline benchmark results

The 2026-10-11 comparison found **no held-out improvement** from scalar CEM or
bounded verification-guided repair. Qwen2.5-Coder-7B-Instruct completed 3/90 cases;
3B completed 2/90. Counts were identical for CaP only, CaP+CEM and CaP+RoboVerify.
No selected hybrid policy had a `valid` training check. This does not establish
that RoboVerify's full synthesis approach outperforms Code as Policies.

## Physical completion

Each cell is successes out of ten cases. CaP+CEM matches CaP only in every row.

| Task | 3B CaP only | 3B hybrid | 7B CaP only | 7B hybrid |
| --- | --- | --- | --- | --- |
| Stack | 0/10 | 0/10 | 3/10 | 3/10 |
| Unstack | 0/10 | 0/10 | 0/10 | 0/10 |
| Reverse | 0/10 | 0/10 | 0/10 | 0/10 |
| Partial Stack | 0/10 | 0/10 | 0/10 | 0/10 |
| Insertion Sort | 2/10 | 2/10 | 0/10 | 0/10 |
| Tower Merge | 0/10 | 0/10 | 0/10 | 0/10 |
| Tower Split | 0/10 | 0/10 | 0/10 | 0/10 |
| Stable Merge | 0/10 | 0/10 | 0/10 | 0/10 |
| Sort Restore | 0/10 | 0/10 | 0/10 | 0/10 |
| Total | 2/90 | 2/90 | 3/90 | 3/90 |

The 3B sorting successes were already sorted inputs, using 12 and 15 primitive
calls. They do not show ability to sort an unsorted row. All four added complex
tasks failed. Exceptions and exhausted budgets remain failures even when the
measured layout happens to satisfy the geometry.

## Failures and proof scope

Counts are identical across the three variants within each model.

| Failure category | 3B | 7B |
| --- | --- | --- |
| Language or unavailable API access | 45 | 7 |
| Primitive or predicate contract misuse | 35 | 33 |
| Final goal not established | 8 | 15 |
| AST budget exhausted | 0 | 30 |
| Controller failure | 0 | 2 |

There were no held-out setup errors or wall-clock timeouts. Common errors were
invented `.position`, `.role` and `.z` attributes; moving without a picked object;
passing None or Target to an object predicate; querying targets after moving
their occupants; and non-progressing loops. All nine selected hybrid policies
per model came from the initial pool, not the repair rounds.

7B training checks were `refuted` on Stack, Partial Stack and all three row
tasks; the others were `unsupported`. 3B checks were `refuted` on row tasks and
`unsupported` on tower tasks. Unsupported means no complete covered primitive
trace was available, not a proof of correctness.

The hybrid is a **verification-guided repair prototype**, not full CFG synthesis,
partition-based invariant learning and unbounded proof. Row checking symbolically
executes generated ASTs with arbitrary integer keys and ties at fixed finite size
and initial permutation; refutations are physically replayed. Tower checking uses
RoboVerify's existing noiseless waypoint backend on observed finite scenes.
Unseen scenes, unbounded counts and controller refinement remain unproved.
A fallback policy is tested when no verified candidate exists; physical successes
are not certified policies. See [the guide](README.md) for exact coverage.

## Frozen protocol and resources

Both runs used clean source commit `4217639`, five training, five validation and
ten test scenes per task. All 180 distinct manifest scenes passed preparation.
Methods restore identical full settled snapshots without extra settling; identical
executions are cached with reuse marked. Policies freeze before held-out testing.
No test feedback enters generation, tuning or repair.

The guide's defaults were used: two initial candidates, two repairs, temperature
0.2, seed 42 offset by 100 per task, 3072 tokens per call and eight recursive
helpers. CEM uses one iteration, four samples, two elites and two training cases,
tuning exposed scalars only. Limits are 100000 AST steps, 300 primitives, 20000
physics steps and 60 seconds per physical run; Unstack retains its 60-second limit.
Row checking permits 512 paths and 15 seconds; tower traces have a 15-second limit.
Hybrid methods have more budget; this is not equal-cost evidence.

| Resource | 3B | 7B |
| --- | --- | --- |
| LLM calls including helpers | 49 | 43 |
| Input tokens | 92458 | 93377 |
| Output tokens | 15962 | 11228 |
| Physical rollouts excluding reuse | 475 | 419 |
| Records including reuse | 914 | 856 |
| Full run time | 481.2 s | 336.5 s |

Models ran in BF16 on separate A100 80 GB GPUs with Transformers 4.51.3 and
Torch 2.5.1+cu124. Revisions: 3B `488639f1ff808d1d3d0ba301aef8c11461451ec5`;
7B `c03e6d358207e414f1eca0bb1891e29f1db0e242`.
Of 36 root candidates including repairs, 7B had 35 generated and one generation
error; 3B had 33 and three. Only one pool per task/model was evaluated; ten scenes
are not ten independent policy-generation trials. Strong comparisons need more
policy seeds, stronger generation and equal-budget controls.

## Artifacts and reproduction

Both completed local runs retain full snapshots, prompts, all model responses,
policies, parameters, traces, verification records and episode indices:

- [3B summary](../../runs/cap-v2/20261011-172657-4217639-qwen2.5-coder-3b-instruct/artifacts/summary.json).
- [7B summary](../../runs/cap-v2/20261011-172656-4217639-qwen2.5-coder-7b-instruct/artifacts/summary.json).

Run directories are not committed. Development and interrupted runs are retained
but excluded. From `roboverify/`, configure MuJoCo and start servers as in the guide:

```bash
uv run python -m synthesis.entry.evaluate_cap --model-url http://127.0.0.1:8123 --run-name cap-v2
uv run python -m synthesis.entry.evaluate_cap --model-url http://127.0.0.1:8124 --run-name cap-v2
uv run python -m synthesis.cap.report --glob 'runs/cap-v2/2026*'
```

Pin source and model revisions. Servers were stopped after evaluation to release
GPU memory; downloaded weights remain cached. Validation passed **32 new baseline
tests and 27 existing motion/BMC/logger tests**. Physical regressions cover buffered
row sorting, primitive Stack plus motion checks, exact replay and five-block
clearance. These validate the harness, not generated policies' task-solving ability.
