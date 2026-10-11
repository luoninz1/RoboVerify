# Google CaP source port and baseline evaluation

The default implementation now runs the actual `LMP`, `LMPFGen` and
`FunctionParser` source from [Google Code as Policies](https://code-as-policies.github.io/),
adapted to RoboVerify's primitives and local inference. This is a restricted
source port, not the original PyBullet robot/perception stack or an independent
reimplementation. Completion measures physical execution, not an LLM's claim.

The [completed 3B and 7B benchmark report](RESULTS.md) records all nine tasks for
the **previous independent implementation**, retained as `--implementation legacy`.
Its hybrid did not improve held-out completion. Those numbers are not results
for this Google source port; a new full comparison requires separate runs.

## Source provenance and replacements

[vendor/google_cap.py](vendor/google_cap.py) contains the unmodified generation
cell from Google's [Interactive Demo at commit 71896d4](https://github.com/google-research/google-research/blob/71896d4b6816983672aa6999e2dc92a3cfe904e4/code_as_policies/Interactive_Demo.ipynb),
with a copyright/SPDX header and `import ast` added. The
[provenance record](vendor/google_cap.provenance.json) pins the full commit,
notebook/cell hashes and original helper prompt for reference. Google's
[Apache-2.0 license](vendor/LICENSE) is included. Do not format the snapshot;
its original trailing whitespace is preserved by a scoped Git attribute.

[google_generator.py](google_generator.py) loads an isolated copy of this trusted
module for each generation and supplies these replacements:

- Legacy `openai.Completion.create` calls use the existing local chat client.
  Requests, responses, seeds and completion settings are archived. Fences are
  removed and next-query stop delimiters applied before restricted AST parsing.
  Local transport failures propagate with the client timeout; the original
  unbounded SDK retry and retired Edit endpoint are not enabled.
- Google `LMPFGen` still discovers calls, prioritizes assignment signatures,
  recursively creates child helpers and rebinds their parents. An adapter
  recognizes local definitions, arguments and aliases so they are not treated
  as missing global functions. A bounded helper budget also covers recursion.
- `var_exists` becomes dictionary name lookup. `exec_safe` becomes a
  definitions-only registrar: it never executes a generated body or expression.
  `LMP` uses generation-only `debug_mode=True`, not a skipped evaluation.
  The complete program executes later in the existing bounded AST interpreter.
- Python's `ast.unparse` replaces `astunparse`; pretty-printing is plain text.
  RoboVerify-specific prompts replace the NumPy/Shapely/perception/macros in
  Google's examples. Prompt version is recorded separately from source hashes.
  Main-program query/stop delimiters use `# task:` rather than any `#` comment.
- A mechanical `policy()` entry wrapper shares one interpreted scope between
  generated and inline helpers. Already supplied `def policy()` programs are
  called once. Raw completions and normalized source are both saved; this is
  interface adaptation, not an algorithmic repair of model output.

The source stays LLM-generated. Neither this replacement nor CEM changes it into
RoboVerify's native CFG/MCMC program-structure synthesis. Google perception,
PyBullet, `put_first_on_second` and raw-position access are deliberately excluded.

## Shared observations and actions

Tower policies receive opaque object identities, task roles and the existing
`ON_star`, frozen `ON_star_zero`, `Higher` and `Scattered` interpretations. Actions
are exactly the named Pick/Move/Release controllers in `synthesis/api/`.
Row policies receive the insertion-sort example's `next_star` and `keyless`
relations, frozen opaque row targets and one buffer target. They use that example's
TableRobot Pick/Move/Release controller. Neither adapter exposes keys as numbers,
raw observations, XYZ coordinates, rewards, goals, images or the simulator.
Tower Move accepts three object references and XYZ offsets, as the DSL does.
Row target handles are coordinate references, not an additional action macro.

The baseline uses oracle relational perception. No VLM is needed. This isolates
program generation from image-recognition errors. Each method restores the same
settled full MuJoCo snapshot, with no extra settling after restore. Generated code
runs in an AST interpreter, not `exec` or `eval`; all helpers and loops share
instruction, primitive, physics-step and wall-clock budgets. This API boundary
is not an operating-system security sandbox for arbitrary untrusted Python.

## Tasks and protocol

The core tasks are Stack, Unstack, Reverse, Partial Stack and Insertion Sort.
The four harder tasks are Tower Merge, Tower Split, Stable Merge and Sort Restore.
Merge and split check object-order preservation and protected objects' positions.
Row tasks include equal and negative keys and require stability. Sort Restore
requires a measured complete sorted checkpoint before the original identity
order is restored; a no-op cannot satisfy an unsorted case.

Defaults are five training scenes, five validation scenes and ten held-out scenes
per task. Seeds are disjoint. Tests include five-object scenes, while training
uses three or four. Policies are selected using validation results and frozen
before testing. Every failed generation and execution remains in the artifacts.
Prompts, helper-generation responses, model revisions, snapshots, selected source,
parameters, execution traces, solver results and per-case statuses are saved.

Three variants share the initial generated candidate pool:

- `cap_only`: two independently generated candidates, selected on validation.
- `cap_cem`: the selected source with bounded scalar-offset optimization using
  RoboVerify's existing CEM implementation on training scenes.
- `cap_roboverify`: CEM plus at most two local-model repairs using training
  failures and model-check feedback; validation selects among the candidates.

Hybrid methods have additional optimization and LLM-call budgets. The budget
artifact reports actual calls and physical rollouts; these runs do **not** support
an equal-cost causal claim. Identical source/parameters/snapshots are physically
executed once and the measured result is reused, with reuse recorded explicitly.
No test failure is used to tune parameters or repair code.
No expert demonstration is given to CaP. Training/validation/test refer to
initial scenes; recorded candidate trajectories are not expert demonstrations.
The examples, candidate selection and feedback rounds mean this is not a strict
single-shot zero-shot evaluation.

## Verification coverage

Tower checks invoke RoboVerify's `PrimitiveMotionProblem` on observed completed
primitive blocks, using every known physical obstacle. These are noiseless
finite-scene waypoint checks, not a proof of generated guards, unseen scenes,
termination, unspecified extra objects or controller tracking.

Row checks symbolically execute the actual generated AST with arbitrary integer
keys, including ties, for a fixed finite object universe and initial spatial
permutation. SAT comparator branches are explored and stable goals checked by Z3.
Transfers assume reliable primitives and empty destinations. A refutation's key
assignment is replayed physically before being reported as an executable witness.
This bounded checker is new; it is not the handwritten insertion-sort example's
invariant proof. `unknown`, `unsupported`, `inconsistent` and exhausted budgets
are never reported as `valid`. No new invariant learner is introduced.

Neither check supplies RoboVerify's full synthesis/learned-invariant/unbounded
proof pipeline for arbitrary Python. `cap_roboverify` means a **verification-guided
repair hybrid**, not independently verified synthesis and not `verified_model`.
All three variants still require measured physical task completion on held-out
cases. Existing Reverse/Partial motion limitations are unchanged.

## Local model deployment

Keep inference dependencies separate from the simulator environment. The tested
server uses Python 3.10, Transformers 4.51.3, Accelerate 1.6.0, Torch 2.5.1 with
CUDA 12.4, FastAPI 0.115.12 and Uvicorn 0.34.2. It runs BF16 on one CUDA GPU.
Models are Qwen2.5-Coder-3B-Instruct and Qwen2.5-Coder-7B-Instruct. The server
records the actual downloaded revision in `/health` and every generation.

From `roboverify/`, after creating a separate inference environment:

```bash
uv venv .runtime/llm --python python3.10
uv pip install --python .runtime/llm/bin/python torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .runtime/llm/bin/python transformers==4.51.3 accelerate==1.6.0 huggingface-hub==0.30.2 fastapi==0.115.12 uvicorn==0.34.2 numpy==1.26.4
CUDA_VISIBLE_DEVICES=0 .runtime/llm/bin/python -m synthesis.cap.serve_model --model Qwen/Qwen2.5-Coder-7B-Instruct --port 8123
```

The server binds localhost only and serializes generation. Use a second GPU and
port 8124 for 3B, or run models sequentially. Set `--revision` to the recorded
commit when reproducing results. API access uses a dependency-free local client.

## Evaluation commands

Use the simulator environment from [AGENTS.md](../../../AGENTS.md). A repository
local MuJoCo installation is supported by setting `MUJOCO_PY_MUJOCO_PATH` and its
`bin` directory in `LD_LIBRARY_PATH`; `patchelf` must be on PATH when compiling
`mujoco_py`. Do not mix the inference and simulator Torch environments.

```bash
uv run python -m synthesis.entry.evaluate_cap --implementation google --model-url http://127.0.0.1:8123 --run-name cap-google
uv run python -m synthesis.entry.evaluate_cap --implementation google --model-url http://127.0.0.1:8124 --run-name cap-google
uv run python -m synthesis.experiment.report --run runs/cap-google/latest
uv run python -m synthesis.cap.report --run runs/cap-google/latest
```

For a pipeline smoke, add `--tasks stack --initial-candidates 1 --repairs 1
--train-cases 1 --validation-cases 1 --test-cases 2 --cem-iterations 0` on one
line. Such a smoke is not the full benchmark.

Fast tests:

```bash
uv run python -m unittest synthesis.cap.test_runtime synthesis.cap.test_generator synthesis.cap.test_google_generator synthesis.cap.test_row_verification synthesis.cap.test_runner synthesis.benchmarks.test_specs -v
```

Physical adapter tests additionally require MuJoCo. Run directories use the shared
logger contract. Read metrics and logs using the bounded experiment report, not
by opening the JSONL or stdout files. `artifacts/summary.json` is the compact task
comparison; `episode_index.json` references individual execution artifacts.
The CaP report adds bounded per-task counts and separates controller, API,
goal, generation and setup failures. Setup errors are harness failures, not
evidence that a method cannot solve the task.
