"""Paired, held-out physical evaluation and bounded verification-guided repair."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from synthesis.benchmarks.specs import ROW_TASKS, TASKS, manifest
from synthesis.cap.generator import Generator
from synthesis.cap.google_generator import GoogleGenerator, provenance
from synthesis.cap.prompts import task_prompt
from synthesis.cap.row_verification import verify_row_policy
from synthesis.cap.verification import verify_tower_trace
from synthesis.experiment.run_logger import RunLogger


@dataclass(frozen=True)
class EvaluationConfig:
    tasks: tuple[str, ...] = TASKS
    implementation: str = "google"
    train_cases: int = 5
    validation_cases: int = 5
    test_cases: int = 10
    initial_candidates: int = 2
    repairs: int = 2
    seed: int = 42
    max_tokens: int = 3072
    helper_budget: int = 8
    temperature: float = 0.2
    cem_iterations: int = 1
    cem_samples: int = 4
    cem_elites: int = 2
    cem_cases: int = 2
    max_calls: int = 300
    max_physics_steps: int = 20000
    deadline_seconds: int = 60
    verification_timeout_seconds: int = 15
    verification_paths: int = 512
    output_root: str = "runs"
    run_name: str = "cap"

    def validate(self):
        if self.implementation not in ("google", "legacy"):
            raise ValueError("Unknown CaP implementation")
        if not self.tasks or any(task not in TASKS for task in self.tasks):
            raise ValueError("Unknown or empty benchmark task set")
        if (
            min(
                self.train_cases,
                self.validation_cases,
                self.test_cases,
                self.initial_candidates,
                self.max_tokens,
                self.cem_cases,
                self.max_calls,
                self.max_physics_steps,
                self.deadline_seconds,
                self.verification_timeout_seconds,
                self.verification_paths,
            )
            < 1
        ):
            raise ValueError("Counts and budgets must be positive")
        if min(self.repairs, self.cem_iterations, self.helper_budget) < 0:
            raise ValueError("Optional budgets cannot be negative")
        if not 1 <= self.cem_elites <= self.cem_samples:
            raise ValueError("CEM needs 1 <= elites <= samples")


def dumps(value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)

    return json.dumps(value, indent=2, default=convert, allow_nan=False) + "\n"


def score(executions):
    if not executions:
        return (0.0, 0.0)
    # Exact physical completion is primary. Partial goal scores only break ties;
    # an exception never earns exact success despite an accidentally satisfied goal.
    return (
        sum(bool(row["success"]) for row in executions) / len(executions),
        sum(row.get("goal", {}).get("score", 0.0) for row in executions)
        / len(executions),
    )


def feedback_for(executions, verification):
    failed = next((row for row in executions if not row["success"]), None)
    compact = {key: value for key, value in verification.items() if key != "cases"}
    if "cases" in verification:
        first_failed = next(
            (row for row in verification["cases"] if row["status"] != "valid"), None
        )
        if first_failed:
            compact["first_failed_trace"] = {
                key: value
                for key, value in first_failed.items()
                if key not in ("checks", "failures")
            }
            compact["first_failed_trace"]["failures"] = first_failed.get(
                "failures", []
            )[:2]
    feedback = {"verification": compact}
    if failed:
        feedback["physical_training_failure"] = {
            key: failed.get(key)
            for key in (
                "case",
                "status",
                "error",
                "goal",
                "scene",
                "final_positions",
                "parameter_values",
            )
        }
        feedback["last_primitives"] = failed.get("trace", [])[-8:]
    # Solver witnesses are geometric abstractions, not physical observations.
    # Rows with arbitrary-key counterexamples are physically replayed separately.
    return feedback


class Evaluation:
    def __init__(self, config, model, suite, logger):
        self.config, self.model, self.suite, self.logger = config, model, suite, logger
        generator = GoogleGenerator if config.implementation == "google" else Generator
        self.generator = generator(
            model,
            helper_budget=config.helper_budget,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )
        self.cache, self.archived_starts = {}, set()
        self.rollout_count = self.generation_count = self.step = 0
        self.rows = []

    def artifact(self, name, data):
        return self.logger.write_artifact(name, dumps(data))

    def generate(self, task, seed, name, *, feedback=None, previous=None):
        self.logger.set_status(phase="generate", task=task, candidate=name)
        start = perf_counter()
        try:
            candidate = self.generator.generate(
                task, seed=seed, feedback=feedback, previous=previous
            )
        except Exception as exc:
            candidate = {
                "task": task,
                "seed": seed,
                "source": None,
                "generation": [],
                "status": "generation_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        self.generation_count += len(candidate["generation"])
        candidate.update(id=name, generation_seconds=perf_counter() - start)
        self.artifact(f"{task}/generation/{name}.json", candidate)
        if candidate["source"]:
            self.logger.write_artifact(
                f"{task}/policies/{name}.py", candidate["source"]
            )
        return candidate

    def execute(self, case, candidate, parameters=None, *, phase, method, keys=None):
        signature = hashlib.sha256(
            dumps(
                [case.as_dict(), candidate.get("source"), parameters or {}, keys]
            ).encode()
        ).hexdigest()
        reused = signature in self.cache
        if not reused:
            self.logger.set_status(
                phase=phase, task=case.task, method=method, case=case.name
            )
            if candidate.get("source") is None:
                result = {
                    "case": case.as_dict(),
                    "status": "generation_error",
                    "success": False,
                    "error": candidate.get("error"),
                    "primitive_calls": 0,
                    "physics_steps": 0,
                    "elapsed_seconds": 0.0,
                }
            else:
                result = self.suite.run(
                    case,
                    candidate["source"],
                    parameters=parameters,
                    keys_override=keys,
                    max_calls=self.config.max_calls,
                    max_physics_steps=self.config.max_physics_steps,
                    deadline_seconds=self.config.deadline_seconds,
                )
                self.rollout_count += 1
            path = f"{case.task}/executions/{signature}.json"
            self.artifact(path, result)
            self.cache[signature] = (result, path)
            if case.name in self.suite.starts and case.name not in self.archived_starts:
                self.suite.save_start(
                    case,
                    self.logger.artifact_dir(case.task, "starts")
                    / (case.name + ".npz"),
                )
                self.artifact(
                    f"{case.task}/starts/{case.name}.json",
                    self.suite.starts[case.name][0].as_dict(),
                )
                self.archived_starts.add(case.name)
        result, path = self.cache[signature]
        self.step += 1
        row = {
            "task": case.task,
            "case": case.name,
            "split": case.split,
            "phase": phase,
            "method": method,
            "candidate": candidate["id"],
            "parameters": parameters or {},
            "success": bool(result["success"]),
            "status": result["status"],
            "primitive_calls": result["primitive_calls"],
            "physics_steps": result["physics_steps"],
            "elapsed_seconds": result["elapsed_seconds"],
            "reused_identical_execution": reused,
            "artifact": path,
        }
        self.rows.append(row)
        self.logger.log_metrics(
            self.step,
            phase=phase,
            task=case.task,
            method=method,
            success=int(result["success"]),
            elapsed_s=result["elapsed_seconds"],
            primitive_calls=result["primitive_calls"],
            physics_steps=result["physics_steps"],
            failure_kind=result["status"],
            reused=int(reused),
        )
        return result

    def evaluate_cases(self, cases, candidate, parameters=None, *, phase, method):
        return [
            self.execute(case, candidate, parameters, phase=phase, method=method)
            for case in cases
        ]

    def tune(self, task, candidate, cases, definitions, seed):
        if (
            not definitions
            or self.config.cem_iterations == 0
            or not candidate["source"]
        ):
            return {}
        from synthesis.mcmc.cem import cem_optimize

        names = sorted(definitions)
        defaults = np.asarray([definitions[name][0] for name in names])
        low = np.asarray([definitions[name][1] for name in names])
        high = np.asarray([definitions[name][2] for name in names])
        initial = (defaults - low) / (high - low)

        def objective(vector):
            values = dict(
                zip(names, (low + np.clip(vector, 0.0, 1.0) * (high - low)).tolist())
            )
            results = self.evaluate_cases(
                cases[: self.config.cem_cases],
                candidate,
                values,
                phase="tuning",
                method="cap_cem",
            )
            exact, partial = score(results)
            return exact + 0.01 * partial

        previous = np.random.get_state()
        try:
            np.random.seed(seed)
            _, vector = cem_optimize(
                objective,
                len(names),
                iterations=self.config.cem_iterations,
                N=self.config.cem_samples,
                K=self.config.cem_elites,
                init_mu=initial,
                init_std=0.15,
                num_workers=0,
            )
        finally:
            np.random.set_state(previous)
        return dict(
            zip(names, (low + np.clip(vector, 0.0, 1.0) * (high - low)).tolist())
        )

    def verify(self, task, candidate, parameters, cases, training):
        self.logger.set_status(
            phase="verification", task=task, candidate=candidate["id"]
        )
        if candidate["source"] is None:
            return {"status": "unsupported", "reason": "No generated executable policy"}
        if task in ROW_TASKS:
            first = next((row for row in training if "scene" in row), None)
            if first is None:
                return {"status": "unsupported", "reason": "No prepared row scene"}
            result = verify_row_policy(
                candidate["source"],
                first["scene"],
                parameters=parameters,
                max_paths=self.config.verification_paths,
                timeout_seconds=self.config.verification_timeout_seconds,
            )
            if result["status"] == "refuted" and result.get("keys"):
                index = training.index(first)
                replay = self.execute(
                    cases[index],
                    candidate,
                    parameters,
                    phase="counterexample_replay",
                    method="cap_roboverify",
                    keys=result["keys"],
                )
                result["physical_witness_replay"] = {
                    "success": replay["success"],
                    "status": replay["status"],
                    "error": replay.get("error"),
                    "goal": replay.get("goal"),
                }
        else:
            from synthesis.cfg.collection import execution_deadline

            results = []
            for row in training:
                try:
                    with execution_deadline(self.config.verification_timeout_seconds):
                        results.append(verify_tower_trace(row, timeout_ms=500))
                except TimeoutError:
                    results.append(
                        {
                            "status": "unknown",
                            "reason": "Trace verification wall-clock budget exhausted",
                        }
                    )
            failures = [row for row in results if row["status"] != "valid"]
            result = {
                "status": failures[0]["status"] if failures else "valid",
                "cases": results,
                "scope": "observed finite-scene traces only; guards and unseen scenes unproved",
            }
        self.artifact(f"{task}/verification/{candidate['id']}.json", result)
        return result

    def task(self, task, task_index):
        cases = manifest(
            task,
            train=self.config.train_cases,
            validation=self.config.validation_cases,
            test=self.config.test_cases,
        )
        self.artifact(f"{task}/manifest.json", [case.as_dict() for case in cases])
        self.logger.write_artifact(f"{task}/prompt.txt", task_prompt(task))
        train = [case for case in cases if case.split == "train"]
        validation = [case for case in cases if case.split == "validation"]
        test = [case for case in cases if case.split == "test"]
        seed = self.config.seed + task_index * 100
        pool = []
        for index in range(self.config.initial_candidates):
            candidate = self.generate(task, seed + index, f"initial-{index}")
            results = self.evaluate_cases(
                validation, candidate, phase="selection", method="cap_only"
            )
            pool.append((score(results), candidate))
        # Stable selection: preserve the earliest candidate for exact ties.
        _, plain = max(pool, key=lambda item: item[0])
        training = self.evaluate_cases(
            train, plain, phase="training", method="cap_only"
        )
        definitions = {}
        for execution in training:
            definitions.update(execution.get("parameters", {}))
        tuned = self.tune(task, plain, train, definitions, seed + 30)
        tuned_training = self.evaluate_cases(
            train, plain, tuned, phase="training", method="cap_cem"
        )
        tuned_validation = self.evaluate_cases(
            validation, plain, tuned, phase="selection", method="cap_cem"
        )
        verification = self.verify(task, plain, tuned, train, tuned_training)
        hybrid_pool = [(score(tuned_validation), plain, tuned, verification)]
        previous, previous_params, previous_training, previous_verification = (
            plain,
            tuned,
            tuned_training,
            verification,
        )
        for repair in range(self.config.repairs):
            if (
                all(row["success"] for row in previous_training)
                and previous_verification["status"] == "valid"
            ):
                break
            feedback = feedback_for(previous_training, previous_verification)
            if previous["source"] is None:
                feedback["generation_failure"] = previous.get("error")
            # Feedback is drawn only from TRAIN scenes and model checks.
            candidate = self.generate(
                task,
                seed + 50 + repair,
                f"repair-{repair}",
                feedback=dumps(feedback),
                previous=previous["source"]
                or (
                    previous["generation"][-1].get("response", {}).get("text", "")
                    if previous["generation"]
                    else ""
                ),
            )
            repaired_train = self.evaluate_cases(
                train, candidate, phase="training", method="cap_roboverify"
            )
            new_definitions = {}
            for execution in repaired_train:
                new_definitions.update(execution.get("parameters", {}))
            params = self.tune(
                task, candidate, train, new_definitions, seed + 70 + repair
            )
            if params:
                repaired_train = self.evaluate_cases(
                    train, candidate, params, phase="training", method="cap_roboverify"
                )
            checked = self.verify(task, candidate, params, train, repaired_train)
            results = self.evaluate_cases(
                validation,
                candidate,
                params,
                phase="selection",
                method="cap_roboverify",
            )
            hybrid_pool.append((score(results), candidate, params, checked))
            previous, previous_params, previous_training, previous_verification = (
                candidate,
                params,
                repaired_train,
                checked,
            )
        # Validation performance is primary; model-check status breaks performance
        # ties but does not establish correctness on new scenes.
        selected = max(
            hybrid_pool, key=lambda item: (*item[0], item[3]["status"] == "valid")
        )
        _, hybrid, hybrid_params, hybrid_verification = selected
        selection = {
            "cap_only": {"candidate": plain["id"], "parameters": {}},
            "cap_cem": {"candidate": plain["id"], "parameters": tuned},
            "cap_roboverify": {
                "candidate": hybrid["id"],
                "parameters": hybrid_params,
                "training_verification": hybrid_verification,
            },
        }
        self.artifact(f"{task}/selection.json", selection)
        summary = {}
        # Policies and parameters are now frozen. Tests cannot trigger repairs.
        for method, candidate, params in (
            ("cap_only", plain, {}),
            ("cap_cem", plain, tuned),
            ("cap_roboverify", hybrid, hybrid_params),
        ):
            results = self.evaluate_cases(
                test, candidate, params, phase="held_out", method=method
            )
            summary[method] = {
                "successes": sum(bool(row["success"]) for row in results),
                "cases": len(results),
                "success_rate": score(results)[0],
                "failures": dict(
                    Counter(row["status"] for row in results if not row["success"])
                ),
                "primitive_calls": sum(row["primitive_calls"] for row in results),
                "physics_steps": sum(row["physics_steps"] for row in results),
                "execution_seconds": sum(row["elapsed_seconds"] for row in results),
                "candidate": candidate["id"],
            }
        summary["verification_status"] = hybrid_verification["status"]
        self.artifact(f"{task}/summary.json", summary)
        self.logger.progress_line(
            f"{task}: CaP {summary['cap_only']['successes']}/{len(test)}, "
            f"CaP+CEM {summary['cap_cem']['successes']}/{len(test)}, "
            f"CaP+RoboVerify {summary['cap_roboverify']['successes']}/{len(test)} "
            f"(model check: {hybrid_verification['status']})"
        )
        return summary


def run_evaluation(config, model, *, suite=None):
    config.validate()
    if suite is None:
        from synthesis.benchmarks.physical import PhysicalSuite

        suite = PhysicalSuite()
    metadata = model.health()
    slug = metadata["model"].split("/")[-1].lower()
    resolved = {
        **asdict(config),
        "generator": (
            provenance()
            if config.implementation == "google"
            else {
                "implementation": "legacy",
                "method": "independent Google-style reimplementation",
            }
        ),
        "model": metadata,
        "protocol": "oracle predicates; paired full snapshots; frozen held-out tests",
        "budget_comparison": "hybrid has additional repair and optimization budget; no equal-cost causal claim",
        "verification_scope": "bounded row-policy paths or observed tower waypoint traces; not unbounded synthesis",
    }
    with RunLogger(config.output_root, config.run_name, resolved, slug=slug) as logger:
        evaluation = Evaluation(config, model, suite, logger)
        logger.progress_line(f"CaP run: {logger.run_dir}")
        summaries = {}
        try:
            for index, task in enumerate(config.tasks):
                summaries[task] = evaluation.task(task, index)
                evaluation.artifact("summary.json", summaries)
                evaluation.artifact("episode_index.json", evaluation.rows)
            evaluation.artifact(
                "budget.json",
                {
                    "llm_calls": evaluation.generation_count,
                    "physical_rollouts": evaluation.rollout_count,
                    "records_including_reuse": len(evaluation.rows),
                },
            )
            logger.finish(
                "completed",
                tasks=len(summaries),
                llm_calls=evaluation.generation_count,
                physical_rollouts=evaluation.rollout_count,
                summary="artifacts/summary.json",
            )
        finally:
            suite.close()
        return Path(logger.run_dir), summaries
