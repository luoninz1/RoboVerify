"""Run the Google CaP port / CEM / RoboVerify-repair comparison."""

import argparse

from synthesis.benchmarks.specs import TASKS
from synthesis.cap.client import LocalModel
from synthesis.cap.runner import EvaluationConfig, run_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-url", default="http://127.0.0.1:8123")
    parser.add_argument(
        "--implementation", choices=("google", "legacy"), default="google"
    )
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--run-name", default="cap")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-cases", type=int, default=5)
    parser.add_argument("--validation-cases", type=int, default=5)
    parser.add_argument("--test-cases", type=int, default=10)
    parser.add_argument("--initial-candidates", type=int, default=2)
    parser.add_argument("--repairs", type=int, default=2)
    parser.add_argument("--cem-iterations", type=int, default=1)
    parser.add_argument("--cem-samples", type=int, default=4)
    parser.add_argument("--cem-elites", type=int, default=2)
    parser.add_argument("--cem-cases", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=3072)
    parser.add_argument("--helper-budget", type=int, default=8)
    parser.add_argument("--max-calls", type=int, default=300)
    parser.add_argument("--max-physics-steps", type=int, default=20000)
    parser.add_argument("--deadline-seconds", type=int, default=60)
    parser.add_argument("--verification-paths", type=int, default=512)
    parser.add_argument("--verification-timeout-seconds", type=int, default=15)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()
    fields = {key: value for key, value in vars(args).items() if key != "model_url"}
    fields["tasks"] = tuple(fields["tasks"])
    run_evaluation(EvaluationConfig(**fields), LocalModel(args.model_url))


if __name__ == "__main__":
    main()
