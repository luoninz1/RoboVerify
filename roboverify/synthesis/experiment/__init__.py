"""Run-directory logging and reporting for RoboVerify experiments.

The package provides a fixed on-disk contract for a single experiment run so that
progress can be inspected at constant cost, regardless of how long the run is:

``config.json``
    Resolved configuration, written once.
``status.json``
    Overwritten every update, so it stays one small JSON object forever.
``metrics.jsonl``
    One flat record per iteration; aggregate it, do not read it line by line.
``events.jsonl``
    Rare, semantically important moments only.
``stdout.log``
    The unfiltered firehose, captured at file-descriptor level; grep it.
``result.json``
    Final verdict, written once at exit.

Read a run with ``python -m synthesis.experiment.report --run <run_dir>`` rather than
by opening these files directly.
"""

from synthesis.experiment.run_logger import RollingRate, RunLogger, git_info

__all__ = ["RunLogger", "RollingRate", "git_info"]
