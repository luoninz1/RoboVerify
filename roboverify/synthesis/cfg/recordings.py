"""Current full-state NPZ archives. Old demonstration formats are not supported."""

import json
from pathlib import Path

import numpy as np

from synthesis.cfg.demos import DemoTrace
from synthesis.cfg.reset import Snapshot

FORMAT = "roboverify-demonstrations"
VERSION = 2


def _json(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def check_trace(trace):
    count = len(trace.states)
    if not count or len(trace.snapshots) != count or len(trace.actions) != 2:
        raise ValueError(
            "Every trajectory needs observations, snapshots, and actions; recollect demonstrations"
        )
    actions, indices = trace.actions
    if (
        len(indices) != count
        or any(not 0 <= i <= len(actions) for i in indices)
        or list(indices) != sorted(indices)
    ):
        raise ValueError("Invalid observation-to-action indices")
    if indices[0] != 0:
        raise ValueError("Recording must start before its first action")
    states = np.asarray(trace.states)
    if states.ndim != 2 or not np.isfinite(states).all():
        raise ValueError("Observations must be a finite two-dimensional array")
    for event in trace.events:
        if not 0 <= event["index"] < count:
            raise ValueError("Event points outside the recording")
        if "entry_index" in event and not 0 <= event["entry_index"] <= event["index"]:
            raise ValueError("Invalid frozen loop-entry index")


def save_traces(path, traces):
    if not traces:
        raise ValueError("Cannot save an empty demonstration archive")
    arrays, metadata = {}, []
    for i, trace in enumerate(traces):
        check_trace(trace)
        key = f"d{i}"
        arrays[key + "_states"] = np.asarray(trace.states)
        arrays[key + "_gt"] = np.stack([s.gt_state for s in trace.snapshots])
        names = sorted(trace.snapshots[0].arrays)
        for name in names:
            arrays[key + "_" + name] = np.stack(
                [s.arrays[name] for s in trace.snapshots]
            )
        actions, indices = trace.actions
        arrays[key + "_actions"] = np.asarray(actions)
        arrays[key + "_indices"] = np.asarray(indices, dtype=np.int64)
        metadata.append(
            dict(
                seed=trace.seed,
                task=trace.task,
                num_blocks=trace.num_blocks,
                bindings=[s.bindings for s in trace.snapshots],
                arrays=names,
                events=trace.events,
                metadata=trace.metadata,
            )
        )
    arrays["metadata"] = np.asarray(
        json.dumps(
            dict(format=FORMAT, version=VERSION, traces=metadata),
            default=_json,
            allow_nan=False,
        )
    )
    path = Path(path)
    # Write atomically and do not append .npz implicitly to user paths.
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_traces(path, *, require_valid=False):
    if Path(path).suffix != ".npz":
        raise ValueError(
            "Use the current .npz archive; recollect with synthesis.entry.collect_demos"
        )
    try:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            if meta.get("format") != FORMAT or meta.get("version") != VERSION:
                raise ValueError(
                    "Unsupported demonstration format; recollect with synthesis.entry.collect_demos"
                )
            traces = []
            for i, row in enumerate(meta["traces"]):
                key = f"d{i}"
                # NPZ indexing decompresses the entire member. Read each array
                # once per trajectory rather than once per observation.
                snapshot_arrays = {
                    name: data[key + "_" + name] for name in row["arrays"]
                }
                snapshots = tuple(
                    Snapshot(
                        gt.copy(),
                        {
                            name: values[t].copy()
                            for name, values in snapshot_arrays.items()
                        },
                        row["bindings"][t],
                    )
                    for t, gt in enumerate(data[key + "_gt"])
                )
                trace = DemoTrace(
                    tuple(data[key + "_states"].copy()),
                    snapshots,
                    (
                        tuple(data[key + "_actions"].copy()),
                        tuple(data[key + "_indices"].tolist()),
                    ),
                    row["seed"],
                    row["task"],
                    row["num_blocks"],
                    tuple(row["events"]),
                    row["metadata"],
                )
                check_trace(trace)
                if require_valid and trace.metadata.get("status") != "valid":
                    raise ValueError(
                        "Archive contains unvalidated or incomplete demonstrations"
                    )
                traces.append(trace)
            if not traces:
                raise ValueError("Empty demonstration archive")
            return traces
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(
            "Invalid demonstration archive; recollect with synthesis.entry.collect_demos"
        ) from exc


def loop_store(traces, *, loop_id=None, names=None):
    """Adapt recorded runtime loop heads and normal exits to the in-memory learner."""
    from synthesis.inference_lib.demo_store import (
        DemoStore,
        LoopHeadState,
        observation_positions,
    )

    store = DemoStore()
    for trace in traces:
        for event in trace.events:
            if event["kind"] not in ("loop_head", "loop_exit") or (
                loop_id is not None and event["path"] != loop_id
            ):
                continue
            bindings = dict(event["bindings"])
            if names is not None:
                bindings = {
                    k: v for k, v in bindings.items() if k in names or k == "tbl"
                }
            else:
                bindings = {
                    k: v
                    for k, v in bindings.items()
                    if k not in event.get("witnesses", ())
                }
            store.add(
                LoopHeadState.from_observation(
                    event["path"],
                    trace.states[event["index"]],
                    observation_positions(
                        trace.states[event["entry_index"]], trace.num_blocks
                    ),
                    bindings,
                    trace.num_blocks,
                )
            )
    return store
