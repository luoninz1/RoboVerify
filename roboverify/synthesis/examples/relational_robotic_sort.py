"""Two-loop, next*-only insertion sort with real pick/move/release actions.

``insertion_sort_relational`` is the restricted program. Everything else is
read-only predicate/perception plumbing, trace recording, or scene setup.
Guard witnesses are selected by quantified formulas, never by a next pointer.
Only setup_scene writes block poses; all sorting movement uses env.step().
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from synthesis.examples.relational_sort_controller import (
    SortVideo, TableRobot, block_position,
)

Vector3 = tuple[float, float, float]
INPUT_FIELDS = (
    "ctrl", "mocap_pos", "mocap_quat", "qacc_warmstart", "qfrc_applied", "xfrc_applied",
)


@dataclass(frozen=True)
class RelationalSortSnapshot:
    """An immutable measured row at a guard, including the false-guard exit.

    Program variables identify objects, not row slots. ``order`` is decoded
    from measured poses solely for the observation/relational semantics layer.
    ``positions`` and ``keys`` are tuples, avoiding mutable nested dictionaries.
    """

    objects: tuple[str, ...]
    keys: tuple[tuple[str, int], ...]
    order: tuple[str, ...]
    b: str
    x: str | None
    r: str | None
    program_point: str
    case_name: str
    positions: tuple[tuple[str, Vector3], ...]
    step: int
    held: str | None = None
    buffered: tuple[str, ...] = ()
    initial_order: tuple[str, ...] = ()

    @property
    def key_map(self) -> dict[str, int]:
        return dict(self.keys)

    @property
    def position_map(self) -> dict[str, Vector3]:
        return dict(self.positions)

    def next_star(self, u: str, v: str) -> bool:
        """Reflexive reachability in this snapshot's measured, finite row."""
        rank = {name: position for position, name in enumerate(self.order)}
        return u in rank and v in rank and rank[u] <= rank[v]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RelationalSortSnapshot:
        fields = dict(data)
        fields["objects"] = tuple(fields["objects"])
        fields["keys"] = tuple((str(k), int(v)) for k, v in fields["keys"])
        fields["order"] = tuple(fields["order"])
        fields["positions"] = tuple((str(k), tuple(float(q) for q in v))
                                    for k, v in fields["positions"])
        fields["buffered"] = tuple(fields.get("buffered", ()))
        fields["initial_order"] = tuple(fields.get("initial_order", ()))
        return cls(**fields)


def decode_locations(positions, slots, buffer_position, *, xy_tolerance=.012,
                     z_tolerance=.005) -> dict[str, int | str]:
    """Require one unambiguous measured row/buffer location for each block."""
    targets = np.asarray([*slots, buffer_position], dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 3 or not np.isfinite(targets).all():
        raise ValueError("Expected finite XYZ row and buffer targets.")
    locations = {}
    for name, position in positions.items():
        point = np.asarray(position, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError(f"Invalid measured position for {name}: {point}")
        matches = np.flatnonzero(
            (np.linalg.norm(targets[:, :2] - point[:2], axis=1) <= xy_tolerance)
            & (np.abs(targets[:, 2] - point[2]) <= z_tolerance))
        if len(matches) != 1:
            raise ValueError(f"Expected one physical location for {name}; got {matches} at {point}")
        location = int(matches[0])
        locations[name] = "buffer" if location == len(slots) else location
    if len(set(locations.values())) != len(locations):
        raise ValueError("Multiple blocks occupy the same measured location.")
    return locations


class RelationalScene:
    """Predicate oracle backed by measured poses, plus frozen action targets.

    Each guard captures a fresh scene. Row targets remain frozen throughout
    one inner body so that p can be placed at x's old slot after x is buffered.
    This is perception infrastructure, not additional program cursor storage.
    Arrays/ranks below implement relations and actuator coordinates only.
    """

    def __init__(self, objects, keys, slots, buffer_position, robot,
                 position_reader: Callable[[str], Sequence[float]], *,
                 initial_order=None, case_name="case"):
        self.objects = tuple(objects)
        self.keys = {name: int(keys[name]) for name in self.objects}
        self.slots = np.asarray(slots, dtype=float).copy()
        self.buffer_position = np.asarray(buffer_position, dtype=float).copy()
        if not self.objects or self.slots.shape != (len(self.objects), 3):
            raise ValueError("The relational program requires a nonempty row with one slot per object.")
        if len(set(self.objects)) != len(self.objects) or set(keys) != set(self.objects):
            raise ValueError("Object identities must be unique and every object needs a key.")
        self.robot, self.position_reader = robot, position_reader
        self.initial_order = tuple(initial_order or self.objects)
        self.case_name = case_name
        self.snapshots: list[RelationalSortSnapshot] = []
        self._rank: dict[str, int] = {}
        self._targets: dict[str, np.ndarray] = {}
        self._positions: dict[str, Vector3] = {}
        self.order: tuple[str, ...] = ()

    def refresh(self) -> None:
        """Read, validate, and freeze occupancy; never change block poses."""
        if self.robot.held is not None:
            raise ValueError("A loop guard must not run while holding a block.")
        positions = {name: tuple(float(q) for q in self.position_reader(name))
                     for name in self.objects}
        locations = decode_locations(positions, self.slots, self.buffer_position)
        if any(location == "buffer" for location in locations.values()):
            raise ValueError("The buffer must be empty at every loop guard.")
        if set(locations.values()) != set(range(len(self.objects))):
            raise ValueError("Every row slot must be occupied at a loop guard.")
        self._positions = positions
        self._rank = {name: int(location) for name, location in locations.items()}
        self.order = tuple(sorted(self.objects, key=self._rank.__getitem__))
        self._targets = {name: self.slots[self._rank[name]].copy() for name in self.objects}

    def next_star(self, u: str, v: str) -> bool:
        return self._rank[u] <= self._rank[v]

    def keyless(self, u: str, v: str) -> bool:
        return self.keys[u] < self.keys[v]

    def row_target(self, name: str) -> np.ndarray:
        """Canonical row target from the last guard, not a newly measured pose."""
        return self._targets[name].copy()

    def _record(self, point: str, b: str, x=None, r=None) -> None:
        self.snapshots.append(RelationalSortSnapshot(
            objects=self.objects, keys=tuple(self.keys.items()), order=self.order,
            b=b, x=x, r=r, program_point=point, case_name=self.case_name,
            positions=tuple(self._positions.items()), step=self.robot.steps,
            held=self.robot.held, buffered=(), initial_order=self.initial_order))

    def outer_guard(self, b: str) -> str | None:
        """∃x. b≠x ∧ b next* x ∧ ∀y. (b≠y ∧ b next* y) ⇒ x next* y."""
        self.refresh()
        witnesses = [x for x in self.objects
                     if b != x and self.next_star(b, x)
                     and all(not (b != y and self.next_star(b, y))
                             or self.next_star(x, y) for y in self.objects)]
        if len(witnesses) > 1:
            raise AssertionError("Outer guard is not functional on the measured list.")
        self._record("outer_head" if witnesses else "outer_exit", b)
        return witnesses[0] if witnesses else None

    def inner_guard(self, b: str, x: str, r: str) -> str | None:
        """∃p. p≠x ∧ p next* x ∧ ∀z. (p next* z ∧ z next* x)
        ⇒ (z=p ∨ z=x), conjoined with keyless(x,p).
        """
        self.refresh()
        witnesses = [p for p in self.objects
                     if p != x and self.next_star(p, x)
                     and all(not (self.next_star(p, z) and self.next_star(z, x))
                             or z == p or z == x for z in self.objects)
                     and self.keyless(x, p)]
        if len(witnesses) > 1:
            raise AssertionError("Inner guard is not functional on the measured list.")
        self._record("inner_head" if witnesses else "inner_exit", b, x, r)
        if witnesses:
            self.robot.description = f"Insert key {self.keys[x]}: pass key {self.keys[witnesses[0]]}"
        return witnesses[0] if witnesses else None


def insertion_sort_relational(head, scene, robot):
    """The exact restricted two-loop program; sorts the measured row in place.

    The walrus assignments bind existential guard witnesses. The only ordinary
    assignments initialize/update b and r at their allowed loop boundaries.
    Targets refer to the measured row frozen by the current inner guard.
    """
    b = head
    while (x := scene.outer_guard(b)) is not None:
        r = x
        while (p := scene.inner_guard(b, x, r)) is not None:
            robot.pick(x)
            robot.move(scene.buffer_position)
            robot.release(scene.buffer_position)
            robot.pick(p)
            robot.move(scene.row_target(x))
            robot.release(scene.row_target(x))
            robot.pick(x)
            robot.move(scene.row_target(p))
            robot.release(scene.row_target(p))
            r = b
        b = r


@dataclass(frozen=True)
class SimulatorBaseline:
    state: Any
    inputs: tuple[tuple[str, np.ndarray], ...]
    goal: np.ndarray


def capture_simulator_state(env) -> SimulatorBaseline:
    """Capture full MuJoCo state plus inputs omitted by legacy get_GT_state."""
    return SimulatorBaseline(deepcopy(env.sim.get_state()),
                             tuple((name, getattr(env.sim.data, name).copy())
                                   for name in INPUT_FIELDS),
                             np.asarray(env.goal).copy())


def restore_simulator_state(env, baseline: SimulatorBaseline) -> None:
    env.sim.set_state(deepcopy(baseline.state))
    for name, values in baseline.inputs:
        getattr(env.sim.data, name)[:] = values
    env.goal = baseline.goal.copy()
    env.sim.forward()


def create_environment(num_blocks=4, seed=23):
    """The reference notebook's Fetch scene and 0.11m-spaced table row."""
    from synthesis.environment.cee_us_env.runtime import configure_local_mujoco
    configure_local_mujoco()
    from synthesis.environment.cee_us_env.fpp_construction_env import FetchPickAndPlaceConstruction
    np.random.seed(seed)
    env = FetchPickAndPlaceConstruction(
        name="relational_robotic_insertion_sort", sparse=False, shaped_reward=False,
        num_blocks=num_blocks, reward_type="sparse", case="PickAndPlace", simple=True,
        visualize_target=False, visualize_mocap=False)
    env.reset()
    slots = np.column_stack((np.full(num_blocks, env.initial_gripper_xpos[0] + .10),
                             env.initial_gripper_xpos[1] + .11 * (np.arange(num_blocks) - (num_blocks - 1) / 2),
                             np.full(num_blocks, env.height_offset)))
    buffer_position = np.array([slots[0, 0] - .18, env.initial_gripper_xpos[1], env.height_offset])
    return env, slots, buffer_position


def setup_scene(env, slots, buffer_position, initial_order, keys, *, baseline=None):
    """Initialize a run; this is the only routine allowed to write object poses."""
    objects = tuple(str(name) for name in env.object_names)
    initial_order = tuple(initial_order)
    slots = np.asarray(slots, dtype=float)
    if sorted(initial_order) != sorted(objects) or set(keys) != set(objects):
        raise ValueError("Initial order and keys must cover the physical objects exactly.")
    if slots.shape != (len(objects), 3) or not np.isfinite(slots).all():
        raise ValueError("Expected one finite XYZ row slot for each physical object.")
    if np.asarray(buffer_position).shape != (3,) or not np.isfinite(buffer_position).all():
        raise ValueError("Expected one finite XYZ buffer position.")
    if baseline is not None:
        restore_simulator_state(env, baseline)
    positions = {name: slots[index] for index, name in enumerate(initial_order)}
    for name in objects:
        env.sim.data.set_joint_qpos(f"{name}:joint", np.r_[positions[name], 1., 0., 0., 0.])
        env.sim.data.set_joint_qvel(f"{name}:joint", np.zeros(6))
    env.goal = np.r_[np.asarray([positions[name] for name in objects]).ravel(), np.zeros(3)]
    env.sim.forward()
    return initial_order


def run_physical_case(env, slots, buffer_position, *, baseline=None, initial_order=None,
                      keys=None, key_sequence=None, seed=0, case_name="case", recorder=None):
    """Run and validate one measured physical simulation; return a result dict."""
    objects = tuple(str(name) for name in env.object_names)
    initial_order = tuple(initial_order) if initial_order is not None else tuple(
        str(name) for name in np.random.default_rng(seed).permutation(objects))
    if keys is not None and key_sequence is not None:
        raise ValueError("Specify keys or key_sequence, not both.")
    if key_sequence is not None:
        if len(key_sequence) != len(objects):
            raise ValueError("Expected one key for each initial row position.")
        keys = dict(zip(initial_order, key_sequence))
    if keys is None:
        keys = {name: index + 1 for index, name in enumerate(objects)}
    setup_scene(env, slots, buffer_position, initial_order, keys, baseline=baseline)
    robot = TableRobot(env, slots, keys, recorder=recorder)
    scene = RelationalScene(objects, keys, slots, buffer_position, robot,
                            lambda name: block_position(env, name),
                            initial_order=initial_order, case_name=case_name)
    if recorder is not None:
        recorder.pause(robot, seconds=.6)
    insertion_sort_relational(initial_order[0], scene, robot)
    final = scene.snapshots[-1]
    expected = tuple(sorted(initial_order, key=keys.__getitem__))
    if final.order != expected or final.buffered or final.held is not None:
        raise AssertionError(f"Physical result differs from stable sorted order: {final.order} != {expected}")
    final_positions = np.asarray([block_position(env, name) for name in final.order])
    position_errors = np.linalg.norm(final_positions - np.asarray(slots), axis=1)
    upright_cosines = {name: float(env.sim.data.get_site_xmat(name)[2, 2]) for name in objects}
    if any(value <= .98 for value in upright_cosines.values()):
        raise AssertionError(f"Final row includes a tilted block: {upright_cosines}")
    if any(np.linalg.norm(block_position(env, name) - buffer_position) <= .05 for name in objects):
        raise AssertionError("A final block remains close to the buffer.")
    robot.description, robot.phase = "Sorted row verified from measured poses", "Complete"
    if recorder is not None:
        recorder.pause(robot, seconds=1.)
    return {"case_name": case_name, "objects": objects, "keys": tuple(keys.items()),
            "initial_order": initial_order, "final_order": final.order,
            "snapshots": tuple(scene.snapshots), "steps": robot.steps,
            "swaps": sum(s.program_point == "inner_head" for s in scene.snapshots),
            "max_position_error": float(np.max(position_errors)),
            "min_upright_cosine": min(upright_cosines.values()),
            "transfers": tuple(deepcopy(robot.transfers))}


def main():
    """Optional standalone physical smoke run / recording entry point."""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keys", default="4,3,2,1", help="Keys in initial row order")
    parser.add_argument("--seed", default=23, type=int)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sequence = tuple(int(value) for value in args.keys.split(","))
    env, slots, buffer_position = create_environment(len(sequence), seed=args.seed)
    recorder = None
    try:
        keys = dict(zip(env.object_names, sequence))
        if args.video:
            recorder = SortVideo(env, keys, slots, buffer_position, args.video, stride=5)
        result = run_physical_case(
            env, slots, buffer_position, baseline=capture_simulator_state(env),
            initial_order=tuple(env.object_names), keys=keys,
            case_name="standalone_" + "_".join(map(str, sequence)), recorder=recorder)
        payload = {**result, "snapshots": [snapshot.as_dict() for snapshot in result["snapshots"]]}
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps({name: value for name, value in payload.items()
                          if name not in ("snapshots", "transfers")}, indent=2))
    finally:
        try:
            if recorder is not None:
                recorder.close()
        finally:
            env.close()


if __name__ == "__main__":
    main()
