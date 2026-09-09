"""Immutable loop-boundary observations from the actual Fetch sorting program.

The observer reads MuJoCo poses. It never derives physical occupancy from the
sorting program's ``order`` list, and it never moves a block. Scene writes in
``run_physical_case`` happen only when initializing a new execution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from synthesis.examples.robotic_insertion_sort import TableRobot, insertion_sort_blocks

Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]
Location = int | str
INPUT_FIELDS = (
    "ctrl", "mocap_pos", "mocap_quat", "qacc_warmstart", "qfrc_applied", "xfrc_applied",
)


def decode_locations(
    positions: Mapping[str, Sequence[float]],
    slots: Sequence[Sequence[float]],
    buffer_position: Sequence[float],
    *,
    xy_tolerance: float = 0.012,
    z_tolerance: float = 0.005,
) -> dict[str, Location]:
    """Decode each measured pose into exactly one row slot or ``"buffer"``.

    Missing or ambiguous matches are errors, rather than silently choosing the
    nearest slot. Distinct blocks may decode to the same location: conservation
    and occupancy uniqueness are separate properties for the caller to check.
    """
    targets = np.asarray([*slots, buffer_position], dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 3 or not np.isfinite(targets).all():
        raise ValueError("Slot and buffer targets must be finite XYZ positions.")
    if xy_tolerance <= 0 or z_tolerance <= 0:
        raise ValueError("Position tolerances must be positive.")
    result = {}
    for name, position in positions.items():
        point = np.asarray(position, dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError(f"Invalid measured position for {name}: {point}")
        matches = np.flatnonzero(
            (np.linalg.norm(targets[:, :2] - point[:2], axis=1) <= xy_tolerance)
            & (np.abs(targets[:, 2] - point[2]) <= z_tolerance)
        )
        if len(matches) != 1:
            reason = "unlocated" if len(matches) == 0 else "ambiguous"
            raise ValueError(f"{name} has {reason} physical occupancy at {point.tolist()}")
        index = int(matches[0])
        result[name] = "buffer" if index == len(targets) - 1 else index
    return result


@dataclass(frozen=True)
class RoboticSortSnapshot:
    """One immutable completed loop boundary; names retain physical identity."""

    program_point: str
    i: int
    j: int | None
    selected: str | None
    order: tuple[str | None, ...]
    keys: tuple[tuple[str, int], ...]
    initial_order: tuple[str, ...]
    positions: tuple[tuple[str, Vector3], ...]
    rotations: tuple[tuple[str, Matrix3], ...]
    held: str | None
    step: int
    slots: tuple[Vector3, ...]
    buffer_position: Vector3
    case_name: str = "case"

    @property
    def objects(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.keys)

    @property
    def key_map(self) -> dict[str, int]:
        return dict(self.keys)

    @property
    def position_map(self) -> dict[str, Vector3]:
        return dict(self.positions)

    @property
    def rotation_map(self) -> dict[str, Matrix3]:
        return dict(self.rotations)

    @property
    def locations(self) -> dict[str, Location]:
        return decode_locations(self.position_map, self.slots, self.buffer_position)

    @property
    def slot_map(self) -> dict[str, int | None]:
        return {name: location if isinstance(location, int) else None
                for name, location in self.locations.items()}

    @property
    def buffered(self) -> frozenset[str]:
        return frozenset(name for name, location in self.locations.items() if location == "buffer")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RoboticSortSnapshot:
        """Restore immutable tuples after a ``dataclasses.asdict`` JSON round trip."""
        def tuples(value):
            return tuple(tuples(item) for item in value) if isinstance(value, (list, tuple)) else value

        return cls(**{name: tuples(value) for name, value in data.items()})


class PhysicalTraceRecorder:
    """Read-only observer for ``insertion_sort_blocks(observer=...)``."""

    def __init__(self, env, slots, buffer_position, keys, initial_order, robot, *, case_name="case"):
        self.env = env
        self.robot = robot
        self.objects = tuple(str(name) for name in env.object_names)
        self.keys = tuple((name, int(keys[name])) for name in self.objects)
        self.initial_order = tuple(initial_order)
        self.slots = tuple(tuple(float(value) for value in slot) for slot in slots)
        self.buffer_position = tuple(float(value) for value in buffer_position)
        self.case_name = case_name
        self._snapshots: list[RoboticSortSnapshot] = []
        if set(keys) != set(self.objects) or sorted(self.initial_order) != sorted(self.objects):
            raise ValueError("Keys and initial order must cover each physical block exactly once.")

    @property
    def snapshots(self) -> tuple[RoboticSortSnapshot, ...]:
        return tuple(self._snapshots)

    def __call__(self, program_point, *, i, j, selected, order) -> None:
        snapshot = RoboticSortSnapshot(
            program_point=program_point, i=i, j=j, selected=selected,
            order=tuple(order), keys=self.keys, initial_order=self.initial_order,
            positions=tuple((name, tuple(float(x) for x in self.env.sim.data.get_site_xpos(name)))
                            for name in self.objects),
            rotations=tuple((name, tuple(tuple(float(x) for x in row)
                                        for row in self.env.sim.data.get_site_xmat(name)))
                            for name in self.objects),
            held=self.robot.held, step=self.robot.steps,
            slots=self.slots, buffer_position=self.buffer_position, case_name=self.case_name,
        )
        # A loop-boundary sample needs an unambiguous physical interpretation.
        # Logical alignment, sorting, and conservation are checked separately.
        snapshot.locations
        self._snapshots.append(snapshot)


@dataclass(frozen=True)
class SimulatorBaseline:
    """Full simulator state plus inputs omitted by legacy ``get_GT_state``."""

    state: Any
    inputs: tuple[tuple[str, np.ndarray], ...]
    goal: np.ndarray


def capture_simulator_state(env) -> SimulatorBaseline:
    return SimulatorBaseline(
        state=deepcopy(env.sim.get_state()),
        inputs=tuple((name, getattr(env.sim.data, name).copy()) for name in INPUT_FIELDS),
        goal=np.asarray(env.goal).copy(),
    )


def restore_simulator_state(env, baseline: SimulatorBaseline) -> None:
    """Restore time, qpos/qvel/act, simulator inputs, and the environment goal."""
    env.sim.set_state(deepcopy(baseline.state))
    for name, values in baseline.inputs:
        getattr(env.sim.data, name)[:] = values
    env.goal = baseline.goal.copy()
    env.sim.forward()


@dataclass(frozen=True)
class PhysicalSortRun:
    case_name: str
    initial_order: tuple[str, ...]
    keys: tuple[tuple[str, int], ...]
    final_order: tuple[str, ...]
    snapshots: tuple[RoboticSortSnapshot, ...]
    steps: int
    transfers: tuple[dict[str, Any], ...]


def run_physical_case(
    env,
    slots,
    buffer_position,
    *,
    baseline: SimulatorBaseline,
    initial_order: Sequence[str] | None = None,
    keys: Mapping[str, int] | None = None,
    key_sequence: Sequence[int] | None = None,
    seed: int = 0,
    case_name: str = "case",
) -> PhysicalSortRun:
    """Reset one simulator, initialize a case, and trace real pick/move/release.

    ``key_sequence`` assigns keys by initial row position, allowing duplicate
    keys. Alternatively, ``keys`` maps physical block names to keys. When no
    ``initial_order`` is supplied, ``seed`` chooses a permutation of the blocks.
    The helper does not render video or replace physical transfers with moves
    of simulator joint positions.
    """
    objects = tuple(str(name) for name in env.object_names)
    slots = np.asarray(slots, dtype=float)
    buffer_position = np.asarray(buffer_position, dtype=float)
    if slots.shape != (len(objects), 3) or buffer_position.shape != (3,):
        raise ValueError("Provide one XYZ row slot per block and one XYZ buffer position.")
    if not np.isfinite(slots).all() or not np.isfinite(buffer_position).all():
        raise ValueError("Row and buffer positions must be finite.")
    initial_order = (tuple(str(name) for name in np.random.default_rng(seed).permutation(objects))
                     if initial_order is None else tuple(initial_order))
    if sorted(initial_order) != sorted(objects):
        raise ValueError("Initial order must be a permutation of all physical blocks.")
    if keys is not None and key_sequence is not None:
        raise ValueError("Supply keys or key_sequence, not both.")
    if key_sequence is not None:
        if len(key_sequence) != len(objects):
            raise ValueError("A key sequence must have one key per initial row slot.")
        keys = dict(zip(initial_order, key_sequence))
    elif keys is None:
        keys = {name: index + 1 for index, name in enumerate(objects)}
    if set(keys) != set(objects) or any(not isinstance(value, (int, np.integer)) for value in keys.values()):
        raise ValueError("Keys must map every physical block to an integer.")
    keys = {name: int(keys[name]) for name in objects}

    restore_simulator_state(env, baseline)
    positions = {name: slots[index] for index, name in enumerate(initial_order)}
    for name in objects:
        env.sim.data.set_joint_qpos(f"{name}:joint", np.r_[positions[name], 1.0, 0.0, 0.0, 0.0])
        env.sim.data.set_joint_qvel(f"{name}:joint", np.zeros(6))
    env.goal = np.r_[np.asarray([positions[name] for name in objects]).ravel(), np.zeros(3)]
    env.sim.forward()

    robot = TableRobot(env, slots, keys)
    trace = PhysicalTraceRecorder(env, slots, buffer_position, keys, initial_order,
                                  robot, case_name=case_name)
    order = list(initial_order)
    insertion_sort_blocks(order, keys, slots, buffer_position, robot, observer=trace)
    # Independently decode the final physical row, rather than trusting order.
    final = trace.snapshots[-1]
    row_locations = [(name, location) for name, location in final.locations.items()
                     if isinstance(location, int)]
    physical = tuple(name for name, _ in sorted(row_locations, key=lambda item: item[1]))
    expected = tuple(sorted(initial_order, key=keys.__getitem__))
    if physical != tuple(order) or physical != expected or final.buffered or final.held is not None:
        raise AssertionError(f"Physical sort did not finish correctly: expected {expected}, got {physical}")
    return PhysicalSortRun(case_name, initial_order, tuple(keys.items()), physical,
                           trace.snapshots, robot.steps, tuple(deepcopy(robot.transfers)))
