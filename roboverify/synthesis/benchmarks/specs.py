"""Independent task evaluators; policies cannot access these functions.

Goals are measured from physical object positions, not declared by a generated
program. Reference policies and CaP policies use the same evaluators.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from synthesis.util.on import (
    BLOCK_LENGTH,
    on_star_implementation,
    scattered_implementation,
)

TASKS = (
    "stack",
    "unstack",
    "reverse",
    "partial_stack",
    "insertion_sort",
    "tower_merge",
    "tower_split",
    "stable_merge",
    "sort_restore",
)
ROW_TASKS = frozenset(("insertion_sort", "stable_merge", "sort_restore"))
DESCRIPTIONS = {
    "stack": "Stack every object in one tower whose bottom object is base.",
    "unstack": "Unstack the input tower: put every object on the table, pairwise Scattered.",
    "reverse": "Reverse the bottom-to-top order of the input tower. Every initial strict ON_star relationship must be reversed in the final single tower.",
    "partial_stack": "Combine the input partial towers into one tower whose bottom object is base. base may initially have objects below it.",
    "insertion_sort": "Sort the row by increasing key using physical transfers. Preserve the initial relative order of objects with equal keys. Leave the buffer empty.",
    "tower_merge": "Merge the two towers rooted at root_a and root_b into one tower: the entire original A tower below the entire original B tower. Preserve the original bottom-to-top order inside each tower and leave A's original objects in place.",
    "tower_split": "Split the input tower at cut: cut and everything initially above it form a separate tower in their original order. Everything initially strictly below cut remains in its original place and order.",
    "stable_merge": "The left and right contiguous row segments are each sorted. Merge them into one increasing row, preserving initial order for equal keys. Leave the buffer empty. Any algorithm using the permitted primitives is allowed.",
    "sort_restore": "First physically reach the stable increasing sorted row with an empty buffer. Then restore the exact original object order, also with an empty buffer. The evaluator checks both stages from measured positions.",
}


@dataclass(frozen=True)
class Case:
    task: str
    split: str
    seed: int
    num_blocks: int
    pattern: str = "random"

    def __post_init__(self):
        if self.task not in TASKS or self.split not in ("train", "validation", "test"):
            raise ValueError("Invalid benchmark task or split")
        if not 2 <= self.num_blocks <= 6:
            raise ValueError("Physical benchmark size must be between 2 and 6")
        if (
            self.task in ("tower_merge", "partial_stack", "stable_merge")
            and self.num_blocks < 4
        ):
            raise ValueError("This task needs at least four objects")

    @property
    def name(self):
        return (
            f"{self.task}-{self.split}-n{self.num_blocks}-s{self.seed}-{self.pattern}"
        )

    def as_dict(self):
        return {**asdict(self), "name": self.name}


def manifest(task, *, train=5, validation=5, test=10):
    """Disjoint case identities with explicit edge cases and size generalization."""
    if task not in TASKS:
        raise ValueError(task)
    result = []
    patterns = ("random", "reverse", "ties", "sorted", "negative")
    for split, count, seed_start in (
        ("train", train, 0),
        ("validation", validation, 100),
        ("test", test, 1000),
    ):
        for index in range(count):
            n = (
                4
                if task in ROW_TASKS or task in ("tower_merge", "partial_stack")
                else 3 + index % 2
            )
            if split == "test" and index % 2:
                n = 5
            result.append(
                Case(
                    task, split, seed_start + index, n, patterns[index % len(patterns)]
                )
            )
    return tuple(result)


@dataclass
class SceneDefinition:
    case: Case
    objects: tuple[str, ...]
    positions: dict[str, tuple[float, float, float]]
    initial_order: tuple[str, ...]
    keys: dict[str, int]
    roles: dict[str, str]
    slots: tuple[tuple[float, float, float], ...] = ()
    buffer: tuple[float, float, float] | None = None
    groups: tuple[tuple[str, ...], ...] = ()
    table_center: float = 0.425

    def as_dict(self):
        result = asdict(self)
        result["case"] = self.case.as_dict()
        return result


def make_scene(case, objects, *, gripper, robot_base_xy, table_center):
    """Construct geometry relative to the actual simulator's robot/table frame."""
    objects = tuple(map(str, objects))
    if len(objects) != case.num_blocks:
        raise ValueError("Object universe does not match the case")
    rng = np.random.default_rng(case.seed)
    order = tuple(str(name) for name in rng.permutation(objects))
    n = len(order)
    values = rng.permutation(np.arange(n)).tolist()
    if case.pattern == "reverse":
        values = list(range(n - 1, -1, -1))
    elif case.pattern == "sorted":
        values = list(range(n))
    elif case.pattern == "ties":
        values = [index % 2 for index in range(n)]
    elif case.pattern == "negative":
        values = [-2, 1, -2, 0, 3, -1][:n]
    if case.task == "stable_merge":
        cut_index = n // 2
        values = sorted(values[:cut_index]) + sorted(values[cut_index:])
    # Avoid a trivial sort/restore checkpoint in initially sorted inputs.
    if case.task == "sort_restore" and values == sorted(values):
        values = list(reversed(values))
    keys = dict(zip(order, values))
    roles = {"base": order[0]}
    slots, buffer, groups = (), None, ()
    if case.task in ROW_TASKS:
        slots = tuple(
            (
                float(gripper[0] + 0.10),
                float(gripper[1] + 0.11 * (i - (n - 1) / 2)),
                float(table_center),
            )
            for i in range(n)
        )
        buffer = (float(slots[0][0] - 0.18), float(gripper[1]), float(table_center))
        positions = dict(zip(order, slots))
        groups = (order[: n // 2], order[n // 2 :])
    elif case.task == "stack":
        from synthesis.environment.stack_reset import sample_stack_xy

        # This sampler uses global numpy RNG. Restore it so manifest generation
        # cannot perturb policy sampling or parameter search.
        previous = np.random.get_state()
        try:
            np.random.seed(case.seed)
            xy = sample_stack_xy(n, robot_base_xy, np.asarray(gripper)[:2])
        finally:
            np.random.set_state(previous)
        positions = {
            obj: (float(p[0]), float(p[1]), float(table_center))
            for obj, p in zip(order, xy)
        }
    else:
        # Reachable, separated columns within the Stack horizontal workspace.
        x = float(robot_base_xy[0] + 0.60)
        y = float(robot_base_xy[1])
        if case.task in ("tower_merge", "partial_stack"):
            groups = (order[: n // 2], order[n // 2 :])
            roles.update(root_a=groups[0][0], root_b=groups[1][0])
            positions = {
                obj: (
                    x,
                    y + (-0.16 if group_index == 0 else 0.16),
                    float(table_center + level * BLOCK_LENGTH),
                )
                for group_index, group in enumerate(groups)
                for level, obj in enumerate(group)
            }
            if case.task == "partial_stack" and case.split == "test" and n >= 5:
                roles["base"] = groups[0][1]
        else:
            positions = {
                obj: (x, y - 0.16, float(table_center + level * BLOCK_LENGTH))
                for level, obj in enumerate(order)
            }
            if case.task == "tower_split":
                groups = (order[: n // 2], order[n // 2 :])
                roles["cut"] = groups[1][0]
    return SceneDefinition(
        case,
        objects,
        positions,
        order,
        keys,
        roles,
        slots,
        buffer,
        groups,
        float(table_center),
    )


def row_order(scene, positions, *, tolerance_xy=0.012, tolerance_z=0.005):
    """Return a complete physical row or None; never infer a missing occupant."""
    occupants = []
    for target in scene.slots:
        matches = [
            obj
            for obj in scene.objects
            if np.linalg.norm(np.asarray(positions[obj])[:2] - np.asarray(target)[:2])
            <= tolerance_xy
            and abs(positions[obj][2] - target[2]) <= tolerance_z
        ]
        if len(matches) != 1:
            return None
        occupants.append(matches[0])
    return tuple(occupants) if len(set(occupants)) == len(scene.objects) else None


@dataclass
class Evaluation:
    success: bool
    score: float
    failed: list[str] = field(default_factory=list)
    observed_order: tuple[str, ...] | None = None


def evaluate(scene, positions, *, held=None, sorted_seen=False, upright=None):
    """Conjunction of the full task goal, identity, and physical completion checks."""
    checks = {
        "identity": set(positions) == set(scene.objects),
        "empty_gripper": held is None,
    }
    if not checks["identity"]:
        return Evaluation(False, 0, ["identity"])
    checks["finite_geometry"] = all(
        np.asarray(p).shape == (3,) and np.isfinite(p).all() for p in positions.values()
    )
    if not checks["finite_geometry"]:
        return Evaluation(False, 0, ["finite_geometry"])
    if upright is not None:
        checks["upright"] = set(upright) == set(scene.objects) and all(
            value > 0.98 for value in upright.values()
        )
    task = scene.case.task
    on = lambda a, b: bool(on_star_implementation(positions[a], positions[b]))
    original_on = lambda a, b: bool(
        on_star_implementation(scene.positions[a], scene.positions[b])
    )
    pairs = [(a, b) for a in scene.objects for b in scene.objects if a != b]
    order = None
    if task in ROW_TASKS:
        order = row_order(scene, positions)
        expected = tuple(sorted(scene.initial_order, key=scene.keys.__getitem__))
        checks["complete_row"] = order is not None
        checks["empty_buffer"] = all(
            np.linalg.norm(np.asarray(pos)[:2] - np.asarray(scene.buffer)[:2]) > 0.05
            for pos in positions.values()
        )
        checks["stable_order"] = order == (
            scene.initial_order if task == "sort_restore" else expected
        )
        if task == "sort_restore":
            checks["sorted_checkpoint"] = sorted_seen
    elif task in ("stack", "partial_stack"):
        checks["one_tower_on_base"] = all(
            on(obj, scene.roles["base"]) for obj in scene.objects
        )
        checks["supported_levels"] = _tower_levels(scene, positions, scene.objects)
    elif task == "unstack":
        checks["on_table"] = all(
            abs(pos[2] - scene.table_center) <= 0.005 for pos in positions.values()
        )
        checks["pairwise_scattered"] = all(
            scattered_implementation(positions[a], positions[b]) for a, b in pairs
        )
    elif task == "reverse":
        checks["reverse_order"] = all(on(a, b) == original_on(b, a) for a, b in pairs)
        checks["supported_levels"] = _tower_levels(scene, positions, scene.objects)
    elif task in ("tower_merge", "tower_split"):
        first, second = scene.groups
        checks["internal_order"] = all(
            on(a, b) == original_on(a, b)
            for group in scene.groups
            for a in group
            for b in group
        )
        checks["protected_prefix"] = all(
            np.linalg.norm(
                np.asarray(positions[obj]) - np.asarray(scene.positions[obj])
            )
            <= 0.005
            for obj in first
        )
        if task == "tower_merge":
            checks["concatenation"] = all(
                on(b, a) and not on(a, b) for a in first for b in second
            )
            checks["supported_levels"] = _tower_levels(scene, positions, scene.objects)
        else:
            checks["separate_towers"] = all(
                not on(a, b) and not on(b, a) for a in first for b in second
            )
            checks["supported_levels"] = all(
                _tower_levels(scene, positions, group) for group in scene.groups
            )
    failed = [name for name, ok in checks.items() if not ok]
    return Evaluation(
        not failed, sum(map(bool, checks.values())) / len(checks), failed, order
    )


def _tower_levels(scene, positions, objects):
    ordered = sorted(objects, key=lambda obj: positions[obj][2])
    root = np.asarray(positions[ordered[0]])
    return all(
        np.linalg.norm(np.asarray(positions[obj])[:2] - root[:2]) <= 0.012
        and abs(positions[obj][2] - (scene.table_center + level * BLOCK_LENGTH))
        <= 0.005
        for level, obj in enumerate(ordered)
    )
