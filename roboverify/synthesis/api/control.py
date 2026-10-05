"""Shared bounded controllers for the primitive manipulation instructions.

Action generation is independent of convergence. Tolerances belong to the
controller which decides whether to take another step, never to the action
helper. ID and named instructions use this same implementation after lookup.
"""

from dataclasses import dataclass
from math import isfinite

import numpy as np

from synthesis.util.actions import bound_delta_action


@dataclass(frozen=True)
class ControlConfig:
    """Distances are metres; these settings are fixed, not CEM parameters."""

    position_tolerance: float = 2e-3
    gain: float = 20.0
    gripper_threshold: float = 0.052
    gripper_tolerance: float = 1e-3

    def __post_init__(self):
        for name, value in vars(self).items():
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


DEFAULT_CONTROL = ControlConfig()
DEFAULT_PICK_CONTROL = ControlConfig(position_tolerance=0.002)


@dataclass(frozen=True)
class ControlResult:
    """Controller convergence only; this does not certify a grasp or task goal."""

    converged: bool
    steps: int
    phase: str
    position_error: float | None = None


def get_move_action(
    observation, target_position, *, gain=DEFAULT_CONTROL.gain, close_gripper=False
):
    """Direction-preserving bounded XYZ command, with independent gripper control."""
    action = gain * (np.asarray(target_position) - np.asarray(observation)[:3])
    return bound_delta_action(np.r_[action, -0.2 if close_gripper else 0.0])


class PrimitiveController:
    """One instruction's shared step budget, observation capture, and convergence."""

    def __init__(
        self, env, trajectory, *, limit=50, control=DEFAULT_CONTROL, render=False
    ):
        if type(limit) is not int or limit < 0:
            raise ValueError("limit must be a nonnegative integer")
        if not isinstance(control, ControlConfig):
            raise TypeError("control must be a ControlConfig")
        self.env, self.trajectory = env, trajectory
        self.limit, self.control, self.render = limit, control, render
        self.steps, self.images = 0, []
        self.observation = self._observe()
        self.result = None

    def _observe(self):
        return np.asarray(self.env.flatten_observation(self.env.env._get_obs())).copy()

    def _step(self, action):
        self.env.step(action)
        self.steps += 1
        self.observation = self._observe()
        if self.render:
            self.images.append(self.env.render())
        self.trajectory.append(self.observation.copy())

    def _until(self, phase, done, action, error=None):
        while not done() and self.steps < self.limit:
            self._step(action())
        converged = bool(done())
        self.result = ControlResult(
            converged, self.steps, phase, None if error is None else float(error())
        )
        return converged

    def box_position(self, box_id):
        num_blocks = (len(self.observation) - 13) // 15
        if not 0 <= box_id < num_blocks:
            raise ValueError(f"Unknown box ID {box_id}")
        return self.observation[10 + 12 * box_id : 13 + 12 * box_id].copy()

    def move(self, target, *, close_gripper=True, vertical_only=False, phase="move"):
        target = np.asarray(target, dtype=float).copy()
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("Motion target must contain three finite coordinates")

        def error():
            delta = target - self.observation[:3]
            return abs(delta[2]) if vertical_only else np.linalg.norm(delta)

        def action():
            command_target = target.copy()
            if vertical_only:
                # Ignore unused axes before scaling, so their error cannot slow Z.
                command_target[:2] = self.observation[:2]
            return get_move_action(
                self.observation,
                command_target,
                gain=self.control.gain,
                close_gripper=close_gripper,
            )

        return self._until(
            phase, lambda: error() <= self.control.position_tolerance, action, error
        )

    def gripper(self, *, opened):
        # One complementary test avoids a gap where neither predicate holds.
        def is_open():
            return float(np.sum(self.observation[3:5])) > (
                self.control.gripper_threshold + self.control.gripper_tolerance
            )

        return self._until(
            "open" if opened else "close",
            lambda: bool(is_open()) == opened,
            lambda: np.array([0.0, 0.0, 0.0, 0.2 if opened else -0.2]),
        )

    def pick(self, box_id):
        target = self.box_position(box_id)
        target[2] = self.observation[2]
        if not self.move(target, close_gripper=False, phase="approach"):
            return False
        if not self.gripper(opened=True):
            return False
        target[2] = self.box_position(box_id)[2]
        if not self.move(target, close_gripper=False, phase="descend"):
            return False
        return self.gripper(opened=False)

    def move_relative(self, reference_ids, offsets):
        target = [
            self.box_position(box_id)[axis] + offsets[axis]
            for axis, box_id in enumerate(reference_ids)
        ]
        return self.move(target)

    def release(self, box_id, offset):
        # Freeze XY before opening; retreat feedback corrects opening/tracking drift.
        target = self.observation[:3].copy()
        target[2] = self.box_position(box_id)[2] + offset
        if not self.gripper(opened=True):
            return False
        return self.move(target, close_gripper=False, phase="retreat")
