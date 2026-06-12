import pdb
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
import z3

from synthesis.util import on as on_util


def grippers_are_closed(obs, atol=1e-3):
    threshold = 0.026
    gripper_state = obs[3:5]
    return (
        abs(np.sum(gripper_state) - 2 * threshold) < atol
        or np.sum(gripper_state) - 2 * threshold < 0
    )


def grippers_are_open(obs, atol=1e-3):
    threshold = 0.026
    gripper_state = obs[3:5]
    # return abs(gripper_state[0] - 0.05) < atol
    # return gripper_state[0] > threshold + atol
    return np.sum(gripper_state) > 2 * threshold + atol


def get_open_gripper_action():
    return np.array([0.0, 0.0, 0.0, 0.2])


def get_close_gripper_action():
    return np.array([0.0, 0.0, 0.0, -0.2])


def get_move_action(
    observation, target_position, atol=1e-3, gain=10.0, close_gripper=False
):
    """
    Move an end effector to a position and orientation.
    """
    # Get the currents
    # current_position = observation['observation'][:3]
    current_position = observation[:3]

    action = gain * np.subtract(target_position, current_position)
    if close_gripper:
        # gripper_action = -1.
        gripper_action = -0.2
    else:
        gripper_action = 0.0
    action = np.hstack((action, gripper_action))

    return action


class Parameter:
    """Scalar program parameter. ``val is None`` means *unspecified* (trainable /
    BMC-solve unknown), not numeric zero. Use ``numeric_val()`` for a float
    suitable before training (unknown → ``0.0``)."""

    def __init__(self, val: float | None = None):
        self.pos: int | None = None
        self.val: float | None = float(val) if val is not None else None

    def numeric_val(self) -> float:
        """Float for runtime / flat parameter vectors; unknown maps to ``0.0``."""
        return 0.0 if self.val is None else float(self.val)

    def concrete_float(self, where: str) -> float:
        """Require a numeric value (e.g. BMC **verify**); raise if still unspecified."""
        if self.val is None:
            raise ValueError(
                f"{where}: offset parameter is unspecified (None). "
                "Pass explicit floats on the instruction for verify mode, "
                "or use BMC solve for existentially quantified offsets."
            )
        return float(self.val)

    def register(self, parameters: List):
        self.pos = len(parameters)
        parameters.append(self.numeric_val())

    def update(self, new_parameter: List[float]):
        if self.pos is not None:
            self.val = new_parameter[self.pos]
        else:
            pdb.set_trace()
            raise ValueError

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Parameter):
            return False
        return self.pos == other.pos and self.val == other.val

    def __str__(self):
        if self.val is None:
            return "?"
        return f"{self.val:.3}"


class Instruction(ABC):
    @abstractmethod
    def eval(self, env, traj, return_image=False) -> List:
        pass

    def register_trainable_parameter(self, parameters: List):
        pass

    def update_trainable_parameter(self, new_parameter: List):
        pass

    def get_operand(self):
        return []

    def set_operand(self, new_operands):
        pass

    @abstractmethod
    def __str__(self):
        pass

    def _resolve(self, env) -> Dict[str, int]:
        mapping = getattr(env, "symbolic_name_to_box_id", None)
        if mapping is None:
            raise ValueError(
                "PickPlaceByName requires env.symbolic_name_to_box_id (e.g. {'b0': 1})."
            )
        if not isinstance(mapping, dict):
            raise TypeError("env.symbolic_name_to_box_id must be a dict[str, int].")
        return mapping


class Skip(Instruction):
    def __init__(self, skip_steps: int = 20):
        self.skip_steps = skip_steps

    def eval(self, env, traj, return_image=False):
        imgs = []
        for _ in range(self.skip_steps):
            obs = env.flatten_observation(env.env._get_obs())
            if return_image:
                imgs.append(env.render())
            traj.append(obs)
        return imgs

    def __str__(self):
        return "Skip"


class Pick(Instruction):
    def __init__(self, grab_box_id: int = 0, limit: int = 50):
        self.limit = limit
        self.grab_box_id = grab_box_id
        self.types = ["Box"]

    def get_box_pos(self, box_id, obs):
        block_num = (obs.shape[0] - 13) // 15
        if 0 <= box_id < block_num:
            return obs[10 + box_id * 12 : 10 + box_id * 12 + 3]
        assert False, f"unknown box id {box_id}"

    def eval(self, env, traj, return_image=False):
        imgs = []
        obs = env.flatten_observation(env.env._get_obs())
        target_box_x = self.get_box_pos(self.grab_box_id, obs)[0]
        target_box_y = self.get_box_pos(self.grab_box_id, obs)[1]
        target_z = obs[2]
        target_position = np.array([target_box_x, target_box_y, target_z])
        steps = 0
        while steps < self.limit and np.linalg.norm(obs[:3] - target_position) > 2e-2:
            # print(np.linalg.norm(obs[:3] - target_position))
            action = get_move_action(obs, target_position, close_gripper=False)
            env.step(action)
            obs = env.flatten_observation(env.env._get_obs())
            steps += 1
            if return_image:
                imgs.append(env.render())
            traj.append(obs)

        while steps < self.limit and grippers_are_closed(obs):
            action = get_open_gripper_action()
            env.step(action)
            obs = env.flatten_observation(env.env._get_obs())
            steps += 1
            if return_image:
                imgs.append(env.render())
            traj.append(obs)

        target_box_z = self.get_box_pos(self.grab_box_id, obs)[2]
        target_position = np.array([target_box_x, target_box_y, target_box_z])
        while steps < self.limit and np.linalg.norm(obs[:3] - target_position) > 2e-2:
            action = get_move_action(obs, target_position, close_gripper=False)
            env.step(action)
            obs = env.flatten_observation(env.env._get_obs())
            steps += 1
            if return_image:
                imgs.append(env.render())
            traj.append(obs)

        while steps < self.limit and grippers_are_open(obs):
            action = get_close_gripper_action()
            env.step(action)
            obs = env.flatten_observation(env.env._get_obs())
            steps += 1
            if return_image:
                imgs.append(env.render())
            traj.append(obs)
        return imgs

    def register_trainable_parameter(self, parameter: List[float]):
        return

    def update_trainable_parameter(self, new_parameter: List[float]):
        return

    def get_operand(self):
        return [{"type": self.types[0], "val": self.grab_box_id}]

    def set_operand(self, new_operands):
        assert new_operands[0]["type"] == "Box"
        self.grab_box_id = new_operands[0]["val"]

    def __eq__(self, other):
        if not isinstance(other, Pick):
            return False
        cond1 = self.get_operand() == other.get_operand()
        return cond1

    def __str__(self):
        return f"Pick({self.grab_box_id})"


class PickByName(Instruction):
    def __init__(self, grab_box_name: str, limit: int = 50):
        self.limit = limit
        self.grab_box_name = grab_box_name
        self.types = ["BoxName"]

    def get_box_pos(self, box_id: int, obs):
        block_num = (obs.shape[0] - 13) // 15
        if 0 <= box_id < block_num:
            return obs[10 + box_id * 12 : 10 + box_id * 12 + 3]
        assert False, f"unknown box id {box_id}"

    def eval(self, env, traj, return_image=False):
        mapping = self._resolve(env)
        imgs = []
        obs = env.flatten_observation(env.env._get_obs())
        target_box_x = self.get_box_pos(mapping[self.grab_box_name], obs)[0]
        target_box_y = self.get_box_pos(mapping[self.grab_box_name], obs)[1]
        target_z = obs[2]
        target_position = np.array([target_box_x, target_box_y, target_z])
        steps = 0
        while steps < self.limit and np.linalg.norm(obs[:3] - target_position) > 2e-2:
            print(np.linalg.norm(obs[:3] - target_position))
            action = get_move_action(obs, target_position, close_gripper=False)
            env.step(action)
            obs = env.flatten_observation(env.env._get_obs())
            steps += 1
            if return_image:
                imgs.append(env.render())
            traj.append(obs)

        while steps < self.limit and grippers_are_closed(obs):
            action = get_open_gripper_action()
            env.step(action)
            obs = env.flatten_observation(env.env._get_obs())
            steps += 1
            if return_image:
                imgs.append(env.render())
            traj.append(obs)

        target_box_z = self.get_box_pos(mapping[self.grab_box_name], obs)[2]
        target_position = np.array([target_box_x, target_box_y, target_box_z])
        while steps < self.limit and np.linalg.norm(obs[:3] - target_position) > 2e-2:
            action = get_move_action(obs, target_position, close_gripper=False)
            env.step(action)
            obs = env.flatten_observation(env.env._get_obs())
            steps += 1
            if return_image:
                imgs.append(env.render())
            traj.append(obs)

        while steps < self.limit and grippers_are_open(obs):
            action = get_close_gripper_action()
            env.step(action)
            obs = env.flatten_observation(env.env._get_obs())
            steps += 1
            if return_image:
                imgs.append(env.render())
            traj.append(obs)
        return imgs

    def register_trainable_parameter(self, parameter: List[float]):
        return

    def update_trainable_parameter(self, new_parameter: List[float]):
        return

    def get_operand(self):
        return [{"type": self.types[0], "val": self.grab_box_name}]

    def set_operand(self, new_operands):
        assert new_operands[0]["type"] == "BoxName"
        self.grab_box_name = new_operands[0]["val"]

    def __eq__(self, other):
        if not isinstance(other, PickByName):
            return False
        cond1 = self.get_operand() == other.get_operand()
        return cond1

    def __str__(self):
        return f"PickByName({self.grab_box_name})"


class Move(Instruction):
    def __init__(
        self,
        target_box_id_x: int = 0,
        target_box_id_y: int = 0,
        target_box_id_z: int = 0,
        limit: int = 50,
        target_offset: Optional[List[float]] = None,
    ):
        self.limit = limit
        self.target_box_id_x = target_box_id_x
        self.target_box_id_y = target_box_id_y
        self.target_box_id_z = target_box_id_z
        self.types = ["Box", "Box", "Box"]
        if target_offset is None:
            self.target_offset = [Parameter() for _ in range(3)]
        else:
            if len(target_offset) != 3:
                raise ValueError("target_offset must be a length-3 list of floats.")
            self.target_offset = [Parameter(float(v)) for v in target_offset]

    def get_box_pos(self, box_id, obs):
        block_num = (obs.shape[0] - 13) // 15
        if 0 <= box_id < block_num:
            return obs[10 + box_id * 12 : 10 + box_id * 12 + 3]
        assert False, f"unknown box id {box_id}"

    def eval(self, env, traj, return_image=False):
        imgs = []
        obs = env.flatten_observation(env.env._get_obs())
        target_box_x = (
            self.get_box_pos(self.target_box_id_x, obs)[0]
            + self.target_offset[0].numeric_val()
        )
        target_box_y = (
            self.get_box_pos(self.target_box_id_y, obs)[1]
            + self.target_offset[1].numeric_val()
        )
        target_box_z = (
            self.get_box_pos(self.target_box_id_z, obs)[2]
            + self.target_offset[2].numeric_val()
        )
        target_position = np.array([target_box_x, target_box_y, target_box_z])
        steps = 0
        while steps < self.limit and np.linalg.norm(obs[:3] - target_position) > 2e-2:
            action = get_move_action(obs, target_position, close_gripper=True)
            env.step(action)
            obs = env.flatten_observation(env.env._get_obs())
            steps += 1
            if return_image:
                imgs.append(env.render())
            traj.append(obs)
        return imgs

    def register_trainable_parameter(self, parameter: List[float]):
        for p in self.target_offset:
            p.register(parameter)

    def update_trainable_parameter(self, new_parameter: List[float]):
        for p in self.target_offset:
            p.update(new_parameter)

    def get_operand(self):
        return [
            {"type": self.types[0], "val": self.target_box_id_x},
            {"type": self.types[1], "val": self.target_box_id_y},
            {"type": self.types[2], "val": self.target_box_id_z},
        ]

    def set_operand(self, new_operands):
        assert new_operands[0]["type"] == "Box"
        assert new_operands[1]["type"] == "Box"
        assert new_operands[2]["type"] == "Box"
        self.target_box_id_x = new_operands[0]["val"]
        self.target_box_id_y = new_operands[1]["val"]
        self.target_box_id_z = new_operands[2]["val"]

    def __eq__(self, other):
        if not isinstance(other, Move):
            return False
        cond1 = self.get_operand() == other.get_operand()
        return cond1

    def __str__(self):
        return f"Move({self.target_box_id_x}({str(self.target_offset[0])}), {self.target_box_id_y}({str(self.target_offset[1])}), {self.target_box_id_z}({str(self.target_offset[2])}))"


class MoveByName(Instruction):
    def __init__(
        self,
        target_box_name_x: str,
        target_box_name_y: str,
        target_box_name_z: str,
        limit: int = 50,
        target_offset: Optional[List[float]] = None,
    ):
        self.limit = limit
        self.target_box_name_x = target_box_name_x
        self.target_box_name_y = target_box_name_y
        self.target_box_name_z = target_box_name_z
        self.types = ["BoxName", "BoxName", "BoxName"]
        if target_offset is None:
            self.target_offset = [Parameter() for _ in range(3)]
        else:
            if len(target_offset) != 3:
                raise ValueError("target_offset must be a length-3 list of floats.")
            self.target_offset = [Parameter(float(v)) for v in target_offset]

    def get_box_pos(self, box_id: int, obs):
        block_num = (obs.shape[0] - 13) // 15
        if 0 <= box_id < block_num:
            return obs[10 + box_id * 12 : 10 + box_id * 12 + 3]
        assert False, f"unknown box id {box_id}"

    def eval(self, env, traj, return_image=False):
        mapping = self._resolve(env)
        imgs = []
        obs = env.flatten_observation(env.env._get_obs())
        target_box_x = (
            self.get_box_pos(mapping[self.target_box_name_x], obs)[0]
            + self.target_offset[0].numeric_val()
        )
        target_box_y = (
            self.get_box_pos(mapping[self.target_box_name_y], obs)[1]
            + self.target_offset[1].numeric_val()
        )
        target_box_z = (
            self.get_box_pos(mapping[self.target_box_name_z], obs)[2]
            + self.target_offset[2].numeric_val()
        )
        target_position = np.array([target_box_x, target_box_y, target_box_z])
        steps = 0
        while steps < self.limit and np.linalg.norm(obs[:3] - target_position) > 2e-2:
            action = get_move_action(obs, target_position, close_gripper=True)
            env.step(action)
            obs = env.flatten_observation(env.env._get_obs())
            steps += 1
            if return_image:
                imgs.append(env.render())
            traj.append(obs)
        return imgs

    def register_trainable_parameter(self, parameter: List[float]):
        for p in self.target_offset:
            p.register(parameter)

    def update_trainable_parameter(self, new_parameter: List[float]):
        for p in self.target_offset:
            p.update(new_parameter)

    def get_operand(self):
        return [
            {"type": self.types[0], "val": self.target_box_name_x},
            {"type": self.types[1], "val": self.target_box_name_y},
            {"type": self.types[2], "val": self.target_box_name_z},
        ]

    def set_operand(self, new_operands):
        assert new_operands[0]["type"] == "BoxName"
        assert new_operands[1]["type"] == "BoxName"
        assert new_operands[2]["type"] == "BoxName"
        self.target_box_name_x = new_operands[0]["val"]
        self.target_box_name_y = new_operands[1]["val"]
        self.target_box_name_z = new_operands[2]["val"]

    def __eq__(self, other):
        if not isinstance(other, MoveByName):
            return False
        cond1 = self.get_operand() == other.get_operand()
        return cond1

    def __str__(self):
        return f"MoveByName({self.target_box_name_x}, {self.target_box_name_y}, {self.target_box_name_z})"


class Release(Instruction):
    def __init__(
        self,
        release_box_id: int = 0,
        limit: int = 50,
        *,
        target_z: float | None = None,
    ):
        self.limit = limit
        self.release_box_id = release_box_id
        self.types = ["Box"]
        self.target_z_offset = (
            Parameter(float(target_z)) if target_z is not None else Parameter()
        )

    def get_box_pos(self, box_id, obs):
        block_num = (obs.shape[0] - 13) // 15
        if 0 <= box_id < block_num:
            return obs[10 + box_id * 12 : 10 + box_id * 12 + 3]
        assert False, f"unknown box id {box_id}"

    def eval(self, env, traj, return_image=False):
        imgs = []
        obs = env.flatten_observation(env.env._get_obs())
        target_z = (
            self.get_box_pos(self.release_box_id, obs)[2]
            + self.target_z_offset.numeric_val()
        )
        steps = 0
        while steps < self.limit and grippers_are_closed(
            env.flatten_observation(env.env._get_obs())
        ):
            action = get_open_gripper_action()
            env.step(action)
            steps += 1
            if return_image:
                imgs.append(env.render())
            traj.append(env.flatten_observation(env.env._get_obs()))
        while (
            steps < self.limit
            and abs(env.flatten_observation(env.env._get_obs())[2] - target_z) > 2e-2
        ):
            action = get_move_action(
                np.array([0.0, 0.0, env.flatten_observation(env.env._get_obs())[2]]),
                np.array([0.0, 0.0, target_z]),
                close_gripper=False,
            )
            env.step(action)
            steps += 1
            if return_image:
                imgs.append(env.render())
            traj.append(env.flatten_observation(env.env._get_obs()))
        return imgs

    def register_trainable_parameter(self, parameter: List[float]):
        self.target_z_offset.register(parameter)

    def update_trainable_parameter(self, new_parameter: List[float]):
        self.target_z_offset.update(new_parameter)

    def get_operand(self):
        return [{"type": self.types[0], "val": self.release_box_id}]

    def set_operand(self, new_operands):
        assert new_operands[0]["type"] == "Box"
        self.release_box_id = new_operands[0]["val"]

    def __eq__(self, other):
        if not isinstance(other, Release):
            return False
        cond1 = self.get_operand() == other.get_operand()
        return cond1

    def __str__(self):
        return f"Release({self.release_box_id}, {self.target_z_offset})"


class ReleaseByName(Instruction):
    def __init__(
        self,
        release_box_name: str,
        limit: int = 50,
        *,
        target_z: float | None = None,
    ):
        self.limit = limit
        self.release_box_name = release_box_name
        self.types = ["BoxName"]
        self.target_z_offset = (
            Parameter(float(target_z)) if target_z is not None else Parameter()
        )

    def get_box_pos(self, box_id: int, obs):
        block_num = (obs.shape[0] - 13) // 15
        if 0 <= box_id < block_num:
            return obs[10 + box_id * 12 : 10 + box_id * 12 + 3]
        assert False, f"unknown box id {box_id}"

    def eval(self, env, traj, return_image=False):
        mapping = self._resolve(env)
        imgs = []
        obs = env.flatten_observation(env.env._get_obs())
        target_z = (
            self.get_box_pos(mapping[self.release_box_name], obs)[2]
            + self.target_z_offset.numeric_val()
        )
        steps = 0
        while steps < self.limit and grippers_are_closed(
            env.flatten_observation(env.env._get_obs())
        ):
            action = get_open_gripper_action()
            env.step(action)
            steps += 1
            if return_image:
                imgs.append(env.render())
            traj.append(env.flatten_observation(env.env._get_obs()))
        while (
            steps < self.limit
            and abs(env.flatten_observation(env.env._get_obs())[2] - target_z) > 1e-3
        ):
            action = get_move_action(
                np.array([0.0, 0.0, env.flatten_observation(env.env._get_obs())[2]]),
                np.array([0.0, 0.0, target_z]),
                close_gripper=False,
            )
            env.step(action)
            steps += 1
            if return_image:
                imgs.append(env.render())
            traj.append(env.flatten_observation(env.env._get_obs()))
        return imgs

    def register_trainable_parameter(self, parameter: List[float]):
        self.target_z_offset.register(parameter)

    def update_trainable_parameter(self, new_parameter: List[float]):
        self.target_z_offset.update(new_parameter)

    def get_operand(self):
        return [{"type": self.types[0], "val": self.release_box_name}]

    def set_operand(self, new_operands):
        assert new_operands[0]["type"] == "BoxName"
        self.release_box_name = new_operands[0]["val"]

    def __eq__(self, other):
        if not isinstance(other, ReleaseByName):
            return False
        cond1 = self.get_operand() == other.get_operand()
        return cond1

    def __str__(self):
        return f"ReleaseByName({self.release_box_name}, {self.target_z_offset})"


class PickPlace(Instruction):
    def __init__(self, grab_box_id: int = 0, target_box_id: int = 0, limit: int = 50):
        self.limit = limit
        self.grab_box_id = grab_box_id
        self.target_box_id = target_box_id
        self.types = ["Box", "Box"]
        self.target_offset = [Parameter() for _ in range(3)]

    def get_box_pos(self, box_id, obs):
        block_num = (obs.shape[0] - 13) // 15
        if 0 <= box_id < block_num:
            return obs[10 + box_id * 12 : 10 + box_id * 12 + 3]
        assert False, f"unknown box id {box_id}"

    def eval(self, env, traj, return_image=False):
        from synthesis.environment.data.pickplace_naive import get_pick_control_naive

        imgs = []
        success = False
        initial_goal_box = self.get_box_pos(self.target_box_id, traj[-1])
        step = 0
        while not success and step < self.limit:
            obs = env.flatten_observation(env.env._get_obs())
            action, success = get_pick_control_naive(
                obs,
                initial_goal_box
                + np.array(
                    [offset.numeric_val() for offset in self.target_offset],
                    dtype=float,
                ),
                block_id=self.grab_box_id,
                last_block=True,
            )
            env.step(action)
            step += 1
            if return_image:
                imgs.append(env.render())
            traj.append(obs)
        return imgs

    def register_trainable_parameter(self, parameter: List[float]):
        for p in self.target_offset:
            p.register(parameter)

    def update_trainable_parameter(self, new_parameter: List[float]):
        for p in self.target_offset:
            p.update(new_parameter)

    def get_operand(self):
        return [
            {"type": self.types[0], "val": self.grab_box_id},
            {"type": self.types[1], "val": self.target_box_id},
        ]

    def set_operand(self, new_operands):
        assert new_operands[0]["type"] == "Box"
        assert new_operands[1]["type"] == "Box"
        self.grab_box_id = new_operands[0]["val"]
        self.target_box_id = new_operands[1]["val"]

    def __eq__(self, other):
        if not isinstance(other, PickPlace):
            return False
        cond1 = self.get_operand() == other.get_operand()
        cond2 = self.target_offset == other.target_offset
        return cond1 and cond2

    def __str__(self):
        return f"PickPlace({self.grab_box_id}, {self.target_box_id}, {[str(x) for x in self.target_offset]})"


class PickPlaceByName(Instruction):
    def __init__(
        self,
        *,
        grab_box_name: str,
        target_box_name_x: str,
        target_box_name_y: str,
        target_box_name_z: str,
        limit: int = 50,
        target_offset: Optional[List[float]] = None,
        release: bool = True,
    ):
        self.limit = limit
        self.grab_box_name = grab_box_name
        self.target_box_name_x = target_box_name_x
        self.target_box_name_y = target_box_name_y
        self.target_box_name_z = target_box_name_z
        self.release = bool(release)
        self.types = ["BoxName", "BoxName", "BoxName", "BoxName"]
        if target_offset is None:
            self.target_offset = [Parameter() for _ in range(3)]
        else:
            if len(target_offset) != 3:
                raise ValueError("target_offset must be a length-3 list of floats.")
            self.target_offset = [Parameter(float(v)) for v in target_offset]

    def _resolve(self, env) -> Dict[str, int]:
        mapping = getattr(env, "symbolic_name_to_box_id", None)
        if mapping is None:
            raise ValueError(
                "PickPlaceByName requires env.symbolic_name_to_box_id (e.g. {'b0': 1})."
            )
        if not isinstance(mapping, dict):
            raise TypeError("env.symbolic_name_to_box_id must be a dict[str, int].")
        return mapping

    @property
    def target_box_names(self) -> tuple[str, str, str]:
        return (self.target_box_name_x, self.target_box_name_y, self.target_box_name_z)

    def get_box_pos(self, box_id: int, obs):
        block_num = (obs.shape[0] - 13) // 15
        if 0 <= box_id < block_num:
            return obs[10 + box_id * 12 : 10 + box_id * 12 + 3]
        assert False, f"unknown box id {box_id}"

    def eval(self, env, traj, return_image=False):
        from synthesis.environment.data.pickplace_naive import get_pick_control_naive

        mapping = self._resolve(env)
        grab_box_id = mapping[self.grab_box_name]
        target_x_id = mapping[self.target_box_name_x]
        target_y_id = mapping[self.target_box_name_y]
        target_z_id = mapping[self.target_box_name_z]

        imgs = []
        success = False

        step = 0
        while not success and step < self.limit:
            obs = env.flatten_observation(env.env._get_obs())
            # Compute the goal from the *current* observation, not only from the
            # initial one. This prevents stale targets causing unnecessary re-grasps
            # when the block is already correctly placed.
            gx = float(self.get_box_pos(target_x_id, obs)[0])
            gy = float(self.get_box_pos(target_y_id, obs)[1])
            gz = float(self.get_box_pos(target_z_id, obs)[2])
            goal = np.array([gx, gy, gz], dtype=float) + np.array(
                [offset.numeric_val() for offset in self.target_offset], dtype=float
            )
            action, success = get_pick_control_naive(
                obs,
                goal,
                block_id=grab_box_id,
                last_block=True,
                release=self.release,
            )
            env.step(action)
            step += 1
            if return_image:
                imgs.append(env.render())
            traj.append(obs)
        return imgs

    def register_trainable_parameter(self, parameter: List[float]):
        for p in self.target_offset:
            p.register(parameter)

    def update_trainable_parameter(self, new_parameter: List[float]):
        for p in self.target_offset:
            p.update(new_parameter)

    def get_operand(self):
        return [
            {"type": self.types[0], "val": self.grab_box_name},
            {"type": self.types[1], "val": self.target_box_name_x},
            {"type": self.types[2], "val": self.target_box_name_y},
            {"type": self.types[3], "val": self.target_box_name_z},
        ]

    def set_operand(self, new_operands):
        assert new_operands[0]["type"] == "BoxName"
        assert new_operands[1]["type"] == "BoxName"
        assert new_operands[2]["type"] == "BoxName"
        assert new_operands[3]["type"] == "BoxName"
        self.grab_box_name = new_operands[0]["val"]
        self.target_box_name_x = new_operands[1]["val"]
        self.target_box_name_y = new_operands[2]["val"]
        self.target_box_name_z = new_operands[3]["val"]

    def __eq__(self, other):
        if not isinstance(other, PickPlaceByName):
            return False
        cond1 = self.get_operand() == other.get_operand()
        cond2 = self.target_offset == other.target_offset
        return cond1 and cond2

    def __str__(self):
        tx, ty, tz = self.target_box_names
        return (
            f"PickPlaceByName({self.grab_box_name}, ({tx}, {ty}, {tz}), "
            f"{[str(x) for x in self.target_offset]}, release={self.release})"
        )


class While(Instruction):
    def __init__(
        self,
        instantiated_cond,
        guard_exists_vars,
        body: List[Instruction],
        invariant,
        max_iters: int = 10,
    ):
        if guard_exists_vars is None:
            raise ValueError("guard_exists_vars must be provided for While.")
        self.body = body
        self.invariant = invariant
        self.guard_exists_vars = guard_exists_vars
        self.instantiated_cond = instantiated_cond
        self.cond = (
            instantiated_cond
            if len(guard_exists_vars) == 0
            else z3.Exists(guard_exists_vars, instantiated_cond)
        )
        self.max_iters = int(max_iters)
        # Optional structured provenance for learned invariants:
        # list of objects with attribute `.expr` (z3.ExprRef) and metadata.
        self.invariant_provenance = (
            list(invariant)
            if isinstance(invariant, list)
            and invariant
            and hasattr(invariant[0], "expr")
            else None
        )

    def _get_num_blocks(self, env, obs) -> int:
        # Prefer environment-provided counts when available.
        num_blocks = getattr(getattr(env, "env", None), "num_blocks", None)
        if num_blocks is None:
            num_blocks = getattr(getattr(env, "unwrapped", None), "num_blocks", None)
        if num_blocks is None:
            num_blocks = getattr(env, "num_blocks", None)
        if num_blocks is not None:
            return int(num_blocks)

        n_obj = getattr(env, "nObj", None)
        if n_obj is None:
            n_obj = getattr(getattr(env, "env", None), "nObj", None)
        if n_obj is None:
            n_obj = getattr(getattr(env, "unwrapped", None), "nObj", None)
        if n_obj is not None:
            return int(n_obj)

        # Fallback: infer from observation layout used elsewhere in this repo.
        # Observation packs agent dims then per-object dims; in this project we
        # index object positions with `10 + 12*i : 10 + 12*i + 3`.
        # Use the same heuristic as PickPlace.get_box_pos.
        return max(0, (int(obs.shape[0]) - 13) // 15)

    def _build_block_positions(self, obs, num_blocks: int):
        return [on_util.get_block_pos(obs, i) for i in range(num_blocks)]

    def _eval_z3_guard(self, z3_expr, bindings, all_block_pos):
        """Evaluate a restricted subset of Z3 Bool formulas under concrete bindings.

        - `bindings` maps variable names (str) -> block index (int)
        - `all_block_pos` is a list of xyz arrays for each block id
        """

        def eval_term(term, bound_vals):
            if z3.is_var(term):
                return bound_vals[z3.get_var_index(term)]
            if z3.is_app(term) and term.num_args() == 0:
                # Free constant like `b0`, `b`, `b_prime`, `tbl` (if present).
                name = term.decl().name()
                if name in bindings:
                    return bindings[name]
                raise KeyError(
                    f"While guard evaluation could not resolve symbol {name!r}; "
                    f"available: {sorted(bindings.keys())}"
                )
            raise TypeError(f"Unsupported Z3 term in guard: {term}")

        def eval_bool(expr, bound_vals):
            if z3.is_true(expr):
                return True
            if z3.is_false(expr):
                return False
            if z3.is_quantifier(expr):
                body = expr.body()
                k = expr.num_vars()
                # De Bruijn index 0 is the innermost variable; Z3 exposes it the same way.
                all_ids = list(range(len(all_block_pos)))

                def rec(i, cur):
                    if i == k:
                        return eval_bool(body, cur)
                    if expr.is_forall():
                        for bid in all_ids:
                            if not rec(i + 1, cur + [bid]):
                                return False
                        return True
                    # Exists
                    for bid in all_ids:
                        if rec(i + 1, cur + [bid]):
                            return True
                    return False

                return rec(0, bound_vals)

            if not z3.is_app(expr):
                raise TypeError(f"Unsupported Z3 guard node: {expr}")

            op = expr.decl().kind()
            if op == z3.Z3_OP_NOT:
                return not eval_bool(expr.arg(0), bound_vals)
            if op == z3.Z3_OP_AND:
                return all(
                    eval_bool(expr.arg(i), bound_vals) for i in range(expr.num_args())
                )
            if op == z3.Z3_OP_OR:
                return any(
                    eval_bool(expr.arg(i), bound_vals) for i in range(expr.num_args())
                )
            if op == z3.Z3_OP_IMPLIES:
                a = eval_bool(expr.arg(0), bound_vals)
                b = eval_bool(expr.arg(1), bound_vals)
                return (not a) or b
            if op == z3.Z3_OP_IFF:
                a = eval_bool(expr.arg(0), bound_vals)
                b = eval_bool(expr.arg(1), bound_vals)
                return a == b
            if op == z3.Z3_OP_EQ:
                return eval_term(expr.arg(0), bound_vals) == eval_term(
                    expr.arg(1), bound_vals
                )
            if op == z3.Z3_OP_DISTINCT:
                vals = [
                    eval_term(expr.arg(i), bound_vals) for i in range(expr.num_args())
                ]
                return len(set(vals)) == len(vals)

            # Uninterpreted predicates from our verification context.
            name = expr.decl().name()
            if name in {"ON_star", "ON_star_zero"}:
                a = eval_term(expr.arg(0), bound_vals)
                b = eval_term(expr.arg(1), bound_vals)
                return on_util.on_star_implementation(
                    all_block_pos[a], all_block_pos[b]
                )
            if name == "Higher":
                a = eval_term(expr.arg(0), bound_vals)
                b = eval_term(expr.arg(1), bound_vals)
                return on_util.higher_implementation(all_block_pos[a], all_block_pos[b])
            if name == "Scattered":
                a = eval_term(expr.arg(0), bound_vals)
                b = eval_term(expr.arg(1), bound_vals)
                return on_util.scattered_implementation(
                    all_block_pos[a], all_block_pos[b]
                )
            if name == "Top":
                a = eval_term(expr.arg(0), bound_vals)
                return on_util.top_implementation(all_block_pos[a], all_block_pos)

            raise TypeError(
                f"Unsupported Z3 operator/predicate in guard: {name} ({expr})"
            )

        return eval_bool(z3_expr, [])

    def _find_and_bind_guard_exists(self, env, traj) -> bool:
        """Try to satisfy `instantiated_cond` by choosing guard_exists_vars.

        If satisfiable, mutate `env.symbolic_name_to_box_id` to bind the chosen
        existential variables (by name) to concrete block IDs, and return True.
        """
        mapping = getattr(env, "symbolic_name_to_box_id", None)
        if mapping is None or not isinstance(mapping, dict):
            raise ValueError(
                "While.eval requires env.symbolic_name_to_box_id to exist as a dict[str, int]."
            )

        obs = traj[-1]
        num_blocks = self._get_num_blocks(env, obs)

        all_block_pos = self._build_block_positions(obs, num_blocks)
        all_ids = list(range(num_blocks))

        # Base bindings come from current symbolic mapping.
        base_bindings = {str(k): int(v) for k, v in mapping.items()}

        # Support multiple existential vars by nested iteration.
        guard_names = [
            v.decl().name() if hasattr(v, "decl") else str(v)
            for v in self.guard_exists_vars
        ]

        def rec(i, cur_bindings):
            if i == len(guard_names):
                return (
                    cur_bindings
                    if self._eval_z3_guard(
                        self.instantiated_cond, cur_bindings, all_block_pos
                    )
                    else None
                )
            name_i = guard_names[i]
            for bid in all_ids:
                nxt = dict(cur_bindings)
                nxt[name_i] = bid
                sol = rec(i + 1, nxt)
                if sol is not None:
                    return sol
            return None

        sol = rec(0, base_bindings)
        if sol is None:
            return False
        for name in guard_names:
            mapping[name] = int(sol[name])
        return True

    def eval(self, env, traj, return_image=False) -> List:
        # Runtime execution: evaluate the synthesized (Z3) guard by searching over
        # concrete blocks, and bind the existential variable(s) into the env mapping.
        imgs: List = []
        iters = 0
        while self._find_and_bind_guard_exists(env, traj):
            iters += 1
            if iters > self.max_iters:
                break
            for instr in self.body:
                imgs.extend(instr.eval(env, traj, return_image=return_image))
        return imgs

    def register_trainable_parameter(self, parameters: List):
        for instr in self.body:
            instr.register_trainable_parameter(parameters)
        return parameters

    def update_trainable_parameter(self, new_parameter: List):
        for instr in self.body:
            instr.update_trainable_parameter(new_parameter)

    def get_operand(self):
        return []

    def set_operand(self, new_operands):
        pass

    def __eq__(self, other):
        if not isinstance(other, While):
            return False
        cond1 = self.instantiated_cond == other.instantiated_cond
        cond2 = self.guard_exists_vars == other.guard_exists_vars
        cond3 = self.body == other.body
        cond4 = self.invariant == other.invariant
        return cond1 and cond2 and cond3 and cond4

    def __str__(self):
        return f"while({self.instantiated_cond}, {self.guard_exists_vars}, {self.body}, {self.invariant})"


class Put(Instruction):
    def __init__(self, upper_block, base_block):
        self.base_block = base_block
        self.upper_block = upper_block

    def __str__(self):
        return f"put({self.base_block}, {self.upper_block})"

    def eval(self, env, traj, return_image=False) -> List:
        # `Put` is a logical/table-level operation in the stack DSL.
        # For physical execution, the program should typically replace
        # `Put`/`Assign`/`While` with concrete pick-and-place instructions.
        #
        # We intentionally do not mutate `env.symbolic_name_to_box_id` here:
        # symbolic aliases are established explicitly via `Assign`.
        raise RuntimeError(
            "Put.eval() was called, but `Put` is a verification-only logical "
            "operation in the stack DSL. It cannot be evaluated at runtime. "
            "Replace `Put`/`Assign`/`While` with concrete pick-and-place instructions "
            "before execution."
        )


class Assign(Instruction):
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __str__(self):
        return f"{self.left} <- {self.right}"

    def eval(self, env, traj, return_image=False) -> List:
        mapping = getattr(env, "symbolic_name_to_box_id", None)
        if mapping is None:
            raise ValueError(
                "Assign.eval requires env.symbolic_name_to_box_id to exist."
            )
        if not isinstance(mapping, dict):
            raise TypeError("env.symbolic_name_to_box_id must be a dict[str, int].")

        if self.right not in mapping:
            raise KeyError(
                f"Assign.eval could not resolve RHS symbolic name {self.right!r}; "
                f"available: {sorted(mapping.keys())}"
            )

        # Create/update the alias: env[left] = env[right].
        mapping[self.left] = mapping[self.right]
        return []


class GoalAssign(Instruction):
    """Verification-only assignment over GoalSort symbols."""

    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __str__(self):
        return f"{self.left} <- {self.right} (goal)"

    def eval(self, env, traj, return_image=False) -> List:
        raise RuntimeError(
            "GoalAssign.eval() is verification-only. "
            "Use concrete low-level instructions for execution."
        )


class MarkGoal(Instruction):
    """Verification-only mark update Mark(x) := Mark(x) or x==target."""

    def __init__(self, target):
        self.target = target

    def __str__(self):
        return f"mark({self.target})"

    def eval(self, env, traj, return_image=False) -> List:
        raise RuntimeError(
            "MarkGoal.eval() is verification-only. "
            "Use concrete low-level instructions for execution."
        )


class MoveRight(Instruction):
    """Verification-only nondeterministic update x := x.r."""

    def __init__(self, var_name):
        self.var_name = var_name

    def __str__(self):
        return f"{self.var_name} := {self.var_name}.r"

    def eval(self, env, traj, return_image=False) -> List:
        raise RuntimeError(
            "MoveRight.eval() is verification-only. "
            "Use concrete low-level instructions for execution."
        )


class MoveDown(Instruction):
    """Verification-only nondeterministic update x := x.d."""

    def __init__(self, var_name):
        self.var_name = var_name

    def __str__(self):
        return f"{self.var_name} := {self.var_name}.d"

    def eval(self, env, traj, return_image=False) -> List:
        raise RuntimeError(
            "MoveDown.eval() is verification-only. "
            "Use concrete low-level instructions for execution."
        )


class Seq:
    def __init__(self, s1, s2):
        self.s1 = s1
        self.s2 = s2

    def __str__(self):
        return f"Seq({self.s1}, {self.s2})"
