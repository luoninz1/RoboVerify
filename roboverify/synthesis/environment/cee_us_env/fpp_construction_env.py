import os
from abc import ABC

import mujoco_py
import numpy as np
import torch
from gym import spaces
from gym.envs.robotics.rotations import euler2quat
from gym.utils import EzPickle

import synthesis.environment.cee_us_env.torch_helpers as torch_helpers
from synthesis.environment.cee_us_env.abstract_environments import MaskedGoalSpaceEnvironmentInterface
from synthesis.environment.cee_us_env.fpp_construction.construction import FetchBlockConstructionEnv
from synthesis.environment.cee_us_env.robotics import GymRoboticsGroundTruthSupportEnv
from synthesis.environment.stack_reset import sample_stack_xy
from synthesis.util.actions import bound_delta_action
from synthesis.util.on import BLOCK_LENGTH, on as on_relation

ROBOVERIFY_PYRAMID_NUM_BLOCKS = 6
ROBOVERIFY_PYRAMID_LAYER_SIZES = (3, 2, 1)

ROBOVERIFY_GRID_WORKSPACE_X_OFFSET = 0.0
ROBOVERIFY_GRID_BASE_Y_CLEARANCE = 0.08
ROBOVERIFY_GRID_BLOCK_Y_OFFSET = 0.22
ROBOVERIFY_GRID_GOAL_Y_OFFSET = 0.10
ROBOVERIFY_GRID_BLOCK_Y_HALF_RANGE = 0.12
ROBOVERIFY_GRID_MIN_X_AHEAD_OF_BASE = 0.35
GRIPPER_XY_CLEARANCE = 0.10

ROBOVERIFY_CASES = (
    "RoboVerifyStack",
    "RoboVerifyUnstack",
    "RoboVerifyReverse",
    "RoboVerifyPartialStack",
)

import pdb

class FetchPickAndPlaceConstruction(
    MaskedGoalSpaceEnvironmentInterface,
    GymRoboticsGroundTruthSupportEnv,
    FetchBlockConstructionEnv,
):
    def __init__(
        self,
        *,
        name,
        sparse,
        shaped_reward,
        simple=False,
        base_block_id=None,
        **kwargs,
    ):
        self.shaped_reward = shaped_reward
        self.sparse = sparse
        self.simple = simple

        FetchBlockConstructionEnv.__init__(self, **kwargs)
        GymRoboticsGroundTruthSupportEnv.__init__(self, name=name, **kwargs)

        if self.case in ROBOVERIFY_CASES:
            # table0 is a horizontal plane; its world z is the physical surface,
            # unlike height_offset, which is the resting block-center height.
            # Keep this motion-level fact separate from observation packing and
            # the relational tbl marker, which has no coordinates.
            self.table_surface_height = float(self.sim.data.get_geom_xpos("table0")[2])
            if base_block_id is None:
                raise ValueError(f"{self.case} requires base_block_id.")
            if not isinstance(base_block_id, int):
                raise TypeError(f"base_block_id must be an int for {self.case}.")
            if not (0 <= base_block_id < self.num_blocks):
                raise ValueError(
                    f"base_block_id must be in [0, {self.num_blocks - 1}] for {self.case}; got {base_block_id}"
                )
            self.roboverify_base_block_id = int(base_block_id)
            # Mapping used by PickPlaceByName and program execution.
            # Requirement: `b0` must refer to the constructor-chosen base block, not
            # necessarily physical index 0.
            #
            # User intent: initially only `b0` is known. Other symbolic names are
            # created only during program execution (e.g. via `Assign(b, b0)`).
            self.symbolic_name_to_box_id = {"b0": self.roboverify_base_block_id}
        else:
            self.roboverify_base_block_id = None

        self.store_init_arguments(locals())
        EzPickle.__init__(
            self,
            name=name,
            sparse=sparse,
            shaped_reward=shaped_reward,
            base_block_id=base_block_id,
            **kwargs,
        )

        # These are the set attributes that will be used by the object-centric world models and controllers
        self.agent_dim = 10
        self.object_dyn_dim = 12
        self.object_stat_dim = 0
        self.nObj = self.num_blocks

        assert isinstance(self.observation_space, spaces.Dict)
        orig_obs_len = self.observation_space.spaces["observation"].shape[0]
        goal_space_size = self.observation_space.spaces["desired_goal"].shape[0]

        goal_idx = np.arange(orig_obs_len, orig_obs_len + goal_space_size)

        if self.case == "Flip":
            # agent_dim is 10, per_object_dim is 12 (The 3-6 elements of each object contain the euler angles!)
            achieved_goal_idx = [np.arange(10 + i * 12 + 3, 10 + i * 12 + 6) for i in range(self.num_blocks)]
        else:
            # agent_dim = 10, per_object_dim = 12 (First three elements of each object contain the locations!)
            achieved_goal_idx = [np.arange(10 + i * 12, 10 + i * 12 + 3) for i in range(self.num_blocks)]
        achieved_goal_idx.append([0, 1, 2])
        achieved_goal_idx = np.asarray(achieved_goal_idx).flatten()

        self.goal_idx = goal_idx
        self.achieved_goal_idx = achieved_goal_idx

        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(orig_obs_len + goal_space_size,), dtype="float32")
        self.observation_space_size_preproc = self.obs_preproc(self.flatten_observation(self._get_obs())).shape[0]
        self.goal_space_size = goal_space_size  # Should we equal to num_objects * 3 + 3 for the gripper pos!

        if (
            "tower" in self.case
            or self.case == "Pyramid"
            or self.case in ROBOVERIFY_CASES
            or self.case == "RoboVerifyGrid"
            or self.case == "RoboVerifyPyramid"
        ):
            self.threshold = 0.02
        elif self.case == "PickAndPlace":
            self.threshold = 0.025
        elif self.case == "Slide":
            self.threshold = 0.1
        elif self.case == "Flip":
            self.threshold = 0.087 * 2  # In radians! this is threshold for the euler angles
        else:
            self.threshold = 0.05

        MaskedGoalSpaceEnvironmentInterface.__init__(
            self,
            name=name,
            goal_idx=goal_idx,
            achieved_goal_idx=achieved_goal_idx,
            sparse=sparse,
            threshold=self.threshold,
        )

        self.goal_idx_tensor = torch.tensor(
            goal_idx,
            dtype=torch.int32,
            requires_grad=False,
            device=torch_helpers.device,
        )
        self.achieved_goal_idx_tensor = torch.tensor(
            achieved_goal_idx,
            dtype=torch.int32,
            requires_grad=False,
            device=torch_helpers.device,
        )

        # Only for Flip and Slide tasks we need these additional parameters
        # For these tasks subgoal_distances_per_xyz is called!
        if self.case == "Flip":
            # Only caring about euler angles for x and y axis
            self.coord_weights = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32).to(torch_helpers.device)
            self.buffer_threshold = torch.tensor(1e-4, dtype=torch.float32).to(torch_helpers.device)
            self.cost_thres = self.coord_weights * 0.087  # 5 degrees per coordinate
        elif self.case == "Slide":
            self.coord_weights = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).to(torch_helpers.device)
            self.buffer_threshold = torch.tensor([0.1, 0.1, self.height_offset], dtype=torch.float32).to(
                torch_helpers.device
            )
            self.cost_thres = self.coord_weights * 0.1  # object center can be 5cm away from center of the goal pad
            self.cost_thres[-1] = self.height_offset
        else:
            self.buffer_threshold = torch.tensor(0.025, dtype=torch.float32).to(torch_helpers.device)

        self.robot_base_xy = np.array([0.69189994, 0.74409998])
        self.manipulability_r = 0.8531

        self.robot_base_xy_tensor = torch.tensor(
            self.robot_base_xy,
            dtype=torch.float32,
            requires_grad=False,
            device=torch_helpers.device,
        )

        self.gripper_init = np.array([1.344, 0.749, 0.5319])
        self.gripper_init_tensor = torch.tensor(
            self.gripper_init,
            dtype=torch.float32,
            requires_grad=False,
            device=torch_helpers.device,
        )

    def obs_preproc(self, obs):
        return self.observation_wo_goal(obs)

    def targ_proc(self, observations, next_observations):
        return next_observations - observations

    def obs_postproc(self, obs, pred=None, out=None):
        if pred is not None:
            obs = obs + pred
        if torch.is_tensor(obs):
            goal_tensor = torch_helpers.to_tensor(self.goal.copy()).to(torch_helpers.device)
            return self.append_goal_to_observation_tensor(obs, goal_tensor)
        else:
            return self.append_goal_to_observation(obs, self.goal.copy())

    def _step_callback(self):
        # we need to call forward because part of the model was overwritten and
        # it is not consistent
        self.sim.forward()

    def get_pos_vel_of_joints(self, names):
        if self.sim.data.qpos is not None and self.sim.model.joint_names:
            return (
                np.array([self.sim.data.get_joint_qpos(name) for name in names]),
                np.array([self.sim.data.get_joint_qvel(name) for name in names]),
            )

    def set_pos_vel_of_joints(self, names, q_pos, q_vel):
        if self.sim.data.qpos is not None and self.sim.model.joint_names:
            for n, p, v in zip(names, q_pos, q_vel):
                self.sim.data.set_joint_qpos(n, p)
                self.sim.data.set_joint_qvel(n, v)

    @staticmethod
    def flatten_observation(obs):
        musk = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                          1.0, 1.0, 1.0, 0.0, 0.0, 0.0
                         ])
        return np.concatenate((obs["observation"], obs["desired_goal"]))

    def step(self, action):
        action = bound_delta_action(action)
        self._set_action(action)
        try:
            self.sim.step()
        except mujoco_py.builder.MujocoException as e:
            print(e)
            print(f"action {action}")
        self._step_callback()
        obs = self._get_obs()

        done = False

        if "image" in self.obs_type:
            reward = self.compute_reward_image()
            if reward < 0.05:
                info = {
                    "is_success": True,
                }
            else:
                info = {
                    "is_success": False,
                }
        elif "state" in self.obs_type:
            info = {
                "is_success": self._is_success(np.concatenate((obs["observation"], self.goal))),
            }
            reward = self.compute_reward(obs)
        else:
            raise ("Obs_type not recognized")
        return self.flatten_observation(obs), reward, done, info

    # rewrite sample goal for simple case
    def _sample_goal_simple(self):
        # get case
        cases = [
            "Singletower",
            "Pyramid",
            "Multitower",
            "Slide",
            "PickAndPlace",
            "Flip",
            "RoboVerifyStack",
            "RoboVerifyUnstack",
            "RoboVerifyReverse",
            "RoboVerifyPartialStack",
            "RoboVerifyGrid",
            "RoboVerifyPyramid",
        ]
        if self.case == "All":
            case_id = np.random.randint(0, len(cases))
            case = cases[case_id]
        elif self.case in cases:
            case = self.case
        else:
            raise NotImplementedError

        if case == "RoboVerifyGrid":
            return self._roboverify_grid_goals_flat().copy()
        if case == "RoboVerifyPyramid":
            return self._roboverify_pyramid_goals_flat().copy()

        goals = []
        objs = []

        # single tower
        if case == "Singletower":
            # get object positions
            for i in range(self.num_blocks):
                object_i_pos = self.sim.data.get_site_xpos(self.object_names[i])
                objs.append(object_i_pos)

            target_offset = np.array([-0.05, 0.0, 0.0])
            goal_object0 = self.initial_gripper_xpos[:3] + np.random.uniform(
                -self.target_range, self.target_range, size=3
            )
            goal_object0 += target_offset
            goal_object0[2] = self.height_offset

            while not (np.all([np.linalg.norm(goal_object0[:2] - obj_pos[:2]) > 0.071 for obj_pos in objs])):
                goal_object0[:2] = self.initial_gripper_xpos[:2] + np.random.uniform(
                    -self.target_range, self.target_range, size=2
                )
                goal_object0[:2] += target_offset[:2]

            if self.target_in_the_air and np.random.uniform() < 0.5 and not self.stack_only:
                # If we're only stacking, do not allow the block0 to be in the air
                goal_object0[2] += np.random.uniform(0, 0.45)

            # Start off goals array with the first block
            goals.append(goal_object0)

            # These below don't have goal object0 because only object0+ can be used for towers in PNP stage. In stack stage,
            previous_xys = [goal_object0[:2]]
            current_tower_heights = [goal_object0[2]]

            num_configured_blocks = self.num_blocks - 1

            for i in range(num_configured_blocks):
                if hasattr(self, "stack_only") and self.stack_only:
                    # If stack only, use the object0 position as a base
                    goal_objecti = goal_object0[:2]
                    objecti_xy = goal_objecti
                else:
                    objecti_xy = self.initial_gripper_xpos[:2] + np.random.uniform(
                        -self.target_range, self.target_range, size=2
                    )
                    # Keep rolling if the other block xys are too close to the green block xy
                    # This is because the green block is sometimes lifted into the air
                    # while np.linalg.norm(objecti_xy - goal_object0[0:2]) < 0.071:
                    #     objecti_xy = self.initial_gripper_xpos[:2] + np.random.uniform(
                    #         -self.target_range, self.target_range, size=2
                    #     )
                    while not (np.all([np.linalg.norm(objecti_xy - goal[:2]) > 0.071 for goal in goals]) and np.all([np.linalg.norm(objecti_xy - obj_pos[:2]) > 0.071 for obj_pos in objs])):
                        objecti_xy = self.initial_gripper_xpos[:2] + np.random.uniform(
                            -self.target_range, self.target_range, size=2
                    )
                    goal_objecti = objecti_xy

                # Check if any of current block xy matches any previous xy's
                for _ in range(len(previous_xys)):
                    previous_xy = previous_xys[_]
                    if np.linalg.norm(previous_xy - objecti_xy) < 0.071:
                        goal_objecti = previous_xy

                        new_height_offset = current_tower_heights[_] + 0.05
                        current_tower_heights[_] = new_height_offset
                        goal_objecti = np.append(goal_objecti, new_height_offset)

                # If we didn't find a previous height at the xy.. just put the block at table height and update the previous xys array
                if len(goal_objecti) == 2:
                    goal_objecti = np.append(goal_objecti, self.height_offset)
                    previous_xys.append(objecti_xy)
                    current_tower_heights.append(self.height_offset)

                goals.append(goal_objecti)

        # pick and place (simple)
        elif case == "PickAndPlace":
            # get object positions
            for i in range(self.num_blocks):
                object_i_pos = self.sim.data.get_site_xpos(self.object_names[i])
                objs.append(object_i_pos)

            target_offset = np.array([-0.05, 0.0, 0.0])  # Added this to be closer to robot base (for manipulability)
            goal_object0 = self.initial_gripper_xpos[:3] + np.random.uniform(
                -self.target_range, self.target_range, size=3
            )
            goal_object0 += target_offset
            goal_object0[2] = self.height_offset

            if np.random.uniform() < 0.5:  # normally 0.5 (fifty-fifty)
                # If we're only stacking, do not allow the block0 to be in the air
                goal_object0[2] += np.random.uniform(0.1, 0.45)

            # Start off goals array with the first block
            goals.append(goal_object0)
            for i in range(self.num_blocks - 1):
                objecti_xy = self.initial_gripper_xpos[:2] + np.random.uniform(
                    -self.target_range, self.target_range, size=2
                )
                while not (np.all([np.linalg.norm(objecti_xy - goal[:2]) > 0.071 for goal in goals]) and np.all([np.linalg.norm(objecti_xy - obj_pos[:2]) > 0.071 for obj_pos in objs])):
                    objecti_xy = self.initial_gripper_xpos[:2] + np.random.uniform(
                        -self.target_range, self.target_range, size=2
                    )
                goal_objecti = np.zeros(3)
                goal_objecti[:2] = objecti_xy
                goal_objecti[2] = self.height_offset
                goals.append(goal_objecti)
            goals[0], goals[-1] = (
                goals[-1],
                goals[0],
            )  # Switch first and last obj xy (last object should be lifted!)
        elif case in ROBOVERIFY_CASES:
            # RoboVerify success/reward ignores the sampled goal; it checks layout
            # from achieved object positions. Use the default goal sampler to ensure
            # correct shape without introducing extra constraints here.
            return self._sample_goal()
        else:
            return self._sample_goal()

        goals.append([0.0, 0.0, 0.0])
        return np.concatenate(goals, axis=0).copy()

    def _reset_sim(self):
        if self.case == "RoboVerifyStack":
            return self._reset_sim_roboverify_stack()
        if self.case == "RoboVerifyPartialStack":
            return self._reset_sim_roboverify_partial_stack()
        if self.case == "RoboVerifyGrid":
            return self._reset_sim_roboverify_grid()
        if self.case == "RoboVerifyPyramid":
            return self._reset_sim_roboverify_pyramid()
        if self.case in ("RoboVerifyUnstack", "RoboVerifyReverse"):
            return self._reset_sim_roboverify_tower()
        return super()._reset_sim()

    def _reset_sim_roboverify_stack(self):
        """
        RoboVerifyStack init:
        - Randomize all blocks on the tabletop within 0.70 m XY of the robot base.
        - Enforce pairwise "scattered" separation in XY for every block pair.
        """
        self.sim.set_state(self.initial_state)
        positions = sample_stack_xy(
            self.num_blocks, self.robot_base_xy, self.initial_gripper_xpos[:2]
        )
        for obj_name, object_xypos in zip(self.object_names, positions):
            object_qpos = self.sim.data.get_joint_qpos(f"{obj_name}:joint")
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xypos
            object_qpos[2] = self.height_offset
            self.sim.data.set_joint_qpos(f"{obj_name}:joint", object_qpos)
        self.sim.forward()
        return True

    def _reset_sim_roboverify_tower(self):
        """
        RoboVerifyUnstack / RoboVerifyReverse init:
        - Place all blocks in a single tower with `base_block_id` (b0) at the bottom.
        """
        self.sim.set_state(self.initial_state)

        base_id = self.roboverify_base_block_id
        tower_order = self._roboverify_canonical_tower_order()
        block_height = BLOCK_LENGTH

        while True:
            tower_xy = self.initial_gripper_xpos[:2] + np.random.uniform(
                -self.obj_range, self.obj_range, size=2
            )
            if np.linalg.norm(tower_xy - self.initial_gripper_xpos[:2]) >= 0.1:
                break

        for level, block_id in enumerate(tower_order):
            obj_name = self.object_names[block_id]
            object_qpos = self.sim.data.get_joint_qpos(f"{obj_name}:joint")
            assert object_qpos.shape == (7,)
            object_qpos[:2] = tower_xy
            object_qpos[2] = self.height_offset + level * block_height
            self.sim.data.set_joint_qpos(f"{obj_name}:joint", object_qpos)

        self.sim.forward()
        return True

    def _roboverify_random_tower_partition(self, num_towers: int) -> list[list[int]]:
        if num_towers < 2:
            raise ValueError("partial stack requires at least two towers.")
        if num_towers > self.num_blocks:
            raise ValueError("cannot have more towers than blocks.")

        order = np.random.permutation(self.num_blocks)
        split_points = sorted(
            np.random.choice(np.arange(1, self.num_blocks), size=num_towers - 1, replace=False)
        )
        towers: list[list[int]] = []
        prev = 0
        for split in split_points:
            towers.append(order[prev:split].tolist())
            prev = int(split)
        towers.append(order[prev:].tolist())
        return towers

    def _roboverify_order_partial_tower_blocks(self, blocks: list[int]) -> list[int]:
        """Allow ``b0`` to appear in the middle of a multi-block tower when possible."""
        base_id = self.roboverify_base_block_id
        ordered = list(blocks)
        if base_id not in ordered or len(ordered) < 3:
            return ordered

        internal_positions = list(range(1, len(ordered) - 1))
        pos = int(np.random.choice(internal_positions))
        ordered.remove(base_id)
        ordered.insert(pos, base_id)
        return ordered

    def _sample_scattered_tower_xy_positions(self, num_towers: int) -> list[np.ndarray]:
        scattered_sep = 2.0 * BLOCK_LENGTH
        tower_xypos: list[np.ndarray] = []

        while len(tower_xypos) < num_towers:
            candidate = self.initial_gripper_xpos[:2] + np.random.uniform(
                -self.obj_range, self.obj_range, size=2
            )
            if np.linalg.norm(candidate - self.initial_gripper_xpos[:2]) < 0.1:
                continue

            ok = True
            for other_xypos in tower_xypos:
                dx = abs(candidate[0] - other_xypos[0])
                dy = abs(candidate[1] - other_xypos[1])
                if not (dx >= scattered_sep or dy >= scattered_sep):
                    ok = False
                    break

            if ok:
                tower_xypos.append(candidate)

        return tower_xypos

    def _reset_sim_roboverify_partial_stack(self):
        """
        RoboVerifyPartialStack init:
        - Partition blocks into at least two partial towers (height >= 1 each).
        - ``b0`` may appear in the middle of a multi-block tower.
        """
        if self.num_blocks < 2:
            raise ValueError("RoboVerifyPartialStack requires at least two blocks.")

        self.sim.set_state(self.initial_state)

        num_towers = int(np.random.randint(2, self.num_blocks + 1))
        towers = self._roboverify_random_tower_partition(num_towers)
        tower_xypos = self._sample_scattered_tower_xy_positions(len(towers))
        block_height = BLOCK_LENGTH

        for tower_blocks, tower_xy in zip(towers, tower_xypos):
            if self.roboverify_base_block_id in tower_blocks:
                tower_blocks = self._roboverify_order_partial_tower_blocks(tower_blocks)

            for level, block_id in enumerate(tower_blocks):
                obj_name = self.object_names[block_id]
                object_qpos = self.sim.data.get_joint_qpos(f"{obj_name}:joint")
                assert object_qpos.shape == (7,)
                object_qpos[:2] = tower_xy
                object_qpos[2] = self.height_offset + level * block_height
                self.sim.data.set_joint_qpos(f"{obj_name}:joint", object_qpos)

        self.sim.forward()
        return True

    def _roboverify_grid_block_accept(self, candidate: np.ndarray) -> bool:
        base_x, base_y = self.robot_base_xy
        if np.linalg.norm(candidate - self.initial_gripper_xpos[:2]) < GRIPPER_XY_CLEARANCE:
            return False
        if candidate[1] >= base_y - ROBOVERIFY_GRID_BASE_Y_CLEARANCE:
            return False
        if candidate[0] < base_x + ROBOVERIFY_GRID_MIN_X_AHEAD_OF_BASE:
            return False
        return True

    def _roboverify_random_scattered_layout(
        self,
        count: int,
        center_xy: np.ndarray,
        *,
        x_half_range: float,
        y_half_range: float,
        max_attempts: int = 200,
        layout_retries: int = 20,
        accept_candidate=None,
    ) -> list[np.ndarray]:
        """Randomly scatter blocks in a rectangular region with pairwise separation."""
        scattered_sep = 2.0 * BLOCK_LENGTH
        center_xy = np.asarray(center_xy, dtype=np.float32)

        for _ in range(layout_retries):
            positions: list[np.ndarray] = []
            for _ in range(count):
                for _ in range(max_attempts):
                    candidate = center_xy + np.random.uniform(
                        [-x_half_range, -y_half_range],
                        [x_half_range, y_half_range],
                        size=2,
                    ).astype(np.float32)
                    if accept_candidate is not None and not accept_candidate(candidate):
                        continue
                    elif accept_candidate is None:
                        if np.linalg.norm(candidate - self.initial_gripper_xpos[:2]) < GRIPPER_XY_CLEARANCE:
                            continue
                    if all(
                        abs(candidate[0] - other[0]) >= scattered_sep
                        or abs(candidate[1] - other[1]) >= scattered_sep
                        for other in positions
                    ):
                        positions.append(candidate)
                        break
                else:
                    break
            else:
                return positions

        fallback = self._roboverify_deterministic_scattered_layout(
            count,
            center_xy,
            cols=max(2, int(np.ceil(np.sqrt(count)))),
            jitter=0.03,
        )
        order = np.random.permutation(count)
        return [fallback[int(i)] for i in order]

    def _roboverify_deterministic_scattered_layout(
        self,
        count: int,
        center_xy: np.ndarray,
        *,
        cols: int | None = None,
        sep: float | None = None,
        jitter: float = 0.0,
    ) -> list[np.ndarray]:
        """Place ``count`` XY positions on a scattered grid without rejection sampling."""
        sep = float(2.0 * BLOCK_LENGTH if sep is None else sep)
        if cols is None:
            cols = int(np.ceil(np.sqrt(count)))
        rows = int(np.ceil(count / cols))
        width = (cols - 1) * sep
        height = (rows - 1) * sep
        origin = np.asarray(center_xy, dtype=np.float32) - np.array(
            [0.5 * width, 0.5 * height], dtype=np.float32
        )

        positions: list[np.ndarray] = []
        for index in range(count):
            row, col = divmod(index, cols)
            xy = origin + np.array([col * sep, row * sep], dtype=np.float32)
            if jitter > 0.0:
                xy += np.random.uniform(-jitter, jitter, size=2).astype(np.float32)
            positions.append(xy)
        return positions

    def _roboverify_grid_goal_origin_xy(self) -> np.ndarray:
        spacing = self._roboverify_grid_cell_spacing()
        grid_width = max(0, self.grid_cols - 1) * spacing
        workspace_x = self.initial_gripper_xpos[0] + ROBOVERIFY_GRID_WORKSPACE_X_OFFSET
        return np.array(
            [
                workspace_x - 0.5 * grid_width,
                self.robot_base_xy[1] + ROBOVERIFY_GRID_GOAL_Y_OFFSET,
            ],
            dtype=np.float32,
        )

    def _roboverify_grid_block_center_xy(self) -> np.ndarray:
        workspace_x = self.initial_gripper_xpos[0] + ROBOVERIFY_GRID_WORKSPACE_X_OFFSET
        return np.array(
            [
                workspace_x,
                self.robot_base_xy[1] - ROBOVERIFY_GRID_BLOCK_Y_OFFSET,
            ],
            dtype=np.float32,
        )

    def _roboverify_pyramid_goal_origin_xy(self) -> np.ndarray:
        row_spacing = self._roboverify_pyramid_row_spacing()
        base_width = (ROBOVERIFY_PYRAMID_LAYER_SIZES[0] - 1) * row_spacing
        workspace_x = self.initial_gripper_xpos[0] + ROBOVERIFY_GRID_WORKSPACE_X_OFFSET
        return np.array(
            [
                workspace_x - 0.5 * base_width,
                self.robot_base_xy[1] + ROBOVERIFY_GRID_GOAL_Y_OFFSET,
            ],
            dtype=np.float32,
        )

    def _roboverify_layout_cell_spacing(self) -> float:
        # Adjacent goal centers are one block length plus a 0.3-block-length gap.
        return BLOCK_LENGTH + 0.3 * BLOCK_LENGTH

    def _roboverify_pyramid_row_spacing(self) -> float:
        return self._roboverify_layout_cell_spacing()

    def _roboverify_pyramid_goal_positions(self) -> np.ndarray:
        row_spacing = self._roboverify_pyramid_row_spacing()
        origin_x, origin_y = self._roboverify_pyramid_goal_origin_xy()
        base_width = (ROBOVERIFY_PYRAMID_LAYER_SIZES[0] - 1) * row_spacing
        goals = np.zeros((self.num_blocks, 3), dtype=np.float32)
        block_id = 0
        for layer_idx, layer_size in enumerate(ROBOVERIFY_PYRAMID_LAYER_SIZES):
            z = self.height_offset + layer_idx * BLOCK_LENGTH
            layer_width = (layer_size - 1) * row_spacing
            start_x = float(origin_x) + 0.5 * (base_width - layer_width)
            for col in range(layer_size):
                goals[block_id] = [start_x + col * row_spacing, float(origin_y), z]
                block_id += 1
        return goals

    def _roboverify_pyramid_goals_flat(self) -> np.ndarray:
        goals = self._roboverify_pyramid_goal_positions()
        return np.concatenate([goals.reshape(-1), np.zeros(3, dtype=np.float32)])

    def _roboverify_grid_cell_spacing(self) -> float:
        return self._roboverify_layout_cell_spacing()

    def _roboverify_grid_goal_positions(self) -> np.ndarray:
        spacing = self._roboverify_grid_cell_spacing()
        goal_origin = self._roboverify_grid_goal_origin_xy()
        origin_x = float(goal_origin[0])
        origin_y = float(goal_origin[1])
        goals = np.zeros((self.num_blocks, 3), dtype=np.float32)
        for block_id in range(self.num_blocks):
            row = block_id // self.grid_cols
            col = block_id % self.grid_cols
            goals[block_id] = [
                origin_x + col * spacing,
                origin_y + row * spacing,
                self.height_offset,
            ]
        return goals

    def _roboverify_grid_goals_flat(self) -> np.ndarray:
        goals = self._roboverify_grid_goal_positions()
        return np.concatenate([goals.reshape(-1), np.zeros(3, dtype=np.float32)])

    def _roboverify_goal_layout_complete(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray
    ) -> bool:
        achieved_goal = np.asarray(achieved_goal, dtype=np.float32)
        desired_goal = np.asarray(desired_goal, dtype=np.float32)
        dists = self.subgoal_distances(achieved_goal, desired_goal)
        return all(float(d) <= float(self.threshold) for d in dists)

    def _reset_sim_roboverify_workspace_blocks(self) -> None:
        """Scatter blocks to the left of the robot base in the shared workspace."""
        left_positions = self._roboverify_random_scattered_layout(
            self.num_blocks,
            self._roboverify_grid_block_center_xy(),
            x_half_range=self.obj_range,
            y_half_range=ROBOVERIFY_GRID_BLOCK_Y_HALF_RANGE,
            accept_candidate=self._roboverify_grid_block_accept,
        )
        for obj_name, object_xypos in zip(self.object_names, left_positions):
            object_qpos = self.sim.data.get_joint_qpos(f"{obj_name}:joint")
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xypos
            object_qpos[2] = self.height_offset
            self.sim.data.set_joint_qpos(f"{obj_name}:joint", object_qpos)

    def _reset_sim_roboverify_grid(self):
        """
        RoboVerifyGrid init:
        - Scatter blocks to the left of the robot base (lower Y), in the workspace
          in front of the base.
        - Place the goal grid to the right of the robot base (higher Y), also in front.
        """
        self.sim.set_state(self.initial_state)

        self.roboverify_grid_origin_xy = self._roboverify_grid_goal_origin_xy()
        self.roboverify_goal_marker_positions = self._roboverify_grid_goal_positions()
        self._reset_sim_roboverify_workspace_blocks()
        self._update_roboverify_goal_markers()
        return True

    def _reset_sim_roboverify_pyramid(self):
        """
        RoboVerifyPyramid init:
        - Scatter six blocks on the left of the robot base.
        - Show a 3+2+1 pyramid goal marker layout on the right of the robot base.
        """
        self.sim.set_state(self.initial_state)

        self.roboverify_goal_marker_positions = self._roboverify_pyramid_goal_positions()
        self._reset_sim_roboverify_workspace_blocks()
        self._update_roboverify_goal_markers()
        return True

    def _update_roboverify_goal_markers(self):
        """Move brown goal-marker mocap bodies to the target layout."""
        if not hasattr(self, "roboverify_goal_marker_positions"):
            return
        for i in range(self.num_blocks):
            body_id = self.sim.model.body_name2id(f"grid_marker{i}")
            mocap_id = self.sim.model.body_mocapid[body_id]
            self.sim.data.mocap_pos[mocap_id] = self.roboverify_goal_marker_positions[i]
            self.sim.data.mocap_quat[mocap_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.sim.forward()

    def _update_roboverify_grid_markers(self):
        self._update_roboverify_goal_markers()

    def _roboverify_grid_complete(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> bool:
        return self._roboverify_goal_layout_complete(achieved_goal, desired_goal)

    def reset(self):
        # Attempt to reset the simulator.
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        if self.simple:
            self.goal = self._sample_goal_simple().copy()
        else:
            self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        return self.flatten_observation(obs)

    def get_GT_state(self):
        return np.concatenate((super().get_GT_state(), self.goal))

    def set_GT_state(self, state):
        mj_state = state[: -self.goal_space_size]
        self.goal = state[-self.goal_space_size :]
        super().set_GT_state(mj_state)

    def set_state_from_observation(self, observation):
        """Compatibility alias for approximate visualization, not segment reset."""
        return self.set_observation_for_visualization(observation)

    def set_observation_for_visualization(self, observation):
        # This is a dummy function to only visualize the object dynamics!
        mj_state = np.zeros_like(np.concatenate((super().get_GT_state(), self.goal)))
        mj_state[-self.goal_space_size :] = observation[-self.goal_space_size :].copy()
        observation_only_objects = observation[self.agent_dim : -self.goal_space_size]
        obj_positions = [
            np.concatenate(
                [
                    observation_only_objects[i * self.object_dyn_dim : i * self.object_dyn_dim + 3],
                    euler2quat(observation_only_objects[i * self.object_dyn_dim + 3 : i * self.object_dyn_dim + 6]),
                ]
            )
            for i in range(self.num_blocks)
        ]
        obj_positions = np.asarray(obj_positions).flatten()
        # Setting object_positions
        mj_state[16 : 16 + 7 * self.num_blocks] = obj_positions
        mj_state[:16] = np.array(
            [
                4.00000000e-01,
                4.04999826e-01,
                4.80000001e-01,
                4.76618422e-05,
                -1.99439740e-05,
                8.29381627e-11,
                6.00288121e-02,
                9.64127884e-03,
                -8.27177223e-01,
                -2.98843692e-03,
                1.45287872e00,
                2.52114206e-03,
                9.32808214e-01,
                5.95256625e-03,
                3.12759151e-06,
                -3.43083151e-08,
            ]
        )
        self.set_GT_state(mj_state)

    def goal_from_observation(self, observations):
        return np.take(observations, self.goal_idx, -1)

    def achieved_goal_from_observation(self, observations):
        return np.take(observations, self.achieved_goal_idx, -1)

    def observation_wo_goal(self, observation):
        mask = np.ones(observation.shape[-1])
        mask[self.goal_idx] = 0
        return observation[..., mask == 1]

    def append_goal_to_observation(self, observation, goal):
        _goal = np.broadcast_to(goal, (list(observation.shape[:-1]) + [goal.shape[-1]]))
        return np.concatenate([observation, _goal], dim=-1)

    def goal_from_observation_tensor(self, observations):
        return torch.index_select(observations, -1, self.goal_idx_tensor)

    def achieved_goal_from_observation_tensor(self, observations):
        return torch.index_select(observations, -1, self.achieved_goal_idx_tensor)

    def observation_wo_goal_tensor(self, observation):
        mask = torch.ones(observation.shape[-1]).to(torch_helpers.device)
        mask[self.goal_idx_tensor] = 0
        return observation[..., mask == 1]

    def append_goal_to_observation_tensor(self, observation, goal):
        _goal = torch.broadcast_to(goal, (list(observation.shape[:-1]) + [goal.shape[-1]]))
        return torch.cat([observation, _goal], dim=-1)

    def subgoal_distances(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        for i in range(self.num_blocks - 1):
            assert goal_a[..., i * 3 : (i + 1) * 3].shape == goal_a[..., (i + 1) * 3 : (i + 2) * 3].shape
        if torch.is_tensor(goal_a):
            return [
                torch.linalg.norm(goal_a[..., i * 3 : (i + 1) * 3] - goal_b[..., i * 3 : (i + 1) * 3], dim=-1)
                for i in range(self.num_blocks)
            ]
        else:
            return [
                np.linalg.norm(goal_a[..., i * 3 : (i + 1) * 3] - goal_b[..., i * 3 : (i + 1) * 3], axis=-1)
                for i in range(self.num_blocks)
            ]

    def subgoal_distances_per_xyz(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        # The maximum operator is kept in case the env is used with dense rewards and acts as a buffer zone!
        for i in range(self.num_blocks - 1):
            assert goal_a[..., i * 3 : (i + 1) * 3].shape == goal_a[..., (i + 1) * 3 : (i + 2) * 3].shape
        if torch.is_tensor(goal_a):
            return [
                torch.maximum(
                    torch.abs(goal_a[..., i * 3 : (i + 1) * 3] - goal_b[..., i * 3 : (i + 1) * 3]),
                    self.buffer_threshold,
                )
                * self.coord_weights
                for i in range(self.num_blocks)
            ]
        else:
            coord_weights = torch_helpers.to_numpy(self.coord_weights)
            buffer_threshold = torch_helpers.to_numpy(self.buffer_threshold)
            return [
                np.maximum(
                    np.abs(goal_a[..., i * 3 : (i + 1) * 3] - goal_b[..., i * 3 : (i + 1) * 3]),
                    buffer_threshold,
                )
                * coord_weights
                for i in range(self.num_blocks)
            ]

    def subgoal_distances_per_xyz_exp(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        # The maximum operator is kept in case the env is used with dense rewards and acts as a buffer zone!
        if torch.is_tensor(goal_a):
            return [
                1.0
                - torch.exp(
                    -0.5
                    * torch.maximum(
                        torch.abs(goal_a[..., i * 3 : (i + 1) * 3] - goal_b[..., i * 3 : (i + 1) * 3]),
                        self.buffer_threshold,
                    )
                    * self.coord_weights
                )
                for i in range(self.num_blocks)
            ]
        else:
            coord_weights = torch_helpers.to_numpy(self.coord_weights)
            buffer_threshold = torch_helpers.to_numpy(self.buffer_threshold)
            return [
                1.0
                - np.exp(
                    -0.5
                    * np.maximum(
                        np.abs(goal_a[..., i * 3 : (i + 1) * 3] - goal_b[..., i * 3 : (i + 1) * 3]),
                        buffer_threshold,
                    )
                    * coord_weights
                )
                for i in range(self.num_blocks)
            ]

    def gripper_pos_distance_from_next_block(self, gripper_pos, block_pos, next_block_id):
        # Block_pos: nB (xnE) x horizon x 3*(nObj+1)
        if torch.is_tensor(gripper_pos):
            return torch.linalg.norm(gripper_pos - block_pos[..., next_block_id * 3 : (next_block_id + 1) * 3], dim=-1)
        else:
            return [
                np.linalg.norm(gripper_pos - block_pos[..., i * 3 : (i + 1) * 3], axis=-1)
                for i in range(self.num_blocks)
            ]

    def get_next_block_id(self, observation):
        # This function is only valid for MPC where the observations are the current state of the env so same across
        # all samples, (ensemble members)
        assert (observation[..., :] == observation).all()
        goal = self.goal_from_observation_tensor(observation)
        achieved_goal = self.achieved_goal_from_observation_tensor(observation)

        subgoal_distances = self.subgoal_distances(achieved_goal, goal)  # List (len num_blocks) of tensors
        costs_per_object = torch.stack(subgoal_distances)  # num_blocks x batch_size x (nE) x 1

        unsolved_per_object = torch.as_tensor(
            costs_per_object > self.threshold, dtype=torch.float32
        )  # nObj x nB x (xnE)
        next_block_id = self.num_blocks - torch.sum(
            unsolved_per_object[..., 0], dim=0
        )  # Check unsolved for only the first time step!
        next_block_id = next_block_id.type(torch.int64)  # size nB (xNE)

        return next_block_id.view(-1)[0]  # No need for the individual samples or ensemble members!

    def cost_env_constraint_fn(self, obj_pos):
        if torch.is_tensor(obj_pos):
            obj_r = [
                torch.linalg.norm(obj_pos[..., i * 3 : i * 3 + 2] - self.robot_base_xy_tensor, dim=-1)
                for i in range(self.num_blocks)
            ]
            manipulability_radius_per_obj = torch.stack(obj_r)  # num_blocks x batch_size x horizon....
            return torch.sum(
                torch.as_tensor(manipulability_radius_per_obj > self.manipulability_r, dtype=torch.float32),
                dim=0,
            )
        else:
            obj_r = [
                np.linalg.norm(obj_pos[..., i * 3 : i * 3 + 2] - self.robot_base_xy, axis=-1)
                for i in range(self.num_blocks)
            ]
            manipulability_radius_per_obj = np.stack(obj_r)  # num_blocks x batch_size x horizon....
            return np.sum(manipulability_radius_per_obj > self.manipulability_r, axis=0)

    def cost_fn(self, observation, action, next_obs):
        """
        In case of controller observations have shape: num_samples x (nE) x horizon x obs_dim
        """
        if torch.is_tensor(observation):
            # Here assumes torch tensor!
            if len(observation.shape) == 1:  # Extending the dimensions to accomodate batch and horizon dimensions!
                observation = observation[None, None, ...]
                action = action[None, None, ...]
                next_obs = next_obs[None, None, ...]

            start_observation = observation[..., 0, :].view(
                observation.shape[:-2] + (1, -1)
            )  # num_samples x (nE) x 1 (for horizon) x obs_dim

            goal = self.goal_from_observation_tensor(next_obs)
            achieved_goal = self.achieved_goal_from_observation_tensor(next_obs)

            if self.case == "Flip" or self.case == "Slide":
                subgoal_distances = self.subgoal_distances_per_xyz(
                    achieved_goal, goal
                )  # List (len num_blocks) of tensors
                costs_per_object_and_coords = torch.stack(
                    subgoal_distances
                )  # num_blocks x batch_size x horizon .. x num_coordinates(3)
                costs_per_object_sparse = torch.as_tensor(
                    torch.any(costs_per_object_and_coords > self.cost_thres, dim=-1),
                    dtype=torch.float32,
                )

                subgoal_distances_exp = self.subgoal_distances_per_xyz_exp(
                    achieved_goal, goal
                )  # List (len num_blocks) of tensors
                costs_per_object_exp_dense = torch.mean(
                    torch.stack(subgoal_distances_exp), dim=-1
                )  # num_blocks x batch_size x horizon .. x num_coordinates(3)
            else:
                subgoal_distances = self.subgoal_distances(achieved_goal, goal)  # List (len num_blocks) of tensors
                costs_per_object = torch.stack(subgoal_distances)  # num_blocks x batch_size x (nE) x horizon....

            gripper_pos = achieved_goal[..., -3:]
            block_pos = torch.cat((achieved_goal[..., :-3], goal[..., -3:]), dim=-1)

            dist_end_eff_to_next_block = torch.zeros(
                action.shape[:-1],
                dtype=torch.float32,
                requires_grad=False,
                device=torch_helpers.device,
            )

            if self.shaped_reward:
                assert self.case != "Slide", "Shaped reward cannot be used with Slide!"
                if self.case == "Flip":
                    dist_end_eff_to_next_block = torch.linalg.norm(gripper_pos - self.gripper_init_tensor, dim=-1)
                else:
                    unsolved_per_object = torch.as_tensor(
                        costs_per_object > self.threshold, dtype=torch.float32
                    )  # nObj x nB x (xnE) x horizon

                    next_block_id = self.get_next_block_id(start_observation)
                    dist_end_eff_to_next_block = self.gripper_pos_distance_from_next_block(
                        gripper_pos, block_pos, next_block_id
                    )  # nB x horizon

                    #  ------------------- Individual timesteps for next object -----------------------------

                    next_block_id_all = self.num_blocks - torch.sum(unsolved_per_object, dim=0)

                    mask = torch.as_tensor(next_block_id_all <= next_block_id, dtype=torch.float32)
                    dist_end_eff_to_next_block = dist_end_eff_to_next_block * mask

            if self.sparse:
                if self.case == "Flip" or self.case == "Slide":
                    cost = (
                        torch.sum(torch.as_tensor(costs_per_object_sparse, dtype=torch.float32), dim=0)
                        + dist_end_eff_to_next_block * 0.001
                    )
                else:
                    cost = (
                        torch.sum(
                            torch.as_tensor(costs_per_object > self.threshold, dtype=torch.float32),
                            dim=0,
                        )
                        + torch.as_tensor(dist_end_eff_to_next_block > self.threshold, dtype=torch.float32) * 0.5
                    )
            else:
                if "tower" in self.case or self.case == "Pyramid":
                    cost = (
                        torch.sum(
                            torch.as_tensor(costs_per_object > self.threshold, dtype=torch.float32),
                            dim=0,
                        )
                        + dist_end_eff_to_next_block * 0.01
                    )
                elif self.case == "Slide" or self.case == "Flip":
                    cost = (
                        torch.sum(costs_per_object_exp_dense, dim=0) * 1e-3
                        + torch.sum(torch.as_tensor(costs_per_object_sparse, dtype=torch.float32), dim=0)
                        + dist_end_eff_to_next_block * 0.01
                    )
                else:
                    cost = (
                        torch.sum(torch.maximum(costs_per_object, self.buffer_threshold), dim=0)
                        + dist_end_eff_to_next_block * 0.01
                    )
        else:
            raise NotImplementedError
        return cost

    def _roboverify_canonical_tower_order(self) -> list[int]:
        base_id = self.roboverify_base_block_id
        if base_id is None:
            raise ValueError("roboverify_base_block_id must be set.")
        return [base_id] + [i for i in range(self.num_blocks) if i != base_id]

    def _roboverify_tower_order_from_positions(self, positions: np.ndarray) -> list[int] | None:
        """Return bottom-to-top block indices if ``positions`` form a single tower."""
        positions = np.asarray(positions, dtype=np.float32)
        if positions.shape != (self.num_blocks, 3):
            positions = positions.reshape(self.num_blocks, 3)

        table_epsilon = float(self.threshold)
        bottom_candidates = []
        for block_id in range(self.num_blocks):
            if abs(float(positions[block_id, 2]) - float(self.height_offset)) > table_epsilon:
                continue
            if any(
                other_id != block_id and on_relation(positions[block_id], positions[other_id])
                for other_id in range(self.num_blocks)
            ):
                continue
            bottom_candidates.append(block_id)

        if len(bottom_candidates) != 1:
            return None

        order = [bottom_candidates[0]]
        remaining = set(range(self.num_blocks)) - set(order)
        current = order[0]

        while remaining:
            candidates = [
                block_id
                for block_id in remaining
                if on_relation(positions[block_id], positions[current])
            ]
            if len(candidates) != 1:
                return None
            current = candidates[0]
            order.append(current)
            remaining.remove(current)

        return order

    def _roboverify_reverse_tower_complete(self, positions: np.ndarray) -> bool:
        """
        Check whether blocks form a tower in the reverse of the canonical init order.

        Init order (bottom to top): ``[b0, ...]`` from
        :meth:`_roboverify_canonical_tower_order`. Success requires the same
        single-tower layout with block order reversed.
        """
        if self.roboverify_base_block_id is None:
            raise ValueError("roboverify_base_block_id must be set for RoboVerifyReverse.")

        order = self._roboverify_tower_order_from_positions(positions)
        if order is None:
            return False
        return order == list(reversed(self._roboverify_canonical_tower_order()))

    def _roboverify_stack_tower_complete(self, positions: np.ndarray) -> bool:
        """
        Check whether blocks form a single tower above `base_block_id`.

        Tower success:
        1. base block is on the tabletop (z approx equals `self.height_offset`)
         2. there exists a linear chain of blocks where each next block sits "on"
           the current block according to `synthesis.util.on.on()`.
        """
        base_id = self.roboverify_base_block_id
        if base_id is None:
            raise ValueError("roboverify_base_block_id must be set for RoboVerifyStack.")

        positions = np.asarray(positions, dtype=np.float32)
        if positions.shape != (self.num_blocks, 3):
            positions = positions.reshape(self.num_blocks, 3)

        # Table tolerance: reuse the same threshold used for tower-like tasks.
        table_epsilon = float(self.threshold)
        if abs(float(positions[base_id, 2]) - float(self.height_offset)) > table_epsilon:
            return False

        remaining = set(range(self.num_blocks))
        remaining.remove(base_id)
        pos_current = positions[base_id]

        while remaining:
            candidates = [i for i in remaining if on_relation(positions[i], pos_current)]
            if len(candidates) != 1:
                return False
            k = candidates[0]
            remaining.remove(k)
            pos_current = positions[k]

        return True

    def _roboverify_unstack_all_on_ground(self, positions: np.ndarray) -> bool:
        """
        Check whether every block rests on the tabletop with no block on another.

        Success:
        1. each block's z is approximately `self.height_offset`
        2. for all i != j, block i is not on block j
        """
        base_id = self.roboverify_base_block_id
        if base_id is None:
            raise ValueError("roboverify_base_block_id must be set for RoboVerifyUnstack.")

        positions = np.asarray(positions, dtype=np.float32)
        if positions.shape != (self.num_blocks, 3):
            positions = positions.reshape(self.num_blocks, 3)

        table_epsilon = float(self.threshold)
        for i in range(self.num_blocks):
            if abs(float(positions[i, 2]) - float(self.height_offset)) > table_epsilon:
                return False

        for i in range(self.num_blocks):
            for j in range(self.num_blocks):
                if i != j and on_relation(positions[i], positions[j]):
                    return False

        return True

    def _roboverify_task_complete(self, positions: np.ndarray) -> bool:
        if self.case in ("RoboVerifyStack", "RoboVerifyPartialStack"):
            return self._roboverify_stack_tower_complete(positions)
        if self.case == "RoboVerifyUnstack":
            return self._roboverify_unstack_all_on_ground(positions)
        if self.case == "RoboVerifyReverse":
            return self._roboverify_reverse_tower_complete(positions)
        raise ValueError(f"unsupported RoboVerify case: {self.case}")

    def compute_reward(self, obs):
        if self.case in ("RoboVerifyGrid", "RoboVerifyPyramid"):
            achieved_goal = obs["achieved_goal"]
            desired_goal = obs["desired_goal"]
            return (
                1.0
                if self._roboverify_goal_layout_complete(achieved_goal, desired_goal)
                else 0.0
            )

        if self.case not in ROBOVERIFY_CASES:
            return super().compute_reward(obs)

        achieved_goal = obs["achieved_goal"]
        achieved_goal = np.asarray(achieved_goal, dtype=np.float32)
        # achieved_goal layout: [obj0(x,y,z), ..., objN-1(x,y,z), grip(x,y,z)]
        block_positions = achieved_goal[:-3].reshape(self.num_blocks, 3)

        return 1.0 if self._roboverify_task_complete(block_positions) else 0.0

    def _is_success(self, obs):
        if self.case in ("RoboVerifyGrid", "RoboVerifyPyramid"):
            achieved_goal = self.achieved_goal_from_observation(obs)
            desired_goal = self.goal_from_observation(obs)
            return self._roboverify_goal_layout_complete(achieved_goal, desired_goal)

        if self.case not in ROBOVERIFY_CASES:
            success_of_blocks = self.eval_success(obs)
            return success_of_blocks == self.num_blocks

        achieved_goal = self.achieved_goal_from_observation(obs)
        achieved_goal = np.asarray(achieved_goal, dtype=np.float32)

        if achieved_goal.ndim == 1:
            block_positions = achieved_goal[:-3].reshape(self.num_blocks, 3)
            return self._roboverify_task_complete(block_positions)

        # Batch case (rare for step()).
        leading_shape = achieved_goal.shape[:-1]
        flat = achieved_goal.reshape(-1, achieved_goal.shape[-1])
        results = []
        for row in flat:
            block_positions = row[:-3].reshape(self.num_blocks, 3)
            results.append(self._roboverify_task_complete(block_positions))
        return np.asarray(results, dtype=np.bool_).reshape(leading_shape)

    def eval_success(self, observation):
        if self.case in ("RoboVerifyGrid", "RoboVerifyPyramid"):
            if torch.is_tensor(observation):
                obs_np = observation.detach().cpu().numpy()
            else:
                obs_np = observation

            achieved_goal = self.achieved_goal_from_observation(obs_np)
            desired_goal = self.goal_from_observation(obs_np)
            ok = self._roboverify_goal_layout_complete(achieved_goal, desired_goal)
            return float(self.num_blocks) if ok else 0.0

        if self.case in ROBOVERIFY_CASES:
            # Return format matches existing semantics: `num_blocks` when success else 0.
            if torch.is_tensor(observation):
                obs_np = observation.detach().cpu().numpy()
            else:
                obs_np = observation

            achieved_goal = self.achieved_goal_from_observation(obs_np)
            achieved_goal = np.asarray(achieved_goal, dtype=np.float32)

            achieved_dim = achieved_goal.shape[-1]
            if achieved_goal.ndim == 1:
                positions = achieved_goal[:-3].reshape(self.num_blocks, 3)
                ok = self._roboverify_task_complete(positions)
                return float(self.num_blocks) if ok else 0.0

            flat = achieved_goal.reshape(-1, achieved_dim)
            results = []
            for row in flat:
                positions = row[:-3].reshape(self.num_blocks, 3)
                results.append(self._roboverify_task_complete(positions))
            success_rate = np.asarray(results, dtype=np.float32).reshape(achieved_goal.shape[:-1])
            success_rate = success_rate * float(self.num_blocks)
            return success_rate

        if torch.is_tensor(observation):
            assert len(observation.shape) < 3

            goal = self.goal_from_observation_tensor(observation)
            achieved_goal = self.achieved_goal_from_observation_tensor(observation)

            if self.case != "Flip" and self.case != "Slide":
                subgoal_distances = self.subgoal_distances(achieved_goal, goal)  # List (len num_blocks) of tensors
                costs_per_object = torch.stack(subgoal_distances)  # num_blocks x batch_size x horizon....
                # evaluation threshold for Pick&Place set to be same as in the original construction environment!
                eval_thres = self.threshold if self.case != "PickAndPlace" else 0.05
                solved_per_object = torch.as_tensor(
                    costs_per_object < eval_thres, dtype=torch.float32
                )  # nObj x nB x (xnE) x horizon
            else:
                subgoal_distances = self.subgoal_distances_per_xyz(
                    achieved_goal, goal
                )  # List (len num_blocks) of tensors
                costs_per_object_and_coords = torch.stack(
                    subgoal_distances
                )  # num_blocks x batch_size x horizon .. x num_coordinates(3)
                solved_per_object = torch.as_tensor(
                    torch.all(costs_per_object_and_coords <= self.cost_thres, dim=-1),
                    dtype=torch.float32,
                )

            success_rate = torch.sum(solved_per_object, dim=0)
            success_rate = torch_helpers.to_numpy(success_rate)
        else:
            goal = self.goal_from_observation(observation)
            achieved_goal = self.achieved_goal_from_observation(observation)

            if self.case != "Flip" and self.case != "Slide":
                subgoal_distances = self.subgoal_distances(achieved_goal, goal)  # List (len num_blocks) of np arrays
                costs_per_object = np.stack(subgoal_distances)
                # evaluation threshold for Pick&Place set to be same as in the original construction environment!
                eval_thres = self.threshold if self.case != "PickAndPlace" else 0.05
                solved_per_object = np.asarray(costs_per_object < eval_thres, dtype=np.float32)
            else:
                # List (len num_blocks) of np arrays
                subgoal_distances = self.subgoal_distances_per_xyz(achieved_goal, goal)

                # num_blocks x batch_size x horizon .. x num_coordinates(3)
                costs_per_object_and_coords = np.stack(subgoal_distances)

                solved_per_object = np.asarray(
                    np.all(costs_per_object_and_coords <= torch_helpers.to_numpy(self.cost_thres), axis=-1),
                    dtype=np.float32,
                )

            success_rate = np.sum(solved_per_object, axis=0)

        return success_rate

    @staticmethod
    def get_object_centric_obs(obs, agent_dim=10, object_dim=12, object_static_dim=0, goal_dim=3):
        """Preprocessing on the observation to make the input suitable for GNNs

        :param obs: N x (nA + nO * nFo + n0 * nSo) Numpy array
        :param agent_dim: State dimension for the agent
        :param object_dim: State dimension for a single object
        """
        if obs.ndim == 3:
            obs = obs.squeeze(1)
        elif obs.ndim == 1:
            obs = np.expand_dims(obs, axis=0)

        batch_size, environment_state_length = obs.shape
        nObj = (environment_state_length - agent_dim - goal_dim) / (object_dim + object_static_dim + goal_dim)

        assert nObj.is_integer()
        nObj = int(nObj)

        start_ind_stat = agent_dim + nObj * object_dim
        start_ind_goal = agent_dim + nObj * object_dim + nObj * object_static_dim
        state_dict = {
            "agent": obs[:, :agent_dim],
            "objects_dyn": np.asarray(
                [obs[:, agent_dim + object_dim * i : agent_dim + object_dim * (i + 1)] for i in range(nObj)]
            ),
            "objects_static": np.asarray(
                [
                    obs[
                        :,
                        start_ind_stat + object_static_dim * i : start_ind_stat + object_static_dim * (i + 1),
                    ]
                    for i in range(nObj)
                ]
            ),
            "objects_goal": np.asarray(
                [obs[:, start_ind_goal + goal_dim * i : start_ind_goal + goal_dim * (i + 1)] for i in range(nObj)]
            ),
        }
        return state_dict
