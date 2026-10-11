"""Full-state scene preparation and execution through the existing controllers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np

from synthesis.benchmarks.specs import ROW_TASKS, evaluate, make_scene, row_order
from synthesis.cap.runtime import PolicyBudgetExceeded, PolicyError, Runtime
from synthesis.util.on import (
    higher_implementation,
    on_star_implementation,
    scattered_implementation,
)


@dataclass(frozen=True)
class Target:
    """Opaque frozen row target; generated policies cannot inspect XYZ fields."""

    xyz: tuple[float, float, float]


class ParameterRegistry:
    def __init__(self, values=None):
        self.values = dict(values or {})
        self.definitions = {}

    def parameter(self, name, default, low, high):
        if (
            not isinstance(name, str)
            or not name
            or not all(np.isfinite(x) for x in (default, low, high))
        ):
            raise PolicyError("Parameters need a name and finite default/bounds")
        if not low <= default <= high or low == high:
            raise PolicyError("Parameter default must lie inside nonempty bounds")
        definition = (float(default), float(low), float(high))
        if name in self.definitions and self.definitions[name] != definition:
            raise PolicyError(f"Inconsistent repeated parameter definition: {name}")
        if len(self.definitions) >= 32 and name not in self.definitions:
            raise PolicyError("At most 32 scalar parameters are allowed")
        self.definitions[name] = definition
        value = float(self.values.get(name, default))
        if not low <= value <= high:
            raise PolicyError(f"Parameter outside bounds: {name}")
        return value


class PhysicalSuite:
    """Cache prepared states; every candidate restores the identical snapshot."""

    def __init__(self):
        self.environments = {}
        self.starts = {}

    def prepare(self, case):
        from synthesis.api.control import get_move_action
        from synthesis.examples.relational_robotic_sort import (
            capture_simulator_state,
            create_environment,
            restore_simulator_state,
        )

        if case.num_blocks not in self.environments:
            env, _, _ = create_environment(case.num_blocks, seed=29)
            self.environments[case.num_blocks] = (env, capture_simulator_state(env))
        env, baseline = self.environments[case.num_blocks]
        if case.name not in self.starts:
            restore_simulator_state(env, baseline)
            scene = make_scene(
                case,
                env.object_names,
                gripper=env.initial_gripper_xpos,
                robot_base_xy=env.robot_base_xy,
                table_center=env.height_offset,
            )
            for obj, point in scene.positions.items():
                env.sim.data.set_joint_qpos(
                    f"{obj}:joint", np.r_[point, 1.0, 0.0, 0.0, 0.0]
                )
                env.sim.data.set_joint_qvel(f"{obj}:joint", np.zeros(6))
            env.sim.forward()
            obs = env.flatten_observation(env._get_obs())
            target = obs[:3].copy()
            # Prepare a clear, empty gripper above tall input scenes. This is
            # initialization shared by every method, excluded from policy steps.
            target[2] = max(
                target[2], max(p[2] for p in scene.positions.values()) + 0.10
            )
            for _ in range(50):
                env.step(get_move_action(obs, target, close_gripper=False))
                obs = env.flatten_observation(env._get_obs())
            actual = {
                obj: tuple(env.sim.data.get_site_xpos(obj)) for obj in scene.objects
            }
            if any(
                np.linalg.norm(np.asarray(actual[obj]) - scene.positions[obj]) > 0.006
                for obj in scene.objects
            ):
                raise RuntimeError(
                    f"Prepared scene failed settling validation: {case.name}"
                )
            scene.positions = actual
            # Validate the empty initial row; missing occupants are a setup error.
            if (
                case.task in ROW_TASKS
                and row_order(scene, actual) != scene.initial_order
            ):
                raise RuntimeError("Prepared row does not match its manifest")
            self.starts[case.name] = (scene, capture_simulator_state(env))
        scene, start = self.starts[case.name]
        restore_simulator_state(env, start)
        return env, deepcopy(scene)

    def close(self):
        for env, _ in self.environments.values():
            env.close()

    def save_start(self, case, path):
        """Archive the prepared state and all inputs needed for exact replay."""
        from pathlib import Path

        _, start = self.starts[case.name]
        if start.state.udd_state:
            raise RuntimeError(
                "Nonempty MuJoCo user state needs an explicit serializer"
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            simulator_state=start.state.flatten(),
            goal=start.goal,
            **dict(start.inputs),
        )

    def load_start(self, case, scene, path):
        """Restore an archived start, without settling or regenerating geometry."""
        from synthesis.examples.relational_robotic_sort import capture_simulator_state

        env, _ = self.prepare(case)
        with np.load(path, allow_pickle=False) as arrays:
            env.sim.set_state_from_flattened(arrays["simulator_state"])
            env.goal = arrays["goal"].copy()
            for name in (
                "ctrl",
                "mocap_pos",
                "mocap_quat",
                "qacc_warmstart",
                "qfrc_applied",
                "xfrc_applied",
            ):
                getattr(env.sim.data, name)[:] = arrays[name]
        env.sim.forward()
        self.starts[case.name] = (deepcopy(scene), capture_simulator_state(env))

    def run(
        self,
        case,
        source,
        *,
        parameters=None,
        keys_override=None,
        max_calls=300,
        max_physics_steps=20000,
        max_ast_steps=100000,
        deadline_seconds=60,
        recorder=None,
    ):
        from synthesis.cfg.collection import execution_deadline

        start = perf_counter()
        # Scene preparation is also within the process deadline.
        try:
            with execution_deadline(
                min(deadline_seconds, 60)
                if case.task == "unstack"
                else deadline_seconds
            ):
                env, scene = self.prepare(case)
                if keys_override is not None:
                    if case.task not in ROW_TASKS or set(keys_override) != set(
                        scene.objects
                    ):
                        raise ValueError(
                            "Key witness must cover exactly the row objects"
                        )
                    scene.keys = {
                        obj: int(value) for obj, value in keys_override.items()
                    }
                session = Session(
                    env,
                    scene,
                    parameters=parameters,
                    max_calls=max_calls,
                    max_physics_steps=max_physics_steps,
                    recorder=recorder,
                )
                original_step = env.step

                def step(action):
                    session.physics_steps += 1
                    if session.physics_steps > max_physics_steps:
                        raise PolicyBudgetExceeded("Physics step budget exhausted")
                    return original_step(action)

                env.step = step
                error = status = None
                try:
                    runtime = Runtime(session.api(), max_steps=max_ast_steps)
                    runtime.run(source, inputs=dict(scene.roles))
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    status = (
                        "timeout"
                        if isinstance(exc, TimeoutError)
                        else (
                            "budget_exhausted"
                            if isinstance(exc, PolicyBudgetExceeded)
                            else "execution_error"
                        )
                    )
                finally:
                    env.step = original_step
                result = evaluate(
                    scene,
                    session.positions(),
                    held=session.held,
                    sorted_seen=session.sorted_seen,
                    upright=session.upright(),
                )
                return {
                    "case": case.as_dict(),
                    "status": status
                    or ("success" if result.success else "goal_failed"),
                    "success": result.success and error is None,
                    "goal": asdict(result),
                    "error": error,
                    "primitive_calls": session.calls,
                    "physics_steps": session.physics_steps,
                    "ast_steps": runtime.steps,
                    "elapsed_seconds": perf_counter() - start,
                    "parameters": session.registry.definitions,
                    "parameter_values": dict(parameters or {}),
                    "scene": scene.as_dict(),
                    "trace": session.trace,
                    "final_positions": session.positions(),
                    "motion_blocks": session.motion_blocks,
                }
        except Exception as exc:
            return {
                "case": case.as_dict(),
                "status": "timeout" if isinstance(exc, TimeoutError) else "setup_error",
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": perf_counter() - start,
                "primitive_calls": 0,
                "physics_steps": 0,
            }


class Session:
    def __init__(
        self,
        env,
        scene,
        *,
        parameters=None,
        max_calls=300,
        max_physics_steps=20000,
        recorder=None,
    ):
        self.env, self.scene = env, scene
        self.registry = ParameterRegistry(parameters)
        self.max_calls, self.max_physics_steps = max_calls, max_physics_steps
        self.calls = self.physics_steps = 0
        self.held = None
        self.sorted_seen = False
        self.trace = []
        self.motion_blocks = []
        self.pending = []
        self.block_start = None
        if scene.case.task in ROW_TASKS:
            from synthesis.examples.relational_sort_controller import TableRobot

            self.robot = TableRobot(env, scene.slots, scene.keys, recorder=recorder)
        else:
            from synthesis.environment.general_env import GymToGymnasium

            self.wrapper = GymToGymnasium(env)
            self.env.symbolic_name_to_box_id = {
                name: index for index, name in enumerate(scene.objects)
            }
            self.trajectory = [env.flatten_observation(env._get_obs())]
        self.observe_checkpoint()

    def positions(self):
        return {
            name: self.env.sim.data.get_site_xpos(name).copy().tolist()
            for name in self.scene.objects
        }

    def upright(self):
        return {
            name: float(self.env.sim.data.get_site_xmat(name)[2, 2])
            for name in self.scene.objects
        }

    def object(self, name):
        if name not in self.scene.objects:
            raise PolicyError(f"Unknown object handle: {name}")
        return name

    def call(self, kind, arguments):
        self.calls += 1
        if self.calls > self.max_calls:
            raise PolicyBudgetExceeded("Primitive call budget exhausted")
        self.trace.append(
            {
                "primitive": kind,
                "arguments": arguments,
                "step": self.physics_steps,
                "positions": self.positions(),
            }
        )

    def observe_checkpoint(self):
        if self.scene.case.task in ROW_TASKS and self.held is None:
            order = row_order(self.scene, self.positions())
            expected = tuple(
                sorted(self.scene.initial_order, key=self.scene.keys.__getitem__)
            )
            if order == expected:
                self.sorted_seen = True

    def execute(self, instruction):
        instruction.eval(self.wrapper, self.trajectory)
        self.pending.append(instruction)
        result = instruction.last_control_result
        if result is None or not result.converged:
            raise PolicyError(f"Unconverged {type(instruction).__name__}: {result}")

    def pick(self, obj):
        self.object(obj)
        if self.held is not None:
            raise PolicyError("Pick requires an empty gripper")
        self.call("pick", [obj])
        self.block_start = {
            "positions": self.positions(),
            "arm": self.env.sim.data.get_site_xpos("robot0:grip").copy().tolist(),
        }
        self.pending = []
        if self.scene.case.task in ROW_TASKS:
            self.robot.pick(obj)
        else:
            from synthesis.api.instructions import PickByName

            self.execute(PickByName(obj))
        self.held = obj

    def move(self, *args, **kwargs):
        if self.scene.case.task in ROW_TASKS:
            if len(args) != 1 or kwargs or not isinstance(args[0], Target):
                raise PolicyError("Row move requires one frozen target handle")
            self.call("move", [list(args[0].xyz)])
            self.robot.move(np.asarray(args[0].xyz))
        else:
            from synthesis.api.instructions import MoveByName

            if len(args) not in (3, 4):
                raise PolicyError(
                    "Tower move needs x_ref, y_ref, z_ref and optional offset"
                )
            references = [self.object(arg) for arg in args[:3]]
            offset = (
                args[3] if len(args) == 4 else kwargs.pop("offset", (0.0, 0.0, 0.0))
            )
            if kwargs or len(offset) != 3 or not np.isfinite(offset).all():
                raise PolicyError("Move offset must have three finite components")
            self.call("move", references + [list(offset)])
            self.execute(MoveByName(*references, target_offset=list(offset)))

    def release(self, target, retreat_offset=0.15):
        if self.held is None:
            raise PolicyError("Release requires a held object")
        if self.scene.case.task in ROW_TASKS:
            if not isinstance(target, Target):
                raise PolicyError("Row release needs one frozen target handle")
            self.call("release", [list(target.xyz)])
            self.robot.release(np.asarray(target.xyz))
        else:
            from synthesis.api.instructions import ReleaseByName

            self.object(target)
            if target != self.held:
                raise PolicyError("Release must name the held object")
            self.call("release", [target, retreat_offset])
            self.execute(ReleaseByName(target, target_z=retreat_offset))
            self.motion_blocks.append(
                {
                    **self.block_start,
                    "source": self.held,
                    "instructions": [encode_instruction(inst) for inst in self.pending],
                }
            )
        self.held = None
        self.observe_checkpoint()

    def target(self, obj):
        self.object(obj)
        if self.scene.case.task not in ROW_TASKS:
            raise PolicyError("Row target API is unavailable for tower tasks")
        positions = self.positions()
        matches = [
            slot
            for slot in self.scene.slots
            if np.linalg.norm(np.asarray(positions[obj]) - np.asarray(slot)) <= 0.015
        ]
        if len(matches) != 1:
            raise PolicyError(
                "target(obj) needs an object currently occupying a row slot; save the target before moving it"
            )
        return Target(tuple(matches[0]))

    def next_star(self, a, b):
        self.object(a), self.object(b)
        positions = self.positions()
        order = row_order(self.scene, positions)
        if order is None or self.held is not None:
            raise PolicyError("next_star is available at complete-row boundaries")
        return order.index(a) <= order.index(b)

    def api(self):
        result = {
            "objects": lambda: list(self.scene.objects),
            "pick": self.pick,
            "move": self.move,
            "release": self.release,
            "parameter": self.registry.parameter,
        }
        if self.scene.case.task in ROW_TASKS:
            result.update(
                next_star=self.next_star,
                keyless=lambda a, b: self.scene.keys[self.object(a)]
                < self.scene.keys[self.object(b)],
                target=self.target,
                buffer_target=lambda: Target(self.scene.buffer),
            )
        else:
            for name, predicate in (
                ("on_star", on_star_implementation),
                ("higher", higher_implementation),
                ("scattered", scattered_implementation),
            ):

                def query(a, b, predicate=predicate):
                    positions = self.positions()
                    return bool(
                        predicate(positions[self.object(a)], positions[self.object(b)])
                    )

                result[name] = query
            result["on_star_zero"] = lambda a, b: bool(
                on_star_implementation(
                    self.scene.positions[self.object(a)],
                    self.scene.positions[self.object(b)],
                )
            )
        return result


def encode_instruction(inst):
    from synthesis.api.instructions import MoveByName, PickByName, ReleaseByName

    if isinstance(inst, PickByName):
        return {"kind": "pick", "object": inst.grab_box_name}
    if isinstance(inst, MoveByName):
        return {
            "kind": "move",
            "references": [
                inst.target_box_name_x,
                inst.target_box_name_y,
                inst.target_box_name_z,
            ],
            "offset": [parameter.numeric_val() for parameter in inst.target_offset],
        }
    if isinstance(inst, ReleaseByName):
        return {
            "kind": "release",
            "object": inst.release_box_name,
            "retreat_offset": inst.target_z_offset.numeric_val(),
        }
    raise ValueError(type(inst).__name__)
