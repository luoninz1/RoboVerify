"""Execution events shared by full rollouts and restored candidate executions."""

from copy import deepcopy
from dataclasses import asdict

from synthesis.api.control import ControlResult


def emit(callback, kind, path, env, trajectory, **fields):
    if callback is not None:
        callback(
            dict(
                kind=kind,
                path=path,
                index=len(trajectory) - 1,
                bindings=deepcopy(env.symbolic_name_to_box_id),
                **fields
            )
        )


def execute_instruction(
    instruction,
    env,
    trajectory,
    *,
    path,
    return_image=False,
    on_event=None,
    on_loop_head=None,
    max_loop_iterations=None
):
    from synthesis.api.instructions import While

    emit(on_event, "instruction_start", path, env, trajectory)
    kwargs = {}
    if isinstance(instruction, While):
        kwargs = dict(
            loop_id=path,
            on_loop_head=on_loop_head,
            on_event=on_event,
            max_loop_iterations=max_loop_iterations,
        )
    images = instruction.eval(env, trajectory, return_image, **kwargs)
    result = getattr(instruction, "last_control_result", None)
    fields = {"control": asdict(result)} if isinstance(result, ControlResult) else {}
    emit(on_event, "instruction_end", path, env, trajectory, **fields)
    return images
