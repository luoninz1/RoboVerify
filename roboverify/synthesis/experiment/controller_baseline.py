"""Historical controller behavior for serial diagnostic comparisons only."""

from contextlib import contextmanager
from unittest.mock import patch

from synthesis.api.control import PrimitiveController
from synthesis.cfg.reset import inner_env


def previous_release(controller, box_id, offset):
    """The former Z-only stopping rule, including its pre-opening target."""
    target = controller.observation[:3].copy()
    target[2] = controller.box_position(box_id)[2] + offset
    if not controller.gripper(opened=True):
        return False
    return controller.move(
        target, close_gripper=False, vertical_only=True, phase="retreat"
    )


@contextmanager
def previous_release_control():
    with patch.object(PrimitiveController, "release", previous_release):
        yield


def stack_env_with_head_contacts(num_blocks=4):
    """Re-enable the former head geometry masks in a fresh diagnostic env."""
    from synthesis.mcmc.synthesis import make_roboverify_stack_env

    env = make_roboverify_stack_env(num_blocks=num_blocks)
    sim = inner_env(env).sim
    for i in range(sim.model.ngeom):
        if sim.model.body_id2name(sim.model.geom_bodyid[i]).startswith("robot0:head"):
            sim.model.geom_contype[i] = sim.model.geom_conaffinity[i] = 1
    sim.forward()
    return env
