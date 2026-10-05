"""Bound Fetch delta actions without changing the Cartesian direction."""

import numpy as np

CARTESIAN_ACTION_MODE = "uniform-xyz-v1"


def bound_delta_action(action):
    """Scale XYZ together into [-1, 1]; bound the independent finger command."""
    bounded = np.asarray(action, dtype=float).copy()
    if bounded.shape != (4,) or not np.all(np.isfinite(bounded)):
        raise ValueError("Delta action must contain four finite coordinates")
    bounded[:3] /= max(1.0, float(np.max(np.abs(bounded[:3]))))
    bounded[3] = np.clip(bounded[3], -1.0, 1.0)
    return bounded
