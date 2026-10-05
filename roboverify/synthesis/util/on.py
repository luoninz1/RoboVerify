from __future__ import annotations

import math
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np
import z3

BLOCK_LENGTH = (
    0.025 * 2
)  # (0.025, 0.025, 0.025) in the xml file of the gym env is half length


DEFAULT_HIGHER_TOLERANCE = 0.001  # metres; small relative to a 50 mm block level
_higher_tolerance = ContextVar("higher_tolerance", default=DEFAULT_HIGHER_TOLERANCE)


def validate_higher_tolerance(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value < BLOCK_LENGTH / 2:
        raise ValueError("Higher tolerance must be finite and in [0, 0.025) metres")
    return value


def get_higher_tolerance():
    return _higher_tolerance.get()


@contextmanager
def using_higher_tolerance(value):
    """Scope predicate semantics to one run, restoring them even after errors."""
    token = _higher_tolerance.set(validate_higher_tolerance(value))
    try:
        yield
    finally:
        _higher_tolerance.reset(token)


def higher_z3(z1, z2, *, tolerance=None):
    """Geometric Higher comparison; table isolation belongs to the caller."""
    tolerance = (
        get_higher_tolerance()
        if tolerance is None
        else validate_higher_tolerance(tolerance)
    )
    return z1 >= z2 - z3.RealVal(repr(tolerance))


class NoGeometry:
    """Placeholder position for a ``Box``-sort element that is not a real block.

    The block algebra quantifies over a ``Box`` sort that contains one such
    element, ``tbl``. It is a null/bottom marker: the axioms in
    ``highlevel_verification_lib`` isolate it from all three relations rather
    than giving it a location. Handing it a coordinate triple (it used to carry
    ``[-100, -100, -100]``) made the isolation depend on that triple landing
    outside every tolerance, so a change to ``BLOCK_LENGTH`` or a block drifting
    below the table plane would have shifted the semantics silently.
    """

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"NoGeometry({self.name!r})"


#: The table, as it appears in a relational state. Its physical surface height is
#: a motion-level fact and is deliberately not recorded here.
TABLE = NoGeometry("tbl")

#: The ``Goal``-sort null element. Goal relations (``d_star``/``r_star``) special-case
#: it by name; it has no position for the same reason ``TABLE`` has none.
NULL = NoGeometry("null")


def is_table(block) -> bool:
    """True for the table marker, which is what the ``tbl`` axioms range over."""
    return block is TABLE


def _xyz(block):
    """Unpack a block position, refusing sort elements that carry no geometry."""
    if isinstance(block, NoGeometry):
        raise TypeError(
            f"{block!r} has no position: a geometric predicate was applied to a "
            "sort element that is not a physical block"
        )
    x, y, z = block
    return x, y, z


def on_star_eval(block1, block2) -> bool:
    """define the numerical interpretation of the on(block1, block2) between two blocks"""
    x1, y1, z1 = block1
    x2, y2, z2 = block2
    return (
        abs(x1 - x2) < BLOCK_LENGTH / 2
        and abs(y1 - y2) < BLOCK_LENGTH / 2
        and 0 <= z1 - z2
    )


def on(block1, block2) -> bool:
    """define the numerical interpretation of the on(block1, block2) between two blocks"""
    x1, y1, z1 = block1
    x2, y2, z2 = block2
    return (
        abs(x1 - x2) < BLOCK_LENGTH / 2
        and abs(y1 - y2) < BLOCK_LENGTH / 2
        and 0 <= z1 - z2 < 1.5 * BLOCK_LENGTH
    )


def _clamped_linear_reward(error: float, tolerance: float) -> float:
    """Return 1 at zero error and linearly decay to 0 beyond ``tolerance``."""
    if tolerance <= 0.0:
        return 1.0 if error <= 0.0 else 0.0
    return float(max(0.0, 1.0 - error / tolerance))


def on_reward(
    block1,
    block2,
    block_length: float = BLOCK_LENGTH,
) -> float:
    """Dense [0, 1] reward aligned with :func:`on` geometry.

    Returns 1.0 when ``on(block1, block2)`` holds; decays smoothly as xy
    misalignment or vertical gap move outside the valid ON band.
    """
    x1, y1, z1 = np.asarray(block1, dtype=float)
    x2, y2, z2 = np.asarray(block2, dtype=float)
    half_xy = block_length / 2.0
    max_dz = 1.5 * block_length

    score_x = _clamped_linear_reward(abs(x1 - x2), half_xy)
    score_y = _clamped_linear_reward(abs(y1 - y2), half_xy)
    score_xy = score_x * score_y

    dz = z1 - z2
    if 0.0 <= dz < max_dz:
        score_z = 1.0
    elif dz < 0.0:
        score_z = _clamped_linear_reward(-dz, half_xy)
    else:
        score_z = _clamped_linear_reward(dz - max_dz, half_xy)

    return float(score_xy * score_z)


def z3_on(
    x1: z3.ArithRef,
    y1: z3.ArithRef,
    z1: z3.ArithRef,
    x2: z3.ArithRef,
    y2: z3.ArithRef,
    z2: z3.ArithRef,
    block_length: float = BLOCK_LENGTH,
) -> z3.BoolRef:
    """Z3 encoding aligned with :func:`on` (same xy tolerance and vertical gap)."""
    half_xy = z3.RealVal(block_length / 2.0)
    max_dz = z3.RealVal(1.5 * block_length)
    return z3.And(
        z3.Abs(x1 - x2) < half_xy,
        z3.Abs(y1 - y2) < half_xy,
        z1 - z2 >= 0,
        z1 - z2 < max_dz,
    )


def on_star_implementation(block1, block2) -> bool:
    """Numeric reading of ``ON*(block1, block2)``.

    The table is ``on*``-isolated. The ``on_tbl`` axiom says
    ``ON*(x, tbl) or ON*(tbl, x)`` implies ``x == tbl``, and ``on2`` makes ``ON*``
    reflexive, so ``(tbl, tbl)`` is the one true pair involving the table. That
    used to hold only because the table's sentinel coordinates fell outside the
    ``BLOCK_LENGTH / 2`` tolerance below -- an accident of the numbers rather
    than a statement of the axiom.
    """
    if is_table(block1) or is_table(block2):
        return is_table(block1) and is_table(block2)
    x1, y1, z1 = _xyz(block1)
    x2, y2, z2 = _xyz(block2)
    return (
        abs(x1 - x2) < BLOCK_LENGTH / 2
        and abs(y1 - y2) < BLOCK_LENGTH / 2
        and 0 <= z1 - z2
    )


def d_star_implementation(block1, block2) -> bool:
    """define the numerical interpretation of the d_star(block1, block2) between two blocks"""
    x1, y1, z1 = block1
    x2, y2, z2 = block2
    return x1 == x2 and z1 == z2 and y1 <= y2


def r_star_implementation(block1, block2) -> bool:
    """define the numerical interpretation of the r_star(block1, block2) between two blocks"""
    x1, y1, z1 = block1
    x2, y2, z2 = block2
    return y1 == y2 and z1 == z2 and x1 <= x2


def higher_implementation(block1, block2, *, tolerance=None) -> bool:
    """Read Higher as ``z1 >= z2 - tolerance``; the default is 1 mm.

    Runtime guards, learning and geometric verification share this tolerance.
    Zero recovers exact ordering. Tolerance treats near-equal resting heights
    as one level; it changes neither coordinates nor control tolerances.
    On arbitrary continuous heights this comparison need not be transitive.
    The abstract ordering axioms therefore still require a suitable domain,
    such as sufficiently separated uniform block levels. The table remains
    isolated: only Higher(tbl, tbl) holds for pairs involving the table.
    """
    tolerance = (
        get_higher_tolerance()
        if tolerance is None
        else validate_higher_tolerance(tolerance)
    )
    if is_table(block1) or is_table(block2):
        return is_table(block1) and is_table(block2)
    _, _, z1 = _xyz(block1)
    _, _, z2 = _xyz(block2)
    return z1 >= z2 - tolerance


def scattered_implementation(block1, block2) -> bool:
    """Numeric reading of ``Scattered(block1, block2)``: well separated in xy.

    ``scattered_not_tbl`` gives ``not Scattered(x, tbl)``; with ``scattered1``
    (symmetry) that also rules out ``Scattered(tbl, x)``, and instantiating it at
    ``x = tbl`` rules out ``Scattered(tbl, tbl)``. So every pair involving the
    table is false -- unlike ``ON*`` and ``Higher``, there is no reflexive case.

    The separation test is non-strict. ``_reset_sim_roboverify_stack`` in
    ``fpp_construction_env`` samples initial layouts by rejecting until
    ``dx >= 2L or dy >= 2L``, so a layout separated by exactly ``2L`` is one the
    environment can hand us and must count as scattered;
    ``LowLevelContext.lowlevel_scattered`` uses the same non-strict test.

    The ``z1 >= 0`` / ``z2 >= 0`` conjuncts this used to carry excluded the
    table's sentinel position, and are removed for the same reason as in
    :func:`higher_implementation`.
    """
    if is_table(block1) or is_table(block2):
        return False
    x1, y1, _ = _xyz(block1)
    x2, y2, _ = _xyz(block2)
    return abs(x1 - x2) >= 2 * BLOCK_LENGTH or abs(y1 - y2) >= 2 * BLOCK_LENGTH


def get_block_pos(obs, block_id):
    start_idx = 10 + 12 * block_id
    end_idx = start_idx + 3
    return np.array(obs[start_idx:end_idx])


def state_comparison_indices(num_blocks: int) -> list[int]:
    """Indices into flattened RoboVerifyStack obs for policy vs expert comparison.

    Includes gripper position (0-2), finger opening (3-4), and every block's xyz.
    """
    agent_dim = 10
    object_dyn_dim = 12
    indices = [0, 1, 2, 3, 4]
    for block_id in range(num_blocks):
        start = agent_dim + block_id * object_dyn_dim
        indices.extend(range(start, start + 3))
    return indices


def print_block_layout(obs, num_block):
    for i in range(0, num_block):
        for j in range(0, num_block):
            print(f"on({i}, {j})", on(get_block_pos(obs, i), get_block_pos(obs, j)))
