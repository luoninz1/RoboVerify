"""Explicit supported-height premises and checked geometric loop invariants.

Height premises restrict the input model; column alignment and arm clearance are
candidate invariants whose entry obligations must be proved. Every quantified
object, including unnamed collision witnesses, is covered. The CFG verifier also
checks preservation of all three facts at loop boundaries.
"""

import z3

from synthesis.util.symbols import fresh_const


def tower_height_invariant(problem, *, fields=None):
    """Sound consequences of supported towers, including every unnamed block.

    Every tower root rests on the common table, all blocks are above it, and
    distinct members of one tower have at least one block-height of separation.
    These weaker consequences suffice for Stack; asserting full recursive support
    chains creates avoidable quantifier alternation. Gaps are deliberately not
    excluded here, so proving a VC is also valid for the documented complete towers.
    """
    low = problem.context
    x, y = [fresh_const(low.BoxSort, name) for name in ("tower_x", "tower_y")]

    def point(obj):
        if fields is None:
            return tuple(f(obj) for f in (low.X, low.Y, low.Z))
        return tuple(z3.substitute_vars(f, obj) for f in fields)

    def physical(obj):
        return z3.Not(low._is_table(obj))

    px, py = point(x), point(y)
    same = low.lowlevel_box_equal(x, y)
    on = z3.And(
        z3.Abs(px[0] - py[0]) < low.L / 2,
        z3.Abs(px[1] - py[1]) < low.L / 2,
        px[2] >= py[2],
    )
    root = z3.ForAll([y], z3.Implies(z3.And(physical(y), on), same))
    return z3.And(
        z3.ForAll(
            [x],
            z3.Implies(
                physical(x),
                z3.And(
                    px[2] >= problem.table_center_height,
                    z3.Implies(root, px[2] == problem.table_center_height),
                ),
            ),
        ),
        z3.ForAll(
            [x, y],
            z3.Implies(
                z3.And(physical(x), physical(y), on, z3.Not(same)),
                px[2] >= py[2] + low.L,
            ),
        ),
    )


def column_alignment(problem, *, fields=None):
    """Candidate loop invariant: members of a column share exact XY coordinates.

    This is not an assumption about arbitrary input towers. The CFG verifier must
    establish it before each loop and preserve it with the actual physical body.
    It captures the ideal, noiseless root-aligned placements made by Stack.
    """
    low = problem.context
    x, y = [fresh_const(low.BoxSort, name) for name in ("column_x", "column_y")]

    def point(obj):
        return (
            tuple(f(obj) for f in (low.X, low.Y, low.Z))
            if fields is None
            else tuple(z3.substitute_vars(f, obj) for f in fields)
        )

    px, py = point(x), point(y)
    on = z3.And(
        z3.Abs(px[0] - py[0]) < low.L / 2,
        z3.Abs(px[1] - py[1]) < low.L / 2,
        px[2] >= py[2],
    )
    return z3.ForAll(
        [x, y],
        z3.Implies(
            z3.And(z3.Not(low._is_table(x)), z3.Not(low._is_table(y)), on),
            z3.And(px[0] == py[0], px[1] == py[1]),
        ),
    )


def arm_clearance(problem, *, fields=None):
    """The empty gripper is at least a half-block above every object center."""
    low = problem.context
    obj = fresh_const(low.BoxSort, "arm_clearance_object")
    height = low.Z(obj) if fields is None else z3.substitute_vars(fields[2], obj)
    return z3.ForAll(
        [obj],
        z3.Implies(z3.Not(low._is_table(obj)), problem.arm[2] >= height + low.L / 2),
    )


def assume_tower_geometry(problem, table_surface_height):
    low = problem.context
    problem.table_center_height = (
        fresh_const(z3.RealSort(), "table_center_height")
        if table_surface_height is None
        else z3.RealVal(str(table_surface_height)) + low.L / 2
    )
    problem.solver.add(tower_height_invariant(problem))
    problem.consistency_candidates = tuple(problem.constants.values())
    # Global auxiliary invariants quantify over objects beyond the named
    # collision witness. Retain the actual relational premises for those proofs;
    # their finite instances alone can lose facts about a newly quantified object.
    problem.solver.add(
        *(
            low.translate_exact(condition, problem.constants)
            for condition in problem.initial_condition
        )
    )
