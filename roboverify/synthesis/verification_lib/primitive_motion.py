"""POPL §5.5 primitive waypoint semantics, with explicit grasp-contact handling.

This proves the stated geometric primitive model, not the MuJoCo feedback
controller. Release requires support; unsupported drops havoc the object's
position and fail a support obligation. The empty gripper is a point; payloads
use the existing swept-cube overapproximation. Intended contact with the Pick
and Release target is excluded explicitly (paper discrepancy 18).
"""

import z3

from synthesis.api.instructions import (
    Assign,
    Get,
    MoveByName,
    PickByName,
    ReleaseByName,
    Skip,
)
from synthesis.util.symbols import fresh_const
from synthesis.verification_lib.motion_verification import (
    MotionCheck,
    MotionProblem,
    _on_star,
)


class PrimitiveMotionProblem(MotionProblem):
    def __init__(self, *args, initial_arm=None, enforce_source=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.enforce_source = enforce_source
        self.arm = tuple(fresh_const(z3.RealSort(), "arm_" + a) for a in "xyz")
        if initial_arm is not None:
            self.solver.add(
                *(a == z3.RealVal(str(v)) for a, v in zip(self.arm, initial_arm))
            )
        self.origin = dict(self.initial)
        self.initial_arm = self.arm
        self.held_name = None
        from z3.z3util import get_vars

        self.used_bindings = {
            str(v) for condition in self.initial_condition for v in get_vars(condition)
        }
        self.step = 0
        obj = z3.Var(0, self.context.get_consts("sym").sort())
        self.fields = tuple(
            axis(obj) for axis in (self.context.X, self.context.Y, self.context.Z)
        )
        self.entry_fields = self.fields

    def counterexample(self, model, obligation):
        from dataclasses import replace

        from synthesis.verification_lib.motion_verification import _number

        before = self.initial
        try:
            self.initial = self.origin
            example = super().counterexample(model, obligation)
            return replace(
                example, initial_arm=[_number(model, v) for v in self.initial_arm]
            )
        finally:
            self.initial = before

    def at(self, expr, *, fields=None, resolve=True):
        mapping = {
            name: self.constants[self.bindings[name] if resolve else name]
            for name in self.constants
        }
        translated = self.context.translate_exact(expr, mapping)
        return z3.substitute_funs(
            translated,
            *zip(
                (self.context.X, self.context.Y, self.context.Z), fields or self.fields
            ),
        )

    def endpoint(self, names, offsets, kind):
        error = (
            (0, 0, 0)
            if self.noise is None
            else self.error(
                getattr(self.noise, "eps_" + kind), f"{self.block_v}_{self.step}_{kind}"
            )
        )
        return tuple(
            self.current[self.bindings[name]][i] + z3.RealVal(str(offset)) + error[i]
            for i, (name, offset) in enumerate(zip(names, offsets))
        )

    def sweep(self, start, end, *, payload=None, contact=None, phase=None):
        for name, position in self.current.items():
            exclude = [
                z3.Not(self.same(name, n)) for n in (payload, contact) if n is not None
            ]
            # An empty gripper is modeled as a point against cubes of half-size L/2.
            # A carried cube uses the existing L Minkowski radius.
            if payload is None:
                t = fresh_const(z3.RealSort(), "arm_t")
                collision = z3.And(
                    t >= 0,
                    t <= 1,
                    *(
                        z3.Abs(position[i] - ((1 - t) * start[i] + t * end[i]))
                        < self.context.L / 2
                        for i in range(3)
                    ),
                )
            else:
                collision = self.context.encode_collision_at(position, start, end)
            label = f"collision_{self.step}"
            if phase is not None:
                label += "_" + phase
            self.check(
                f"{label}_{name}",
                z3.And(self.physical(name), *exclude, collision),
            )

    def transport(self, name, endpoint):
        before = dict(self.current)
        start = before[name]
        for other, position in before.items():
            supported = z3.And(
                self.physical(other),
                z3.Not(self.same(other, name)),
                _on_star(position, start, self.context.L),
                position[2] > start[2],
            )
            self.check(f"support_{self.step}_{other}", supported)
            self.current[other] = tuple(
                z3.If(self.same(other, name), end, old)
                for end, old in zip(endpoint, position)
            )
        obj = z3.Var(0, self.constants[name].sort())
        alias = self.context.lowlevel_box_equal(obj, self.constants[name])
        self.fields = tuple(
            z3.If(alias, end, old) for end, old in zip(endpoint, self.fields)
        )

    def execute(self, body):
        for instruction in body:
            self.step += 1
            if isinstance(instruction, Skip):
                continue
            if isinstance(instruction, Assign):
                self.used_bindings.update((instruction.left, instruction.right))
                self.bindings[instruction.left] = self.bindings[instruction.right]
                continue
            if isinstance(instruction, Get):
                variables = instruction.guard_exists_vars
                if self.used_bindings.intersection(map(str, variables)):
                    self.checks.append(
                        MotionCheck(
                            f"get_{self.step}",
                            "unsupported",
                            reason="Rebinding an existing Get name requires a fresh symbolic version",
                        )
                    )
                    return
                self.used_bindings.update(map(str, variables))
                condition = instruction.instantiated_cond
                self.check(
                    f"get_exists_{self.step}",
                    z3.Not(self.at(z3.Exists(variables, condition))),
                )
                # A fresh free witness is universally checked; never choose the demo ID.
                self.solver.add(self.at(condition))
                continue
            self.used_bindings.update(
                o["val"] for o in instruction.get_operand() if o["type"] == "BoxName"
            )
            if isinstance(instruction, PickByName):
                if self.held_name is not None:
                    self.checks.append(
                        MotionCheck(
                            f"pick_{self.step}",
                            "unsupported",
                            reason="Pick while already holding an object",
                        )
                    )
                    return
                name = self.bindings[instruction.grab_box_name]
                if (
                    self.enforce_source
                    and self.check(
                        f"source_{self.step}",
                        z3.Not(self.same(name, self.contract.source)),
                    ).status
                    != "valid"
                ):
                    return
                self.check(f"physical_{self.step}", z3.Not(self.physical(name)))
                end = self.endpoint([instruction.grab_box_name] * 3, [0, 0, 0], "grasp")
                # PrimitiveController.pick first aligns XY at the current arm
                # height, then descends vertically. A single diagonal segment
                # invents collisions the actual waypoint program does not take.
                approach = (end[0], end[1], self.arm[2])
                self.sweep(self.arm, approach, contact=name, phase="approach")
                self.sweep(approach, end, contact=name, phase="descend")
                self.arm, self.held_name = end, name
                self.released = False
            elif isinstance(instruction, MoveByName):
                names = [
                    instruction.target_box_name_x,
                    instruction.target_box_name_y,
                    instruction.target_box_name_z,
                ]
                self.check(
                    f"physical_{self.step}",
                    z3.Not(z3.And(*(self.physical(self.bindings[n]) for n in names))),
                )
                end = self.endpoint(
                    names,
                    [
                        p.concrete_float("primitive move")
                        for p in instruction.target_offset
                    ],
                    "move",
                )
                self.sweep(self.arm, end, payload=self.held_name)
                if self.held_name is not None:
                    self.transport(self.held_name, end)
                self.arm = end
            elif isinstance(instruction, ReleaseByName):
                name = self.bindings[instruction.release_box_name]
                if self.held_name is None:
                    self.checks.append(
                        MotionCheck(
                            f"release_{self.step}",
                            "unsupported",
                            reason="Release without a held object",
                        )
                    )
                    return
                self.check(
                    f"release_object_{self.step}",
                    z3.Not(self.same(name, self.held_name)),
                )
                point = self.current[self.held_name]
                support = [
                    z3.And(
                        self.physical(other),
                        z3.Not(self.same(other, self.held_name)),
                        z3.Abs(point[0] - p[0]) < self.context.L / 2,
                        z3.Abs(point[1] - p[1]) < self.context.L / 2,
                        point[2] - p[2] == self.context.L,
                    )
                    for other, p in self.current.items()
                ]
                if self.contract.table_surface_height is not None:
                    support.append(
                        point[2]
                        == z3.RealVal(str(self.contract.table_surface_height))
                        + self.context.L / 2
                    )
                supported = z3.Or(*support)
                self.check(f"release_support_{self.step}", z3.Not(supported))
                fallen = tuple(
                    z3.If(supported, p, fresh_const(z3.RealSort(), "fall_" + a))
                    for a, p in zip("xyz", point)
                )
                # Release's fall update does not transport supporting objects.
                for other, p in list(self.current.items()):
                    self.current[other] = tuple(
                        z3.If(self.same(other, self.held_name), q, old)
                        for q, old in zip(fallen, p)
                    )
                obj = z3.Var(0, self.constants[name].sort())
                alias = self.context.lowlevel_box_equal(
                    obj, self.constants[self.held_name]
                )
                self.fields = tuple(
                    z3.If(alias, q, old) for q, old in zip(fallen, self.fields)
                )
                end = self.endpoint(
                    [instruction.release_box_name] * 3,
                    [
                        0,
                        0,
                        instruction.target_z_offset.concrete_float("primitive release"),
                    ],
                    "release",
                )
                self.sweep(self.arm, end, contact=self.held_name)
                self.arm, self.held_name, self.released = end, None, True
            else:
                self.checks.append(
                    MotionCheck(
                        f"instruction_{self.step}",
                        "unsupported",
                        reason=f"No primitive encoding for {type(instruction).__name__}",
                    )
                )
                return
