"""Bounded symbolic execution of the *generated* row policy.

This checks a fixed finite object universe and initial spatial permutation, with
arbitrary integer keys (including ties). It assumes successful row transfers.
It is not the existing example's invariant proof, an unbounded proof, or a
physical-controller proof. No abstract successor is added to training traces.
"""

from dataclasses import dataclass
from time import perf_counter

import z3

from synthesis.cap.runtime import PolicyBudgetExceeded, PolicyError, Runtime
from synthesis.util.symbols import fresh_const


class BranchRequest(Exception):
    def __init__(self, expression, constraints, choices):
        self.expression = expression
        self.constraints = constraints
        self.choices = choices


class Decision:
    def __init__(self, session, expression):
        self.session, self.expression = session, expression

    def __bool__(self):
        session = self.session
        expression = z3.simplify(self.expression)
        if z3.is_true(expression):
            return True
        if z3.is_false(expression):
            return False
        index = len(session.choices)
        if index == len(session.prefix):
            raise BranchRequest(
                expression, list(session.constraints), tuple(session.choices)
            )
        value = session.prefix[index]
        session.choices.append(value)
        session.constraints.append(expression if value else z3.Not(expression))
        return value

    def __eq__(self, other):
        expression = other.expression if isinstance(other, Decision) else other
        return Decision(self.session, self.expression == expression)

    def __ne__(self, other):
        expression = other.expression if isinstance(other, Decision) else other
        return Decision(self.session, self.expression != expression)

    def __int__(self):
        return int(bool(self))

    def __float__(self):
        return float(bool(self))

    def __add__(self, other):
        return int(self) + other

    def __radd__(self, other):
        return other + int(self)

    def __mul__(self, other):
        return int(self) * other

    def __rmul__(self, other):
        return other * int(self)


@dataclass(frozen=True)
class Slot:
    index: int


class RowState:
    def __init__(self, task, objects, initial_order, keys, prefix, parameters):
        self.task, self.objects = task, tuple(objects)
        self.initial_order = tuple(initial_order)
        self.keys, self.prefix = keys, prefix
        self.choices, self.constraints = [], []
        self.locations = {obj: index for index, obj in enumerate(initial_order)}
        self.held = self.destination = None
        self.checkpoints = []
        self.calls = 0
        self.parameters = parameters or {}
        if task == "stable_merge":
            cut = len(initial_order) // 2
            for half in (initial_order[:cut], initial_order[cut:]):
                self.constraints.extend(
                    keys[a] <= keys[b] for a, b in zip(half, half[1:])
                )
        self.checkpoint()

    def object(self, obj):
        if obj not in self.objects:
            raise PolicyError("Unknown object handle")
        return obj

    def complete_order(self):
        if self.held is not None or set(self.locations.values()) != set(
            range(len(self.objects))
        ):
            raise PolicyError("Complete-row boundary required")
        return tuple(sorted(self.objects, key=self.locations.__getitem__))

    def stable_condition(self, order):
        comparisons = [self.keys[a] <= self.keys[b] for a, b in zip(order, order[1:])]
        original = {obj: index for index, obj in enumerate(self.initial_order)}
        comparisons.extend(
            self.keys[a] != self.keys[b]
            for i, a in enumerate(order)
            for b in order[i + 1 :]
            if original[a] > original[b]
        )
        return z3.And(*comparisons)

    def checkpoint(self):
        if self.held is None and set(self.locations.values()) == set(
            range(len(self.objects))
        ):
            self.checkpoints.append(self.stable_condition(self.complete_order()))

    def count(self):
        self.calls += 1
        if self.calls > 300:
            raise PolicyBudgetExceeded("Symbolic primitive budget exhausted")

    def pick(self, obj):
        self.count()
        self.object(obj)
        if self.held is not None or self.locations[obj] is None:
            raise PolicyError("Pick requires empty gripper and available object")
        self.held, self.locations[obj] = obj, None
        self.destination = None

    def target(self, obj):
        location = self.locations[self.object(obj)]
        if location is None or not 0 <= location < len(self.objects):
            raise PolicyError("target requires current row occupant")
        return Slot(location)

    def move(self, target):
        self.count()
        if self.held is None or not isinstance(target, Slot):
            raise PolicyError("move requires a held object and frozen target")
        self.destination = target

    def release(self, target):
        self.count()
        if self.held is None or not isinstance(target, Slot):
            raise PolicyError("release requires a held object and frozen target")
        # The physical release controller also moves to its target. Do not impose
        # a move-before-release requirement absent from the physical API.
        if target.index in self.locations.values():
            raise PolicyError("Release destination is occupied")
        self.locations[self.held] = target.index
        self.held = self.destination = None
        self.checkpoint()

    def parameter(self, name, default, low, high):
        if not low <= default <= high or low == high:
            raise PolicyError("Invalid parameter bounds")
        value = self.parameters.get(name, default)
        if not low <= value <= high:
            raise PolicyError("Parameter outside bounds")
        return value

    def api(self):
        def next_star(a, b):
            order = self.complete_order()
            return order.index(self.object(a)) <= order.index(self.object(b))

        return {
            "objects": lambda: list(self.objects),
            "next_star": next_star,
            "keyless": lambda a, b: Decision(
                self, self.keys[self.object(a)] < self.keys[self.object(b)]
            ),
            "target": self.target,
            "buffer_target": lambda: Slot(len(self.objects)),
            "pick": self.pick,
            "move": self.move,
            "release": self.release,
            "parameter": self.parameter,
        }

    def goal(self):
        order = self.complete_order()
        if self.task == "sort_restore":
            return z3.And(
                z3.BoolVal(order == self.initial_order), z3.Or(*self.checkpoints)
            )
        return self.stable_condition(order)


def verify_row_policy(
    source,
    scene,
    *,
    parameters=None,
    max_paths=512,
    timeout_seconds=15,
    solver_timeout_ms=1000,
):
    objects, initial_order = scene["objects"], scene["initial_order"]
    task = scene["case"]["task"]
    scope = {
        "kind": "bounded generated-policy symbolic execution",
        "num_objects": len(objects),
        "initial_order": initial_order,
        "keys": "arbitrary integers including ties",
        "assumptions": "successful empty-destination row transfers; no controller proof",
    }
    keys = {obj: fresh_const(z3.IntSort(), "cap_key") for obj in objects}
    pending, visited, finished = [()], 0, 0
    start = perf_counter()

    def solve(constraints):
        solver = z3.Solver()
        solver.set(timeout=solver_timeout_ms)
        solver.add(*constraints)
        return solver, solver.check()

    def result(status, **details):
        return {
            "status": status,
            "scope": scope,
            "paths": visited,
            "completed_paths": finished,
            "elapsed_seconds": perf_counter() - start,
            **details,
        }

    def witness(solver, state, **details):
        model = solver.model()
        return result(
            "refuted",
            keys={
                obj: model.eval(key, model_completion=True).as_long()
                for obj, key in keys.items()
            },
            choices=state.choices,
            **details,
        )

    while pending:
        if visited >= max_paths or perf_counter() - start > timeout_seconds:
            return result("budget_exhausted", remaining_paths=len(pending))
        state = RowState(task, objects, initial_order, keys, pending.pop(), parameters)
        visited += 1
        try:
            Runtime(state.api(), max_steps=30000).run(source, inputs=scene["roles"])
            goal = state.goal()
        except BranchRequest as branch:
            for value in (False, True):
                solver, status = solve(
                    branch.constraints
                    + [branch.expression if value else z3.Not(branch.expression)]
                )
                if status == z3.unknown:
                    return result("unknown", reason=solver.reason_unknown())
                if status == z3.sat:
                    pending.append(branch.choices + (value,))
            continue
        except PolicyBudgetExceeded as exc:
            return result("budget_exhausted", error=str(exc))
        except Exception as exc:
            solver, status = solve(state.constraints)
            if status == z3.sat:
                return witness(solver, state, error=f"{type(exc).__name__}: {exc}")
            if status == z3.unknown:
                return result("unknown", reason=solver.reason_unknown())
            continue
        finished += 1
        solver, status = solve(state.constraints + [z3.Not(goal)])
        if status == z3.sat:
            return witness(
                solver, state, error="Generated policy does not establish the task goal"
            )
        if status == z3.unknown:
            return result("unknown", reason=solver.reason_unknown())
    return result("valid" if finished else "inconsistent")
