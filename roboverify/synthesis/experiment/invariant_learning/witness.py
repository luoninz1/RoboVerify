"""Reachable VC failures with bounded, unordered execution witnesses."""

import itertools
from dataclasses import dataclass, field

import z3

from synthesis.api.instructions import Assign, Get, Put, Skip, While
from synthesis.api.program import wp
from synthesis.util.symbols import fresh_const


def finite_formula(expr, context):
    """Expand finite Box quantifiers with capture-safe de Bruijn substitution."""
    memo = {}

    def visit(node):
        # Retain ASTs: numeric Z3 IDs can be reused after temporary binders die.
        key = node
        if key in memo:
            return memo[key]
        if z3.is_quantifier(node):
            if any(node.var_sort(i) != context.BoxSort for i in range(node.num_vars())):
                raise ValueError("Witness queries support Box quantifiers only")
            rows = [
                visit(z3.substitute_vars(node.body(), *reversed(values)))
                for values in itertools.product(
                    context.enum_blocks, repeat=node.num_vars()
                )
            ]
            result = (z3.And if node.is_forall() else z3.Or)(*rows)
        elif z3.is_app(node) and node.num_args():
            result = node.decl()(*(visit(child) for child in node.children()))
        else:
            result = node
        result = z3.simplify(result)
        memo[key] = result
        return result

    return visit(expr)


@dataclass
class GuardChoice:
    kind: str
    names: tuple
    values: tuple


def bind_choice(instruction, target, context):
    """One existential solver choice, with no physical-ID ordering constraint."""
    variables = tuple(context.get_consts(str(v)) for v in instruction.guard_exists_vars)
    values = tuple(
        fresh_const(
            context.BoxSort, "witness", avoid=(target, instruction.instantiated_cond)
        )
        for _ in variables
    )
    expr = (
        z3.substitute(
            z3.And(instruction.instantiated_cond, target), *zip(variables, values)
        )
        if variables
        else z3.And(instruction.instantiated_cond, target)
    )
    return expr, GuardChoice(
        type(instruction).__name__, tuple(map(str, variables)), values
    )


def execution_preimage(instructions, target, context):
    """Existential preimage of a straight-line body; return replay choices too.

    Once Get/While witnesses are fixed, Assign/Put/Skip use the shared WP rules.
    Never apply wp to While: that would assume the candidate invariant.
    """
    choices = []
    for instruction in reversed(instructions):
        if isinstance(instruction, Get):
            target, choice = bind_choice(instruction, target, context)
            choices.insert(0, choice)
        elif isinstance(instruction, (Assign, Put, Skip)):
            target = wp(instruction, target, context)
        else:
            raise ValueError(
                f"Unsupported bounded instruction: {type(instruction).__name__}"
            )
        target = finite_formula(target, context)
    return target, choices


@dataclass
class ReplayPlan:
    failure_kind: str
    loop_id: str
    head_iteration: int
    guard_choices: list
    condition: object

    def metadata(self):
        return dict(
            failure_kind=self.failure_kind,
            loop_id=self.loop_id,
            head_iteration=self.head_iteration,
            guard_choices=self.guard_choices,
        )


@dataclass
class ReachPath:
    condition: object
    choices: list
    failure_kind: str
    loop_id: str
    head_iteration: int

    def decode(self, model, context):
        ids = {str(value): i for i, value in enumerate(context.enum_blocks)}
        substitutions, choices = [], []
        for choice in self.choices:
            values = [model.eval(v, model_completion=True) for v in choice.values]
            substitutions.extend(zip(choice.values, values))
            choices.append(
                dict(
                    kind=choice.kind,
                    bindings={
                        name: ids[str(value)]
                        for name, value in zip(choice.names, values)
                    },
                )
            )
        condition = (
            z3.substitute(self.condition, *substitutions)
            if substitutions
            else self.condition
        )
        return ReplayPlan(
            self.failure_kind, self.loop_id, self.head_iteration, choices, condition
        )


def failure_paths(program, context, vcs, iterations):
    """Paths to the selected failed VCs, bounded in total executed loop bodies.

    Establishment targets the first head. Preservation reaches I & guard &
    NOT wp(body, I), including a replay choice for the failing body. Prefix
    execution and all earlier iterations are explicit; I is not assumed there.
    """
    if context.mode != "enum" or iterations < 0:
        raise ValueError(
            "A finite context and nonnegative iteration bound are required"
        )
    loops = [
        (i, row) for i, row in enumerate(program.instructions) if isinstance(row, While)
    ]
    if len(loops) != 1 or any(isinstance(row, While) for row in loops[0][1].body):
        raise ValueError("Witness queries require one non-nested loop")
    index, loop = loops[0]
    paths = []
    for vc in vcs:
        if vc.loop_id != str(index):
            raise ValueError("Selected VC does not belong to the supported loop")
        if vc.kind not in ("establish", "preserve"):
            raise ValueError(f"Unsupported reachable failure: {vc.kind}")
        for depth in ([0] if vc.kind == "establish" else range(iterations)):
            choices = []
            if vc.kind == "establish":
                target = z3.Not(loop.invariant)
            else:
                # Negated VC is the target, rather than any missing invariant state.
                target, choice = bind_choice(loop, z3.Not(vc.expr), context)
                choices.append(choice)
                # Preserve body Get choices if present: a concrete successful body
                # must actually reach NOT I. Also excludes undefined Put executions.
                successor, body_choices = execution_preimage(
                    loop.body, z3.Not(loop.invariant), context
                )
                successor = (
                    z3.substitute(
                        successor,
                        *zip(
                            [
                                context.get_consts(str(v))
                                for v in loop.guard_exists_vars
                            ],
                            choice.values,
                        ),
                    )
                    if choice.values
                    else successor
                )
                target = z3.And(target, successor)
                choices.extend(body_choices)
            for _ in range(depth):
                target, body_choices = execution_preimage(loop.body, target, context)
                target, choice = bind_choice(loop, target, context)
                choices = [choice, *body_choices, *choices]
            target, prefix_choices = execution_preimage(
                program.instructions[:index], target, context
            )
            paths.append(
                ReachPath(
                    finite_formula(target, context),
                    prefix_choices + choices,
                    vc.kind,
                    vc.loop_id,
                    depth,
                )
            )
    return paths


@dataclass
class WitnessQuery:
    solver: object
    coverage: object
    decode: object
    iterations: int
    context: object = None
    paths: list = field(default_factory=list)

    def replay_plan(self, model):
        for path in self.paths:
            if z3.is_true(model.eval(path.condition, model_completion=True)):
                return path.decode(model, self.context)
        raise ValueError("SAT reachability query has no satisfied execution path")


@dataclass
class WitnessSearch:
    status: str
    attempts: list = field(default_factory=list)
    size: int | None = None
    witness: object = None
    query: object = None
    reason: str = ""
    plan: object = None


def find_witness(build_query, max_blocks, *, record=None):
    """Smallest reachable failure within the initial domain and execution bounds.

    Query initial geometry, execution, and the failed VC together. No separate
    finite inductiveness checks or unreachable countermodels enter this search.
    """
    attempts = []
    for size in range(1, max_blocks + 1):
        query = build_query(size)
        feasible = query.solver.check()
        row = dict(
            num_blocks=size, iterations=query.iterations, initial_domain=str(feasible)
        )
        if feasible == z3.unknown:
            row.update(status="unknown", reason=query.solver.reason_unknown())
        elif feasible == z3.unsat:
            row.update(status="empty_initial_domain")
        else:
            query.solver.add(query.coverage)
            answer = query.solver.check()
            row.update(status=str(answer))
            if answer == z3.unknown:
                row["reason"] = query.solver.reason_unknown()
        attempts.append(row)
        if record:
            record(size, query, row)
        if row["status"] == "unknown":
            return WitnessSearch("unknown", attempts, reason=row["reason"])
        if row["status"] == "sat":
            model = query.solver.model()
            return WitnessSearch(
                "found",
                attempts,
                size,
                query.decode(model),
                query,
                plan=query.replay_plan(model),
            )
    return WitnessSearch(
        "no_reachable_counterexample",
        attempts,
        reason="No reachable failure found within the block, initial-environment, and execution bounds",
    )
