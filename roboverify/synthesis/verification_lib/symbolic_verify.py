"""Labeled symbolic obligations and exact, fail-closed vacuity detection."""

from dataclasses import dataclass, field
from typing import Optional

import z3

from synthesis.util.symbols import fresh_const
from synthesis.verification_lib.highlevel_verification_lib import VC_CHECK_TIMEOUT_MS


@dataclass(frozen=True)
class VC:
    kind: str
    loop_id: Optional[str]
    expr: z3.BoolRef


@dataclass
class VCResult:
    vc: VC
    status: str  # valid / invalid / vacuous / unknown
    model: object = None
    reason: str = ""
    queries: int = 1


@dataclass
class SymbolicVerificationResult:
    checks: list = field(default_factory=list)
    loop_head_state: object = None
    context: object = None
    num_blocks: Optional[int] = None
    reason: str = ""
    scope: str = "context"

    @property
    def ok(self):
        return bool(self.checks) and all(c.status == "valid" for c in self.checks)

    @property
    def failure(self):
        return next((c for c in self.checks if c.status != "valid"), None)

    @property
    def failed_vc_kind(self):
        return self.failure.vc.kind if self.failure else None

    @property
    def model(self):
        return self.failure.model if self.failure else None

    def __bool__(self):
        return self.ok

    def __str__(self):
        return f"{self.ok} ({self.scope}, {[c.status for c in self.checks]})"


def discharge_vc(vc, context, timeout_ms=VC_CHECK_TIMEOUT_MS):
    """A core omitting the negated conclusion proves vacuity; its presence doesn't.

    UNKNOWN is separate from the three semantic verdicts: no model or proof may
    be manufactured from a solver timeout.
    """
    expr = vc.expr
    premise, conclusion = (
        (expr.arg(0), expr.arg(1)) if z3.is_implies(expr) else (z3.BoolVal(True), expr)
    )
    solver = context.new_solver(timeout_ms)
    premise_label, conclusion_label = fresh_const(
        z3.BoolSort(), "premise", avoid=(expr,)
    ), fresh_const(z3.BoolSort(), "neg_conclusion", avoid=(expr,))
    solver.assert_and_track(premise, premise_label)
    solver.push()
    solver.assert_and_track(z3.Not(conclusion), conclusion_label)
    answer = solver.check()
    if answer == z3.sat:
        return VCResult(vc, "invalid", solver.model())
    if answer == z3.unknown:
        return VCResult(vc, "unknown", reason=solver.reason_unknown())
    if not any(label.eq(conclusion_label) for label in solver.unsat_core()):
        return VCResult(vc, "vacuous")
    solver.pop()
    answer = solver.check()
    if answer == z3.unknown:
        return VCResult(vc, "unknown", reason=solver.reason_unknown(), queries=2)
    return VCResult(vc, "valid" if answer == z3.sat else "vacuous", queries=2)


def symbolic_verify(
    build_problem,
    *,
    min_blocks=2,
    max_blocks=4,
    timeout_ms=5000,
    prove_unbounded=True,
    constants=(),
):
    """Find the smallest finite counterexample, then optionally prove generically.

    ``build_problem(n)`` returns (program, pre, post, context); ``n=None`` requests
    DeclareSort. Successful finite checking is explicitly labeled bounded and
    never promoted to an unbounded proof. An unknown at a smaller size stops the
    search, since a later model could no longer be called the smallest.
    """
    from synthesis.verification_lib.counterexamples import (
        UnrealizableCounterexample,
        model_to_loop_head,
    )

    if not 1 <= min_blocks <= max_blocks:
        raise ValueError("Require 1 <= min_blocks <= max_blocks")
    sizes = list(range(min_blocks, max_blocks + 1))
    if prove_unbounded:
        sizes.append(None)
    all_checks = []
    for size in sizes:
        program, pre, post, context = build_problem(size)
        checks = [
            discharge_vc(vc, context, timeout_ms)
            for vc in program.VC_gen(pre, post, context)
        ]
        result = SymbolicVerificationResult(
            checks,
            context=context,
            num_blocks=size,
            scope="unbounded" if size is None else f"finite:{size}",
        )
        if not result:
            if result.model is not None and size is not None:
                try:
                    result.loop_head_state = model_to_loop_head(
                        context,
                        result.model,
                        result.failure.vc.loop_id,
                        constants,
                        timeout_ms=timeout_ms,
                    )
                except UnrealizableCounterexample as exc:
                    result.reason = str(exc)
            return result
        all_checks.extend(checks)
    result.checks = all_checks
    if not prove_unbounded:
        result.scope = f"finite:{min_blocks}..{max_blocks}"
    return result


SymbolicVerify = symbolic_verify
