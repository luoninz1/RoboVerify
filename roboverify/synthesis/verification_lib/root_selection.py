"""Prove tower roots using POPL §5.5's universally quantified criterion.

Candidate names are only a search space, never evidence. The entry context and
weakest-precondition history cover arbitrary objects, not just named blocks.
"""

from dataclasses import dataclass, field

import z3

from synthesis.api.program import to_seq, wp
from synthesis.util.symbols import fresh_const


@dataclass(frozen=True)
class RootSelection:
    status: str
    root: str = None
    reason: str = ""


@dataclass
class RootContext:
    context: object
    entry: object
    timeout_ms: int = 5000
    history: list = field(default_factory=list)

    def before(self, condition):
        return (
            wp(to_seq(self.history), condition, self.context)
            if self.history
            else condition
        )

    def prove(self, condition):
        solver = self.context.new_solver(self.timeout_ms)
        solver.add(self.entry)
        consistency = solver.check()
        if consistency != z3.sat:
            return RootSelection(
                "inconsistent" if consistency == z3.unsat else "unknown",
                reason=(
                    "Inconsistent root context"
                    if consistency == z3.unsat
                    else solver.reason_unknown()
                ),
            )
        solver.add(z3.Not(self.before(condition)))
        answer = solver.check()
        return RootSelection(
            (
                "valid"
                if answer == z3.unsat
                else "unproved" if answer == z3.sat else "unknown"
            ),
            reason=solver.reason_unknown() if answer == z3.unknown else "",
        )

    def select(self, target, candidates, *, continuation=(), postcondition=None):
        """Prove P => forall u. ON*(target,u) => ON*(u,root).

        P is wp(continuation, postcondition), when supplied, in the established
        entry context transported through history. Checking that context implies
        P prevents assuming a desired postcondition to manufacture a root proof.
        Unrestricted u covers unnamed objects. Table and diagnostic witnesses
        cannot be chosen as physical roots.
        """
        required = z3.BoolVal(True)
        if postcondition is not None:
            required = (
                wp(to_seq(continuation), postcondition, self.context)
                if continuation
                else postcondition
            )
            applicable = self.prove(required)
            if applicable.status != "valid":
                return RootSelection(
                    applicable.status,
                    reason=applicable.reason
                    or "Entry context does not establish the continuation's weakest precondition",
                )
        candidates = tuple(candidates)
        u = fresh_const(
            self.context.BoxSort,
            prefix="root_member",
            avoid=(required, target, *candidates),
        )
        y = self.context.get_consts(target)
        unknown = None
        for name in sorted(set(candidates) - {"tbl", "sym"}):
            root = self.context.get_consts(name)
            criterion = z3.ForAll(
                [u],
                z3.Implies(self.context.ON_star(y, u), self.context.ON_star(u, root)),
            )
            if self.context.use_tbl:
                criterion = z3.And(root != self.context.get_consts("tbl"), criterion)
            result = self.prove(z3.Implies(required, criterion))
            if result.status == "valid":
                return RootSelection("valid", name, "Proved root for " + target)
            if result.status == "inconsistent":
                return result
            if result.status == "unknown":
                unknown = result.reason
        return RootSelection(
            "unknown" if unknown is not None else "unproved",
            reason=unknown or "No in-scope object is provably the tower root",
        )

    def initial_roots(self, candidates):
        """Named input roots whose tight alignment may be assumed at entry."""
        roots = []
        for name in sorted(set(candidates) - {"tbl", "sym"}):
            result = self.select(name, [name])
            if result.status == "valid":
                roots.append(name)
        return roots
