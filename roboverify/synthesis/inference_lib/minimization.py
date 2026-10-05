"""Selectable Boolean minimization within the fixed partition learner."""

from contextlib import contextmanager
from contextvars import ContextVar

import sympy

DEFAULT_INVARIANT_MINIMIZER = "sympy"
INVARIANT_MINIMIZERS = ("sympy", "pyeda")
_minimizer = ContextVar("invariant_minimizer", default=DEFAULT_INVARIANT_MINIMIZER)


def get_invariant_minimizer():
    return _minimizer.get()


@contextmanager
def using_invariant_minimizer(backend=None):
    """Select a backend for all nested inference calls, restoring it on exit.

    None inherits the enclosing scope (SymPy by default).
    """
    if backend is None:
        backend = get_invariant_minimizer()
    if backend not in INVARIANT_MINIMIZERS:
        raise ValueError(f"Unknown invariant minimizer: {backend!r}")
    token = _minimizer.set(backend)
    try:
        yield
    finally:
        _minimizer.reset(token)


def espresso_form(var_symbols, rows, *, complement=False):
    """Minimize a fully specified on-set, optionally returning its CNF complement.

    Every omitted row is false, never a don't-care. For POS, callers pass the
    rejected rows: complementing their minimized DNF by De Morgan yields CNF
    without an exponential DNF-to-CNF distribution. SymPy is only the returned
    expression representation; it performs no minimization on this path.
    """
    from pyeda.boolalg.expr import And, Or, exprvar
    from pyeda.boolalg.minimization import espresso_exprs

    variables = tuple(exprvar("term", i) for i in range(len(var_symbols)))
    expression = Or(
        *(
            And(*(var if bit else ~var for var, bit in zip(variables, row)))
            for row in rows
        )
    )
    # Espresso requires at least one input; constant functions need no work.
    if not expression.is_zero() and not expression.is_one():
        (expression,) = espresso_exprs(expression)
    symbols_by_id = {var.uniqid: symbol for var, symbol in zip(variables, var_symbols)}

    def convert(node):
        kind, *args = node
        if kind == "const":
            return sympy.true if bool(args[0]) != complement else sympy.false
        if kind == "lit":
            literal = args[0]
            symbol = symbols_by_id[abs(literal)]
            return sympy.Not(symbol) if (literal < 0) != complement else symbol
        if kind in ("and", "or"):
            operator = sympy.And if (kind == "and") != complement else sympy.Or
            return operator(*(convert(arg) for arg in args))
        raise TypeError(f"Unsupported Espresso expression node: {kind}")

    return convert(expression.to_ast())
