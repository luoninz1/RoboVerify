"""Shared allocation and capture-safe handling of auxiliary symbolic variables.

Use fresh_const for solver auxiliaries, passing surrounding formulas/operands in
avoid when introducing a binder. Keep get_consts/Const for persistent program
identities, vocabulary names, and intentionally shared state symbols.
Use rewrite_quantifier when transforming an existing quantified formula: its
printed binder names are not identities and must not be reconstructed as Consts.
"""

import z3


def symbol_names(expressions):
    """Collect declaration and binder names, including beneath quantifiers."""
    names, visited = set(), set()
    pending = list(expressions)
    while pending:
        expr = pending.pop()
        if isinstance(expr, str):
            names.add(expr)
            continue
        key = (expr.ctx_ref().value, expr.get_id())
        if key in visited:
            continue
        visited.add(key)
        if z3.is_quantifier(expr):
            names.update(str(expr.var_name(i)) for i in range(expr.num_vars()))
            pending.append(expr.body())
        elif z3.is_app(expr):
            if expr.decl().kind() == z3.Z3_OP_UNINTERPRETED:
                names.add(str(expr.decl().name()))
            pending.extend(expr.children())
    return names


def fresh_const(sort, prefix="aux", *, avoid=()):
    """Allocate a new typed symbol, also avoiding names already in input formulas.

    Z3 supplies the allocation counter; avoid additionally protects against user
    names that happen to spell a generated name. Never recover this variable by
    guessing its printed name: retain and use the returned expression.
    """
    occupied = symbol_names(avoid)
    while True:
        result = z3.FreshConst(sort, prefix=prefix)
        if str(result.decl().name()) not in occupied:
            return result


def fresh_name(preferred, occupied):
    """Allocate and reserve a deterministic program-level name in a mutable set."""
    name, index = preferred, 0
    while name in occupied:
        index += 1
        name = f"{preferred}_{index}"
    occupied.add(name)
    return name


def open_quantifier(expr, *, avoid=()):
    """Return fresh binders in declaration order and the instantiated body.

    substitute_vars handles de Bruijn indices and nested shadowing. The reversal
    matters: Z3's index zero denotes the last declared variable.
    """
    if not z3.is_quantifier(expr) or not (expr.is_forall() or expr.is_exists()):
        raise ValueError("Expected a universal or existential quantifier")
    surroundings = (expr, *avoid)
    variables = [
        fresh_const(expr.var_sort(i), str(expr.var_name(i)), avoid=surroundings)
        for i in range(expr.num_vars())
    ]
    return variables, z3.substitute_vars(expr.body(), *reversed(variables))


def rewrite_quantifier(expr, transform, *, avoid=()):
    """Transform a body without capturing surrounding free or nested variables.

    Preserves logical quantifier kind and variable sorts. Solver annotations
    (patterns/weights) are omitted because the body is being rewritten.
    """
    variables, body = open_quantifier(expr, avoid=avoid)
    constructor = z3.ForAll if expr.is_forall() else z3.Exists
    return constructor(variables, transform(body))
