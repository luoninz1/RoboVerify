import itertools
import os
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Union

from z3 import (
    Z3_OP_UNINTERPRETED,
    And,
    Const,
    Consts,
    Exists,
    ForAll,
    Implies,
    Not,
    Or,
    is_and,
    is_app,
    is_false,
    is_implies,
    is_not,
    is_or,
    is_quantifier,
    is_true,
    sat,
    simplify,
    substitute,
    substitute_vars,
    unsat,
)

import synthesis.inference_lib.inference
import synthesis.verification_lib.highlevel_verification_lib as highlevel_verification_lib
import synthesis.verification_lib.lowlevel_verification_lib as lowlevel_verification_lib
from synthesis.api.instructions import (
    Assign,
    GoalAssign,
    Instruction,
    MarkGoal,
    MoveDown,
    MoveRight,
    PickPlace,
    PickPlaceByName,
    Put,
    Seq,
    Skip,
    While,
)


def _outer_while_direct_body_move_down_var(
    instructions: List[Instruction],
) -> Optional[str]:
    """
    For nested goal programs, return the ``var_name`` of ``MoveDown`` in the
    *first* top-level ``While``'s *direct* body (last such instruction wins).

    Used for VC diagnosis: preserve obligations use ``wp(MoveDown(x), inv)`` on
    each outer conjunct, not the raw conjunct at the pre-step state.
    """
    for inst in instructions:
        if isinstance(inst, While):
            last: Optional[str] = None
            for b in inst.body:
                if isinstance(b, MoveDown):
                    last = b.var_name
            return last
    return None


def _sexpr_trunc(expr, max_len: int = 420) -> str:
    s = expr.sexpr()
    if len(s) <= max_len:
        return s
    return s[:max_len] + " ..."


_MAX_FINITE_FORALL_DIAG_DEPTH = 16
_MAX_GOAL_FORALL_PRODUCT = 25000


def _goal_universe_for_diag(solver: highlevel_verification_lib.HighLevelContext):
    """Finite Goal domain (null + enum goals), or None if not available."""
    if getattr(solver, "GoalSort", None) is None:
        return None
    u = [solver.null] + list(getattr(solver, "enum_goals", []) or [])
    return u if u else None


def _is_goal_only_forall(solver, expr) -> bool:
    if not (is_quantifier(expr) and expr.is_forall()):
        return False
    if _goal_universe_for_diag(solver) is None:
        return False
    gsort = solver.GoalSort
    return all(expr.var_sort(i) == gsort for i in range(expr.num_vars()))


def _tri_goal_forall_by_finite_expansion(
    model,
    solver: highlevel_verification_lib.HighLevelContext,
    q,
    _depth: int,
) -> str:
    """
    For ``ForAll`` over finite Goal only: truth = all instantiations true (finite
    semantics). Z3 often leaves ``model.eval(ForAll...)`` non-Boolean; this forces
    a definite true/false when every instantiated body evaluates to a constant.
    """
    universe = _goal_universe_for_diag(solver)
    n = q.num_vars()
    body = q.body()
    seen = 0
    for tup in itertools.product(universe, repeat=n):
        seen += 1
        if seen > _MAX_GOAL_FORALL_PRODUCT:
            return "unknown"
        # For ``ForAll([v0,...,v_{n-1}], body)``, DB index 0 is innermost = v_{n-1}.
        inst = substitute_vars(body, *reversed(tup))
        st = _tri_status_under_model(model, inst, solver, _depth + 1)
        if st == "false":
            return "false"
        if st == "unknown":
            return "unknown"
    return "true"


def _flatten_or(expr):
    if is_or(expr):
        out: List = []
        for ch in expr.children():
            out.extend(_flatten_or(ch))
        return out
    return [expr]


def _tri_status_under_model(model, expr, solver=None, _depth: int = 0) -> str:
    """
    Whether ``expr`` holds in ``model``: 'true', 'false', or 'unknown'.

    When ``solver`` is a goals-mode context with finite Goal, top-level
    ``ForAll`` over Goal is decided by enumerating all tuples (finite expansion),
    so Z3 does not need to reduce the quantifier node to a Boolean constant.
    """
    try:
        expr = simplify(expr)
    except Exception:
        pass

    if (
        solver is not None
        and _depth < _MAX_FINITE_FORALL_DIAG_DEPTH
        and _is_goal_only_forall(solver, expr)
    ):
        return _tri_goal_forall_by_finite_expansion(model, solver, expr, _depth)

    # ``wp(MoveDown(i), Q)`` is ``And(i != null, ForAll z. ...)``. ``model.eval`` on the
    # whole ``And`` often stays non-Boolean when ``dtot`` expands under ``z``; peel
    # conjuncts so inner ``ForAll`` over Goal still gets finite expansion.
    if solver is not None and _depth < _MAX_FINITE_FORALL_DIAG_DEPTH and is_and(expr):
        chs = _flatten_and(expr)
        if len(chs) > 1:
            saw_unknown = False
            for ch in chs:
                st = _tri_status_under_model(model, ch, solver, _depth + 1)
                if st == "false":
                    return "false"
                if st == "unknown":
                    saw_unknown = True
            return "unknown" if saw_unknown else "true"

    # Finite boolean structure (common under Goal ForAll bodies, e.g. ``Implies(dtot, …)``).
    if solver is not None and _depth < _MAX_FINITE_FORALL_DIAG_DEPTH:
        if is_implies(expr):
            a, b = expr.arg(0), expr.arg(1)
            st_b = _tri_status_under_model(model, b, solver, _depth + 1)
            if st_b == "true":
                return "true"
            st_a = _tri_status_under_model(model, a, solver, _depth + 1)
            if st_a == "false":
                return "true"
            if st_a == "true":
                return st_b
            return "unknown"
        if is_not(expr):
            st = _tri_status_under_model(model, expr.arg(0), solver, _depth + 1)
            if st == "true":
                return "false"
            if st == "false":
                return "true"
            return "unknown"
        if is_or(expr):
            chs = _flatten_or(expr)
            if len(chs) > 1:
                saw_unknown = False
                any_true = False
                for ch in chs:
                    st = _tri_status_under_model(model, ch, solver, _depth + 1)
                    if st == "true":
                        any_true = True
                        break
                    if st == "unknown":
                        saw_unknown = True
                if any_true:
                    return "true"
                if saw_unknown:
                    return "unknown"
                return "false"

    v = simplify(model.eval(expr, True))
    if is_true(v):
        return "true"
    if is_false(v):
        return "false"
    vn = simplify(model.eval(Not(expr), True))
    if is_true(vn):
        return "false"
    if is_false(vn):
        return "true"
    return "unknown"


def _flatten_and(expr):
    """Flatten nested And into a list of conjuncts (single non-And node stays one element)."""
    if is_and(expr):
        out: List = []
        for ch in expr.children():
            out.extend(_flatten_and(ch))
        return out
    return [expr]


def _forall_goal_find_falsifying_witness(
    solver: highlevel_verification_lib.HighLevelContext, model, q, _depth: int = 0
):
    """
    If ``q`` is ForAll over GoalSort only, enumerate finite Goal tuples and return
    (witness_tuple, instantiated_body) for the first instantiation that is false in model.
    """
    if not (is_quantifier(q) and q.is_forall()):
        return None
    gsort = getattr(solver, "GoalSort", None)
    if gsort is None:
        return None
    n = q.num_vars()
    if n == 0 or any(q.var_sort(i) != gsort for i in range(n)):
        return None
    universe = _goal_universe_for_diag(solver)
    if not universe:
        return None
    body = q.body()
    seen = 0
    for tup in itertools.product(universe, repeat=n):
        seen += 1
        if seen > _MAX_GOAL_FORALL_PRODUCT:
            break
        inst = substitute_vars(body, *reversed(tup))
        if _tri_status_under_model(model, inst, solver, _depth + 1) == "false":
            return tup, inst
    return None


def print_where_conclusion_fails(
    solver: highlevel_verification_lib.HighLevelContext,
    model,
    conclusion,
    max_lines: int = 96,
    max_conjuncts_to_list: int = 64,
    outer_move_down_var: Optional[str] = None,
) -> None:
    """
    When VC check-2 is SAT, the model satisfies premise ∧ ¬conclusion, so ``conclusion``
    is false. Flatten nested And, then report each conjunct that is false (or unknown)
    under the model; drill into Implies / nested And / Goal ForAll when possible.

    When ``outer_move_down_var`` is set and the solver carries ``_learned_clause_provenance``,
    also print per-conjunct ``wp(MoveDown(outer_move_down_var), C)`` under the model
    (same encoding as ``wp()`` in this module), instead of evaluating raw ``C`` at the CE.
    """
    state = {"n": 0}

    def emit(msg: str) -> bool:
        if state["n"] >= max_lines:
            return False
        print(msg)
        state["n"] += 1
        return True

    def walk(expr, path: str) -> None:
        if state["n"] >= max_lines:
            return
        truth = _tri_status_under_model(model, expr, solver)
        if truth == "true":
            return
        if truth == "unknown":
            if is_quantifier(expr) and expr.is_forall():
                wit = _forall_goal_find_falsifying_witness(solver, model, expr)
                if wit is not None:
                    tup, inst = wit
                    emit(f"{path}: ForAll falsified at Goal witness {tup}")
                    walk(inst, f"{path}@{tup}")
                    return
            emit(f"{path}: unknown truth under model; {_sexpr_trunc(expr)}")
            return

        if is_implies(expr):
            ant, cons = expr.arg(0), expr.arg(1)
            if (
                _tri_status_under_model(model, ant, solver) == "true"
                and _tri_status_under_model(model, cons, solver) == "false"
            ):
                emit(
                    f"{path}: Implies with TRUE antecedent and FALSE consequent "
                    f"(antecedent: {_sexpr_trunc(ant, 200)})"
                )
                walk(cons, f"{path}/consequent")
                return
            emit(f"{path}: false Implies (unexpected shape): {_sexpr_trunc(expr)}")
            return

        if is_and(expr):
            chs = expr.children()
            false_idxs = [
                i
                for i, ch in enumerate(chs)
                if _tri_status_under_model(model, ch, solver) != "true"
            ]
            strict_false = [
                i
                for i, ch in enumerate(chs)
                if _tri_status_under_model(model, ch, solver) == "false"
            ]
            emit(
                f"{path}: And — {len(strict_false)} definitely false, "
                f"{len(false_idxs) - len(strict_false)} unknown among {len(chs)} children "
                f"(non-true indices {false_idxs[:30]}{'...' if len(false_idxs) > 30 else ''})"
            )
            for i in false_idxs:
                if state["n"] >= max_lines:
                    break
                walk(chs[i], f"{path}/[{i}]")
            return

        if (
            is_quantifier(expr)
            and expr.is_forall()
            and getattr(solver, "GoalSort", None) is not None
        ):
            wit = _forall_goal_find_falsifying_witness(solver, model, expr)
            if wit is not None:
                tup, inst = wit
                emit(f"{path}: ForAll falsified at Goal witness {tup}")
                walk(inst, f"{path}@{tup}")
                return

        emit(f"{path}: FALSE: {_sexpr_trunc(expr)}")

    print(
        "[diagnosis] conclusion is false in the counterexample model; "
        "failing conjunct(s) below (nested And flattened; truncated s-exprs). "
        "ForAll over finite Goal is evaluated by full tuple enumeration (not model.eval)."
    )
    parts = _flatten_and(conclusion)
    n_true = n_false = n_unk = 0
    bad_rows: List[tuple] = []
    # One-shot debug aid for "unknown under model" ForAll in Goal-mode.
    dbg_forall_printed = False
    for i, p in enumerate(parts):
        st = _tri_status_under_model(model, p, solver)
        if st == "true":
            n_true += 1
        elif st == "false":
            n_false += 1
            bad_rows.append((i, p, st))
        else:
            n_unk += 1
            bad_rows.append((i, p, st))
            if (
                not dbg_forall_printed
                and st == "unknown"
                and is_quantifier(p)
                and p.is_forall()
                and getattr(solver, "GoalSort", None) is not None
            ):
                dbg_forall_printed = True
                u = _goal_universe_for_diag(solver)
                gsort = getattr(solver, "GoalSort", None)
                # Print only metadata (no huge s-expr dumps).
                emit("[diagnosis-debug] first unknown ForAll summary:")
                emit(
                    f"[diagnosis-debug]  enum_goals={len(getattr(solver, 'enum_goals', []) or [])}, "
                    f"universe_size={(len(u) if u else 0)} (includes null), "
                    f"num_vars={p.num_vars()}, "
                    f"product={( (len(u) ** p.num_vars()) if u else 0 )}, "
                    f"cutoff={_MAX_GOAL_FORALL_PRODUCT}"
                )
                emit(
                    f"[diagnosis-debug]  is_goal_only_forall={_is_goal_only_forall(solver, p)}; "
                    f"GoalSort={gsort}; "
                    f"var_sorts={[p.var_sort(j) for j in range(p.num_vars())]}"
                )
    emit(
        f"Flattened conjunction: {len(parts)} conjunct(s) — "
        f"{n_true} true, {n_false} false, {n_unk} unknown under model."
    )
    listed = 0
    for i, p, st in bad_rows:
        if listed >= max_conjuncts_to_list or state["n"] >= max_lines:
            break
        emit(f"  conjunct[{i}] ({st}): {_sexpr_trunc(p)}")
        # Robust provenance lookup: match against clauses carried on While nodes.
        provs = getattr(solver, "_learned_clause_provenance", []) or []
        matched = 0
        for c in provs:
            try:
                if p.eq(c.expr):
                    emit(
                        "    -> learned-from: "
                        f"omega_index={getattr(c, 'omega_index', None)} "
                        f"target={getattr(c, 'target_predicate', None)} "
                        f"via={getattr(c, 'learned_via', None)}"
                    )
                    matched += 1
                    if matched >= 3:
                        break
            except Exception:
                continue
        if st == "unknown" and is_quantifier(p) and p.is_forall():
            wit = _forall_goal_find_falsifying_witness(solver, model, p)
            if wit is not None:
                tup, inst = wit
                emit(
                    f"    -> ForAll broken at Goal witness {tup}; body: {_sexpr_trunc(inst, 300)}"
                )
        listed += 1
    if len(bad_rows) > listed:
        emit(
            f"  ... ({len(bad_rows) - listed} more false/unknown conjunct(s) not shown; "
            f"raise max_conjuncts_to_list={max_conjuncts_to_list})"
        )

    provs = getattr(solver, "_learned_clause_provenance", []) or []
    if provs and outer_move_down_var and getattr(solver, "GoalSort", None) is not None:
        emit(
            "[diagnosis] outer invariant by conjunct: "
            f"wp(MoveDown({outer_move_down_var!r}), C_j) under the CE model "
            "(library wp / ctx.dtot; same as synthesis.api.program.wp for MoveDown)."
        )
        n_wp_true = n_wp_false = n_wp_unk = 0
        for j, c in enumerate(provs):
            expr = getattr(c, "expr", None)
            if expr is None:
                continue
            obl = _wp_move_down_on_goal_var(expr, solver, outer_move_down_var)
            st_wp = _tri_status_under_model(model, obl, solver)
            if st_wp == "true":
                n_wp_true += 1
            elif st_wp == "false":
                n_wp_false += 1
            else:
                n_wp_unk += 1
        emit(
            f"  {len(provs)} provenance conjunct(s): "
            f"wp holds (true) on {n_wp_true}, false on {n_wp_false}, unknown on {n_wp_unk}."
        )
        shown_wp = 0
        for j, c in enumerate(provs):
            if state["n"] >= max_lines or shown_wp >= max_conjuncts_to_list:
                break
            expr = getattr(c, "expr", None)
            if expr is None:
                continue
            obl = _wp_move_down_on_goal_var(expr, solver, outer_move_down_var)
            st_wp = _tri_status_under_model(model, obl, solver)
            if st_wp == "true":
                continue
            emit(
                f"  outer_inv[{j}] wp→({st_wp}): "
                f"omega_index={getattr(c, 'omega_index', None)} "
                f"target={getattr(c, 'target_predicate', None)} "
                f"via={getattr(c, 'learned_via', None)} "
                f"wp_obl={_sexpr_trunc(obl, 220)}"
            )
            shown_wp += 1
        failing_wp = n_wp_false + n_wp_unk
        if failing_wp > shown_wp:
            emit(
                f"  ... ({failing_wp - shown_wp} more false/unknown wp row(s) not shown; "
                f"raise max_conjuncts_to_list={max_conjuncts_to_list})"
            )
    elif (
        provs
        and getattr(solver, "GoalSort", None) is not None
        and not outer_move_down_var
    ):
        emit(
            "[diagnosis] outer invariant provenance present but no MoveDown in the first "
            "While's direct body — skipping wp-per-conjunct table."
        )
    emit("[diagnosis] structured walk (nested detail):")
    walk(conclusion, "conclusion")


def rewrite_for_put_for_ON_star(expr, b_prime, b, context):
    """Rewrite every possible occurrence of alpha<on*> beta to
    Or(alpha<on*>beta, And(alpha<on*>b_prime, b<on*>beta))
    """
    # Case 1: Quantifier
    if is_quantifier(expr):
        # Extract info about the quantifier
        num_vars = expr.num_vars()
        var_sorts = [expr.var_sort(i) for i in range(num_vars)]
        var_names = [expr.var_name(i) for i in range(num_vars)]

        # Extract and rewrite the body
        body = expr.body()
        rewritten_body = rewrite_for_put_for_ON_star(body, b_prime, b, context)

        # Rebuild the quantifier (keep same type)
        if expr.is_forall():
            return ForAll(
                list(map(lambda n_s: Const(n_s[0], n_s[1]), zip(var_names, var_sorts))),
                rewritten_body,
            )
        else:
            return Exists(
                list(map(lambda n_s: Const(n_s[0], n_s[1]), zip(var_names, var_sorts))),
                rewritten_body,
            )

    # Case 2: Function application (And, Or, ON_star, etc.)
    elif is_app(expr):
        decl = expr.decl()

        # Match ON_star(a,b)
        if decl.kind() == Z3_OP_UNINTERPRETED and decl.name() == "ON_star":
            alpha, beta = expr.children()
            return Or(
                context.ON_star(alpha, beta),
                And(
                    context.ON_star(alpha, b_prime),
                    context.ON_star(b, beta),
                ),
            )

        # Otherwise rebuild recursively
        new_children = [
            rewrite_for_put_for_ON_star(c, b_prime, b, context) for c in expr.children()
        ]
        return decl(*new_children)

    # Case 3: Constants, bound variables, etc.
    else:
        return expr


def rewrite_for_put_on_tbl_for_ON_star(expr, b_prime, context):
    """Weakest-precondition rewrite for ON_star after put(upper, tbl).

    Matches highlevel/test_unstack_single_tower_b0_bottom_exists_top_inferred.py
    ``wp_for_put_on_tbl`` / ``ON_func_substituted``:

        ON'(alpha, beta) =
            ON(alpha, beta) ∧ (¬ON(alpha, b_prime) ∨ ON(beta, b_prime))

    No acyclicity guard (unlike put on a box).
    """
    if is_quantifier(expr):
        num_vars = expr.num_vars()
        var_sorts = [expr.var_sort(i) for i in range(num_vars)]
        var_names = [expr.var_name(i) for i in range(num_vars)]
        body = expr.body()
        rewritten_body = rewrite_for_put_on_tbl_for_ON_star(body, b_prime, context)
        if expr.is_forall():
            return ForAll(
                list(map(lambda n_s: Const(n_s[0], n_s[1]), zip(var_names, var_sorts))),
                rewritten_body,
            )
        return Exists(
            list(map(lambda n_s: Const(n_s[0], n_s[1]), zip(var_names, var_sorts))),
            rewritten_body,
        )

    if is_app(expr):
        decl = expr.decl()
        if decl.kind() == Z3_OP_UNINTERPRETED and decl.name() == "ON_star":
            alpha, beta = expr.children()
            return And(
                context.ON_star(alpha, beta),
                Or(
                    Not(context.ON_star(alpha, b_prime)),
                    context.ON_star(beta, b_prime),
                ),
            )
        new_children = [
            rewrite_for_put_on_tbl_for_ON_star(c, b_prime, context)
            for c in expr.children()
        ]
        return decl(*new_children)

    return expr


def rewrite_for_put_on_tbl_for_Higher(expr, placed_block, context):
    """Weakest-precondition rewrite for Higher after put(placed_block, tbl).

    Parallel to ``rewrite_for_put_on_tbl_for_ON_star`` / ``ON_func_substituted``:

        Higher'(m, n) =
            Higher(m, n) ∧ (¬Higher(m, placed_block) ∨ Higher(n, placed_block))

    For ``Put("b", "tbl")``, ``placed_block`` is the constant for ``b``.
    """
    if is_quantifier(expr):
        num_vars = expr.num_vars()
        var_sorts = [expr.var_sort(i) for i in range(num_vars)]
        var_names = [expr.var_name(i) for i in range(num_vars)]
        body = expr.body()
        rewritten_body = rewrite_for_put_on_tbl_for_Higher(body, placed_block, context)
        if expr.is_forall():
            return ForAll(
                list(map(lambda n_s: Const(n_s[0], n_s[1]), zip(var_names, var_sorts))),
                rewritten_body,
            )
        return Exists(
            list(map(lambda n_s: Const(n_s[0], n_s[1]), zip(var_names, var_sorts))),
            rewritten_body,
        )

    if is_app(expr):
        decl = expr.decl()
        if decl.kind() == Z3_OP_UNINTERPRETED and decl.name() == "Higher":
            m, n = expr.children()
            t = Const("t", context.BoxSort)
            tbl = Const("tbl", context.BoxSort)
            return Or(
                And(m == placed_block, n == placed_block),
                And(m != placed_block, n != placed_block, context.Higher(m, n)),
                And(
                    m == placed_block,
                    n != placed_block,
                    ForAll([t], Implies(t != tbl, context.Higher(t, n))),
                ),
                And(m != placed_block, n == placed_block, n != tbl),
            )
        new_children = [
            rewrite_for_put_on_tbl_for_Higher(c, placed_block, context)
            for c in expr.children()
        ]
        return decl(*new_children)

    return expr


def _put_base_is_tbl(seq_instruction: Put) -> bool:
    return seq_instruction.base_block == "tbl"


def rewrite_for_put_for_higher(expr, b_prime, b, context):
    """Rewrite every possible occurrence of alpha<higher> beta to
    Or(alpha<higher>beta, And(alpha<higher>b_prime, b<higher>beta))
    """
    # Case 1: Quantifier
    if is_quantifier(expr):
        # Extract info about the quantifier
        num_vars = expr.num_vars()
        var_sorts = [expr.var_sort(i) for i in range(num_vars)]
        var_names = [expr.var_name(i) for i in range(num_vars)]

        # Extract and rewrite the body
        body = expr.body()
        rewritten_body = rewrite_for_put_for_higher(body, b_prime, b, context)

        # Rebuild the quantifier (keep same type)
        if expr.is_forall():
            return ForAll(
                list(map(lambda n_s: Const(n_s[0], n_s[1]), zip(var_names, var_sorts))),
                rewritten_body,
            )
        else:
            return Exists(
                list(map(lambda n_s: Const(n_s[0], n_s[1]), zip(var_names, var_sorts))),
                rewritten_body,
            )

    # Case 2: Function application (And, Or, ON_star, etc.)
    elif is_app(expr):
        decl = expr.decl()

        # Match ON_star(a,b)
        if decl.kind() == Z3_OP_UNINTERPRETED and decl.name() == "Higher":
            m, n = expr.children()
            t = Const("t", context.BoxSort)
            return Or(
                And(m != b_prime, m != b, n != b_prime, n != b, context.Higher(m, n)),
                And(
                    m != b_prime,
                    m != b,
                    n == b_prime,
                    Exists(
                        [t], And(t != n, context.Higher(n, t), context.Higher(t, b))
                    ),
                ),
                And(m != b_prime, m != b, n == b, context.Higher(m, n)),
                And(m == b, n != b_prime, n != b, context.Higher(m, n)),
                And(m == b, n == b),
                And(
                    m == b_prime,
                    n != b_prime,
                    n != b,
                    Or(
                        context.Higher(b, n),
                        ForAll(
                            [t],
                            And(
                                context.Higher(n, b),
                                Implies(
                                    And(t != n, context.Higher(n, t)),
                                    context.Higher(b, t),
                                ),
                            ),
                        ),
                    ),
                ),
                And(m == b_prime, n == b),
                And(m == b_prime, n == b_prime),
            )

        # Otherwise rebuild recursively
        new_children = [
            rewrite_for_put_for_higher(c, b_prime, b, context) for c in expr.children()
        ]
        return decl(*new_children)

    # Case 3: Constants, bound variables, etc.
    else:
        return expr


def rewrite_for_put_for_scattered(expr, b_prime, b, context):
    """Weakest-precondition rewrite for Scattered(m, n) after put(b', b).

    Verbose DNF (user specification): only pairs involving b' or b change;
    b' on b redirects scattered-with-b' to scattered-with-b.
    """
    if is_quantifier(expr):
        num_vars = expr.num_vars()
        var_sorts = [expr.var_sort(i) for i in range(num_vars)]
        var_names = [expr.var_name(i) for i in range(num_vars)]
        body = expr.body()
        rewritten_body = rewrite_for_put_for_scattered(body, b_prime, b, context)
        if expr.is_forall():
            return ForAll(
                list(map(lambda n_s: Const(n_s[0], n_s[1]), zip(var_names, var_sorts))),
                rewritten_body,
            )
        return Exists(
            list(map(lambda n_s: Const(n_s[0], n_s[1]), zip(var_names, var_sorts))),
            rewritten_body,
        )

    if is_app(expr):
        decl = expr.decl()
        if decl.kind() == Z3_OP_UNINTERPRETED and decl.name() == "Scattered":
            m, n = expr.children()
            return Or(
                And(
                    m != b_prime,
                    m != b,
                    n != b_prime,
                    n != b,
                    context.Scattered(m, n),
                ),
                And(
                    m != b_prime,
                    m != b,
                    n == b,
                    context.Scattered(m, n),
                ),
                And(
                    m != b_prime,
                    m != b,
                    n == b_prime,
                    context.Scattered(m, b),
                ),
                And(
                    m == b_prime,
                    n != b_prime,
                    n != b,
                    context.Scattered(b, n),
                ),
                And(
                    m == b,
                    n != b_prime,
                    n != b,
                    context.Scattered(m, n),
                ),
            )
        new_children = [
            rewrite_for_put_for_scattered(c, b_prime, b, context)
            for c in expr.children()
        ]
        return decl(*new_children)

    return expr


def rewrite_for_mark_goal(expr, target, context):
    """Rewrite every Mark(alpha) as Or(Mark(alpha), alpha == target)."""
    if is_quantifier(expr):
        num_vars = expr.num_vars()
        var_sorts = [expr.var_sort(i) for i in range(num_vars)]
        var_names = [expr.var_name(i) for i in range(num_vars)]
        body = expr.body()
        rewritten_body = rewrite_for_mark_goal(body, target, context)
        if expr.is_forall():
            return ForAll(
                list(map(lambda n_s: Const(n_s[0], n_s[1]), zip(var_names, var_sorts))),
                rewritten_body,
            )
        return Exists(
            list(map(lambda n_s: Const(n_s[0], n_s[1]), zip(var_names, var_sorts))),
            rewritten_body,
        )

    if is_app(expr):
        decl = expr.decl()
        if decl.kind() == Z3_OP_UNINTERPRETED and decl.name() == "Mark":
            (alpha,) = expr.children()
            return Or(context.Mark(alpha), alpha == target)
        new_children = [
            rewrite_for_mark_goal(c, target, context) for c in expr.children()
        ]
        return decl(*new_children)

    return expr


Stmt = Union[Instruction, While]


class Program:
    length: int
    instructions: List[Stmt]

    def __init__(self, length: int, instructions: Union[List[Stmt], None] = None):
        self.length = length
        if instructions:
            assert len(instructions) == length
            self.instructions = deepcopy(instructions)
        else:
            self.instructions = [Skip() for _ in range(self.length)]

    def eval(self, env, return_img: bool = False):
        """evaluate the program in the environment and return the trajectories"""
        traj = [env.reset()[0]]
        if return_img:
            imgs = [env.render()]
        for line in self.instructions:
            line_imgs = line.eval(env, traj, return_img)
            if return_img:
                imgs.extend(line_imgs)
        if return_img:
            return traj, imgs
        return traj

    def register_trainable_parameter(self):
        parameters = []
        for line in self.instructions:
            line.register_trainable_parameter(parameters)
        return parameters

    def update_trainable_parameter(self, new_parameter):
        for line in self.instructions:
            line.update_trainable_parameter(new_parameter)

    def __str__(self):
        instruction_str = [f"\t{inst}" for inst in self.instructions]
        return "\n".join(["begin", *instruction_str, "end"])

    def VC_gen(self, P, Q, context):
        # P, Q are z3 formula
        seq_instruction = to_seq(self.instructions)
        return [Implies(P, self.wp(Q, context))] + VC_aux(seq_instruction, Q, context)

    def wp(self, Q, context):
        seq_instruction = to_seq(self.instructions)
        return wp(seq_instruction, Q, context)

    def highlevel_verification(
        self,
        P,
        Q,
        context: Union[highlevel_verification_lib.HighLevelContext, None] = None,
        use_tbl: bool = False,
        box_sort_mode: str = "declare",
        num_blocks: Union[int, None] = None,
        enum_names: Union[List[str], None] = None,
        num_goals: Union[int, None] = None,
        goal_enum_names: Union[List[str], None] = None,
        verification_mode: str = "box",
        visualize_enum_scene: bool = False,
        visualization_prefix: str = "highlevel_scene",
        counterexample_image_dir: Optional[Union[str, Path]] = None,
    ):
        """High-level VC checks.

        When ``verification_mode='goals'`` and ``counterexample_image_dir`` is set
        (or env ``ROBOVERIFY_HL_COUNTEREXAMPLE_DIR``), any **failed** check that
        yields a satisfying Z3 model saves a two-panel PNG under that directory
        (e.g. ``vc_0_check2.png``). Implication VC: check 1 expects ``sat``;
        check 2 expects ``unsat`` (counterexample = ``sat`` model).
        """
        solver = (
            context
            if context is not None
            else highlevel_verification_lib.HighLevelContext(
                mode=box_sort_mode,
                num_blocks=num_blocks,
                enum_names=enum_names,
                num_goals=num_goals,
                goal_enum_names=goal_enum_names,
                use_tbl=use_tbl,
                visualize_enum_scene=visualize_enum_scene,
                visualization_prefix=visualization_prefix,
                verification_mode=verification_mode,
            )
        )
        # Collect structured learned-clause provenance from While invariants (if present).
        prov_clauses = []
        for inst in self.instructions:
            if isinstance(inst, While) and getattr(inst, "invariant_provenance", None):
                prov_clauses.extend(inst.invariant_provenance)
        setattr(solver, "_learned_clause_provenance", prov_clauses)
        vcs = self.VC_gen(P, Q, solver)
        outer_md_var = _outer_while_direct_body_move_down_var(self.instructions)
        ok = True

        img_dir: Optional[Path] = None
        ce_dir = counterexample_image_dir
        if ce_dir is None and getattr(solver, "verification_mode", None) == "goals":
            env_ce = os.environ.get("ROBOVERIFY_HL_COUNTEREXAMPLE_DIR")
            if env_ce:
                ce_dir = env_ce
        if ce_dir is not None and str(ce_dir).strip() != "":
            img_dir = Path(ce_dir).expanduser()
            if not img_dir.is_absolute():
                img_dir = (Path.cwd() / img_dir).resolve()
            else:
                img_dir = img_dir.resolve()

        def save_goal_counterexample_on_failure(viz_tag: str, model) -> None:
            if img_dir is None or model is None:
                return
            if getattr(solver, "verification_mode", "box") != "goals":
                return
            if solver.GoalSort is None:
                return
            img_dir.mkdir(parents=True, exist_ok=True)
            safe = viz_tag.replace(os.sep, "_").replace("/", "_")
            out_path = img_dir / f"{safe}.png"
            from synthesis.verification_lib import goal_model_diagram

            written = goal_model_diagram.save_goal_counterexample_figure(
                solver,
                model,
                str(out_path),
                diagram_title=f"Failure model ({viz_tag})",
            )
            print(f"[counterexample diagram] saved {written}")

        print("testing axioms")
        axiom_check, axiom_model = solver.check_satisfiable(
            None, visualize_model=True, viz_tag="axioms_consistency"
        )
        if axiom_check != sat:
            ok = False
            print(
                f"[FAIL] axioms consistency check returned {axiom_check}; expected sat"
            )
            save_goal_counterexample_on_failure("axioms_consistency", axiom_model)
        print("=====================")

        print("total number of VCs:", len(vcs))
        for idx, vc in enumerate(vcs):
            print(f"verifying VC {idx}", vc)
            if is_implies(vc):
                premise = vc.arg(0)
                conclusion = vc.arg(1)
                print("check 1: axioms + premise")
                check1, model1 = solver.check_satisfiable(
                    premise,
                    visualize_model=True,
                    viz_tag=f"vc_{idx}_check1",
                )
                if check1 != sat:
                    ok = False
                    print(f"[FAIL] VC {idx} check 1 returned {check1}; expected sat")
                    save_goal_counterexample_on_failure(f"vc_{idx}_check1", model1)
                print("---------------------")
                print("check 2: axioms + premise + not(conclusion)")
                check2, model2 = solver.check_satisfiable(
                    And(premise, Not(conclusion)),
                    visualize_model=True,
                    viz_tag=f"vc_{idx}_check2",
                )
                if check2 != unsat:
                    ok = False
                    print(f"[FAIL] VC {idx} check 2 returned {check2}; expected unsat")
                    if check2 == sat and model2 is not None:
                        print_where_conclusion_fails(
                            solver,
                            model2,
                            conclusion,
                            outer_move_down_var=outer_md_var,
                        )
                    save_goal_counterexample_on_failure(f"vc_{idx}_check2", model2)
            else:
                print(
                    "non-implication VC; using check 2 style: axioms + not(VC) should be unsat"
                )
                check, model_n = solver.check_satisfiable(
                    Not(vc),
                    visualize_model=True,
                    viz_tag=f"vc_{idx}_not_vc",
                )
                if check != unsat:
                    ok = False
                    print(
                        f"[FAIL] VC {idx} non-implication check returned {check}; expected unsat"
                    )
                    save_goal_counterexample_on_failure(f"vc_{idx}_not_vc", model_n)
            print("=====================")
        return ok

    def lowlevel_verification(
        self,
        context: Union[lowlevel_verification_lib.LowLevelContext, None] = None,
        sort_name: str = "Box",
    ):
        solver = (
            context
            if context is not None
            else lowlevel_verification_lib.LowLevelContext(sort_name=sort_name)
        )
        ok = True
        found_while = False
        for idx, inst in enumerate(self.instructions):
            if isinstance(inst, While):
                found_while = True
                print(
                    f"starting low-level verification for while loop with index {idx}"
                )
                print(f"invariant: {inst.invariant}")
                print(f"body: {inst.body}")
                print(f"instantiated_cond: {inst.instantiated_cond}")
                loop_ok = solver.start_verification(
                    [*inst.invariant, inst.instantiated_cond],
                    inst.body,
                    constants=["b0", "b", "b_prime"],
                )
                if not loop_ok:
                    ok = False
                    print(f"[FAIL] low-level verification failed for while index {idx}")
        if not found_while:
            print("[WARN] low-level verification found no while loops to check")
        return ok


def to_seq(instructions):
    """convert a list of instructions to Seq connected expression (Seq is left associate)
    That is x;y;z is equivalent to ((x;y);z)
    """
    if len(instructions) == 1:
        return instructions[0]
    elif len(instructions) > 1:
        return Seq(to_seq(instructions[:-1]), instructions[-1])
    else:
        assert False, "unable to convert to seq for empty instructions"


def wp(seq_instruction, Q, context):
    """calculate weakest precondition"""

    def inv_expr(inv):
        # Invariant may be a Z3 expr or a list of ProvenancedClause-like objects.
        if isinstance(inv, list) and inv and hasattr(inv[0], "expr"):
            return And(*[c.expr for c in inv])
        return inv

    if isinstance(seq_instruction, Skip):
        return Q
    elif isinstance(seq_instruction, Seq):
        return wp(seq_instruction.s1, wp(seq_instruction.s2, Q, context), context)
    elif isinstance(seq_instruction, While):
        return inv_expr(seq_instruction.invariant)
    elif isinstance(seq_instruction, Assign):
        return substitute(
            Q,
            (
                context.get_consts(seq_instruction.left),
                context.get_consts(seq_instruction.right),
            ),
        )
    elif isinstance(seq_instruction, GoalAssign):
        return substitute(
            Q,
            (
                context.get_goal_consts(seq_instruction.left),
                context.get_goal_consts(seq_instruction.right),
            ),
        )
    elif isinstance(seq_instruction, MarkGoal):
        target = context.get_goal_consts(seq_instruction.target)
        return rewrite_for_mark_goal(Q, target, context)
    elif isinstance(seq_instruction, MoveRight):
        curr = context.get_goal_consts(seq_instruction.var_name)
        z = Const(f"z_r_{seq_instruction.var_name}", context.GoalSort)
        return And(
            curr != context.null,
            ForAll([z], Implies(context.rtot(curr, z), substitute(Q, (curr, z)))),
        )
    elif isinstance(seq_instruction, MoveDown):
        curr = context.get_goal_consts(seq_instruction.var_name)
        z = Const(f"z_d_{seq_instruction.var_name}", context.GoalSort)
        return And(
            curr != context.null,
            ForAll([z], Implies(context.dtot(curr, z), substitute(Q, (curr, z)))),
        )
    elif isinstance(seq_instruction, Put):
        placed = context.get_consts(seq_instruction.upper_block)
        if _put_base_is_tbl(seq_instruction):
            # put(upper, tbl): e.g. Put("b", "tbl") places block b on the table.
            Q = rewrite_for_put_on_tbl_for_ON_star(Q, placed, context)
            return rewrite_for_put_on_tbl_for_Higher(Q, placed, context)
        b = context.get_consts(seq_instruction.base_block)
        b_prime = placed
        Q = And(
            Not(context.ON_star(b, b_prime)),
            rewrite_for_put_for_ON_star(Q, b_prime, b, context),
        )
        Q = rewrite_for_put_for_higher(Q, b_prime, b, context)
        return rewrite_for_put_for_scattered(Q, b_prime, b, context)
    assert (
        False
    ), f"Unrecognized seq instruction {type(seq_instruction)} to calculate wp"


def _wp_move_down_on_goal_var(
    post, context: highlevel_verification_lib.HighLevelContext, var_name: str
):
    """``wp(MoveDown(var_name), post, context)`` — single-step MoveDown as in ``wp()`` above."""
    return wp(MoveDown(var_name), post, context)


def VC_aux(seq_instruction, Q, context) -> List:
    """generate auxiliary verification conditions"""

    def inv_expr(inv):
        # Invariant may be a Z3 expr or a list of ProvenancedClause-like objects.
        if isinstance(inv, list) and inv and hasattr(inv[0], "expr"):
            return And(*[c.expr for c in inv])
        return inv

    if isinstance(seq_instruction, Seq):
        return VC_aux(
            seq_instruction.s1, wp(seq_instruction.s2, Q, context), context
        ) + VC_aux(seq_instruction.s2, Q, context)
    elif isinstance(seq_instruction, While):
        return VC_aux(
            to_seq(seq_instruction.body), inv_expr(seq_instruction.invariant), context
        ) + [
            Implies(
                And(
                    seq_instruction.instantiated_cond,
                    inv_expr(seq_instruction.invariant),
                ),
                wp(
                    to_seq(seq_instruction.body),
                    inv_expr(seq_instruction.invariant),
                    context,
                ),
            ),
            Implies(
                And(Not(seq_instruction.cond), inv_expr(seq_instruction.invariant)), Q
            ),
        ]
    elif isinstance(seq_instruction, Instruction):
        return []
    assert False, "Unrecognized seq instruction for VC_aux"


def run_stack_example_with_only_ON_star():
    context = highlevel_verification_lib.HighLevelContext(mode="declare")
    inferred_invariant, candidate_lists = (
        synthesis.inference_lib.inference.run_proposal_example(context=context)
    )
    b_prime, b, n, b0, a = Consts("b_prime b n b0 a", context.BoxSort)
    instructions = [
        Assign("b", "b0"),
        While(
            instantiated_cond=And(
                ForAll([n], Or(b_prime == n, Not(context.ON_star(n, b_prime)))),
                b_prime != b,
            ),
            guard_exists_vars=[b_prime],
            body=[Put("b_prime", "b"), Assign("b", "b_prime")],
            invariant=inferred_invariant,
            # invariant=And(*candidate_lists)
        ),
    ]
    p = Program(2, instructions=instructions)

    m, n = Consts("m n", context.BoxSort)
    precondition = ForAll(
        [m],
        ForAll([n], Or(m == n, Not(context.ON_star(n, m)))),
    )

    m, n, b0 = Consts("m n b0", context.BoxSort)
    postcondition = ForAll([m], context.ON_star(m, b0))

    # print(p.VC_gen(precondition, postcondition))
    p.highlevel_verification(precondition, postcondition, context=context)


def run_unstack_example():
    pass
