import itertools
import re
from typing import Any, Callable, Dict, List, Set, Tuple

import z3

from synthesis.util.symbols import fresh_const, open_quantifier, rewrite_quantifier


def z3_to_python_expr(expr: z3.ExprRef) -> Any:
    # ---------- Quantifiers ----------
    if isinstance(expr, z3.QuantifierRef):
        if expr.is_forall():
            op = "ForAll"
        elif expr.is_exists():
            op = "Exists"
        else:
            raise ValueError(f"Unknown quantifier: {expr}")

        vars = [expr.var_name(i) for i in range(expr.num_vars())]

        return {"op": op, "vars": vars, "body": z3_to_python_expr(expr.body())}

    # ---------- Bound variable (inside quantifiers) ----------
    if z3.is_var(expr):
        # Z3 bound vars are De Bruijn indexed; we'll name them by the var_name for simplicity
        # var_name(i) works only in Quantifier body, already handled
        return {"op": "Var", "name": str(expr)}

    # ---------- Literals ----------
    if z3.is_true(expr):
        return True
    if z3.is_false(expr):
        return False
    if z3.is_int_value(expr):
        return {"op": "Const", "value": expr.as_long()}

    # ---------- Variable (uninterpreted constant) ----------
    if (
        z3.is_const(expr)
        and expr.num_args() == 0
        and expr.decl().kind() == z3.Z3_OP_UNINTERPRETED
    ):
        return {"op": "Var", "name": expr.decl().name()}

    # ---------- Function call ----------
    if (
        z3.is_app(expr)
        and expr.num_args() > 0
        and expr.decl().kind() == z3.Z3_OP_UNINTERPRETED
    ):
        return {
            "op": "Func",
            "name": expr.decl().name(),
            "args": [z3_to_python_expr(expr.arg(i)) for i in range(expr.num_args())],
        }

    # ---------- Other applications (Boolean logic / comparisons) ----------
    if not z3.is_app(expr):
        raise ValueError(f"Unsupported expression: {expr}")

    decl = expr.decl()
    kind = decl.kind()
    args = [z3_to_python_expr(a) for a in expr.children()]

    # Boolean logic
    if kind == z3.Z3_OP_AND:
        return {"op": "And", "args": args}
    if kind == z3.Z3_OP_OR:
        return {"op": "Or", "args": args}
    if kind == z3.Z3_OP_NOT:
        return {"op": "Not", "args": args}

    # Comparisons
    if kind == z3.Z3_OP_EQ:
        return {"op": "Eq", "args": args}
    if kind == z3.Z3_OP_LE:
        return {"op": "Le", "args": args}
    if kind == z3.Z3_OP_LT:
        return {"op": "Lt", "args": args}
    if kind == z3.Z3_OP_GE:
        return {"op": "Ge", "args": args}
    if kind == z3.Z3_OP_GT:
        return {"op": "Gt", "args": args}

    # Arithmetic
    if kind == z3.Z3_OP_ADD:
        return {"op": "Add", "args": args}
    if kind == z3.Z3_OP_SUB:
        return {"op": "Sub", "args": args}
    if kind == z3.Z3_OP_MUL:
        return {"op": "Mul", "args": args}
    if kind == z3.Z3_OP_DIV:
        return {"op": "Div", "args": args}

    raise ValueError(f"Unsupported Z3 operation: {decl.name()}")


# --------------------------
# Eval Quantified Expression
# --------------------------
def eval_quantified_expr(
    expr: Any,
    env: Dict[str, Any],
    domain: List,
    function_impls: Dict[str, Callable],
    bound_vars: list = None,
) -> bool:
    if bound_vars is None:
        bound_vars = []

    if expr is True or expr is False:
        return expr

    op = expr.get("op") if isinstance(expr, dict) and "op" in expr else None

    # ---------- Bound Variable ----------
    if op == "Var" and expr["name"].startswith("Var("):
        idx = int(expr["name"][4:-1])
        var_name = bound_vars[-(idx + 1)]
        return env[var_name]

    # ---------- Named Variable ----------
    if op == "Var":
        return env[expr["name"]]

    # ---------- Constant ----------
    if op == "Const":
        return expr["value"]

    # ---------- Function ----------
    if op == "Func":
        args = [
            eval_quantified_expr(a, env, domain, function_impls, bound_vars)
            for a in expr["args"]
        ]
        return function_impls[expr["name"]](*args)

    # ---------- Boolean ----------
    if op == "And":
        return all(
            eval_quantified_expr(a, env, domain, function_impls, bound_vars)
            for a in expr["args"]
        )
    if op == "Or":
        return any(
            eval_quantified_expr(a, env, domain, function_impls, bound_vars)
            for a in expr["args"]
        )
    if op == "Not":
        return not eval_quantified_expr(
            expr["args"][0], env, domain, function_impls, bound_vars
        )

    # ---------- Comparisons ----------
    if op == "Eq":
        return eval_quantified_expr(
            expr["args"][0], env, domain, function_impls, bound_vars
        ) == eval_quantified_expr(
            expr["args"][1], env, domain, function_impls, bound_vars
        )
    if op == "Gt":
        return eval_quantified_expr(
            expr["args"][0], env, domain, function_impls, bound_vars
        ) > eval_quantified_expr(
            expr["args"][1], env, domain, function_impls, bound_vars
        )
    if op == "Lt":
        return eval_quantified_expr(
            expr["args"][0], env, domain, function_impls, bound_vars
        ) < eval_quantified_expr(
            expr["args"][1], env, domain, function_impls, bound_vars
        )
    if op == "Ge":
        return eval_quantified_expr(
            expr["args"][0], env, domain, function_impls, bound_vars
        ) >= eval_quantified_expr(
            expr["args"][1], env, domain, function_impls, bound_vars
        )
    if op == "Le":
        return eval_quantified_expr(
            expr["args"][0], env, domain, function_impls, bound_vars
        ) <= eval_quantified_expr(
            expr["args"][1], env, domain, function_impls, bound_vars
        )

    # ---------- Arithmetic ----------
    if op == "Add":
        return sum(
            eval_quantified_expr(a, env, domain, function_impls, bound_vars)
            for a in expr["args"]
        )
    if op == "Sub":
        a0 = eval_quantified_expr(
            expr["args"][0], env, domain, function_impls, bound_vars
        )
        a1 = eval_quantified_expr(
            expr["args"][1], env, domain, function_impls, bound_vars
        )
        return a0 - a1
    if op == "Mul":
        res = 1
        for a in expr["args"]:
            res *= eval_quantified_expr(a, env, domain, function_impls, bound_vars)
        return res
    if op == "Div":
        a0 = eval_quantified_expr(
            expr["args"][0], env, domain, function_impls, bound_vars
        )
        a1 = eval_quantified_expr(
            expr["args"][1], env, domain, function_impls, bound_vars
        )
        return a0 // a1

    # ---------- Quantifiers ----------
    if op == "ForAll":
        x_vars = expr["vars"]
        x_domains = [domain for _ in x_vars]
        for x_vals in itertools.product(*x_domains):
            new_env = env.copy()
            for i, v in enumerate(x_vars):
                new_env[v] = x_vals[i]
            if not eval_quantified_expr(
                expr["body"], new_env, domain, function_impls, bound_vars + x_vars
            ):
                return False
        return True

    if op == "Exists":
        y_vars = expr["vars"]
        y_domains = [domain for _ in y_vars]
        for y_vals in itertools.product(*y_domains):
            new_env = env.copy()
            for i, v in enumerate(y_vars):
                new_env[v] = y_vals[i]
            if eval_quantified_expr(
                expr["body"], new_env, domain, function_impls, bound_vars + y_vars
            ):
                return True
        return False

    raise ValueError(f"Unknown op: {op}")


# --------------------------
# Witness Map
# --------------------------
def compute_witness_map(
    expr: Any,
    base_env: Dict[str, Any],
    domain: List,
    function_impls: Dict[str, Callable],
) -> Dict[Tuple, Set[Tuple]]:
    """
    Compute witness map for a quantified expression.

    Args:
        expr: Python AST of the expression (ForAll/Exists or Exists)
        base_env: dictionary of free variables with their current values
        domain: list represeting all values in the domain
        function_impls: mapping of function names to Python callables

    Returns:
        witness_map: dict mapping universal assignments (tuple) to set of witness tuples
    """
    if base_env is None:
        base_env = {}

    # Determine universal and existential variables
    if expr.get("op") == "ForAll":
        x_vars = expr["vars"]
        inner = expr["body"]
    else:
        x_vars = []
        inner = expr

    assert (
        inner.get("op") == "Exists"
    ), "compute_witness_map currently expects Exists or ForAll→Exists"
    y_vars = inner["vars"]
    body = inner["body"]

    # Prepare domain lists for bound variables
    x_domains = [domain for v in x_vars] if x_vars else [[()]]

    witness_map: Dict[Tuple, Set[Tuple]] = {}

    # Enumerate all universal assignments
    for x_vals in itertools.product(*x_domains):
        # Start from base environment (free variables)
        env = base_env.copy()
        # Assign universal variables
        for i, v in enumerate(x_vars):
            env[v] = x_vals[i]

        witnesses: Set[Tuple] = set()

        # Enumerate all possible candidates for existential variables
        for y_vals in itertools.product(*(domain for v in y_vars)):
            for i, v in enumerate(y_vars):
                env[v] = y_vals[i]

            # Evaluate the body with current env
            if eval_quantified_expr(
                body, env, domain, function_impls, bound_vars=x_vars + y_vars
            ):
                witnesses.add(tuple(y_vals))

        witness_map[x_vals] = witnesses

    return witness_map


# --------------------------
# Merge helpers
# --------------------------
def can_merge(w1: Dict[Tuple, Set[Tuple]], w2: Dict[Tuple, Set[Tuple]]) -> bool:
    # Check that both have the same set of keys
    if set(w1.keys()) != set(w2.keys()):
        return False

    # Check that for each key, there is at least one overlapping witness
    for x in w1:
        if len(w1[x].intersection(w2[x])) == 0:
            return False
    return True


def merge_bodies(e1: Any, e2: Any) -> Any:
    """Merge two ASTs after checking they can be merged"""
    if e1.get("op") == "ForAll":
        x_vars = e1["vars"]
        y_vars = e1["body"]["vars"]
        b1 = e1["body"]["body"]
        b2 = e2["body"]["body"]
        return {
            "op": "ForAll",
            "vars": x_vars,
            "body": {
                "op": "Exists",
                "vars": y_vars,
                "body": {"op": "And", "args": [b1, b2]},
            },
        }
    else:
        y_vars = e1["vars"]
        b1 = e1["body"]
        b2 = e2["body"]
        return {"op": "Exists", "vars": y_vars, "body": {"op": "And", "args": [b1, b2]}}


# --------------------------
# Merge one round
# --------------------------
def merge_once_multi_env(
    exprs: List[Any],
    base_envs: List[Dict[str, Any]],
    domain: List,
    function_impls: Dict[str, Callable],
) -> List[Any]:
    """
    Attempt one round of merging expressions, considering multiple base_envs.
    Merge only if merge succeeds for all base_envs.
    """
    merged = [False] * len(exprs)
    new_exprs = []

    i = 0
    while i < len(exprs):
        if merged[i]:
            i += 1
            continue

        e1 = exprs[i]
        merged_flag = False

        for j in range(i + 1, len(exprs)):
            if merged[j]:
                continue
            e2 = exprs[j]

            # Check merge across all base_envs
            can_merge_all_envs = True
            for base_env in base_envs:
                w1 = compute_witness_map(e1, base_env, domain, function_impls)
                w2 = compute_witness_map(e2, base_env, domain, function_impls)

                if not can_merge(w1, w2):
                    can_merge_all_envs = False
                    break

            if can_merge_all_envs:
                # Merge expressions
                merged_expr = merge_bodies(e1, e2)
                new_exprs.append(merged_expr)
                merged[i] = True
                merged[j] = True
                merged_flag = True
                break  # merge only one pair at a time

        if not merged_flag:
            new_exprs.append(e1)
            merged[i] = True

        i += 1

    return new_exprs


# --------------------------
# Fixpoint merge
# --------------------------
def fixpoint_merge_multi_env(
    exprs: List[Any],
    base_envs: List[Dict[str, Any]],
    domain: List,
    function_impls: Dict[str, Callable],
) -> List[Any]:
    """
    Repeatedly merge expressions until no more merges can happen.
    """
    prev_len = -1
    current_exprs = exprs
    while prev_len != len(current_exprs):
        prev_len = len(current_exprs)
        current_exprs = merge_once_multi_env(
            current_exprs, base_envs, domain, function_impls
        )
    return current_exprs


# Regex for De Bruijn variables like Var(0), Var(1), etc.
VAR_RE = re.compile(r"Var\((\d+)\)")


# Helper: fresh variable generators
def fresh_univ(n, BoxSort, start, avoid=()):
    vars_ = [
        fresh_const(BoxSort, f"ux{i}", avoid=avoid)
        for i in range(start + 1, start + n + 1)
    ]
    return vars_, start + n


def fresh_exist(n, BoxSort, start, avoid=()):
    vars_ = [
        fresh_const(BoxSort, f"ex{i}", avoid=avoid)
        for i in range(start + 1, start + n + 1)
    ]
    return vars_, start + n


# Main function
def python_expr_to_z3(
    expr, var_map, func_map, BoxSort, bound_depth=0, ux_counter=0, ex_counter=0
):
    """
    Convert AST to Z3 expression with De Bruijn indices and unique bound variable names.

    Returns: (z3.ExprRef, ux_counter, ex_counter)
    """
    # Base cases
    if expr is True:
        return z3.BoolVal(True), ux_counter, ex_counter
    if expr is False:
        return z3.BoolVal(False), ux_counter, ex_counter

    op = expr.get("op") if isinstance(expr, dict) else None

    # ---------- Variable ----------
    if op == "Var":
        name = expr["name"]
        m = VAR_RE.fullmatch(name)
        if m:
            idx = int(m.group(1))
            if idx >= bound_depth:
                raise ValueError(f"De Bruijn index {idx} out of scope")
            return z3.Var(idx, BoxSort), ux_counter, ex_counter
        # free variable
        return var_map[name], ux_counter, ex_counter

    # ---------- Function ----------
    if op == "Func":
        args = []
        for a in expr["args"]:
            arg_z3, ux_counter, ex_counter = python_expr_to_z3(
                a, var_map, func_map, BoxSort, bound_depth, ux_counter, ex_counter
            )
            args.append(arg_z3)
        return func_map[expr["name"]](*args), ux_counter, ex_counter

    # ---------- Boolean ----------
    if op == "And":
        args = []
        for a in expr["args"]:
            arg_z3, ux_counter, ex_counter = python_expr_to_z3(
                a, var_map, func_map, BoxSort, bound_depth, ux_counter, ex_counter
            )
            args.append(arg_z3)
        return z3.And(args), ux_counter, ex_counter

    if op == "Or":
        args = []
        for a in expr["args"]:
            arg_z3, ux_counter, ex_counter = python_expr_to_z3(
                a, var_map, func_map, BoxSort, bound_depth, ux_counter, ex_counter
            )
            args.append(arg_z3)
        return z3.Or(args), ux_counter, ex_counter

    if op == "Not":
        body_z3, ux_counter, ex_counter = python_expr_to_z3(
            expr["args"][0],
            var_map,
            func_map,
            BoxSort,
            bound_depth,
            ux_counter,
            ex_counter,
        )
        return z3.Not(body_z3), ux_counter, ex_counter

    # ---------- Comparisons ----------
    if op in ("Eq", "Gt", "Lt", "Ge", "Le"):
        a0, a1 = expr["args"]
        left, ux_counter, ex_counter = python_expr_to_z3(
            a0, var_map, func_map, BoxSort, bound_depth, ux_counter, ex_counter
        )
        right, ux_counter, ex_counter = python_expr_to_z3(
            a1, var_map, func_map, BoxSort, bound_depth, ux_counter, ex_counter
        )
        if op == "Eq":
            return left == right, ux_counter, ex_counter
        if op == "Gt":
            return left > right, ux_counter, ex_counter
        if op == "Lt":
            return left < right, ux_counter, ex_counter
        if op == "Ge":
            return left >= right, ux_counter, ex_counter
        if op == "Le":
            return left <= right, ux_counter, ex_counter

    # ---------- Arithmetic ----------
    if op == "Add":
        args = []
        for a in expr["args"]:
            arg_z3, ux_counter, ex_counter = python_expr_to_z3(
                a, var_map, func_map, BoxSort, bound_depth, ux_counter, ex_counter
            )
            args.append(arg_z3)
        return sum(args), ux_counter, ex_counter

    if op == "Sub":
        a0, a1 = expr["args"]
        left, ux_counter, ex_counter = python_expr_to_z3(
            a0, var_map, func_map, BoxSort, bound_depth, ux_counter, ex_counter
        )
        right, ux_counter, ex_counter = python_expr_to_z3(
            a1, var_map, func_map, BoxSort, bound_depth, ux_counter, ex_counter
        )
        return left - right, ux_counter, ex_counter

    if op == "Mul":
        args = expr["args"]
        res, ux_counter, ex_counter = python_expr_to_z3(
            args[0], var_map, func_map, BoxSort, bound_depth, ux_counter, ex_counter
        )
        for a in args[1:]:
            a_z3, ux_counter, ex_counter = python_expr_to_z3(
                a, var_map, func_map, BoxSort, bound_depth, ux_counter, ex_counter
            )
            res *= a_z3
        return res, ux_counter, ex_counter

    if op == "Div":
        a0, a1 = expr["args"]
        left, ux_counter, ex_counter = python_expr_to_z3(
            a0, var_map, func_map, BoxSort, bound_depth, ux_counter, ex_counter
        )
        right, ux_counter, ex_counter = python_expr_to_z3(
            a1, var_map, func_map, BoxSort, bound_depth, ux_counter, ex_counter
        )
        return left / right, ux_counter, ex_counter

    # ---------- Quantifiers ----------
    if op in ("ForAll", "Exists"):
        vars_ = expr["vars"]
        n = len(vars_)

        # generate fresh bound variables
        if op == "ForAll":
            bound_vars, ux_counter = fresh_univ(
                n, BoxSort, ux_counter, var_map.values()
            )
        else:
            bound_vars, ex_counter = fresh_exist(
                n, BoxSort, ex_counter, var_map.values()
            )

        # build body recursively
        body_z3, ux_counter, ex_counter = python_expr_to_z3(
            expr["body"],
            var_map,
            func_map,
            BoxSort,
            bound_depth + n,
            ux_counter,
            ex_counter,
        )

        if op == "ForAll":
            return z3.ForAll(bound_vars, body_z3), ux_counter, ex_counter
        else:
            return z3.Exists(bound_vars, body_z3), ux_counter, ex_counter

    raise ValueError(f"Unknown op: {op}")


#################### forall exists forall promotion code ##################
def promote_exists_to_forall_right_z3(
    expr: z3.ExprRef,
    base_envs: List[Dict[str, Any]],
    domain: List[Any],
    function_impls: Dict[str, Callable],
    ux_prefix: str = "ux",
    ex_prefix: str = "ex",
) -> z3.ExprRef:
    """
    Promote one existential variable on the right (∃Y -> ∃Y_except_j ∀Yj),
    only if it is promotable under all provided base_envs.
    Assign systematic names:
      - universals: ux1, ux2, …
      - existentials: ex1, ex2, …
    """

    # Convert Z3 formula to AST for compute_witness_map
    ast = z3_to_python_expr(expr)

    # Count number of existential variables
    num_existentials = count_existentials(ast)

    # Test each existential individually
    for candidate_index in range(num_existentials):
        promotable_in_all_states = all(
            promotable_right_single(
                compute_witness_map(
                    ast, base_env=env, domain=domain, function_impls=function_impls
                ),
                domain,
                candidate_index,
            )
            for env in base_envs
        )

        if promotable_in_all_states:
            # Promote the first candidate safe across all states
            print("promote succeed with index", candidate_index)
            print(
                rebuild_exists_forall_right(expr, candidate_index, ux_prefix, ex_prefix)
            )

    # No promotion possible
    print("promote fail")
    return expr


def count_existentials(ast: Dict) -> int:
    if ast.get("op") == "ForAll":
        inner = ast["body"]
    else:
        inner = ast
    assert inner.get("op") == "Exists"
    return len(inner["vars"])


def promotable_right_single(
    witness_map: Dict[Tuple, Set[Tuple]], domain: List[Any], existential_index: int
) -> bool:
    """
    Check if a single existential variable can be promoted to the right ∀.
    """
    domain_set = set(domain)

    for witnesses in witness_map.values():
        num_existentials = len(next(iter(witnesses))) if witnesses else 0
        buckets = {}
        for w in witnesses:
            key = tuple(w[i] for i in range(num_existentials) if i != existential_index)
            buckets.setdefault(key, set()).add(w[existential_index])

        if not any(domain_set.issubset(vals) for vals in buckets.values()):
            return False

    return True


import z3


def rebuild_exists_forall_right(
    expr, promote_index, ux_prefix="ux", ex_prefix="ex", outer_ux_offset=0
):
    """
    Promote a single existential to a rightmost universal, preserving outer ForAll,
    and avoiding name collisions.

    Args:
        expr: Z3 expression (Exists or ForAll->Exists)
        promote_index: index of existential to promote
        ux_prefix, ex_prefix, outer_ux_offset: retained for caller compatibility;
            binder identities now come from the shared fresh-symbol allocator.
    """
    if z3.is_quantifier(expr) and expr.is_forall():
        return rewrite_quantifier(
            expr,
            lambda body: rebuild_exists_forall_right(
                body,
                promote_index,
                ux_prefix,
                ex_prefix,
                outer_ux_offset=outer_ux_offset + expr.num_vars(),
            ),
        )

    if not z3.is_quantifier(expr) or not expr.is_exists():
        raise ValueError("Expected Exists or ForAll followed by Exists")
    if not 0 <= promote_index < expr.num_vars():
        raise ValueError("Existential promotion index out of range")
    variables, body = open_quantifier(expr)
    result = z3.ForAll([variables[promote_index]], body)
    remaining = [v for i, v in enumerate(variables) if i != promote_index]
    return z3.Exists(remaining, result) if remaining else result
