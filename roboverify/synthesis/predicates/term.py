"""Immutable, interned formulas; named binders canonicalize alpha-equivalence."""

from dataclasses import dataclass
from functools import lru_cache

from synthesis.util.symbols import fresh_const, fresh_name


@dataclass(frozen=True)
class Term:
    op: str
    args: tuple = ()
    value: object = None

    def __deepcopy__(self, memo):
        return self

    @property
    def depth(self):
        return 1 + max((arg.depth for arg in self.args), default=0)

    @property
    def quantifier_count(self):
        return int(self.op in ("exists", "forall")) + sum(
            a.quantifier_count for a in self.args
        )

    def __str__(self):
        if self.op in ("ref", "id"):
            return str(self.value)
        if self.op == "bool":
            return str(self.value)
        if self.op in ("exists", "forall"):
            return f"{self.op} {','.join(self.value)}.({self.args[0]})"
        return f"{self.op}({', '.join(map(str, self.args))})"


@lru_cache(maxsize=100000)
def _term(op, args=(), value=None):
    return Term(op, args, value)


def ref(name):
    if not isinstance(name, str) or not name or name.startswith("@"):
        raise ValueError("Reference names must be nonempty and cannot start with @")
    return _term("ref", value=name)


def block_id(value):
    """A concrete physical ID, never a free variable or existential witness."""
    if type(value) is not int or value < 0:
        raise ValueError("Block IDs must be nonnegative integers")
    return _term("id", value=value)


def boolean(value):
    if not isinstance(value, bool):
        raise TypeError("Boolean literal required")
    return _term("bool", value=value)


def atom(name, left, right):
    if name not in {"ON_star", "ON_star_zero", "Higher", "Scattered", "ON", "eq"}:
        raise ValueError(f"Unknown predicate: {name}")
    if left.op not in ("ref", "id") or right.op not in ("ref", "id"):
        raise TypeError("Predicate arguments must be object references")
    return _term(name, (left, right))


def negate(term):
    return _term("not", (term,))


def conjunction(*terms):
    return _term("and", tuple(terms)) if terms else boolean(True)


def disjunction(*terms):
    return _term("or", tuple(terms)) if terms else boolean(False)


def implies(left, right):
    return _term("implies", (left, right))


def canonical(term, env=None, depth=0):
    env = {} if env is None else env
    if term.op == "ref":
        return _term("ref", value=env.get(term.value, term.value))
    if term.op in ("forall", "exists"):
        names = tuple(f"@{depth+i}" for i in range(len(term.value)))
        nested = dict(env, **dict(zip(term.value, names)))
        body = canonical(term.args[0], nested, depth + len(names))
        return _term(term.op, (body,), names)
    return _term(
        term.op, tuple(canonical(a, env, depth) for a in term.args), term.value
    )


def quantify(kind, names, body):
    names = tuple(names)
    if kind not in ("exists", "forall") or not names or len(set(names)) != len(names):
        raise ValueError("Quantifiers require a kind and distinct binders")
    for name in names:
        ref(name)
    return canonical(_term(kind, (body,), names))


def exists(names, body):
    return quantify("exists", names, body)


def forall(names, body):
    return quantify("forall", names, body)


def free_names(term, bound=frozenset()):
    if term.op == "ref":
        return frozenset() if term.value in bound else frozenset([term.value])
    if term.op in ("forall", "exists"):
        bound = bound | frozenset(term.value)
    return frozenset().union(*(free_names(a, bound) for a in term.args))


def substitute(term, mapping, bound=frozenset()):
    """Capture-free substitution of free object references, not formula holes."""

    def walk(node, hidden):
        if node.op == "ref":
            return node if node.value in hidden else mapping.get(node.value, node)
        if node.op in ("forall", "exists"):
            hidden = hidden | frozenset(node.value)
        return _term(node.op, tuple(walk(a, hidden) for a in node.args), node.value)

    return canonical(walk(term, bound))


def to_z3(term, context, bindings=None):
    import z3

    bindings = {} if bindings is None else bindings
    if term.op == "ref":
        return (
            bindings[term.value]
            if term.value in bindings
            else context.get_consts(term.value)
        )
    if term.op == "id":
        raise ValueError("Concrete IDs must be named before symbolic lowering")
    if term.op == "bool":
        return z3.BoolVal(term.value)
    if term.op in ("exists", "forall"):
        variables = [
            fresh_const(
                context.BoxSort, name, avoid=(*bindings.values(), *free_names(term))
            )
            for name in term.value
        ]
        body = to_z3(
            term.args[0], context, dict(bindings, **dict(zip(term.value, variables)))
        )
        return (z3.Exists if term.op == "exists" else z3.ForAll)(variables, body)
    args = [to_z3(a, context, bindings) for a in term.args]
    operations = {
        "and": z3.And,
        "or": z3.Or,
        "not": z3.Not,
        "implies": z3.Implies,
        "eq": lambda a, b: a == b,
    }
    if term.op in operations:
        return operations[term.op](*args)
    if term.op == "ON":
        a, b = args
        mid = fresh_const(context.BoxSort, "direct_middle", avoid=(a, b))
        return z3.And(
            a != b,
            context.ON_star(a, b),
            z3.ForAll(
                [mid],
                z3.Implies(
                    z3.And(context.ON_star(a, mid), context.ON_star(mid, b)),
                    z3.Or(mid == a, mid == b),
                ),
            ),
        )
    return getattr(context, term.op)(*args)


def open_existentials(term, prefix, occupied=()):
    """Open a classifier prefix into fresh program-level Get binders."""
    names, mapping = [], {}
    occupied = set(occupied) | set(free_names(term))
    while term.op == "exists":
        for old in term.value:
            name = fresh_name(f"{prefix}_{len(names)}", occupied)
            names.append(name)
            mapping[old] = ref(name)
        term = term.args[0]
    return tuple(names), substitute(term, mapping)


def from_z3(expr):
    """Convert the executable relational guard subset, preserving binder scope."""
    import z3
    from z3.z3util import get_vars

    from synthesis.util.symbols import fresh_name

    occupied = {str(v) for v in get_vars(expr)}

    def convert(node, bound=()):
        if z3.is_var(node):
            return ref(bound[z3.get_var_index(node)])
        if z3.is_true(node):
            return boolean(True)
        if z3.is_false(node):
            return boolean(False)
        if z3.is_quantifier(node):
            names = tuple(
                fresh_name("guard_var", occupied) for _ in range(node.num_vars())
            )
            body = convert(node.body(), tuple(reversed(names)) + bound)
            return (forall if node.is_forall() else exists)(names, body)
        if z3.is_const(node) and node.sort().kind() != z3.Z3_BOOL_SORT:
            return ref(str(node))
        args = tuple(convert(a, bound) for a in node.children())
        if z3.is_and(node):
            return conjunction(*args)
        if z3.is_or(node):
            return disjunction(*args)
        if z3.is_not(node):
            return negate(args[0])
        if z3.is_implies(node):
            return implies(*args)
        if z3.is_eq(node):
            if node.arg(0).sort().kind() == z3.Z3_BOOL_SORT:
                return conjunction(implies(*args), implies(*reversed(args)))
            return atom("eq", *args)
        if z3.is_distinct(node):
            import itertools

            return conjunction(
                *(negate(atom("eq", a, b)) for a, b in itertools.combinations(args, 2))
            )
        if str(node.decl().name()) in {
            "ON",
            "ON_star",
            "ON_star_zero",
            "Higher",
            "Scattered",
        }:
            return atom(str(node.decl().name()), *args)
        raise ValueError(f"Unsupported DSL guard: {node}")

    return convert(expr)
