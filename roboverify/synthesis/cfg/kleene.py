"""Flat words of relational letters; starred regions are explicitly deferred."""

from dataclasses import dataclass

from synthesis.predicates.term import (
    Term,
    _term,
    canonical,
    free_names,
    ref,
    substitute,
)
from synthesis.util.symbols import fresh_name


@dataclass(frozen=True)
class Letter:
    predicate: Term
    get_bound: frozenset = frozenset()
    starred: bool = False


def encode_fragment(labels, get_bound=()):
    return tuple(Letter(label, frozenset(get_bound)) for label in labels)


def alpha_equal(left, right):
    return canonical(left) == canonical(right)


@dataclass
class Template:
    word: tuple
    first: dict
    second: dict


def anti_unify(first, second):
    if len(first) != len(second):
        return None
    memo, left, right = {}, {}, {}
    occupied = set().union(
        *(free_names(w.predicate) | w.get_bound for w in (*first, *second))
    )

    def match(a, b, bound=frozenset()):
        if a == b:
            return a
        if (
            a.op in ("ref", "id")
            and b.op in ("ref", "id")
            and (a.op == "id" or a.value not in bound)
            and (b.op == "id" or b.value not in bound)
        ):
            pair = (a, b)
            if pair not in memo:
                name = fresh_name(f"p{len(memo)}", occupied)
                memo[pair] = ref(name)
                left[name] = a
                right[name] = b
            return memo[pair]
        if a.op != b.op or a.value != b.value or len(a.args) != len(b.args):
            raise ValueError("Different relational shapes")
        if a.op in ("forall", "exists"):
            bound = bound | frozenset(a.value)
        return _term(
            a.op, tuple(match(x, y, bound) for x, y in zip(a.args, b.args)), a.value
        )

    try:
        word = []
        for a, b in zip(first, second):
            if a.starred or b.starred:
                return None
            predicate = match(canonical(a.predicate), canonical(b.predicate))
            left_names = {v.value: k for k, v in left.items()}
            right_names = {v.value: k for k, v in right.items()}
            left_bound = frozenset(left_names.get(n, n) for n in a.get_bound)
            right_bound = frozenset(right_names.get(n, n) for n in b.get_bound)
            if left_bound != right_bound:
                return None
            word.append(Letter(predicate, left_bound))
        return Template(tuple(word), left, right)
    except ValueError:
        return None


def match_template(template, word):
    result = {}
    variables = set(template.first)

    def match(pattern, actual):
        if pattern.op == "ref" and pattern.value in variables:
            if actual.op not in ("ref", "id"):
                return False
            existing = result.setdefault(pattern.value, actual)
            return existing == actual
        return (
            pattern.op == actual.op
            and pattern.value == actual.value
            and len(pattern.args) == len(actual.args)
            and all(match(a, b) for a, b in zip(pattern.args, actual.args))
        )

    if len(template.word) != len(word) or any(w.starred for w in word):
        return None
    if all(match(p.predicate, w.predicate) for p, w in zip(template.word, word)):
        if all(
            frozenset(result[n].value if n in result else n for n in p.get_bound)
            == w.get_bound
            for p, w in zip(template.word, word)
        ):
            return result
    return None


def carried_bindings(template):
    """Partial inverse composition: undefined mappings must be guard-rebound."""
    inverse = {}
    for name, value in template.first.items():
        if value in inverse:
            raise ValueError("Non-injective first substitution")
        inverse[value] = name
    carried, rebound = {}, []
    for name, value in template.second.items():
        if value in inverse:
            carried[name] = inverse[value]
        else:
            rebound.append(name)
    return carried, tuple(rebound)
