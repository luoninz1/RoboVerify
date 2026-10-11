"""An interpreted Python subset for policies, without raw environment access.

This is an API boundary, not a security sandbox for arbitrary Python. No
generated Python is passed to exec/eval. Every expression and statement consumes
the same bounded instruction budget, including helper functions/comprehensions.
"""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass


class PolicyError(RuntimeError):
    pass


class PolicyBudgetExceeded(PolicyError):
    pass


class _Return(BaseException):
    def __init__(self, value):
        self.value = value


class _Break(BaseException):
    pass


class _Continue(BaseException):
    pass


@dataclass
class Scope:
    values: dict
    parent: Scope | None = None

    def get(self, name):
        if name in self.values:
            return self.values[name]
        if self.parent is not None:
            return self.parent.get(name)
        raise PolicyError(f"Undefined name: {name}")


_BIN = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_CMP = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}
_METHODS = {
    list: {"append", "extend", "pop", "index", "remove", "copy", "reverse"},
    dict: {"get", "keys", "values", "items", "copy"},
    tuple: {"index", "count"},
}


def _next(items, *default):
    iterator = iter(items)
    return next(iterator, *default)


class Runtime:
    def __init__(self, api, *, max_steps=100000, max_call_depth=32):
        self.api = dict(api)
        self.max_steps = max_steps
        self.max_call_depth = max_call_depth
        self.steps = self.depth = 0
        self.functions = {}
        self.builtins = {
            "all": all,
            "any": any,
            "len": len,
            "range": self.bounded_range,
            "enumerate": enumerate,
            "list": lambda items=(): list(self.bounded_items(items)),
            "tuple": lambda items=(): tuple(self.bounded_items(items)),
            "reversed": reversed,
            "zip": zip,
            "next": _next,
            "min": min,
            "max": max,
            "abs": abs,
            "bool": bool,
            "int": int,
            "float": float,
            "dict": dict,
            "ValueError": ValueError,
            "RuntimeError": RuntimeError,
        }

    def tick(self):
        self.steps += 1
        if self.steps > self.max_steps:
            raise PolicyBudgetExceeded(f"Policy exceeded {self.max_steps} AST steps")

    def bounded_range(self, *args):
        result = range(*args)
        if len(result) > self.max_steps:
            raise PolicyBudgetExceeded("Range exceeds the policy iteration budget")
        return result

    def bounded_items(self, items):
        for item in items:
            self.tick()
            yield item

    @staticmethod
    def parse(source):
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            raise PolicyError(f"Invalid Python: {exc}") from exc
        forbidden = (
            ast.Import,
            ast.ImportFrom,
            ast.With,
            ast.AsyncWith,
            ast.Try,
            ast.ClassDef,
            ast.Global,
            ast.Nonlocal,
            ast.Await,
            ast.Yield,
            ast.YieldFrom,
            ast.AsyncFunctionDef,
            ast.AsyncFor,
            ast.Delete,
            ast.Lambda,
        )
        for node in ast.walk(tree):
            if isinstance(node, forbidden):
                raise PolicyError(f"Unsupported policy syntax: {type(node).__name__}")
            if isinstance(node, ast.Name) and node.id.startswith("_"):
                raise PolicyError(f"Private name is unavailable: {node.id}")
            if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
                raise PolicyError("Private attributes are unavailable")
            if isinstance(node, ast.FunctionDef) and node.decorator_list:
                raise PolicyError("Decorators are unavailable")
        return tree

    def run(self, source, *, inputs=None, entry="policy"):
        self.steps = self.depth = 0
        tree = self.parse(source)
        scope = Scope({**self.builtins, **self.api, **dict(inputs or {})})
        protected = set(self.builtins) | set(self.api)
        # Prevent rebinding capabilities, including from helper scopes.
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                if node.id in protected:
                    raise PolicyError(f"Cannot replace API/builtin: {node.id}")
            if isinstance(node, ast.FunctionDef) and node.name in protected:
                raise PolicyError(f"Cannot replace API/builtin: {node.name}")
            if isinstance(node, ast.arg) and node.arg in protected:
                raise PolicyError(f"Cannot shadow API/builtin: {node.arg}")
        self.block(tree.body, scope)
        function = scope.get(entry)
        if not isinstance(function, _Function):
            raise PolicyError(f"Expected def {entry}()")
        return function()

    def bind(self, target, value, scope):
        if isinstance(target, ast.Name):
            scope.values[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            values = list(value)
            if len(values) != len(target.elts):
                raise PolicyError("Unpacking size mismatch")
            for child, item in zip(target.elts, values):
                self.bind(child, item, scope)
        elif isinstance(target, ast.Subscript):
            container = self.expr(target.value, scope)
            if type(container) not in (list, dict):
                raise PolicyError("Only local lists/dicts can be updated")
            container[self.expr(target.slice, scope)] = value
        else:
            raise PolicyError(f"Unsupported assignment: {type(target).__name__}")

    def block(self, body, scope):
        for node in body:
            self.stmt(node, scope)

    def stmt(self, node, scope):
        self.tick()
        if isinstance(node, ast.FunctionDef):
            scope.values[node.name] = _Function(self, node, scope)
        elif isinstance(node, ast.Assign):
            value = self.expr(node.value, scope)
            for target in node.targets:
                self.bind(target, value, scope)
        elif isinstance(node, ast.AnnAssign):
            if node.value is not None:
                self.bind(node.target, self.expr(node.value, scope), scope)
        elif isinstance(node, ast.AugAssign):
            value = _BIN[type(node.op)](
                self.expr(node.target, scope), self.expr(node.value, scope)
            )
            self.bind(node.target, value, scope)
        elif isinstance(node, ast.Expr):
            self.expr(node.value, scope)
        elif isinstance(node, ast.If):
            self.block(node.body if self.expr(node.test, scope) else node.orelse, scope)
        elif isinstance(node, (ast.For, ast.While)):
            exhausted = True
            iterable = (
                iter(self.expr(node.iter, scope)) if isinstance(node, ast.For) else None
            )
            while True:
                self.tick()
                if iterable is not None:
                    try:
                        self.bind(node.target, next(iterable), scope)
                    except StopIteration:
                        break
                elif not self.expr(node.test, scope):
                    break
                try:
                    self.block(node.body, scope)
                except _Continue:
                    continue
                except _Break:
                    exhausted = False
                    break
            if exhausted:
                self.block(node.orelse, scope)
        elif isinstance(node, ast.Return):
            raise _Return(None if node.value is None else self.expr(node.value, scope))
        elif isinstance(node, ast.Break):
            raise _Break()
        elif isinstance(node, ast.Continue):
            raise _Continue()
        elif isinstance(node, ast.Assert):
            if not self.expr(node.test, scope):
                raise PolicyError(
                    str(self.expr(node.msg, scope))
                    if node.msg
                    else "Policy assertion failed"
                )
        elif isinstance(node, ast.Raise):
            exception = self.expr(node.exc, scope)
            raise PolicyError(str(exception))
        elif not isinstance(node, ast.Pass):
            raise PolicyError(f"Unsupported statement: {type(node).__name__}")

    def expr(self, node, scope):
        self.tick()
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return scope.get(node.id)
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            result = [self.expr(item, scope) for item in node.elts]
            return {ast.List: list, ast.Tuple: tuple, ast.Set: set}[type(node)](result)
        if isinstance(node, ast.Dict):
            return {
                self.expr(k, scope): self.expr(v, scope)
                for k, v in zip(node.keys, node.values)
            }
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
            left, right = self.expr(node.left, scope), self.expr(node.right, scope)
            if isinstance(node.op, ast.Mult):
                for sequence, count in ((left, right), (right, left)):
                    if type(sequence) in (list, tuple, str) and isinstance(count, int):
                        if len(sequence) * count > self.max_steps:
                            raise PolicyBudgetExceeded(
                                "Sequence allocation exceeds policy budget"
                            )
            return _BIN[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp):
            value = self.expr(node.operand, scope)
            return {
                ast.Not: operator.not_,
                ast.USub: operator.neg,
                ast.UAdd: operator.pos,
            }[type(node.op)](value)
        if isinstance(node, ast.BoolOp):
            value = None
            for item in node.values:
                value = self.expr(item, scope)
                if isinstance(node.op, ast.And) and not value:
                    break
                if isinstance(node.op, ast.Or) and value:
                    break
            return value
        if isinstance(node, ast.Compare):
            left = self.expr(node.left, scope)
            for op, right_node in zip(node.ops, node.comparators):
                right = self.expr(right_node, scope)
                if not _CMP[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            return self.expr(
                node.body if self.expr(node.test, scope) else node.orelse, scope
            )
        if isinstance(node, ast.NamedExpr):
            value = self.expr(node.value, scope)
            self.bind(node.target, value, scope)
            return value
        if isinstance(node, ast.Subscript):
            return self.expr(node.value, scope)[self.expr(node.slice, scope)]
        if isinstance(node, ast.Slice):
            return slice(
                *(
                    self.expr(value, scope) if value else None
                    for value in (node.lower, node.upper, node.step)
                )
            )
        if isinstance(node, ast.Attribute):
            value = self.expr(node.value, scope)
            if node.attr not in _METHODS.get(type(value), set()):
                raise PolicyError(f"Unavailable attribute: {node.attr}")
            return getattr(value, node.attr)
        if isinstance(node, ast.Call):
            function = self.expr(node.func, scope)
            arguments = []
            for arg in node.args:
                if isinstance(arg, ast.Starred):
                    arguments.extend(self.expr(arg.value, scope))
                else:
                    arguments.append(self.expr(arg, scope))
            keywords = {}
            for kw in node.keywords:
                if kw.arg is None:
                    keywords.update(self.expr(kw.value, scope))
                else:
                    keywords[kw.arg] = self.expr(kw.value, scope)
            return function(*arguments, **keywords)
        if isinstance(
            node, (ast.ListComp, ast.GeneratorExp, ast.SetComp, ast.DictComp)
        ):

            def visit(index, local):
                if index == len(node.generators):
                    yield (
                        (self.expr(node.key, local), self.expr(node.value, local))
                        if isinstance(node, ast.DictComp)
                        else self.expr(node.elt, local)
                    )
                    return
                generator = node.generators[index]
                for value in self.expr(generator.iter, local):
                    self.tick()
                    child = Scope({}, local)
                    self.bind(generator.target, value, child)
                    if all(self.expr(test, child) for test in generator.ifs):
                        yield from visit(index + 1, child)

            iterator = visit(0, Scope({}, scope))
            if isinstance(node, ast.GeneratorExp):
                return iterator
            result = list(iterator)
            if isinstance(node, ast.DictComp):
                return dict(result)
            if isinstance(node, ast.SetComp):
                return set(result)
            return result
        raise PolicyError(f"Unsupported expression: {type(node).__name__}")


class _Function:
    def __init__(self, runtime, node, closure):
        self.runtime, self.node, self.closure = runtime, node, closure
        self.defaults = [runtime.expr(value, closure) for value in node.args.defaults]

    def __call__(self, *args, **kwargs):
        runtime, node = self.runtime, self.node
        runtime.depth += 1
        try:
            if runtime.depth > runtime.max_call_depth:
                raise PolicyBudgetExceeded("Helper call depth exhausted")
            names = [arg.arg for arg in (*node.args.posonlyargs, *node.args.args)]
            if len(args) > len(names) and node.args.vararg is None:
                raise PolicyError(f"Too many arguments to {node.name}")
            values = dict(zip(names, args))
            for name, value in zip(
                names[len(names) - len(self.defaults) :], self.defaults
            ):
                values.setdefault(name, value)
            for name, value in kwargs.items():
                if name not in names and name not in [
                    arg.arg for arg in node.args.kwonlyargs
                ]:
                    raise PolicyError(f"Unexpected argument {name}")
                if name in names[: len(args)]:
                    raise PolicyError(f"Duplicate argument {name}")
                values[name] = value
            for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults):
                if default is not None:
                    values.setdefault(arg.arg, runtime.expr(default, self.closure))
            missing = [
                name
                for name in names + [arg.arg for arg in node.args.kwonlyargs]
                if name not in values
            ]
            if missing:
                raise PolicyError(f"Missing arguments to {node.name}: {missing}")
            if node.args.vararg:
                values[node.args.vararg.arg] = args[len(names) :]
            try:
                runtime.block(node.body, Scope(values, self.closure))
            except _Return as result:
                return result.value
            return None
        finally:
            runtime.depth -= 1
