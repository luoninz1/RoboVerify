"""Adapt a supplied primitive Program to the same CFG used by synthesis."""

from copy import deepcopy

from synthesis.api.instructions import Assign, Get, PickByName, ReleaseByName, While
from synthesis.cfg.bindings import require_closed
from synthesis.cfg.demos import DemoAssignment
from synthesis.cfg.graph import Edge, Node, RelationalCFG
from synthesis.cfg.program_source import guard_term
from synthesis.cfg.region import BlockRegion, LoopRegion
from synthesis.predicates.term import atom, boolean, conjunction, negate, ref


def program_to_cfg(definition, segments, pre, post):
    def graph(rows, available, entry, exit):
        nodes, order = {}, []
        pending = []
        names = set(available)
        held = False

        def flush():
            nonlocal pending, names
            if not pending:
                return
            exported = require_closed(pending, names) - names
            name = f"v{len(order)}"
            nodes[name] = Node(name, BlockRegion(None, tuple(pending), exported))
            order.append(name)
            names.update(exported)
            pending = []

        for original in rows:
            i = deepcopy(original)
            if isinstance(i, While):
                if held:
                    raise ValueError("A loop cannot split an unfinished placement")
                flush()
                guard_names = tuple(map(str, i.guard_exists_vars))
                body = graph(
                    i.body, names | set(guard_names), boolean(True), boolean(True)
                )
                name = f"v{len(order)}"
                region = LoopRegion(
                    guard_term(i),
                    guard_names,
                    tuple(body.nodes[n].region for n in body.order),
                    invariant=None,
                    body_cfg=body,
                    iteration_limit=i.max_iters,
                )
                nodes[name] = Node(name, region)
                order.append(name)
            else:
                if isinstance(i, (Assign, Get)):
                    if held:
                        raise ValueError(
                            "Binding inside an unfinished placement is unsupported"
                        )
                    flush()
                if isinstance(i, PickByName):
                    if held:
                        raise ValueError("Pick while holding is unsupported")
                    held = True
                pending.append(i)
                if isinstance(i, ReleaseByName):
                    held = False
                    flush()
                elif isinstance(i, (Assign, Get)):
                    flush()
        if held:
            raise ValueError("Unfinished placement at program or loop exit")
        flush()
        if not order:
            raise ValueError("Empty program or loop body is unsupported")
        pairs = list(zip(["entry"] + order, order + ["exit"]))
        edges = [
            Edge(
                a,
                b,
                entry if k == 0 else exit if k == len(pairs) - 1 else boolean(True),
            )
            for k, (a, b) in enumerate(pairs)
        ]
        return RelationalCFG(
            nodes,
            edges,
            order,
            DemoAssignment({n: [] for n in order}),
            entry,
            exit,
            initial_scope=frozenset(available),
        )

    # Fixed numeric aliases denote the same/distinct physical IDs at entry.
    import itertools

    alias_facts = []
    for (left, left_id), (right, right_id) in itertools.combinations(
        definition.initial_bindings.items(), 2
    ):
        equality = atom("eq", ref(left), ref(right))
        alias_facts.append(equality if left_id == right_id else negate(equality))
    pre = conjunction(pre, *alias_facts) if alias_facts else pre
    cfg = graph(definition.program.instructions, definition.initial_bindings, pre, post)
    cfg._task_demos = list(segments)
    cfg._initial_bindings = dict(definition.initial_bindings)
    return cfg
