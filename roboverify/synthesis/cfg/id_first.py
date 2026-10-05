"""Finish ID-first synthesis with an entirely named executable and CFG.

Quotienting creates relational loop variables. IDs that survive outside that
abstraction become fixed entry aliases, never arbitrary Get(True) witnesses.
This module belongs to synthesis; inference and verification need no ID support.
"""

import itertools
from dataclasses import replace

from synthesis.cfg.bindings import require_closed
from synthesis.cfg.physical import name_operands
from synthesis.cfg.region import BlockRegion, LoopRegion
from synthesis.cfg.scope import scope
from synthesis.predicates.term import _term, atom, conjunction, free_names, negate, ref
from synthesis.util.symbols import fresh_name


def _graphs(cfg):
    yield cfg
    for node in cfg.nodes.values():
        if isinstance(node.region, LoopRegion):
            yield from _graphs(node.region.body_cfg)


def _terms(graph):
    yield graph.precondition
    yield graph.postcondition
    for edge in graph.edges:
        yield edge.label
        if edge.binding_condition is not None:
            yield edge.binding_condition
    for node in graph.nodes.values():
        if isinstance(node.region, LoopRegion):
            yield node.region.guard
            yield from node.region.postconditions


def _ids(term):
    if term.op == "id":
        yield term.value
    for child in term.args:
        yield from _ids(child)


def close_id_candidate(cfg):
    """Resolve all residual IDs before a successful synthesis result is returned.

    Entry bindings and their equality/distinctness facts preserve identity when
    the named candidate is executed and proved. No claim of generalizing these
    residual constants to arbitrarily many blocks is introduced by this pass.
    """
    graphs = list(_graphs(cfg))
    rows = getattr(cfg, "_task_demos", cfg.demos.for_node(cfg.order[0]))
    if not rows:
        raise ValueError("Naming concrete IDs requires entry demonstrations")
    ids, occupied = set(), set()
    for row in rows:
        occupied.update(row.bindings)
    for graph in graphs:
        occupied.update(graph.initial_scope)
        for term in _terms(graph):
            ids.update(_ids(term))
            occupied.update(free_names(term))
        for segments in graph.demos.segments.values():
            for segment in segments:
                occupied.update(segment.bindings)
        for node in graph.nodes.values():
            region = node.region
            if isinstance(region, LoopRegion):
                occupied.update(region.exists_vars)
                for left, right in (*region.init, *region.update):
                    occupied.update((left, right))
            elif isinstance(region, BlockRegion):
                for instruction in region.physical:
                    for operand in instruction.get_operand():
                        if operand["type"] == "Box":
                            ids.add(operand["val"])
                        elif operand["type"] == "BoxName":
                            occupied.add(operand["val"])
            else:
                raise ValueError("Cannot return an unresolved ID-first candidate")
    fixed = dict(getattr(cfg, "_fixed_id_bindings", {}))
    known = dict(fixed)
    for name in sorted(cfg.initial_scope):
        values = {row.bindings.get(name) for row in rows}
        if len(values) == 1 and None not in values:
            known[name] = values.pop()
    aliases = {}
    for name, value in known.items():
        aliases.setdefault(value, name)
    new = {}
    for object_id in sorted(ids):
        if type(object_id) is not int or object_id < 0:
            raise ValueError("Invalid physical block ID")
        if any(object_id not in _physical_ids(row) for row in rows):
            raise ValueError(
                f"Block ID {object_id} is absent from an entry demonstration"
            )
        if object_id not in aliases:
            name = fresh_name(f"block_{object_id}", occupied)
            aliases[object_id] = name
            new[name] = known[name] = object_id
    fixed.update(new)

    def named(term):
        if term.op == "id":
            return ref(aliases[term.value])
        return _term(term.op, tuple(named(a) for a in term.args), term.value)

    def bind(segment):
        segment.bindings.update(fixed)
        if segment.final_bindings is not None:
            segment.final_bindings.update(fixed)

    for graph in reversed(graphs):
        graph.initial_scope = frozenset(set(graph.initial_scope) | set(fixed))
        graph.precondition, graph.postcondition = named(graph.precondition), named(
            graph.postcondition
        )
        graph.edges = [
            replace(
                e,
                label=named(e.label),
                binding_condition=(
                    None if e.binding_condition is None else named(e.binding_condition)
                ),
            )
            for e in graph.edges
        ]
        for segments in graph.demos.segments.values():
            for segment in segments:
                bind(segment)
        for node in graph.nodes.values():
            # Synthesis has returned: any subsequent motion repair is named too.
            node.synthesis_approach = "relational"
            region = node.region
            if isinstance(region, BlockRegion):
                region.physical = tuple(
                    name_operands(i, aliases) for i in region.physical
                )
            else:
                region.guard = named(region.guard)
                region.postconditions = tuple(named(t) for t in region.postconditions)
                region.body = tuple(
                    region.body_cfg.nodes[n].region for n in region.body_cfg.order
                )
                for segment in region.exit_demos:
                    bind(segment)
    for row in rows:
        bind(row)
    if new:
        facts = []
        for (left, a), (right, b) in itertools.combinations(known.items(), 2):
            equality = atom("eq", ref(left), ref(right))
            facts.append(equality if a == b else negate(equality))
        cfg.precondition = conjunction(cfg.precondition, *facts)
        cfg.edges[0] = replace(cfg.edges[0], label=cfg.precondition)
    cfg._fixed_id_bindings = fixed
    for graph in graphs:
        scopes = scope(graph)
        for name, node in graph.nodes.items():
            node.available_scope = scopes[name]
            if isinstance(node.region, BlockRegion):
                node.region.bindings = (
                    require_closed(node.region.physical, scopes[name]) - scopes[name]
                )
    return cfg


def _physical_ids(segment):
    from synthesis.predicates.scene import Scene

    state = segment.trace.states[segment.t_start]
    return (
        set(state.positions)
        if isinstance(state, Scene)
        else set(range(segment.trace.num_blocks))
    )
