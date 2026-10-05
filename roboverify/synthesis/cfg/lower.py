"""Fold stored regions into the existing execution/verification Program IR."""

from copy import deepcopy

import z3

from synthesis.api.instructions import Assign, Get, While
from synthesis.api.program import Program
from synthesis.cfg.region import BlockRegion, LoopRegion
from synthesis.predicates.term import to_z3


def lower_region(region, context, *, physical=False):
    if isinstance(region, BlockRegion):
        if not physical and region.symbolic is None:
            raise ValueError("Physical candidate has no established relational summary")
        if physical and not region.physical:
            raise ValueError("Region has no physical implementation")
        instructions = deepcopy(list(region.physical if physical else region.symbolic))
        for i, instruction in enumerate(instructions):
            if isinstance(instruction, Get) and instruction.guard_term is not None:
                names = [str(v) for v in instruction.guard_exists_vars]
                instructions[i] = Get(
                    names[0],
                    to_z3(instruction.guard_term, context),
                    [context.get_consts(n) for n in names],
                    guard_term=instruction.guard_term,
                )
        return instructions
    if not isinstance(region, LoopRegion):
        raise ValueError("Cannot lower an unresolved region")
    if region.invariant is None and not physical:
        raise ValueError("Loop lowering requires an explicit inferred invariant")
    body = (
        list(lower(region.body_cfg, context, physical=physical).instructions)
        if region.body_cfg is not None
        else [
            instruction
            for child in region.body
            for instruction in lower_region(child, context, physical=physical)
        ]
    )
    body.extend(Assign(a, b) for a, b in region.update)
    guard = to_z3(region.guard, context)
    loop = While(
        guard,
        [context.get_consts(v) for v in region.exists_vars],
        body,
        region.invariant,
        max_iters=region.max_iters,
    )
    loop.guard_term = region.guard
    return [*(Assign(a, b) for a, b in region.init), loop]


def lower(cfg, context, *, physical=False):
    cfg.validate_structure()
    instructions = []
    for key in cfg.order:
        edge = cfg.incoming(key)[0]
        if edge.binds:
            if edge.binding_condition is None:
                raise ValueError("Bound edge requires a Get condition")
            names = sorted(edge.binds)
            instructions.append(
                Get(
                    names[0],
                    to_z3(edge.binding_condition, context),
                    [context.get_consts(n) for n in names],
                    guard_term=edge.binding_condition,
                )
            )
        instructions.extend(
            lower_region(cfg.nodes[key].region, context, physical=physical)
        )
    return Program(len(instructions), instructions)


def lower_with_locations(cfg, context, *, physical=True):
    """Map actual physical or symbolic instruction paths to their CFG regions."""
    program = lower(cfg, context, physical=physical)
    locations = {"loops": {}, "blocks": {}}

    def locate(graph, instructions, prefix="", graph_prefix=""):
        cursor = 0
        for name in graph.order:
            region = graph.nodes[name].region
            if graph.incoming(name)[0].binds:
                cursor += 1
            path = graph_prefix + name
            if isinstance(region, LoopRegion):
                cursor += len(region.init)
                loop_path = prefix + str(cursor)
                instruction = instructions[cursor]
                if not isinstance(instruction, While):
                    raise ValueError("Loop location differs from lowered executable")
                locations["loops"][loop_path] = (path, region)
                locate(region.body_cfg, instruction.body, loop_path + ".", path + "/")
                cursor += 1
            else:
                count = len(region.physical if physical else region.symbolic)
                paths = tuple(prefix + str(i) for i in range(cursor, cursor + count))
                locations["blocks"][path] = (graph, name, paths)
                cursor += count

    locate(cfg, program.instructions)
    return program, locations
