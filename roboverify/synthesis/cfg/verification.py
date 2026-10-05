"""Verify the same structured CFG and physical primitives produced by synthesis.

Abstract summaries are candidates until motion checking establishes their full
WP effects. A predicate observed in a demo alone never establishes a summary.
Unsupported summary shapes remain explicit failures.
"""

from copy import deepcopy

import z3

from synthesis.api.instructions import (
    Assign,
    Get,
    MoveByName,
    PickByName,
    Put,
    ReleaseByName,
    Skip,
)
from synthesis.cfg.bindings import require_closed
from synthesis.cfg.lower import lower, lower_region
from synthesis.cfg.region import BlockRegion, LoopRegion
from synthesis.cfg.scope import scope
from synthesis.predicates.term import to_z3
from synthesis.verification_lib.lowlevel_verification_lib import LowLevelContext
from synthesis.verification_lib.motion_verification import (
    MotionCheck,
    MotionContract,
    MotionVerificationResult,
    assume_input_alignment,
    check_abstract_effects,
    check_alignment,
    check_contract_realization,
    check_frame_preservation,
    prepare_alignment,
)
from synthesis.verification_lib.primitive_motion import PrimitiveMotionProblem
from synthesis.verification_lib.root_selection import RootContext


def propose_summaries(cfg, context):
    """Attach a Put only to a structurally identified completed placement.

    Pure binding/arm operations have no relational object effect. Transfers may
    span adjacent blocks, but no relational observation/control boundary may
    intervene while their abstract Put is pending.
    """
    from synthesis.cfg.scope import scope

    held = None
    last_move = None
    for name in cfg.order:
        node, edge = cfg.nodes[name], cfg.incoming(name)[0]
        region = node.region
        if edge.binds and held is not None:
            raise ValueError("Get observes a partially completed abstract placement")
        if isinstance(region, LoopRegion):
            if held is not None:
                raise ValueError("A loop cannot split an abstract placement")
            if region.body_cfg is None:
                raise ValueError("Verification requires the recovered loop body CFG")
            propose_summaries(region.body_cfg, context)
            continue
        if not isinstance(region, BlockRegion):
            raise ValueError("Cannot verify unresolved blocks")
        available = scope(cfg)[name]
        region.bindings = require_closed(region.physical, available) - available
        summary = []
        post = cfg.outgoing(name)[0].label
        for i in region.physical:
            if isinstance(i, Get):
                if held is not None:
                    raise ValueError("Get inside an incomplete placement")
                summary.append(deepcopy(i))
            elif isinstance(i, Assign):
                if held is not None:
                    raise ValueError(
                        "Assignment inside an incomplete placement requires an explicit summary"
                    )
                summary.append(deepcopy(i))
            elif isinstance(i, PickByName):
                if held is not None:
                    raise ValueError("Pick while holding")
                held = i.grab_box_name
                last_move = None
            elif isinstance(i, ReleaseByName):
                if held is None or held != i.release_box_name:
                    raise ValueError("Release must refer to the currently held object")
                if post.op in ("ON", "ON_star") and post.args[0].value == held:
                    target = post.args[1].value
                elif last_move is not None:
                    # This is only a proposed abstraction. The motion stage must
                    # prove all its effects even when the edge is a global goal.
                    target = last_move.target_box_name_z
                    if target == held and context.use_tbl:
                        target = "tbl"
                else:
                    raise ValueError(
                        "Placement needs an explicit ON edge or a transport target"
                    )
                summary.append(Put(held, target))
                held = None
            elif isinstance(i, MoveByName):
                if held is not None:
                    last_move = i
            elif not isinstance(i, Skip):
                # Explicit established IR is allowed only if its physical verifier
                # supports it; the primitive pipeline refuses unknown instructions.
                raise ValueError(f"Unsupported synthesis primitive {type(i).__name__}")
        if region.symbolic is None:
            region.symbolic = tuple(summary or [Skip(0)])
        else:
            # Compare semantically relevant instruction text, not object repr.
            actual = tuple(str(i) for i in region.symbolic if not isinstance(i, Skip))
            proposed = tuple(str(i) for i in summary if not isinstance(i, Skip))
            if actual != proposed:
                raise ValueError(
                    "Declared summary differs from the structural placement candidate"
                )
    if held is not None:
        raise ValueError("Unfinished placement at CFG exit")
    return cfg


def _all_names(cfg):
    names = set(cfg.initial_scope)
    for edge in cfg.edges:
        names.update(edge.binds)
    for node in cfg.nodes.values():
        region = node.region
        if isinstance(region, LoopRegion):
            names.update(region.exists_vars)
            for a, b in (*region.init, *region.update):
                names.update((a, b))
            names.update(_all_names(region.body_cfg))
        elif isinstance(region, BlockRegion):
            for inst in region.physical:
                if isinstance(inst, Get):
                    names.update(str(v) for v in inst.guard_exists_vars)
                elif isinstance(inst, Assign):
                    names.update((inst.left, inst.right))
                else:
                    names.update(
                        o["val"] for o in inst.get_operand() if o["type"] == "BoxName"
                    )
    return names


def verify_cfg_motion(
    cfg,
    context,
    *,
    noise=None,
    timeout_ms=5000,
    initial_positions=None,
    initial_arm=None,
    table_surface_height=None,
    supported_tower_model=False,
    entry_conditions=None,
):
    """Thread geometry/arm/held state through blocks; fresh states at loop boundaries.

    Each loop body is checked from invariant + witnessed guard, and continuation
    starts from a fresh invariant + guard-false state. A failed/unknown premise or
    unsupported primitive is never a successful proof.
    """
    from synthesis.verification_lib.tower_geometry import (
        arm_clearance,
        assume_tower_geometry,
        column_alignment,
        tower_height_invariant,
    )

    constants = sorted(_all_names(cfg))
    low = LowLevelContext(default_L=0.05, use_tbl=context.use_tbl)
    checks = []
    count = 0
    physical_names = [n for n in constants if n != "tbl"]
    if not physical_names:
        return MotionVerificationResult(
            [MotionCheck("coverage", "unsupported", reason="No object bindings")], noise
        )
    # Placeholder source/target for the state machine; no reference root is chosen.
    dummy = MotionContract(
        physical_names[0], physical_names[0], table_surface_height=table_surface_height
    )

    def fresh(
        conditions, path, positions=None, arm=None, available=(), clear_arm=False
    ):
        p = PrimitiveMotionProblem(
            low,
            conditions,
            constants,
            dummy,
            noise,
            path,
            timeout_ms,
            initial_positions=positions,
            initial_arm=arm,
            enforce_source=False,
        )
        if supported_tower_model:
            assume_tower_geometry(p, table_surface_height)
            if clear_arm:
                p.solver.add(arm_clearance(p), column_alignment(p))
        p.root_context = RootContext(context, z3.And(*conditions), timeout_ms)
        assume_input_alignment(p, p.root_context.initial_roots(available))
        p.check("initial_consistency")
        return p

    def append_new(p, offset, path):
        checks.extend(
            MotionCheck(
                path + "/" + c.obligation,
                c.status,
                c.elapsed_seconds,
                c.counterexample,
                c.reason,
            )
            for c in p.checks[offset:]
        )

    def walk(graph, p, prefix="", *, postcondition=None, tail=()):
        nonlocal count
        pending = None
        scopes = scope(graph)
        for index, name in enumerate(graph.order):
            path = prefix + name
            edge, region = graph.incoming(name)[0], graph.nodes[name].region
            p.block_v = path
            offset = len(p.checks)
            available = set(scopes[name])
            if edge.binds:
                names = sorted(edge.binds)
                binding = Get(
                    names[0],
                    to_z3(edge.binding_condition, context),
                    [context.get_consts(n) for n in names],
                    guard_term=edge.binding_condition,
                )
                p.execute([binding])
                p.root_context.history.append(binding)
            if isinstance(region, LoopRegion):
                if p.held_name is not None:
                    checks.append(
                        MotionCheck(
                            path, "unsupported", reason="Held object at loop boundary"
                        )
                    )
                    return p
                initializers = [Assign(a, b) for a, b in region.init]
                p.execute(initializers)
                p.root_context.history.extend(initializers)
                if supported_tower_model:
                    p.check(
                        "arm_clearance_entry", z3.Not(arm_clearance(p, fields=p.fields))
                    )
                    p.check(
                        "column_alignment_entry",
                        z3.Not(column_alignment(p, fields=p.fields)),
                    )
                    p.check(
                        "tower_height_entry",
                        z3.Not(tower_height_invariant(p, fields=p.fields)),
                    )
                append_new(p, offset, path)
                guard = to_z3(region.guard, context)
                inv = region.invariant
                inv = (
                    z3.And(*[c.expr if hasattr(c, "expr") else c for c in inv])
                    if isinstance(inv, list)
                    else inv
                )
                child = fresh(
                    [inv, guard],
                    path,
                    available=region.body_cfg.initial_scope,
                    clear_arm=supported_tower_model,
                )
                append_new(child, 0, path)
                child = walk(
                    region.body_cfg,
                    child,
                    path + "/",
                    postcondition=inv,
                    tail=tuple(Assign(a, b) for a, b in region.update),
                )
                if supported_tower_model:
                    finish_offset = len(child.checks)
                    child.check(
                        "arm_clearance_preserved",
                        z3.Not(arm_clearance(child, fields=child.fields)),
                    )
                    child.check(
                        "column_alignment_preserved",
                        z3.Not(column_alignment(child, fields=child.fields)),
                    )
                    child.check(
                        "tower_height_preserved",
                        z3.Not(tower_height_invariant(child, fields=child.fields)),
                    )
                    append_new(child, finish_offset, path)
                exists_vars = [context.get_consts(n) for n in region.exists_vars]
                no_guard = (
                    z3.Not(z3.Exists(exists_vars, guard))
                    if exists_vars
                    else z3.Not(guard)
                )
                p = fresh(
                    [inv, no_guard],
                    path + "/exit",
                    available=available | {a for a, _ in region.init},
                    clear_arm=supported_tower_model,
                )
                append_new(p, 0, path + "/exit")
                continue
            count += 1
            remaining = list(region.symbolic or ())
            for following in graph.order[index + 1 :]:
                following_edge = graph.incoming(following)[0]
                if following_edge.binds:
                    bound = sorted(following_edge.binds)
                    remaining.append(
                        Get(
                            bound[0],
                            to_z3(following_edge.binding_condition, context),
                            [context.get_consts(n) for n in bound],
                        )
                    )
                remaining.extend(lower_region(graph.nodes[following].region, context))
            remaining.extend(tail)
            for instruction in region.physical:
                if isinstance(instruction, PickByName):
                    proposed = next((i for i in remaining if isinstance(i, Put)), None)
                    if proposed is None:
                        raise ValueError("Pick has no completed symbolic placement")
                    p.contract = MotionContract(
                        p.bindings[proposed.upper_block],
                        p.bindings[proposed.base_block],
                        table_surface_height=table_surface_height,
                    )
                    prepare_alignment(
                        p,
                        p.root_context,
                        available,
                        target=proposed.base_block,
                        continuation=remaining,
                        postcondition=postcondition,
                    )
                    pending = (
                        dict(p.current),
                        p.fields,
                        p.contract,
                        p.alignment_root,
                        proposed,
                    )
                p.execute([instruction])
                if isinstance(instruction, (Get, Assign)):
                    p.root_context.history.append(deepcopy(instruction))
                    if isinstance(instruction, Get):
                        available.update(map(str, instruction.guard_exists_vars))
                    else:
                        available.add(instruction.left)
                    if remaining and isinstance(remaining[0], type(instruction)):
                        remaining.pop(0)
                elif isinstance(instruction, ReleaseByName) and pending is not None:
                    p.root_context.history.append(pending[4])
            puts = [i for i in region.symbolic or () if isinstance(i, Put)]
            if puts:
                if len(puts) != 1 or pending is None:
                    checks.append(
                        MotionCheck(
                            path,
                            "unsupported",
                            reason="Expected one completed placement",
                        )
                    )
                    return p
                old_initial = p.initial
                p.initial, p.effect_entry_fields, p.contract, p.alignment_root, _ = (
                    pending
                )
                check_contract_realization(p)
                check_frame_preservation(p)
                check_abstract_effects(p)
                check_alignment(p)
                p.initial = old_initial
                pending = None
            p.check("transition_consistency")
            append_new(p, offset, path)
        if p.held_name is not None:
            checks.append(
                MotionCheck(
                    prefix + "exit",
                    "unsupported",
                    reason="Unreleased object at CFG exit",
                )
            )
        if tail:
            p.execute(tail)
            p.root_context.history.extend(tail)
        return p

    try:
        p = fresh(
            (
                entry_conditions
                if entry_conditions is not None
                else [to_z3(cfg.precondition, context)]
            ),
            "entry",
            initial_positions,
            initial_arm,
            available=cfg.initial_scope,
        )
        append_new(p, 0, "entry")
        walk(cfg, p, postcondition=to_z3(cfg.postcondition, context))
    except (ValueError, KeyError, TypeError, NotImplementedError) as exc:
        checks.append(MotionCheck("coverage", "unsupported", reason=str(exc)))
    return MotionVerificationResult(checks, noise, count)


def symbolic_problem(cfg, original_context, size=None):
    """Rebuild formulas in each finite/unbounded context without changing the CFG."""
    from uuid import uuid4

    from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext

    context = HighLevelContext(
        mode="declare" if size is None else "enum",
        num_blocks=size,
        use_tbl=original_context.use_tbl,
        sort_name="Box" if size is None else "CFGBox_" + uuid4().hex,
    )
    clone = deepcopy(cfg)
    names = sorted(_all_names(cfg))

    def rewrite(graph):
        for node in graph.nodes.values():
            region = node.region
            if isinstance(region, LoopRegion):
                invariant = region.invariant
                if isinstance(invariant, list):
                    invariant = z3.And(
                        *(c.expr if hasattr(c, "expr") else c for c in invariant)
                    )
                region.invariant = context.spec_to_expr(
                    original_context.expr_to_spec(invariant), known_const_names=names
                )
                rewrite(region.body_cfg)

    rewrite(clone)
    return (
        lower(clone, context),
        to_z3(cfg.precondition, context),
        to_z3(cfg.postcondition, context),
        context,
    )


def verify_cfg_symbolic(cfg, context, *, min_blocks=2, max_blocks=4, timeout_ms=5000):
    from synthesis.verification_lib.symbolic_verify import symbolic_verify

    return symbolic_verify(
        lambda size: symbolic_problem(cfg, context, size),
        min_blocks=min_blocks,
        max_blocks=max_blocks,
        timeout_ms=timeout_ms,
        constants=tuple(sorted(_all_names(cfg))),
    )
