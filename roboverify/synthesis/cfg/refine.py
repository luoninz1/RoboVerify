"""Classifier-guided CFG splitting and absolute demo partitioning in lockstep."""

import itertools
from copy import copy, deepcopy

from synthesis.cfg.graph import Edge, Node
from synthesis.cfg.validate import first_true, last_true, validate_cfg, validate_split
from synthesis.predicates.classifier import learn_classifier, learn_ground_classifier
from synthesis.predicates.scene import Scene, evaluate, scene_from_obs
from synthesis.predicates.term import open_existentials


def scene_at(segment, t):
    state = segment.trace.states[t]
    bindings = dict(segment.bindings)
    if t == segment.t_end and segment.final_bindings is not None:
        bindings.update(segment.final_bindings)
    if isinstance(state, Scene):
        return Scene(
            state.positions,
            dict(state.bindings, **bindings),
            segment.trace.states[segment.entry_index].entry_positions,
        )
    return scene_from_obs(
        state,
        segment.trace.num_blocks,
        bindings,
        include_table="tbl" in bindings,
        entry_obs=segment.trace.states[segment.entry_index],
    )


def transition_witnesses(segments, post):
    result = []
    for segment in segments:
        for t in range(segment.t_start, segment.t_end):
            if not evaluate(post, scene_at(segment, t)) and evaluate(
                post, scene_at(segment, t + 1)
            ):
                result.append(scene_at(segment, t))
    return result


def refine_cfg(cfg, node, negative, available_scope, *, language=None, on_split=None):
    """Atomic refinement: validation failure leaves both CFG and D_V unchanged."""
    cfg.validate_structure()
    incoming, outgoing = cfg.incoming(node), cfg.outgoing(node)
    if len(incoming) != 1 or len(outgoing) != 1:
        raise ValueError("Refinement requires a single-entry/exit block")
    segments = cfg.demos.for_node(node)
    positive = transition_witnesses(segments, outgoing[0].label)
    # Prefix execution also contains states before the Get that introduces this
    # block's aliases. Those states are not in this classifier's scoped domain.
    negative = [
        scene for scene in negative if set(available_scope) <= set(scene.bindings)
    ]
    classifier = (
        learn_ground_classifier
        if cfg.synthesis_approach == "id-first"
        else learn_classifier
    )
    # Fixed aliases introduced at a previous synthesis boundary are encoding
    # details, not new object variables for ID-first refinement.
    classifier_scope = set(available_scope) - set(
        getattr(cfg, "_fixed_id_bindings", {})
    )
    result = classifier(positive, negative, classifier_scope, language=language)
    if not result:
        return result
    predicate = result.term
    binders, condition = open_existentials(
        predicate, "g_" + node.replace(".", "_"), available_scope
    )
    cut_times, target_last_times, splits = {}, {}, []
    for index, segment in enumerate(segments):
        cut_times[index] = first_true(
            segment, lambda s: evaluate(predicate, s), scene_at
        )
        # Compare the new cut C with the original block target Q on the original
        # segment. The incoming predicate P may be destroyed before C is reached.
        target_last_times[index] = last_true(
            segment, lambda s: evaluate(outgoing[0].label, s), scene_at
        )
        cut = cut_times[index]
        if cut is None or not segment.t_start < cut < segment.t_end:
            result.status = "validate_reject"
            return result
        before, after = segment.split(cut)
        if binders:
            scene = scene_at(segment, cut)
            witness = next(
                (
                    dict(zip(binders, values))
                    for values in itertools.product(
                        scene.positions, repeat=len(binders)
                    )
                    if evaluate(condition, scene, dict(zip(binders, values)))
                ),
                None,
            )
            if witness is None:
                raise ValueError("Classifier witness disappeared during splitting")
            after.bindings.update(witness)
        splits.append((before, after))
    if not validate_split(cut_times, target_last_times):
        result.status = "validate_reject"
        return result
    left, right = node + ".0", node + ".1"
    if left in cfg.nodes or right in cfg.nodes:
        raise ValueError("Refinement node collision")
    original = cfg
    cfg = copy(original)
    cfg.nodes, cfg.edges, cfg.order = (
        dict(original.nodes),
        list(original.edges),
        list(original.order),
    )
    cfg.demos = copy(original.demos)
    cfg.demos.segments = dict(original.demos.segments)
    position = cfg.order.index(node)
    cfg.order[position : position + 1] = [left, right]
    del cfg.nodes[node]
    cfg.nodes[left], cfg.nodes[right] = Node(left), Node(right)
    edge_index = cfg.edges.index(incoming[0])
    cfg.edges[edge_index : edge_index + 2] = [
        Edge(
            incoming[0].source,
            left,
            incoming[0].label,
            incoming[0].binds,
            incoming[0].kills,
            incoming[0].binding_condition,
        ),
        Edge(
            left,
            right,
            predicate,
            frozenset(binders),
            binding_condition=condition if binders else None,
        ),
        Edge(
            right,
            outgoing[0].target,
            outgoing[0].label,
            outgoing[0].binds,
            outgoing[0].kills,
            outgoing[0].binding_condition,
        ),
    ]
    del cfg.demos.segments[node]
    cfg.demos.segments[left] = [a for a, b in splits]
    cfg.demos.segments[right] = [b for a, b in splits]
    if not validate_cfg(cfg):
        result.status = "validate_reject"
        return result
    original.__dict__.update(cfg.__dict__)
    if on_split is not None:
        on_split(node, left, right, splits)
    return result


RefineCFG = refine_cfg
