"""Validate temporal ordering over absolute demonstration indices."""


def first_true(segment, predicate, scene_at):
    return next(
        (
            t
            for t in range(segment.t_start, segment.t_end + 1)
            if predicate(scene_at(segment, t))
        ),
        None,
    )


def last_true(segment, predicate, scene_at):
    return next(
        (
            t
            for t in range(segment.t_end, segment.t_start - 1, -1)
            if predicate(scene_at(segment, t))
        ),
        None,
    )


def validate_split(cut_times, target_last_times):
    """Require first(C) <= last(Q) on each original segment being split.

    C is the proposed intermediate condition; Q is the original block's outgoing
    target, not its incoming predicate. Entry and later blocks use the same rule.
    Refinement additionally requires an interior cut, and CFG validation checks
    the incoming condition at entry and Q at the segment's end.
    """
    if not cut_times or set(cut_times) != set(target_last_times):
        return False
    return all(
        cut is not None
        and target_last_times[key] is not None
        and cut <= target_last_times[key]
        for key, cut in cut_times.items()
    )


def validate_cfg(cfg):
    """Validate the complete recorded partition, including neighboring blocks.

    An edge condition is the source block's exit and destination block's entry
    condition. Incoming and outgoing predicates need not overlap or persist.
    The first(C) <= last(Q) refinement check belongs to the original unsplit
    segment, not to comparisons between neighboring block conditions.
    Bindings and absolute boundaries are checked on both sides of each cut.
    """
    from collections import Counter

    from synthesis.cfg.refine import scene_at
    from synthesis.predicates.scene import evaluate

    cfg.validate_structure()
    for name in cfg.order:
        incoming, outgoing = cfg.incoming(name)[0], cfg.outgoing(name)[0]
        rows = cfg.demos.for_node(name)
        if not rows:
            return False
        for row in rows:
            try:
                if not evaluate(incoming.label, scene_at(row, row.t_start)):
                    return False
                if incoming.binding_condition is not None and not evaluate(
                    incoming.binding_condition, scene_at(row, row.t_start)
                ):
                    return False
                if not evaluate(outgoing.label, scene_at(row, row.t_end)):
                    return False
                if outgoing.target != cfg.exit:
                    first = first_true(
                        row, lambda state: evaluate(outgoing.label, state), scene_at
                    )
                    if first != row.t_end or row.t_end <= row.t_start:
                        return False
            except (KeyError, ValueError):
                return False
        if outgoing.target != cfg.exit:
            following = cfg.demos.for_node(outgoing.target)
            ends = Counter((id(r.trace), r.demo_idx, r.t_end) for r in rows)
            starts = Counter((id(r.trace), r.demo_idx, r.t_start) for r in following)
            if ends != starts:
                return False
    return True


Validate = validate_cfg
