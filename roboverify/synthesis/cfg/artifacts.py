"""Structured CFG descriptions that keep trajectory arrays in their archives."""

from synthesis.cfg.region import BlockRegion, LoopRegion


def describe_region(region):
    if isinstance(region, BlockRegion):
        return dict(
            kind="block",
            symbolic=(
                None if region.symbolic is None else [str(i) for i in region.symbolic]
            ),
            physical=[str(i) for i in region.physical],
            bindings=sorted(region.bindings),
        )
    if isinstance(region, LoopRegion):
        return dict(
            kind="loop",
            guard=str(region.guard),
            exists_vars=list(region.exists_vars),
            init=list(region.init),
            update=list(region.update),
            invariant=None if region.invariant is None else region.invariant.sexpr(),
            iteration_counts=list(region.iteration_counts),
            iteration_limit=region.iteration_limit,
            body=(
                describe_cfg(region.body_cfg)
                if region.body_cfg is not None
                else [describe_region(r) for r in region.body]
            ),
        )
    if region is None:
        return dict(kind="unresolved")
    raise TypeError(f"Unsupported CFG region: {type(region).__name__}")


def describe_cfg(cfg):
    return dict(
        synthesis_approach=cfg.synthesis_approach,
        fixed_id_bindings=getattr(cfg, "_fixed_id_bindings", {}),
        order=list(cfg.order),
        nodes={name: describe_region(node.region) for name, node in cfg.nodes.items()},
        edges=[
            dict(
                source=e.source,
                target=e.target,
                label=str(e.label),
                binds=sorted(e.binds),
            )
            for e in cfg.edges
        ],
        segments={
            name: [
                dict(demo=s.demo_idx, start=s.t_start, end=s.t_end, bindings=s.bindings)
                for s in rows
            ]
            for name, rows in cfg.demos.segments.items()
        },
    )
