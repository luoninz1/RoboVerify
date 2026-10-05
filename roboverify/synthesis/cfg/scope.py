"""All-paths must-reach scope; guard bindings disappear on bypass/exit edges."""


def scope(cfg):
    vertices = set(cfg.nodes) | {cfg.entry, cfg.exit}
    reachable = {cfg.entry}
    while True:
        expanded = reachable | {e.target for e in cfg.edges if e.source in reachable}
        if expanded == reachable:
            break
        reachable = expanded
    from synthesis.cfg.region import BlockRegion, LoopRegion

    exports = {
        name: (
            {left for left, _ in node.region.init}
            if isinstance(node.region, LoopRegion)
            else (
                set(node.region.bindings)
                if isinstance(node.region, BlockRegion)
                else set()
            )
        )
        for name, node in cfg.nodes.items()
    }
    universe = set(cfg.initial_scope)
    for names in exports.values():
        universe.update(names)
    for edge in cfg.edges:
        universe.update(edge.binds)
    values = {v: set(universe) if v in reachable else set() for v in vertices}
    values[cfg.entry] = set(cfg.initial_scope)
    changed = True
    while changed:
        changed = False
        for vertex in sorted(reachable - {cfg.entry}):
            paths = [
                (values[e.source] | exports.get(e.source, set()) | set(e.binds))
                - set(e.kills)
                for e in cfg.incoming(vertex)
                if e.source in reachable
            ]
            new = set.intersection(*paths) if paths else set()
            if new != values[vertex]:
                values[vertex], changed = new, True
    return {v: frozenset(names) for v, names in values.items()}
