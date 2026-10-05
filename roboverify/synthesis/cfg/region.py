from dataclasses import dataclass, field


@dataclass
class BlockRegion:
    symbolic: tuple
    physical: tuple = ()
    bindings: frozenset = frozenset()


@dataclass
class LoopRegion:
    guard: object
    exists_vars: tuple
    body: tuple
    init: tuple = ()
    update: tuple = ()
    invariant: object = None
    iteration_counts: tuple = ()
    postconditions: tuple = ()
    body_demos: tuple = ()
    iteration_limit: object = None
    body_cfg: object = None
    exit_demos: tuple = ()

    @property
    def max_iters(self):
        # Observation counts are evidence, not executable semantics. A caller
        # may impose a resource budget, whose exhaustion raises explicitly.
        return self.iteration_limit
