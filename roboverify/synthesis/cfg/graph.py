from dataclasses import dataclass, field

from synthesis.cfg.demos import DemoAssignment
from synthesis.predicates.term import boolean


@dataclass
class Node:
    name: str
    region: object = None
    available_scope: frozenset = frozenset()
    synthesis_approach: str = "relational"


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    label: object
    binds: frozenset = frozenset()
    kills: frozenset = frozenset()
    binding_condition: object = None


@dataclass
class RelationalCFG:
    nodes: dict
    edges: list
    order: list
    demos: DemoAssignment
    precondition: object = field(default_factory=lambda: boolean(True))
    postcondition: object = field(default_factory=lambda: boolean(True))
    entry: str = "entry"
    exit: str = "exit"
    initial_scope: frozenset = frozenset()
    synthesis_approach: str = "relational"

    @classmethod
    def initial(cls, demos, pre, post, scope=(), *, synthesis_approach="relational"):
        return cls(
            {"v0": Node("v0")},
            [Edge("entry", "v0", pre), Edge("v0", "exit", post)],
            ["v0"],
            DemoAssignment({"v0": list(demos)}),
            pre,
            post,
            initial_scope=frozenset(scope),
            synthesis_approach=synthesis_approach,
        )

    def incoming(self, node):
        return [e for e in self.edges if e.target == node]

    def outgoing(self, node):
        return [e for e in self.edges if e.source == node]

    def validate_structure(self):
        if len(set(self.order)) != len(self.order) or set(self.order) != set(
            self.nodes
        ):
            raise ValueError("Every CFG node must occur once in the region order")
        known = set(self.nodes) | {self.entry, self.exit}
        if any(e.source not in known or e.target not in known for e in self.edges):
            raise ValueError("Edge references an unknown node")
        expected = list(zip([self.entry] + self.order, self.order + [self.exit]))
        if [(e.source, e.target) for e in self.edges] != expected:
            raise ValueError(
                "Only single-entry chains with structured loop regions are supported"
            )
