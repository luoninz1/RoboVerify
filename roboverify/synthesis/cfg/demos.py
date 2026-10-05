"""Demonstrations retain absolute, inclusive boundaries through every split."""

from dataclasses import dataclass, field


@dataclass
class DemoTrace:
    states: tuple
    snapshots: tuple = ()
    actions: tuple = ()
    seed: int = 0
    task: str = "stack"
    num_blocks: int = 0
    events: tuple = ()
    metadata: dict = field(default_factory=dict)


@dataclass
class DemoSegment:
    demo_idx: int
    t_start: int
    t_end: int
    trace: DemoTrace
    bindings: dict = field(default_factory=dict)
    parent: object = None
    entry_index: int = None
    final_bindings: object = None

    def __post_init__(self):
        if not 0 <= self.t_start <= self.t_end < len(self.trace.states):
            raise ValueError("Segment bounds must be absolute and inside the trace")
        if self.entry_index is None:
            self.entry_index = self.t_start
        if not 0 <= self.entry_index <= self.t_start:
            raise ValueError("Entry geometry must precede the segment")
        self.bindings = dict(self.bindings)

    @property
    def states(self):
        return self.trace.states[self.t_start : self.t_end + 1]

    def split(self, absolute_index):
        if not self.t_start < absolute_index < self.t_end:
            raise ValueError("A split must leave two nonempty transition segments")
        return (
            DemoSegment(
                self.demo_idx,
                self.t_start,
                absolute_index,
                self.trace,
                self.bindings,
                self,
                self.entry_index,
            ),
            DemoSegment(
                self.demo_idx,
                absolute_index,
                self.t_end,
                self.trace,
                self.bindings,
                self,
                self.entry_index,
                self.final_bindings,
            ),
        )


@dataclass
class DemoAssignment:
    segments: dict = field(default_factory=dict)

    def for_node(self, node):
        return list(self.segments.get(node, ()))
