"""Re-scan complete recordings for successive flat template executions."""

import itertools
from dataclasses import dataclass

from synthesis.cfg.demos import DemoSegment
from synthesis.cfg.refine import scene_at
from synthesis.predicates.scene import Scene, evaluate


@dataclass
class Iterations:
    bodies: list
    terminal: DemoSegment


def extract_iterations(segment, predicates, rebound, updates):
    """Earliest complete template match, followed by a parallel carried update.

    The input includes the entire available suffix, not only already split nodes.
    Each returned body is a tuple of per-block segments. Rebound witnesses are
    selected from each recorded scene; demonstration object names are not inputs.
    """
    if not predicates:
        raise ValueError("A loop template must have at least one milestone")
    cursor = segment.t_start
    bindings = dict(segment.bindings)
    bodies = []
    while cursor < segment.t_end:
        found = None
        for first in range(cursor + 1, segment.t_end + 1):
            scene = scene_at(segment, first)
            for values in itertools.product(scene.positions, repeat=len(rebound)):
                local = dict(bindings, **dict(zip(rebound, values)))
                if not evaluate(predicates[0], scene, local):
                    continue
                boundaries = [cursor, first]
                for predicate in predicates[1:]:
                    next_time = next(
                        (
                            t
                            for t in range(boundaries[-1] + 1, segment.t_end + 1)
                            if evaluate(predicate, scene_at(segment, t), local)
                        ),
                        None,
                    )
                    if next_time is None:
                        break
                    boundaries.append(next_time)
                if len(boundaries) == len(predicates) + 1:
                    found = boundaries, local
                    break
            if found is not None:
                break
        if found is None:
            break
        boundaries, local = found
        bodies.append(
            tuple(
                DemoSegment(
                    segment.demo_idx,
                    a,
                    b,
                    segment.trace,
                    local,
                    segment,
                    segment.entry_index,
                )
                for a, b in zip(boundaries, boundaries[1:])
            )
        )
        bindings.update({left: local[right] for left, right in updates})
        cursor = boundaries[-1]
    terminal = DemoSegment(
        segment.demo_idx,
        cursor,
        cursor,
        segment.trace,
        bindings,
        segment,
        segment.entry_index,
    )
    return Iterations(bodies, terminal)
