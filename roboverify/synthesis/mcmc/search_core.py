"""Shared annealing, candidate pool, and imitation/penalty objective decisions."""

import math
from copy import deepcopy
from dataclasses import dataclass


def acceptance_probability(delta, temperature=1.0):
    if not math.isfinite(temperature) or temperature <= 0 or math.isnan(delta):
        raise ValueError("Temperature must be finite and positive; delta cannot be NaN")
    return 1.0 if delta >= 0 else math.exp(delta / temperature)


def temperature_at(iteration, initial=1.0, decay=0.99, floor=0.01):
    if initial <= 0 or not 0 < decay <= 1 or floor <= 0:
        raise ValueError("Invalid annealing schedule")
    return max(floor, initial * decay**iteration)


def imitation_objective(distance, failures=0.0, weight=0.0):
    return -float(distance) - float(weight) * float(failures)


@dataclass
class PoolEntry:
    program: object
    distance: float


class CandidatePool:
    def __init__(self, epsilon=0.05, limit=10):
        if epsilon < 0 or limit < 1:
            raise ValueError("Invalid pool bounds")
        self.epsilon, self.limit, self.entries = epsilon, limit, []
        self.best = math.inf

    def add(self, program, distance):
        if not math.isfinite(distance):
            return
        self.best = min(self.best, distance)
        items = self.entries + [PoolEntry(deepcopy(program), float(distance))]
        items = [entry for entry in items if entry.distance <= self.best + self.epsilon]
        # Stable repr is sufficient for candidate de-duplication, not proof equality.
        unique = {}
        for entry in items:
            key = str(entry.program)
            if key not in unique or entry.distance < unique[key].distance:
                unique[key] = entry
        self.entries = sorted(unique.values(), key=lambda entry: entry.distance)[
            : self.limit
        ]

    def select(self, post_score):
        if not self.entries:
            raise ValueError("Cannot rank an empty candidate pool")
        scored = [
            (post_score(entry.program), -entry.distance, entry)
            for entry in self.entries
        ]
        score, _, entry = max(scored, key=lambda value: value[:2])
        return deepcopy(entry.program), float(score)
