from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    relations: tuple = ("ON", "ON_star", "Higher", "Scattered", "eq")
    max_depth: int = 4
    max_variables: int = 2
    max_candidates: int = 20000
    timeout_seconds: float = 5.0
