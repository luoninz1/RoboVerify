"""One observation adapter and numeric interpretation for predicate search."""

import itertools
from dataclasses import dataclass, field

from synthesis.util import on


@dataclass
class Scene:
    positions: dict
    bindings: dict = field(default_factory=dict)
    entry_positions: dict = None
    observation: object = field(default=None, compare=False, repr=False)

    def __post_init__(self):
        if self.observation is not None:
            self.observation = self.observation.copy()
        self.positions = {
            key: on.TABLE if on.is_table(value) else tuple(value)
            for key, value in self.positions.items()
        }
        self.bindings = dict(self.bindings)
        if any(value not in self.positions for value in self.bindings.values()):
            raise ValueError("Scene contains an unresolved binding")
        self.entry_positions = (
            dict(self.positions)
            if self.entry_positions is None
            else dict(self.entry_positions)
        )
        if set(self.entry_positions) != set(self.positions):
            raise ValueError(
                "Current and frozen geometry must describe the same universe"
            )

    def __deepcopy__(self, memo):
        # Preserve identity of the relational table marker.
        return Scene(
            self.positions, self.bindings, self.entry_positions, self.observation
        )


def scene_from_obs(
    obs, num_blocks, bindings=None, *, include_table=False, entry_obs=None
):
    positions = {i: on.get_block_pos(obs, i).copy() for i in range(num_blocks)}
    initial = (
        {i: on.get_block_pos(entry_obs, i).copy() for i in range(num_blocks)}
        if entry_obs is not None
        else dict(positions)
    )
    aliases = dict(bindings or {})
    if include_table:
        positions["tbl"] = initial["tbl"] = on.TABLE
        aliases["tbl"] = "tbl"
    return Scene(positions, aliases, initial, obs)


def evaluate(term, scene, bindings=None):
    env = dict(scene.bindings)
    env.update(bindings or {})

    def ev(node, values):
        if node.op == "ref":
            return values[node.value]
        if node.op == "id":
            if node.value not in scene.positions:
                raise ValueError(f"Unknown physical block ID: {node.value}")
            return node.value
        if node.op == "bool":
            return node.value
        if node.op in ("exists", "forall"):
            outcomes = (
                ev(node.args[0], dict(values, **dict(zip(node.value, combo))))
                for combo in itertools.product(scene.positions, repeat=len(node.value))
            )
            return any(outcomes) if node.op == "exists" else all(outcomes)
        if node.op == "not":
            return not ev(node.args[0], values)
        if node.op == "and":
            return all(ev(a, values) for a in node.args)
        if node.op == "or":
            return any(ev(a, values) for a in node.args)
        if node.op == "implies":
            return not ev(node.args[0], values) or ev(node.args[1], values)
        a, b = (ev(arg, values) for arg in node.args)
        if node.op == "eq":
            return a == b
        positions = (
            scene.entry_positions if node.op == "ON_star_zero" else scene.positions
        )
        if node.op == "ON":
            # Relational direct-on, distinct from the tolerant motion placement band.
            return (
                a != b
                and on.on_star_implementation(positions[a], positions[b])
                and all(
                    c in (a, b)
                    or not (
                        on.on_star_implementation(positions[a], pos)
                        and on.on_star_implementation(pos, positions[b])
                    )
                    for c, pos in positions.items()
                )
            )
        predicate = {
            "ON_star": on.on_star_implementation,
            "ON_star_zero": on.on_star_implementation,
            "Higher": on.higher_implementation,
            "Scattered": on.scattered_implementation,
        }[node.op]
        return bool(predicate(positions[a], positions[b]))

    return bool(ev(term, env))
