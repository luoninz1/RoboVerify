"""Ground ON compatibility feature for existing BMC and goal reward callers."""

from dataclasses import dataclass

from synthesis.util import on


@dataclass(frozen=True)
class GroundON:
    b1: int
    b2: int

    def __call__(self, obs):
        return on.on(on.get_block_pos(obs, self.b1), on.get_block_pos(obs, self.b2))

    def reward(self, obs):
        return on.on_reward(
            on.get_block_pos(obs, self.b1), on.get_block_pos(obs, self.b2)
        )

    def __str__(self):
        return f"ON({self.b1}, {self.b2})"
