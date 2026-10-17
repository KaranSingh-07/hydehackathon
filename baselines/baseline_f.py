"""Preserved Snapshot of Reference Bot F.

Git Commit: 1bb7449e4da4ec666882d16c4c4c26c3da9955e4
Do not modify this file. Used as immutable benchmark reference.
"""
import random

from catanatron.players.value import DEFAULT_WEIGHTS, get_value_fn

VALUE_FN = get_value_fn("base_fn", DEFAULT_WEIGHTS)


class Agent:
    def __init__(self):
        self.rng = random.Random(0)

    def decide(self, obs, playable_actions) -> int:
        world = obs.sample_world(self.rng)
        best_index, best_value = 0, None
        for index, action in enumerate(playable_actions):
            successor = world.copy()
            try:
                successor.execute(action)
            except Exception:
                continue
            value = VALUE_FN(successor, obs.me)
            if best_value is None or value > best_value:
                best_index, best_value = index, value
        return best_index
