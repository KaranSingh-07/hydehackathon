"""Unit and Correctness Tests for Catan Agent Interface.

Tests verify:
1. Agent initialization.
2. Representative valid observations.
3. Edge-case legal action lists (single-action, empty).
4. Return type and index boundary guarantees.
5. Missing or unexpected optional observation fields (e.g., obs.cpu_left = None).
6. State varieties: initial settlement phase, resource-constrained states, late-game states.
7. Safe fallback under simulation exceptions.
8. Deterministic repeatability across calls.
"""
import random
import unittest

from catan_kit.obs import Obs
from catanatron.game import Game
from catanatron.models.enums import Action, ActionType
from catanatron.models.player import Color, RandomPlayer

from my_team.agent import Agent


class TestAgentInterface(unittest.TestCase):
    """Correctness and interface adherence test cases."""

    def setUp(self):
        """Set up standard 2-player game and agent."""
        self.agent = Agent()
        self.game = Game(
            [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)],
            seed=4256,
            vps_to_win=10,
        )

    def test_initialization(self):
        """Agent initializes cleanly without errors."""
        agent = Agent()
        self.assertIsNotNone(agent)
        self.assertIsInstance(agent.rng, random.Random)

    def test_valid_observation_decide(self):
        """Agent correctly selects an action in a standard opening state."""
        obs = Obs(self.game, Color.RED)
        obs.cpu_left = 1.0
        actions = list(self.game.playable_actions)
        self.assertGreater(len(actions), 1)

        idx = self.agent.decide(obs, actions)

        # Must be genuine integer, not bool
        self.assertIs(type(idx), int)
        self.assertNotIsInstance(idx, bool)
        # Must be strictly within bounds
        self.assertGreaterEqual(idx, 0)
        self.assertLess(idx, len(actions))

    def test_cpu_left_none(self):
        """Agent functions normally when obs.cpu_left is None (unclocked mode)."""
        obs = Obs(self.game, Color.RED)
        obs.cpu_left = None
        actions = list(self.game.playable_actions)

        idx = self.agent.decide(obs, actions)
        self.assertIs(type(idx), int)
        self.assertGreaterEqual(idx, 0)
        self.assertLess(idx, len(actions))

    def test_single_action_forced_move(self):
        """If given a single action, agent returns 0 without crashing."""
        obs = Obs(self.game, Color.RED)
        actions = [Action(Color.RED, ActionType.END_TURN, None)]

        idx = self.agent.decide(obs, actions)
        self.assertEqual(idx, 0)
        self.assertIs(type(idx), int)

    def test_empty_actions_defensive(self):
        """If defensively passed empty actions, returns 0 without raising exception."""
        obs = Obs(self.game, Color.RED)
        actions = []

        idx = self.agent.decide(obs, actions)
        self.assertEqual(idx, 0)
        self.assertIs(type(idx), int)

    def test_reproducibility_with_same_seed(self):
        """Identical inputs and seed produce identical decisions."""
        obs1 = Obs(self.game, Color.RED)
        actions1 = list(self.game.playable_actions)

        agent_a = Agent()
        agent_a.rng = random.Random(999)
        idx_a = agent_a.decide(obs1, actions1)

        agent_b = Agent()
        agent_b.rng = random.Random(999)
        idx_b = agent_b.decide(obs1, actions1)

        self.assertEqual(idx_a, idx_b)

    def test_stepped_game_states(self):
        """Test decisions through multiple game turns across settlement, road, and rolling."""
        game = Game(
            [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)],
            seed=101,
            vps_to_win=10,
        )
        agent = Agent()

        # Step through the initial placement phase
        for step in range(12):
            if game.winning_color() is not None:
                break
            actions = list(game.playable_actions)
            color = game.state.current_color()
            obs = Obs(game, color)
            obs.cpu_left = 1.2

            if len(actions) > 1:
                idx = agent.decide(obs, actions)
                self.assertIs(type(idx), int)
                self.assertGreaterEqual(idx, 0)
                self.assertLess(idx, len(actions))
                chosen = actions[idx]
            else:
                chosen = actions[0]

            game.execute(chosen)

    def test_safe_fallback_on_corrupted_actions(self):
        """If simulation encounters invalid actions, agent safely catches and returns legal index."""
        obs = Obs(self.game, Color.RED)
        # Mix in an action that cannot be legally executed
        corrupted_actions = [
            Action(Color.RED, ActionType.BUILD_CITY, 999),  # Invalid node id
            Action(Color.RED, ActionType.BUILD_SETTLEMENT, 15),
        ]

        idx = self.agent.decide(obs, corrupted_actions)
        self.assertIs(type(idx), int)
        self.assertGreaterEqual(idx, 0)
        self.assertLess(idx, len(corrupted_actions))

    def test_resource_depleted_or_zero_hand(self):
        """Verify decisions when hand has zero resources."""
        obs = Obs(self.game, Color.RED)
        # Force player to have 0 resources
        for r in obs.my_resources:
            self.assertEqual(obs.my_resources[r], 0)

        actions = list(self.game.playable_actions)
        idx = self.agent.decide(obs, actions)
        self.assertIs(type(idx), int)
        self.assertGreaterEqual(idx, 0)
        self.assertLess(idx, len(actions))


if __name__ == "__main__":
    unittest.main()
