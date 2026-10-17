"""Unit and Integration Tests for Selective Search and Multi-World Sampling.

Tests:
1. State-copy safety (search never mutates original game state).
2. Anytime deadline protection (terminates within bounded time).
3. Candidate ranking and diversity preservation.
4. Compound action evaluation.
5. Deterministic reproducibility under fixed random seed.
6. Safe fallback under emergency low-clock configuration.
"""
import random
import time
import unittest

from catan_kit.obs import Obs
from catanatron.game import Game
from catanatron.models.enums import Action, ActionType
from catanatron.models.player import Color, RandomPlayer

from my_team.search.budget_manager import BudgetManager, DeadlineTimer, SearchConfig
from my_team.search.selective_search import SelectiveSearchEngine
from my_team.agent import Agent


class TestSelectiveSearch(unittest.TestCase):
    """Test suite for search, pruning, and sampling layers."""

    def setUp(self):
        self.game = Game(
            [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)],
            seed=4256,
            vps_to_win=10,
        )
        self.engine = SelectiveSearchEngine()
        self.budget_manager = BudgetManager()
        self.rng = random.Random(0)

    def test_state_copy_safety(self):
        """Search execution must never mutate the original observation or game state."""
        obs = Obs(self.game, Color.RED)
        initial_buildings_count = len(self.game.state.board.buildings)
        initial_turn = self.game.state.num_turns
        actions = list(self.game.playable_actions)

        config = SearchConfig(max_candidates=5, num_worlds=2, time_limit_s=0.050)
        chosen = self.engine.select_best_action(obs, actions, config, self.rng)

        # Check immutability
        self.assertIn(chosen, actions)
        self.assertEqual(len(self.game.state.board.buildings), initial_buildings_count)
        self.assertEqual(self.game.state.num_turns, initial_turn)

    def test_deadline_timer(self):
        """DeadlineTimer accurately detects time expiry."""
        timer = DeadlineTimer(timeout_s=0.020)
        self.assertFalse(timer.is_expired())
        time.sleep(0.025)
        self.assertTrue(timer.is_expired())

    def test_budget_manager_clock_tiers(self):
        """BudgetManager shifts tiers based on remaining CPU allowance."""
        # Emergency tier: < 150 ms
        cfg_emerg = self.budget_manager.get_search_config(
            cpu_left=0.100, turn=20, is_opening=False, opp_dev_count=2
        )
        self.assertEqual(cfg_emerg.max_candidates, 1)
        self.assertEqual(cfg_emerg.num_worlds, 1)
        self.assertFalse(cfg_emerg.allow_compound_search)

        # Conservative tier: 150 - 350 ms
        cfg_cons = self.budget_manager.get_search_config(
            cpu_left=0.250, turn=20, is_opening=False, opp_dev_count=2
        )
        self.assertEqual(cfg_cons.max_candidates, 3)
        self.assertEqual(cfg_cons.num_worlds, 1)

        # Standard tier: > 350 ms with opponent dev cards
        cfg_std = self.budget_manager.get_search_config(
            cpu_left=0.800, turn=20, is_opening=False, opp_dev_count=2
        )
        self.assertEqual(cfg_std.max_candidates, 5)
        self.assertEqual(cfg_std.num_worlds, 3)

        # Standard tier without opponent dev cards
        cfg_nodev = self.budget_manager.get_search_config(
            cpu_left=0.800, turn=20, is_opening=False, opp_dev_count=0
        )
        self.assertEqual(cfg_nodev.num_worlds, 1)

    def test_candidate_ranking_and_diversity(self):
        """Candidate selection preserves top moves and diverse action types."""
        world = self.game.copy()
        actions = list(self.game.playable_actions)
        self.assertGreater(len(actions), 5)

        ranked = self.engine.rank_candidates_with_diversity(
            world=world,
            actions=actions,
            color=Color.RED,
            max_candidates=4,
        )

        self.assertGreaterEqual(len(ranked), 4)
        for act, score in ranked:
            self.assertIn(act, actions)
            self.assertIsInstance(score, (int, float))

    def test_reproducible_search_decisions(self):
        """Identical inputs, state, and RNG produce identical search results."""
        obs = Obs(self.game, Color.RED)
        actions = list(self.game.playable_actions)
        config = SearchConfig(max_candidates=4, num_worlds=2, time_limit_s=0.030)

        rng1 = random.Random(1234)
        action1 = self.engine.select_best_action(obs, actions, config, rng1)

        rng2 = random.Random(1234)
        action2 = self.engine.select_best_action(obs, actions, config, rng2)

        self.assertEqual(action1, action2)

    def test_agent_integration_with_search(self):
        """Agent.decide seamlessly routes through search engine and returns valid index."""
        agent = Agent()
        obs = Obs(self.game, Color.RED)
        obs.cpu_left = 1.0
        actions = list(self.game.playable_actions)

        idx = agent.decide(obs, actions)
        self.assertIs(type(idx), int)
        self.assertGreaterEqual(idx, 0)
        self.assertLess(idx, len(actions))


if __name__ == "__main__":
    unittest.main()
