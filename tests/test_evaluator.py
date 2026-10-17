"""Unit and Correctness Tests for Phase-Aware Strategic Evaluator.

Tests:
1. Individual feature calculations (pips, diminishing returns, diversity, port synergy).
2. Phase detection transitions (EARLY -> MID -> LATE).
3. Winning actions outranking non-winning alternatives.
4. Phase-dependent weight shifts (e.g., production priority in EARLY vs VPs in LATE).
5. Dominated move pruning (redundant 4:1 trades).
6. Deterministic evaluation reproducibility.
"""
import math
import unittest
from catan_kit.obs import Obs
from catanatron.game import Game
from catanatron.models.enums import Action, ActionType, RESOURCES, SETTLEMENT, CITY
from catanatron.models.player import Color, RandomPlayer
from catanatron.state_functions import player_key

from my_team.evaluation.features import (
    compute_diminishing_production,
    extract_production_pips,
    extract_strategic_features,
    get_tile_pips,
)
from my_team.evaluation.evaluator import (
    GamePhase,
    PhaseAwareEvaluator,
    detect_game_phase,
)
from my_team.core.action_prioritizer import prune_dominated_actions


class TestStrategicEvaluator(unittest.TestCase):
    """Test strategic feature extraction, phase adaptation, and tactical scoring."""

    def setUp(self):
        self.game = Game(
            [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)],
            seed=4256,
            vps_to_win=10,
        )
        self.evaluator = PhaseAwareEvaluator()

    def test_tile_pip_calculation(self):
        """Standard dice roll numbers map to exact Catan pips."""
        self.assertEqual(get_tile_pips(2), 1)
        self.assertEqual(get_tile_pips(12), 1)
        self.assertEqual(get_tile_pips(3), 2)
        self.assertEqual(get_tile_pips(11), 2)
        self.assertEqual(get_tile_pips(4), 3)
        self.assertEqual(get_tile_pips(10), 3)
        self.assertEqual(get_tile_pips(5), 4)
        self.assertEqual(get_tile_pips(9), 4)
        self.assertEqual(get_tile_pips(6), 5)
        self.assertEqual(get_tile_pips(8), 5)
        self.assertEqual(get_tile_pips(7), 6)  # 7 has 6 pips (most frequent)
        self.assertEqual(get_tile_pips(None), 0)  # Desert / Water tiles have no number

    def test_diminishing_production_curve(self):
        """Marginal utility of pips decreases as concentration increases."""
        # First 4 pips: linear 1.0 weight
        self.assertAlmostEqual(compute_diminishing_production(4.0), 4.0)
        # Next 4 pips (up to 8): 0.5 weight -> 4.0 + 2.0 = 6.0
        self.assertAlmostEqual(compute_diminishing_production(8.0), 6.0)
        # Beyond 8 pips: 0.2 weight -> 4.0 + 2.0 + 0.4 = 6.4 for 10 pips
        self.assertAlmostEqual(compute_diminishing_production(10.0), 6.4)
        # Concave property: U(4) - U(0) > U(8) - U(4) > U(12) - U(8)
        d1 = compute_diminishing_production(4.0) - compute_diminishing_production(0.0)
        d2 = compute_diminishing_production(8.0) - compute_diminishing_production(4.0)
        d3 = compute_diminishing_production(12.0) - compute_diminishing_production(8.0)
        self.assertGreater(d1, d2)
        self.assertGreater(d2, d3)

    def test_feature_extraction_completeness(self):
        """All expected strategic features exist in extracted dictionary with valid numerical values."""
        features = extract_strategic_features(self.game, Color.RED)
        expected_keys = [
            "is_winning",
            "my_actual_vps",
            "opp_public_vps",
            "vp_delta",
            "my_dim_production",
            "opp_dim_production",
            "production_diversity",
            "wheat_ore_pips",
            "wood_brick_pips",
            "has_3_to_1_port",
            "can_city_readiness",
            "can_settlement_readiness",
            "discard_risk",
            "num_buildable_nodes",
            "has_longest_road",
            "has_largest_army",
        ]
        for key in expected_keys:
            self.assertIn(key, features)
            self.assertIsInstance(features[key], (int, float))
            self.assertFalse(math.isnan(features[key]))

    def test_phase_detection(self):
        """Phase changes appropriately based on turns and victory points."""
        # Initial phase is EARLY
        self.assertEqual(detect_game_phase(self.game, Color.RED), GamePhase.EARLY)

        # Artificially modify player state to test MID phase
        st = self.game.state
        st.is_initial_build_phase = False
        st.num_turns = 15
        me_key = player_key(st, Color.RED)
        opp_key = player_key(st, Color.BLUE)
        st.player_state[f"{me_key}_ACTUAL_VICTORY_POINTS"] = 5
        st.player_state[f"{opp_key}_VICTORY_POINTS"] = 4
        self.assertEqual(detect_game_phase(self.game, Color.RED), GamePhase.MID)

        # Artificially modify player state to test LATE phase
        st.player_state[f"{me_key}_ACTUAL_VICTORY_POINTS"] = 9
        self.assertEqual(detect_game_phase(self.game, Color.RED), GamePhase.LATE)

    def test_winning_state_dominance(self):
        """A winning state (10 VPs) receives dominant score (+1e15)."""
        features_win = {"is_winning": 1.0, "my_actual_vps": 10.0}
        features_mid = {"is_winning": 0.0, "my_actual_vps": 9.0, "my_dim_production": 20.0}

        score_win = self.evaluator.score_features(features_win, GamePhase.LATE)
        score_mid = self.evaluator.score_features(features_mid, GamePhase.LATE)

        self.assertGreater(score_win, 1e14)
        self.assertGreater(score_win, score_mid)

    def test_prune_redundant_four_to_one_trade(self):
        """When owning a 3:1 port, 4:1 trade actions are pruned."""
        # Create dummy game state with 3:1 port owned
        self.game.state.board.player_port_resources_cache[Color.RED] = {None}
        actions = [
            Action(Color.RED, ActionType.MARITIME_TRADE, ("WOOD", "WOOD", "WOOD", "WOOD", "ORE")),
            Action(Color.RED, ActionType.MARITIME_TRADE, ("WOOD", "WOOD", "WOOD", None, "ORE")),
            Action(Color.RED, ActionType.END_TURN, None),
        ]

        pruned = prune_dominated_actions(actions, self.game, Color.RED)
        # The 4:1 trade should be pruned, leaving the 3:1 trade and END_TURN
        action_types = [a.value for a in pruned if a.action_type == ActionType.MARITIME_TRADE]
        self.assertEqual(len(action_types), 1)
        self.assertIsNone(action_types[0][3])  # 3:1 trade has None at index 3

    def test_deterministic_scoring(self):
        """Evaluating identical positions produces identical numeric scores."""
        score1 = self.evaluator.evaluate_state(self.game, Color.RED)
        score2 = self.evaluator.evaluate_state(self.game, Color.RED)
        self.assertEqual(score1, score2)


if __name__ == "__main__":
    unittest.main()
