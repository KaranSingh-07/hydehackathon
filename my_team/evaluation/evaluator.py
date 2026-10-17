"""Phase-Aware Strategic Evaluator for Catan.

Combines extracted strategic features with phase-dependent weightings:
  1. EARLY Phase: Focus on high-diversity production, settlement spots, expansion flexibility.
  2. MID Phase: Focus on city upgrades (wheat/ore), settlement completion, port synergy, dev cards.
  3. LATE Phase: Dominant focus on victory points, longest road/largest army locks, and opponent denial.

Provides tactical overrides for winning moves and dangerous over-holding.
"""
from enum import Enum
from typing import Dict, List, Optional, Tuple

from catan_kit.obs import Obs
from catanatron.game import Game
from catanatron.models.enums import Action, ActionType
try:
    from my_team.evaluation.features import extract_strategic_features
except ImportError:
    from evaluation.features import extract_strategic_features



class GamePhase(Enum):
    EARLY = "EARLY"
    MID = "MID"
    LATE = "LATE"


# Phase-specific feature weights
# Normalized to human-interpretable scales
EARLY_WEIGHTS: Dict[str, float] = {
    "my_actual_vps": 200.0,
    "vp_delta": 50.0,
    "my_dim_production": 45.0,
    "opp_dim_production": -30.0,
    "production_diversity": 60.0,
    "wheat_ore_pips": 15.0,
    "wood_brick_pips": 30.0,        # Crucial for early road/settlement expansion
    "has_3_to_1_port": 15.0,
    "has_special_port": 10.0,
    "port_synergy": 25.0,
    "can_city_readiness": 10.0,
    "can_settlement_readiness": 35.0,
    "can_dev_readiness": 5.0,
    "discard_risk": -25.0,
    "num_buildable_nodes": 15.0,
    "blocked_penalty": -80.0,
    "my_road_length": 5.0,
    "has_longest_road": 50.0,
    "road_lead": 3.0,
    "my_knights_played": 5.0,
    "has_largest_army": 50.0,
    "army_lead": 3.0,
    "unplayed_devs": 10.0,
    "robber_on_enemy": 15.0,
    "robber_on_self": -20.0,
}

MID_WEIGHTS: Dict[str, float] = {
    "my_actual_vps": 500.0,
    "vp_delta": 150.0,
    "my_dim_production": 35.0,
    "opp_dim_production": -35.0,
    "production_diversity": 40.0,
    "wheat_ore_pips": 30.0,        # Wheat & Ore drive Cities & Dev cards
    "wood_brick_pips": 15.0,
    "has_3_to_1_port": 25.0,
    "has_special_port": 20.0,
    "port_synergy": 40.0,
    "can_city_readiness": 45.0,
    "can_settlement_readiness": 35.0,
    "can_dev_readiness": 20.0,
    "discard_risk": -35.0,
    "num_buildable_nodes": 10.0,
    "blocked_penalty": -100.0,
    "my_road_length": 10.0,
    "has_longest_road": 120.0,
    "road_lead": 8.0,
    "my_knights_played": 15.0,
    "has_largest_army": 120.0,
    "army_lead": 10.0,
    "unplayed_devs": 25.0,
    "robber_on_enemy": 25.0,
    "robber_on_self": -30.0,
}

LATE_WEIGHTS: Dict[str, float] = {
    "my_actual_vps": 2000.0,       # Victory points dominate late game
    "vp_delta": 500.0,
    "my_dim_production": 10.0,
    "opp_dim_production": -25.0,
    "production_diversity": 10.0,
    "wheat_ore_pips": 15.0,
    "wood_brick_pips": 10.0,
    "has_3_to_1_port": 15.0,
    "has_special_port": 15.0,
    "port_synergy": 20.0,
    "can_city_readiness": 40.0,
    "can_settlement_readiness": 30.0,
    "can_dev_readiness": 25.0,
    "discard_risk": -40.0,
    "num_buildable_nodes": 5.0,
    "blocked_penalty": -150.0,
    "my_road_length": 15.0,
    "has_longest_road": 300.0,      # 2 VP swing is massive
    "road_lead": 15.0,
    "my_knights_played": 20.0,
    "has_largest_army": 300.0,      # 2 VP swing is massive
    "army_lead": 15.0,
    "unplayed_devs": 35.0,
    "robber_on_enemy": 50.0,        # Choke opponent's winning production
    "robber_on_self": -50.0,
}


from catanatron.state_functions import player_key


def detect_game_phase(game: Game, color) -> GamePhase:
    """Determine strategic game phase from board state and victory points."""
    st = game.state
    # If in initial placement phase or turn is very young
    if st.is_initial_build_phase or st.num_turns < 8:
        return GamePhase.EARLY

    me_key = player_key(st, color)
    opp_colors = [c for c in st.colors if c != color]
    opp_key = player_key(st, opp_colors[0]) if opp_colors else None

    me_vps = float(st.player_state.get(f"{me_key}_ACTUAL_VICTORY_POINTS", 0))
    opp_vps = float(st.player_state.get(f"{opp_key}_VICTORY_POINTS", 0)) if opp_key else 0.0

    # Late game if either player is within striking distance of 10 VPs
    if me_vps >= 8.0 or opp_vps >= 8.0:
        return GamePhase.LATE
    if me_vps <= 3.0 and opp_vps <= 3.0:
        return GamePhase.EARLY

    return GamePhase.MID


import json
import os
from pathlib import Path


def get_default_weights() -> Dict[GamePhase, Dict[str, float]]:
    """Return a deep copy of the hardcoded baseline weights."""
    return {
        GamePhase.EARLY: dict(EARLY_WEIGHTS),
        GamePhase.MID: dict(MID_WEIGHTS),
        GamePhase.LATE: dict(LATE_WEIGHTS),
    }


def load_weights_from_file(filepath: str) -> Dict[GamePhase, Dict[str, float]]:
    """Load weights dictionary from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    weights = {}
    phase_map = {
        "EARLY": GamePhase.EARLY,
        "MID": GamePhase.MID,
        "LATE": GamePhase.LATE,
    }
    for phase_str, p_enum in phase_map.items():
        if phase_str in data:
            weights[p_enum] = {k: float(v) for k, v in data[phase_str].items()}
        else:
            weights[p_enum] = dict(get_default_weights()[p_enum])
    return weights


def save_weights_to_file(weights: Dict[GamePhase, Dict[str, float]], filepath: str):
    """Serialize weights dictionary to a JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    serializable = {
        phase.value: {k: float(v) for k, v in w.items()}
        for phase, w in weights.items()
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


class PhaseAwareEvaluator:
    """Evaluates Catan positions with phase-adaptive strategic feature weighting."""

    def __init__(
        self,
        early_weights: Optional[Dict[str, float]] = None,
        mid_weights: Optional[Dict[str, float]] = None,
        late_weights: Optional[Dict[str, float]] = None,
        weights_file: Optional[str] = None,
    ):
        # 1. Direct explicit weight dicts if passed
        if early_weights is not None or mid_weights is not None or late_weights is not None:
            self.weights = {
                GamePhase.EARLY: early_weights or dict(EARLY_WEIGHTS),
                GamePhase.MID: mid_weights or dict(MID_WEIGHTS),
                GamePhase.LATE: late_weights or dict(LATE_WEIGHTS),
            }
            return

        # 2. Check explicit weights_file argument or environment variable
        target_file = weights_file or os.environ.get("CATAN_WEIGHTS_FILE")
        if not target_file:
            # Check default location in package directory
            pkg_weights = Path(__file__).resolve().parent / "weights.json"
            if pkg_weights.is_file():
                target_file = str(pkg_weights)

        if target_file and os.path.isfile(target_file):
            try:
                self.weights = load_weights_from_file(target_file)
                return
            except Exception:
                pass

        # 3. Default fallback to hardcoded weights
        self.weights = get_default_weights()

    def score_features(self, features: Dict[str, float], phase: GamePhase) -> float:
        """Compute weighted linear evaluation with non-linear tactical overrides."""
        # Dominant Tactical Override 1: Immediate Victory (10+ VPs)
        if features.get("is_winning", 0.0) == 1.0:
            return 1e15

        w = self.weights[phase]
        score = 0.0
        for feat_name, val in features.items():
            if feat_name in w:
                score += val * w[feat_name]

        return score

    def evaluate_state(self, game: Game, color) -> float:
        """Evaluate a game state for `color`."""
        phase = detect_game_phase(game, color)
        features = extract_strategic_features(game, color)
        return self.score_features(features, phase)

    def evaluate_candidate_action(
        self, world: Game, action: Action, color
    ) -> float:
        """Simulate an action on a copy of `world` and return its evaluated strategic score."""
        successor = world.copy()
        try:
            successor.execute(action)
        except Exception:
            return float("-inf")

        return self.evaluate_state(successor, color)
