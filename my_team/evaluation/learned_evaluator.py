"""Compact Learned Value Evaluator for Catan.

Loads offline-trained linear or shallow MLP parameters and performs lightweight
NumPy inference (< 0.05 ms) to score successor positions.
Preserves dominant tactical overrides and provides safe fallback to the strategic evaluator.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from catanatron.game import Game
from catanatron.models.enums import Action

from my_team.evaluation.evaluator import PhaseAwareEvaluator
from my_team.evaluation.features import extract_strategic_features


class LearnedValueEvaluator:
    """Evaluates Catan positions using an offline-trained compact value model."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_type: str = "linear",  # "linear" or "mlp"
        fallback_evaluator: Optional[PhaseAwareEvaluator] = None,
    ):
        self.model_type = model_type
        self.fallback = fallback_evaluator or PhaseAwareEvaluator()
        self.is_loaded = False

        target_path = model_path or os.environ.get("CATAN_LEARNED_MODEL")
        if not target_path:
            pkg_path = Path(__file__).resolve().parent / "learned_model.json"
            if pkg_path.is_file():
                target_path = str(pkg_path)

        if target_path and os.path.isfile(target_path):
            try:
                self._load_model(target_path)
            except Exception:
                self.is_loaded = False

    def _load_model(self, path: str):
        """Parse exported weights from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.feature_names = data.get("feature_names", [])

        if self.model_type == "linear" and "linear_model" in data:
            lm = data["linear_model"]
            self.w = np.array(lm["weights"], dtype=np.float32)
            self.b = float(lm["intercept"])
            self.is_loaded = True
        elif self.model_type == "mlp" and "mlp_model" in data:
            mm = data["mlp_model"]
            self.w1 = np.array(mm["weights"][0], dtype=np.float32)
            self.b1 = np.array(mm["biases"][0], dtype=np.float32)
            self.w2 = np.array(mm["weights"][1], dtype=np.float32)
            self.b2 = np.array(mm["biases"][1], dtype=np.float32)
            self.is_loaded = True

    def forward(self, features: Dict[str, float]) -> float:
        """Run pure NumPy inference forward pass."""
        # Dominant Tactical Override: Immediate Victory (10+ VPs)
        if features.get("is_winning", 0.0) == 1.0:
            return 1e15

        if not self.is_loaded:
            return 0.0

        x = np.array([float(features.get(k, 0.0)) for k in self.feature_names], dtype=np.float32)

        if self.model_type == "linear":
            val = float(np.dot(self.w, x) + self.b)
        else:
            h = np.maximum(0.0, np.dot(x, self.w1) + self.b1)
            val = float(np.dot(h, self.w2) + self.b2)

        return val

    def evaluate_state(self, game: Game, color) -> float:
        """Evaluate a game state for `color`."""
        if not self.is_loaded:
            return self.fallback.evaluate_state(game, color)

        features = extract_strategic_features(game, color)
        return self.forward(features)

    def evaluate_candidate_action(self, world: Game, action: Action, color) -> float:
        """Simulate an action on a copy of `world` and return learned value score."""
        successor = world.copy()
        try:
            successor.execute(action)
        except Exception:
            return float("-inf")

        return self.evaluate_state(successor, color)
