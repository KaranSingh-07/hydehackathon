"""Selective Search Catan Agent (Phase 3 Implementation).

Integrates:
  1. Budgeted Selective Search with anytime deadline protection.
  2. Same-Turn Compound Action Lookahead (Maritime Trade & Dev Card chains).
  3. Multi-World Imperfect Information Sampling across opponent dev card distributions.
  4. Phase-Aware Strategic Evaluation (EARLY, MID, LATE dynamic weighting).
  5. Fallback safety and guaranteed legal index boundaries.
"""
import random
from typing import List, Optional

from catan_kit.obs import Obs
from catanatron.models.enums import Action
try:
    from my_team.evaluation.evaluator import PhaseAwareEvaluator
    from my_team.search.budget_manager import BudgetManager, SearchConfig
    from my_team.search.selective_search import SelectiveSearchEngine
except ImportError:
    from evaluation.evaluator import PhaseAwareEvaluator
    from search.budget_manager import BudgetManager, SearchConfig
    from search.selective_search import SelectiveSearchEngine



class Agent:
    """Competition agent combining Phase-Aware evaluation with Selective Search."""

    def __init__(
        self,
        evaluator: Optional[PhaseAwareEvaluator] = None,
        search_engine: Optional[SelectiveSearchEngine] = None,
        budget_manager: Optional[BudgetManager] = None,
    ):
        """Initialize models, search engine, and budget manager once per process."""
        self.rng = random.Random(0)
        self.evaluator = evaluator or PhaseAwareEvaluator()
        self.search_engine = search_engine or SelectiveSearchEngine(self.evaluator)
        self.budget_manager = budget_manager or BudgetManager()

    def decide(self, obs: Obs, playable_actions: List[Action]) -> int:
        """Select the highest-utility action index within the allocated time budget."""
        # Defensive bounds check
        if not playable_actions:
            return 0
        if len(playable_actions) == 1:
            return 0

        fallback_index = 0

        try:
            # 1. Determine dynamic search configuration based on clock and game situation
            is_opening = obs.turn < 2 or obs._game.state.is_initial_build_phase
            config = self.budget_manager.get_search_config(
                cpu_left=obs.cpu_left,
                turn=obs.turn,
                is_opening=is_opening,
                opp_dev_count=obs.opp_dev_count,
            )

            # 2. Execute bounded selective search
            best_action = self.search_engine.select_best_action(
                obs=obs,
                playable_actions=playable_actions,
                config=config,
                rng=self.rng,
            )

            # 3. Locate chosen action index in original list
            try:
                selected_index = playable_actions.index(best_action)
            except ValueError:
                selected_index = 0

            if 0 <= selected_index < len(playable_actions):
                return int(selected_index)
            return fallback_index

        except Exception:
            # Absolute top-level fallback
            return fallback_index
