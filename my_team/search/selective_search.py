"""Selective Search and Multi-World Planning Engine.

Combines:
  1. Diversity-Preserving Candidate Ranking: Prunes low-value moves while maintaining tactical breadth.
  2. Same-Turn Compound Action Search: Evaluates trades and dev plays by looking ahead to follow-up builds.
  3. Multi-World Imperfect Information Sampling: Averages outcomes across plausible hidden card allocations.
  4. Anytime Deadline Protection: Terminates gracefully when time budget expires.
"""
from typing import Dict, List, Optional, Set, Tuple
import random

from catan_kit.obs import Obs
from catanatron.game import Game
from catanatron.models.enums import Action, ActionType
try:
    from my_team.core.action_prioritizer import is_winning_move_heuristic, prune_dominated_actions
    from my_team.evaluation.evaluator import PhaseAwareEvaluator
    from my_team.search.budget_manager import DeadlineTimer, SearchConfig
except ImportError:
    from core.action_prioritizer import is_winning_move_heuristic, prune_dominated_actions
    from evaluation.evaluator import PhaseAwareEvaluator
    from search.budget_manager import DeadlineTimer, SearchConfig


# Actions that do not end the turn and enable follow-up builds
COMPOUND_ACTION_TYPES = {
    ActionType.MARITIME_TRADE,
    ActionType.PLAY_YEAR_OF_PLENTY,
    ActionType.PLAY_ROAD_BUILDING,
}

# High-value build actions worth looking ahead for after a compound action
FOLLOWUP_BUILD_TYPES = {
    ActionType.BUILD_SETTLEMENT,
    ActionType.BUILD_CITY,
    ActionType.BUY_DEVELOPMENT_CARD,
    ActionType.BUILD_ROAD,
}


class SelectiveSearchEngine:
    """Bounded, anytime search orchestrator."""

    def __init__(self, evaluator: Optional[PhaseAwareEvaluator] = None):
        self.evaluator = evaluator or PhaseAwareEvaluator()

    def rank_candidates_with_diversity(
        self,
        world: Game,
        actions: List[Action],
        color,
        max_candidates: int,
    ) -> List[Tuple[Action, float]]:
        """1-ply score all actions and retain the top scoring actions with diversity preservation."""
        scored: List[Tuple[Action, float]] = []

        for action in actions:
            successor = world.copy()
            try:
                successor.execute(action)
                score = self.evaluator.evaluate_state(successor, color)
            except Exception:
                score = float("-inf")
            scored.append((action, score))

        # Sort descending by 1-ply score
        scored.sort(key=lambda x: x[1], reverse=True)

        # Retain top candidates while ensuring diverse action types are represented
        selected: List[Tuple[Action, float]] = []
        seen_types: Set[ActionType] = set()

        # Pass 1: Select top overall moves and record their action types
        for action, score in scored:
            if len(selected) >= max_candidates:
                break
            selected.append((action, score))
            seen_types.add(action.action_type)

        # Pass 2: Ensure distinct high-impact types (e.g., Settlement, City, Trade) get a seat
        if len(selected) < len(scored):
            for action, score in scored[max_candidates:]:
                if action.action_type not in seen_types and len(selected) < max_candidates + 2:
                    selected.append((action, score))
                    seen_types.add(action.action_type)

        return selected

    def evaluate_compound_action(self, world: Game, action: Action, color) -> float:
        """Evaluate a compound action (e.g., trade) by searching the best same-turn follow-up build."""
        successor = world.copy()
        try:
            successor.execute(action)
        except Exception:
            return float("-inf")

        base_score = self.evaluator.evaluate_state(successor, color)

        # If not a compound action or turn ended, return 1-ply score
        if action.action_type not in COMPOUND_ACTION_TYPES:
            return base_score

        # Check follow-up options in the same turn
        followup_actions = successor.playable_actions
        best_followup_score = base_score

        for f_action in followup_actions:
            if f_action.action_type in FOLLOWUP_BUILD_TYPES:
                f_succ = successor.copy()
                try:
                    f_succ.execute(f_action)
                    f_score = self.evaluator.evaluate_state(f_succ, color)
                    if f_score > best_followup_score:
                        best_followup_score = f_score
                except Exception:
                    continue

        return best_followup_score

    def select_best_action(
        self,
        obs: Obs,
        playable_actions: List[Action],
        config: SearchConfig,
        rng: random.Random,
    ) -> Action:
        """Run bounded selective search and return the chosen legal Action."""
        if not playable_actions:
            raise ValueError("playable_actions cannot be empty")
        if len(playable_actions) == 1:
            return playable_actions[0]

        timer = DeadlineTimer(config.time_limit_s)

        # 1. Fast Path: Immediate Victory Detection
        for action in playable_actions:
            if is_winning_move_heuristic(action, obs._game, obs.me):
                return action

        # 2. Prune dominated moves before search
        primary_world = obs.sample_world(rng)
        candidate_actions = prune_dominated_actions(playable_actions, primary_world, obs.me)
        if not candidate_actions:
            candidate_actions = playable_actions

        # 3. Rank candidates on primary world
        ranked_candidates = self.rank_candidates_with_diversity(
            primary_world,
            candidate_actions,
            obs.me,
            max_candidates=config.max_candidates,
        )

        if not ranked_candidates:
            return playable_actions[0]

        # Anytime guarantee: Initialize best to the top 1-ply ranked candidate
        best_action = ranked_candidates[0][0]
        best_score = ranked_candidates[0][1]

        # If only 1 candidate requested or emergency clock, return top 1-ply immediately
        if config.max_candidates <= 1 or timer.is_expired():
            return best_action

        # 4. Multi-World Sampling Setup
        worlds = [primary_world]
        if config.num_worlds > 1 and obs.opp_dev_count > 0:
            for _ in range(config.num_worlds - 1):
                if timer.is_expired():
                    break
                worlds.append(obs.sample_world(rng))

        # 5. Deep Evaluation of Top Candidates across Sampled Worlds
        candidate_subset = [item[0] for item in ranked_candidates]

        for cand_action in candidate_subset:
            if timer.is_expired():
                break

            total_utility = 0.0
            valid_world_count = 0

            for world in worlds:
                if timer.is_expired():
                    break

                if config.allow_compound_search and cand_action.action_type in COMPOUND_ACTION_TYPES:
                    score = self.evaluate_compound_action(world, cand_action, obs.me)
                else:
                    score = self.evaluator.evaluate_candidate_action(world, cand_action, obs.me)

                if score != float("-inf"):
                    total_utility += score
                    valid_world_count += 1

            if valid_world_count > 0:
                avg_utility = total_utility / valid_world_count
                if avg_utility > best_score:
                    best_score = avg_utility
                    best_action = cand_action

        return best_action
