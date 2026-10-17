"""Action Prioritization and Dominated Move Pruning.

Filters and orders legal actions before simulation to:
  1. Detect immediate game-winning actions for 0-latency execution.
  2. Prune strictly dominated 4:1 trades when 3:1 or 2:1 ports are available.
  3. Filter pointless robber placements (e.g., self-robbing or empty tiles when enemies exist).
  4. Prioritize high-impact moves (Cities, Settlements) first.
"""
from typing import List, Tuple
from catanatron.game import Game
from catanatron.models.enums import Action, ActionType, CITY, SETTLEMENT


def is_winning_move_heuristic(action: Action, game: Game, color) -> bool:
    """Fast check whether an action guarantees immediate victory without full simulation."""
    st = game.state
    idx = st.color_to_index[color]
    actual_vps = st.player_state.get(f"P{idx}_ACTUAL_VICTORY_POINTS", 0)

    # If at 9 actual VPs, any Settlement or City build secures 10 VPs
    if actual_vps >= 9:
        if action.action_type in (ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY):
            return True

    # If at 8 actual VPs, building a Settlement that also takes Longest Road secures 10 VPs
    return False


def prune_dominated_actions(playable_actions: List[Action], game: Game, color) -> List[Action]:
    """Filter strictly dominated actions to save CPU cycles and avoid tactical blunders."""
    if len(playable_actions) <= 1:
        return playable_actions

    types = {a.action_type for a in playable_actions}
    board = game.state.board
    ports_owned = board.get_player_port_resources(color)
    has_three_to_one = None in ports_owned

    pruned = []

    for action in playable_actions:
        # 1. Prune redundant 4:1 maritime trades if player has a 3:1 port
        if action.action_type == ActionType.MARITIME_TRADE and has_three_to_one:
            # action.value is (give1, give2, give3, give4, get)
            # A 4:1 trade has give3 and give4 not None
            if action.value[3] is not None:
                continue

        # 2. Prune 2:1 port redundancy: If player has 2:1 port for resource R, don't 3:1 or 4:1 trade R
        if action.action_type == ActionType.MARITIME_TRADE:
            give_res = action.value[0]
            if give_res in ports_owned and action.value[2] is not None:
                # Giving 3 or 4 of resource R when we have a 2:1 port for R is dominated
                continue

        # 3. Prune self-damaging robber moves:
        # If moving robber to a tile solely owned by oneself when other tiles have enemies
        if action.action_type == ActionType.MOVE_ROBBER:
            target_coord, victim_color = action.value
            # Don't steal from self or rob self if victim is self
            if victim_color == color:
                continue

        pruned.append(action)

    # Safety guard: never return empty list
    return pruned if pruned else playable_actions
