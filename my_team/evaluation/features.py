"""Strategic Feature Extraction for Catan States.

Extracts normalized, interpretable strategic features from catanatron Game/State objects.
Features cover:
  1. Victory points progress & win detection
  2. Production potential, pip distribution, and 5-resource diversity
  3. Port ownership and resource-port synergy
  4. Hand synergy and build distances (City, Settlement, Dev Card, Road)
  5. Expansion network and buildable node availability
  6. Development cards and Army/Road races
  7. Robber choking and defense
"""
import math
from typing import Dict, List, Set, Tuple

from catanatron.models.enums import CITY, RESOURCES, SETTLEMENT, FastResource
from catanatron.models.map import number_probability
from catanatron.state_functions import (
    get_actual_victory_points,
    get_longest_road_length,
    get_played_dev_cards,
    get_player_buildings,
    player_key,
    player_num_dev_cards,
    player_num_resource_cards,
)

# Resource costs for standard structures
COST_ROAD = {"WOOD": 1, "BRICK": 1}
COST_SETTLEMENT = {"WOOD": 1, "BRICK": 1, "SHEEP": 1, "WHEAT": 1}
COST_CITY = {"WHEAT": 2, "ORE": 3}
COST_DEV_CARD = {"SHEEP": 1, "WHEAT": 1, "ORE": 1}


def get_tile_pips(number: int | None) -> int:
    """Return standard Catan pip count (1 to 5) for a dice roll number."""
    if number is None or number < 2 or number > 12:
        return 0
    return round(number_probability(number) * 36)


def extract_production_pips(board, color) -> Dict[str, float]:
    """Calculate effective production pips per resource for a player.
    
    Settlements yield 1x, Cities yield 2x.
    If the robber is on a tile, that tile's effective yield is 0.
    """
    production = {r: 0.0 for r in RESOURCES}
    robber_tile = board.map.tiles.get(board.robber_coordinate)
    robber_tile_id = robber_tile.id if robber_tile else None

    for node_id, building in board.buildings.items():
        if building[0] != color:
            continue
        multiplier = 2.0 if building[1] == CITY else 1.0

        for tile in board.map.adjacent_tiles[node_id]:
            if tile.resource is None or tile.resource not in production:
                continue
            # If robber is on tile, production is blocked
            if robber_tile_id is not None and tile.id == robber_tile_id:
                continue
            pips = get_tile_pips(tile.number)
            production[tile.resource] += pips * multiplier

    return production


def compute_diminishing_production(pips: float) -> float:
    """Concave utility function modeling diminishing marginal returns of single-resource pips.
    
    First 4 pips: full value (1.0 weight)
    Next 4 pips (4-8): moderate value (0.5 weight)
    Excess pips (>8): low value (0.2 weight)
    """
    tier1 = min(pips, 4.0)
    tier2 = min(max(0.0, pips - 4.0), 4.0) * 0.5
    tier3 = max(0.0, pips - 8.0) * 0.2
    return tier1 + tier2 + tier3


def extract_strategic_features(game, color) -> Dict[str, float]:
    """Extract full vector of normalized strategic features from game state for player `color`."""
    st = game.state
    board = st.board
    me_key = player_key(st, color)
    opp_colors = [c for c in st.colors if c != color]
    opp_color = opp_colors[0] if opp_colors else None
    opp_key = player_key(st, opp_color) if opp_color else None

    # 1. Victory Points
    my_actual_vps = float(st.player_state.get(f"{me_key}_ACTUAL_VICTORY_POINTS", 0))
    my_public_vps = float(st.player_state.get(f"{me_key}_VICTORY_POINTS", 0))
    opp_public_vps = float(st.player_state.get(f"{opp_key}_VICTORY_POINTS", 0)) if opp_key else 0.0
    vp_delta = my_actual_vps - opp_public_vps
    is_winning = 1.0 if my_actual_vps >= 10.0 else 0.0

    # 2. Production Metrics
    my_prod = extract_production_pips(board, color)
    opp_prod = extract_production_pips(board, opp_color) if opp_color else {r: 0.0 for r in RESOURCES}

    my_dim_prod = sum(compute_diminishing_production(v) for v in my_prod.values())
    opp_dim_prod = sum(compute_diminishing_production(v) for v in opp_prod.values())

    # Resource Diversity (count of resources with > 0 pips, out of 5)
    distinct_resources = sum(1.0 for v in my_prod.values() if v > 0.0)
    diversity_ratio = distinct_resources / 5.0

    # Key resource bottlenecks: Wheat and Ore (needed for Cities & Devs)
    wheat_ore_pips = my_prod["WHEAT"] + my_prod["ORE"]
    wood_brick_pips = my_prod["WOOD"] + my_prod["BRICK"]

    # 3. Ports & Port Synergy
    ports_owned = board.get_player_port_resources(color)
    has_three_to_one = 1.0 if (None in ports_owned) else 0.0
    has_special_port = 1.0 if any(p is not None for p in ports_owned) else 0.0

    # 2:1 Port Synergy: player produces 4+ pips of resource R and owns port for R
    port_synergy = 0.0
    for res in RESOURCES:
        if res in ports_owned:
            port_synergy += min(my_prod[res], 8.0) * 0.25

    # 4. Hand Synergy & Build Distance
    hand = {r: st.player_state.get(f"{me_key}_{r}_IN_HAND", 0) for r in RESOURCES}
    total_hand = sum(hand.values())

    # Distance to key structures (0 = can build now, larger = farther)
    dist_city = max(0, 2 - hand["WHEAT"]) + max(0, 3 - hand["ORE"])
    dist_settlement = (
        max(0, 1 - hand["WOOD"])
        + max(0, 1 - hand["BRICK"])
        + max(0, 1 - hand["SHEEP"])
        + max(0, 1 - hand["WHEAT"])
    )
    dist_dev = max(0, 1 - hand["SHEEP"]) + max(0, 1 - hand["WHEAT"]) + max(0, 1 - hand["ORE"])
    dist_road = max(0, 1 - hand["WOOD"]) + max(0, 1 - hand["BRICK"])

    # Inverted build readiness in [0.0, 1.0]
    can_city = 1.0 if dist_city == 0 else (1.0 - dist_city / 5.0)
    can_settlement = 1.0 if dist_settlement == 0 else (1.0 - dist_settlement / 4.0)
    can_dev = 1.0 if dist_dev == 0 else (1.0 - dist_dev / 3.0)

    # Discard Risk: Holding > 7 cards before rolling exposes half the hand to loss
    has_rolled = st.player_state.get(f"{me_key}_HAS_ROLLED", True)
    discard_risk = (max(0.0, float(total_hand - 7)) / 4.0) if not has_rolled else 0.0

    # 5. Expansion & Connectivity
    buildable_nodes = board.buildable_node_ids(color)
    num_buildable_nodes = float(len(buildable_nodes))
    blocked_penalty = 1.0 if num_buildable_nodes == 0.0 else 0.0

    # Cities remaining to build
    cities_left = float(st.player_state.get(f"{me_key}_CITIES_AVAILABLE", 0))
    settlements_left = float(st.player_state.get(f"{me_key}_SETTLEMENTS_AVAILABLE", 0))

    # 6. Special Objectives: Longest Road & Largest Army
    my_road_len = float(get_longest_road_length(st, color))
    opp_road_len = float(get_longest_road_length(st, opp_color)) if opp_color else 0.0
    has_longest_road = 1.0 if st.player_state.get(f"{me_key}_HAS_ROAD", False) else 0.0

    my_knights = float(get_played_dev_cards(st, color, "KNIGHT"))
    opp_knights = float(get_played_dev_cards(st, opp_color, "KNIGHT")) if opp_color else 0.0
    has_largest_army = 1.0 if st.player_state.get(f"{me_key}_HAS_ARMY", False) else 0.0
    unplayed_devs = float(player_num_dev_cards(st, color))

    # 7. Robber Positioning
    robber_on_opp = 0.0
    robber_on_me = 0.0
    robber_tile = board.map.tiles.get(board.robber_coordinate)
    if robber_tile:
        for node_id in robber_tile.nodes.values():
            building = board.buildings.get(node_id)
            if building:
                if building[0] == color:
                    robber_on_me = 1.0
                elif opp_color and building[0] == opp_color:
                    robber_on_opp = 1.0

    return {
        "is_winning": is_winning,
        "my_actual_vps": my_actual_vps,
        "opp_public_vps": opp_public_vps,
        "vp_delta": vp_delta,
        "my_dim_production": my_dim_prod,
        "opp_dim_production": opp_dim_prod,
        "production_diversity": diversity_ratio,
        "wheat_ore_pips": wheat_ore_pips,
        "wood_brick_pips": wood_brick_pips,
        "has_3_to_1_port": has_three_to_one,
        "has_special_port": has_special_port,
        "port_synergy": port_synergy,
        "can_city_readiness": can_city,
        "can_settlement_readiness": can_settlement,
        "can_dev_readiness": can_dev,
        "discard_risk": discard_risk,
        "num_buildable_nodes": num_buildable_nodes,
        "blocked_penalty": blocked_penalty,
        "my_road_length": my_road_len,
        "has_longest_road": has_longest_road,
        "road_lead": max(0.0, my_road_len - opp_road_len),
        "my_knights_played": my_knights,
        "has_largest_army": has_largest_army,
        "army_lead": max(0.0, my_knights - opp_knights),
        "unplayed_devs": unplayed_devs,
        "robber_on_enemy": robber_on_opp,
        "robber_on_self": robber_on_me,
    }
