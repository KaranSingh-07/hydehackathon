# Catan Game-Data & Strategic Decision Dataset Schema

**Schema Version**: `1.0.0`  
**Storage Format**: JSON Lines (`.jsonl`) with initial metadata header line  
**Generator Source**: [`scripts/generate_dataset.py`](file:///c:/Users/KARAN%20singh/Desktop/hyde_ai_hackathon/scripts/generate_dataset.py)

---

## 1. File Structure

Each generated dataset file begins with a single metadata header record (`_metadata`), followed by one JSON record per decision:

```json
{"_metadata": {"schema_version": "1.0.0", "timestamp": "...", "agent_a": "my_team", "agent_b": "f", "vps_to_win": 10, "total_games": 40, "seeds": [...]}}
{"turn": 0, "phase": "EARLY", "player_label": "A", ...}
{"turn": 1, "phase": "EARLY", "player_label": "B", ...}
```

---

## 2. Field Dictionary

| Field | Type | Description |
|---|---|---|
| `game_id` | integer | Unique match index within the generation run. |
| `board_seed` | integer | Catanatron board generation random seed. |
| `a_first` | boolean | `true` if Agent A sat in Seat 0 (Red / first player); `false` if Agent B sat in Seat 0. |
| `seat` | integer | `0` for the opening player; `1` for the second player. |
| `player_label` | string | `"A"` for Agent A; `"B"` for Agent B. |
| `color` | string | Color assigned to the player (`"RED"` or `"BLUE"`). |
| `turn` | integer | Game turn counter at the moment of decision. |
| `phase` | string | Strategic phase category: `"EARLY"`, `"MID"`, or `"LATE"`. |
| `num_actions` | integer | Total number of legal actions available to the agent. |
| `action_type` | string | Catanatron `ActionType` name (e.g. `BUILD_SETTLEMENT`, `BUILD_CITY`, `MARITIME_TRADE`, `MOVE_ROBBER`). |
| `action_value` | string | Stringified target node, edge, tile coordinate, or card value. |
| `chosen_index` | integer | Zero-based index within the legal action array selected by the policy. |
| `latency_ms` | float | Wall-clock decision time in milliseconds taken by the agent. |
| `final_vps_me` | integer | Total actual victory points accumulated by this player at game termination. |
| `final_vps_opp` | integer | Total actual victory points accumulated by the opponent at game termination. |
| `vps_delta` | integer | Margin of victory/defeat: $\text{VPs}_{\text{me}} - \text{VPs}_{\text{opp}}$. |
| `outcome` | string | Termination result for this player: `"WIN"`, `"LOSS"`, or `"DRAW"`. |
| `is_win` | integer | Binary label: `1` if this player won; `0` otherwise. |
| `total_game_turns` | integer | Total turns elapsed in the complete game. |

---

## 3. Extracted Feature Vector (`features`)

The `features` object contains 25 normalized strategic features extracted from the observation view:

| Feature Name | Theoretical Range | Strategic Role |
|---|---|---|
| `is_winning` | `[0.0, 1.0]` | Indicator if this state achieves $\ge 10$ VPs. |
| `my_actual_vps` | `[0.0, 10.0]` | Current true victory points owned. |
| `opp_public_vps` | `[0.0, 10.0]` | Publicly visible opponent victory points. |
| `vp_delta` | `[-10.0, 10.0]` | Lead over opponent: $\text{VPs}_{\text{me}} - \text{VPs}_{\text{opp}}$. |
| `my_dim_production` | `[0.0, ~25.0]` | Diminishing-returns production sum across 5 resources. |
| `opp_dim_production` | `[0.0, ~25.0]` | Opponent's total diminishing-returns production sum. |
| `production_diversity` | `[0.0, 5.0]` | Number of distinct resources with non-zero production. |
| `wheat_ore_pips` | `[0.0, ~20.0]` | Combined wheat + ore dice roll pips (city/dev fuel). |
| `wood_brick_pips` | `[0.0, ~20.0]` | Combined wood + brick dice roll pips (expansion fuel). |
| `has_3_to_1_port` | `[0.0, 1.0]` | Ownership of a 3:1 generic maritime trade port. |
| `has_special_port` | `[0.0, 1.0]` | Ownership of at least one 2:1 resource-specific port. |
| `port_synergy` | `[0.0, ~15.0]` | Production pips feeding owned 2:1 ports. |
| `can_city_readiness` | `[0.0, 1.0]` | Resource card readiness ratio toward 2 Wheat + 3 Ore. |
| `can_settlement_readiness` | `[0.0, 1.0]` | Readiness toward 1 Wood + 1 Brick + 1 Wheat + 1 Sheep. |
| `can_dev_readiness` | `[0.0, 1.0]` | Readiness toward 1 Wheat + 1 Sheep + 1 Ore. |
| `discard_risk` | `[0.0, 1.0]` | Danger penalty when holding $>7$ cards on roll 7. |
| `num_buildable_nodes` | `[0.0, ~15.0]` | Reachable legal settlement spots on the road network. |
| `blocked_penalty` | `[0.0, 1.0]` | Severe expansion blockage indicator. |
| `my_road_length` | `[0.0, ~15.0]` | Current continuous road segment length. |
| `has_longest_road` | `[0.0, 1.0]` | Indicator if this player holds Longest Road (+2 VPs). |
| `road_lead` | `[-15.0, 15.0]` | Differential in longest road length vs opponent. |
| `my_knights_played` | `[0.0, ~14.0]` | Total Knight development cards played. |
| `has_largest_army` | `[0.0, 1.0]` | Indicator if this player holds Largest Army (+2 VPs). |
| `army_lead` | `[-14.0, 14.0]` | Differential in knights played vs opponent. |
| `unplayed_devs` | `[0.0, ~5.0]` | Number of unplayed development cards in hand. |
| `robber_on_enemy` | `[0.0, ~10.0]` | Opponent production pips disabled by the robber. |
| `robber_on_self` | `[0.0, ~10.0]` | Self production pips disabled by the robber. |

---

## 4. Learning Targets & Optimization Objectives

The dataset directly supports multiple supervised and reinforcement learning targets:

1. **State Value Function ($V(s)$)**:
   Predict `is_win` ($y \in \{0, 1\}$) or normalized final score $\frac{\text{final\_vps}}{10}$ using features $\mathbf{x} \in \mathbb{R}^{25}$.
2. **Action Advantage ($Q(s, a) - V(s)$)**:
   Compare candidate actions by relative final victory point differential `vps_delta`.
3. **Weight Sensitivity Analysis**:
   Fit ridge/logistic regression on $(\mathbf{x}, y)$ per phase (`EARLY`, `MID`, `LATE`) to estimate feature influence coefficients.

---

## 5. Usage & Reproduction

```powershell
# Generate training dataset: 20 seeds (40 games) vs Bot F
.venv\Scripts\python scripts/generate_dataset.py --a my_team --b f --partition train --tag train_vs_f

# Generate validation dataset: 20 seeds (40 games) vs Bot F
.venv\Scripts\python scripts/generate_dataset.py --a my_team --b f --partition val --tag val_vs_f

# Generate self-play dataset: 20 seeds (40 games)
.venv\Scripts\python scripts/generate_dataset.py --a my_team --b my_team --partition train --tag self_play
```
