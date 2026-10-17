# Strategic Evaluator Architecture & Feature Engineering

## 1. Overview & Evaluator Contract

The Phase 2 Strategic Evaluator ([`my_team/evaluation/evaluator.py`](file:///c:/Users/KARAN%20singh/Desktop/hyde_ai_hackathon/my_team/evaluation/evaluator.py)) replaces arbitrary heuristic weights with a structured, phase-aware strategic utility model. It evaluates candidate actions and resulting game states using only legitimately observable information provided by the competition SDK.

### Core Contract
```python
class PhaseAwareEvaluator:
    def evaluate_state(self, game: Game, color: Color) -> float:
        """Returns strategic utility score for `color` in `game`."""

    def evaluate_candidate_action(self, world: Game, action: Action, color: Color) -> float:
        """Simulates `action` on a copy of `world` and returns evaluated utility."""
```

The output score represents holistic game trajectory utility: immediate tactical value, production growth, port synergy, building readiness, and opponent denial.

---

## 2. Feature Definitions & Normalization Scales

Every feature in [`my_team/evaluation/features.py`](file:///c:/Users/KARAN%20singh/Desktop/hyde_ai_hackathon/my_team/evaluation/features.py) is computed cleanly from game state:

| Feature Name | Mathematical / Rule Definition | Unit / Range | Sign | Strategic Purpose |
|---|---|---|---|---|
| `is_winning` | $1.0 \text{ if } \text{my\_vps} \ge 10 \text{ else } 0.0$ | $[0.0, 1.0]$ | $+$ | Dominant tactical win priority ($+10^{15}$). |
| `my_actual_vps` | Actual victory points (includes hidden VP cards). | $[2, 10]$ | $+$ | Primary goal progression. |
| `opp_public_vps` | Opponent visible victory points. | $[2, 10]$ | $-$ | Threat detection & urgency. |
| `vp_delta` | $\text{my\_actual\_vps} - \text{opp\_public\_vps}$ | $[-8, +8]$ | $+$ | Lead differential. |
| `my_dim_production` | Sum of concave-utility pips across all 5 resources. | $[0.0, 30.0]$ | $+$ | Core engine yield with diminishing returns. |
| `opp_dim_production`| Sum of concave-utility pips for opponent. | $[0.0, 30.0]$ | $-$ | Minimizes enemy resource generation. |
| `production_diversity`| $\frac{\text{count}(\text{pips}(R) > 0)}{5.0}$ | $[0.0, 1.0]$ | $+$ | Rewards covering all 5 resource types. |
| `wheat_ore_pips` | Pips of Wheat + Ore. | $[0.0, 16.0]$ | $+$ | Evaluates capability for Cities & Dev Cards. |
| `wood_brick_pips` | Pips of Wood + Brick. | $[0.0, 16.0]$ | $+$ | Evaluates capability for Roads & Settlements. |
| `has_3_to_1_port` | $1.0 \text{ if owns 3:1 port else } 0.0$ | $[0.0, 1.0]$ | $+$ | Universal trade conversion efficiency. |
| `has_special_port` | $1.0 \text{ if owns any 2:1 port else } 0.0$ | $[0.0, 1.0]$ | $+$ | Specialized trading capability. |
| `port_synergy` | $\sum_{R} (\min(\text{pips}(R), 8) \times 0.25) \cdot \mathbb{I}(R \text{ port owned})$ | $[0.0, 4.0]$ | $+$ | High production paired with matching 2:1 port. |
| `can_city_readiness` | Inverted distance: $1.0 - \frac{\text{dist\_city}}{5.0}$ | $[0.0, 1.0]$ | $+$ | Proximity to immediate City build. |
| `can_settlement_readiness` | Inverted distance: $1.0 - \frac{\text{dist\_settlement}}{4.0}$ | $[0.0, 1.0]$ | $+$ | Proximity to immediate Settlement build. |
| `can_dev_readiness` | Inverted distance: $1.0 - \frac{\text{dist\_dev}}{3.0}$ | $[0.0, 1.0]$ | $+$ | Proximity to immediate Dev Card purchase. |
| `discard_risk` | $\max(0, \text{hand} - 7) / 4.0 \text{ (if pre-roll)}$ | $[0.0, 3.0]$ | $-$ | Penalizes overholding resources on roll. |
| `num_buildable_nodes`| Number of legal settlement locations. | $[0, 20]$ | $+$ | Expansion freedom and blocking avoidance. |
| `blocked_penalty` | $1.0 \text{ if buildable nodes == 0 else } 0.0$ | $[0.0, 1.0]$ | $-$ | Severe penalty for getting boxed in. |
| `my_road_length` | Longest road segment count. | $[0, 15]$ | $+$ | Longest Road progression. |
| `has_longest_road` | $1.0 \text{ if owns Longest Road else } 0.0$ | $[0.0, 1.0]$ | $+$ | 2 VP award status. |
| `road_lead` | $\max(0, \text{my\_road} - \text{opp\_road})$ | $[0, 10]$ | $+$ | Cushion against opponent stealing road. |
| `my_knights_played` | Number of played Knights. | $[0, 14]$ | $+$ | Largest Army progression. |
| `has_largest_army` | $1.0 \text{ if owns Largest Army else } 0.0$ | $[0.0, 1.0]$ | $+$ | 2 VP award status. |
| `army_lead` | $\max(0, \text{my\_knights} - \text{opp\_knights})$ | $[0, 10]$ | $+$ | Cushion against opponent stealing army. |
| `unplayed_devs` | Count of unplayed development cards held. | $[0, 10]$ | $+$ | Strategic flexibility & hidden VP value. |
| `robber_on_enemy` | $1.0 \text{ if robber is on enemy tile else } 0.0$ | $[0.0, 1.0]$ | $+$ | Offensive resource disruption. |
| `robber_on_self` | $1.0 \text{ if robber is on own tile else } 0.0$ | $[0.0, 1.0]$ | $-$ | Choke penalty on own economy. |

---

## 3. Mathematical Models & Curves

### 3.1 Diminishing Marginal Production Utility
In tabletop Catan, having 10 pips of Sheep and 0 pips of Ore is fatal. Concentrated single-resource production suffers steep diminishing returns:
$$U(\text{pips}) = \begin{cases} 
\text{pips} & \text{if } \text{pips} \le 4.0 \\
4.0 + 0.5 \times (\text{pips} - 4.0) & \text{if } 4.0 < \text{pips} \le 8.0 \\
6.0 + 0.2 \times (\text{pips} - 8.0) & \text{if } \text{pips} > 8.0 
\end{cases}$$
This encourages settlements that open up missing resources rather than overloading already abundant ones.

### 3.2 2:1 Port Synergy Multiplier
A 2:1 port is only valuable if the player produces excess quantities of that specific resource. If player owns port $R$, we reward:
$$\text{Synergy}(R) = \min(\text{pips}(R), 8.0) \times 0.25$$
This properly prioritizes expanding toward the 2:1 Ore port when Ore production is strong.

---

## 4. Phase-Aware Strategy & Dynamic Weighting

Games transition through three distinct phases:
- **`EARLY` Phase**: $\text{Turns} < 8$ or Initial Build Phase or ($\text{my\_vps} \le 3$ and $\text{opp\_vps} \le 3$).
  - *Strategic Priority*: Maximize total pip production, 5-resource diversity, and road/settlement expansion flexibility.
- **`MID` Phase**: $\text{Turns} \ge 8$ and $4 \le \text{my\_vps} < 8$ and $\text{opp\_vps} < 8$.
  - *Strategic Priority*: Pivot to Cities (Wheat & Ore), Dev Card acquisition, Largest Army & Longest Road contests, and 2:1 port monetization.
- **`LATE` Phase**: $\text{my\_actual\_vps} \ge 8$ or $\text{opp\_public\_vps} \ge 8$.
  - *Strategic Priority*: Victory points dominate all other factors. Absolute focus on the shortest path to 10 VPs, locking Longest Road/Largest Army, and aggressively placing the robber to choke the opponent's winning hexes.

---

## 5. Action Prioritization & Dominated Move Pruning

Before executing expensive `world.copy()` operations, [`my_team/core/action_prioritizer.py`](file:///c:/Users/KARAN%20singh/Desktop/hyde_ai_hackathon/my_team/core/action_prioritizer.py) applies tactical filters:
1. **Immediate Victory Fast-Path**: If at 9 VPs, any legal Settlement or City build is recognized as an immediate win and chosen with zero search delay.
2. **Redundant 4:1 Trade Pruning**: If player owns a 3:1 port, 4:1 trades are strictly dominated and pruned.
3. **2:1 Port Trade Pruning**: If player owns the 2:1 Ore port, trading 3 or 4 Ore is strictly dominated and pruned.
4. **Self-Robbing Prevention**: Moving the robber to a tile solely owned by oneself is filtered out.
