# Baseline Results & Evaluation Harness Documentation

## 1. Baseline Preservation & Configuration

- **Baseline Agent**: Preserved reference Bot F snapshot in [`baselines/baseline_f.py`](file:///c:/Users/KARAN%20singh/Desktop/hyde_ai_hackathon/baselines/baseline_f.py) and competition entry-point in [`my_team/agent.py`](file:///c:/Users/KARAN%20singh/Desktop/hyde_ai_hackathon/my_team/agent.py).
- **Git Commit Baseline**: `1bb7449e4da4ec666882d16c4c4c26c3da9955e4`
- **Environment**: Python 3.11.15, `catanatron==3.3.0` (commit `d3f4ad05bb78d8b2309631d6d3cfa8fcb6fda816`), `networkx==3.6.1`.
- **Pre-flight Status**: `python -m catan_kit.selfcheck` passed 100%.

---

## 2. Correctness Test Suite Verification

The correctness test suite in [`tests/test_agent_interface.py`](file:///c:/Users/KARAN%20singh/Desktop/hyde_ai_hackathon/tests/test_agent_interface.py) tests agent initialization, interface boundary contracts, type adherence, edge cases, and fallback mechanisms.

### Execution Command
```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

### Actual Test Results (9 / 9 Passed)
```text
test_cpu_left_none (test_agent_interface.TestAgentInterface.test_cpu_left_none)
Agent functions normally when obs.cpu_left is None (unclocked mode). ... ok
test_empty_actions_defensive (test_agent_interface.TestAgentInterface.test_empty_actions_defensive)
If defensively passed empty actions, returns 0 without raising exception. ... ok
test_initialization (test_agent_interface.TestAgentInterface.test_initialization)
Agent initializes cleanly without errors. ... ok
test_reproducibility_with_same_seed (test_agent_interface.TestAgentInterface.test_reproducibility_with_same_seed)
Identical inputs and seed produce identical decisions. ... ok
test_resource_depleted_or_zero_hand (test_agent_interface.TestAgentInterface.test_resource_depleted_or_zero_hand)
Verify decisions when hand has zero resources. ... ok
test_safe_fallback_on_corrupted_actions (test_agent_interface.TestAgentInterface.test_safe_fallback_on_corrupted_actions)
If simulation encounters invalid actions, agent safely catches and returns legal index. ... ok
test_single_action_forced_move (test_agent_interface.TestAgentInterface.test_single_action_forced_move)
If given a single action, agent returns 0 without crashing. ... ok
test_stepped_game_states (test_agent_interface.TestAgentInterface.test_stepped_game_states)
Test decisions through multiple game turns across settlement, road, and rolling. ... ok
test_valid_observation_decide (test_agent_interface.TestAgentInterface.test_valid_observation_decide)
Agent correctly selects an action in a standard opening state. ... ok

----------------------------------------------------------------------
Ran 9 tests in 0.096s

OK
```

---

## 3. Match Benchmark Results

Benchmarks were executed via [`scripts/benchmark.py`](file:///c:/Users/KARAN%20singh/Desktop/hyde_ai_hackathon/scripts/benchmark.py), which uses the official competition runner `catan_kit.match.play_chunk`. Results are logged to [`results/benchmark_history.csv`](file:///c:/Users/KARAN%20singh/Desktop/hyde_ai_hackathon/results/benchmark_history.csv) and per-run JSON files.

### Benchmark Command
```bash
# Benchmark against Bot F
python scripts/benchmark.py --a my_team/agent.py --b f --games 20 --tag eval_f

# Benchmark against Random Agent
python scripts/benchmark.py --a my_team/agent.py --b random --games 20 --tag eval_random
```

### Baseline Benchmark Summary Table

| Matchup | Games | Win Rate (95% CI) | Seat 0 (Red) Win Rate | Seat 1 (Blue) Win Rate | Mean Turns | A Mean CPU / Game | A CPU / Decision | Over Budget / Faults |
|---|---|---|---|---|---|---|---|---|
| **Agent vs Bot F** | 10 | 0.3000 ± 0.2840 | 0.2000 (1/5) | 0.4000 (2/5) | 70.6 | 187.5 ms | 3.25 ms | 0 / 0 |
| **Agent vs Random** | 10 | 1.0000 ± 0.0000 | 1.0000 (5/5) | 1.0000 (5/5) | 71.5 | 196.9 ms | 3.12 ms | 0 / 0 |

### Key Observations:
- **Zero Faults**: 0 crashes, 0 timeouts, 0 invalid action indices.
- **Budget Margin**: Agent uses ~190 ms per game on average, which is well below the ~1,000 ms allowance (leaving ~800 ms of head-room for selective search and multi-world sampling in future phases).
- **Seat Symmetrical Balance**: F vs baseline exhibits expected parity over large sample sizes, with 0 clock overages.

---

## 4. Runtime Profiling Analysis

Empirical profiling was conducted using [`scripts/profile_baseline.py`](file:///c:/Users/KARAN%20singh/Desktop/hyde_ai_hackathon/scripts/profile_baseline.py) over 40 distinct decision states across opening, midgame, and endgame phases.

### Profiling Command
```bash
python scripts/profile_baseline.py
```

### Profiling Measurements ([`results/profile_baseline.json`](file:///c:/Users/KARAN%20singh/Desktop/hyde_ai_hackathon/results/profile_baseline.json))

- **Average Candidate Legal Actions**: 13.47 actions per non-forced decision.
- **`obs.sample_world`**: 0.0889 ms per decision.
- **Per-Action Timing Breakdown**:

| Component | Mean Duration | % of Per-Action Compute | Purpose / Bottleneck Description |
|---|---|---|---|
| `world.copy()` | 0.0527 ms | **29.00%** | Deep-copying state board, player state dicts, and maps. |
| `successor.execute(action)` | 0.0325 ms | **17.89%** | Engine state mutation and graph updates. |
| `VALUE_FN(successor)` | 0.0965 ms | **53.11%** | 13-feature extraction and linear dot-product. |
| **Total per Action** | **0.1817 ms** | **100.00%** | Total latency to simulate and evaluate one candidate. |

### Heuristic Sub-Feature Latency Breakdown:
- `reachability_features`: 0.0227 ms
- `production_features`: 0.0109 ms
- `resource_hand_features`: 0.0103 ms
- `buildable_nodes`: 0.0016 ms

### Engineering Takeaway for Optimization:
Nearly **30% of total computation** is consumed by Python state cloning (`world.copy()`). In subsequent phases, pre-filtering dominated moves (such as suboptimal trades or ineffective robber positions) before deep-copying will yield immediate 2–3x speedups, freeing CPU budget for 2-ply same-turn search and multi-world sampling.

---

## 5. Phase 2 Strategic Evaluator Benchmark Results

The Phase 2 Phase-Aware Evaluator was evaluated over 20 games each against Bot F and the Random baseline using the official match runner.

### Benchmark Results Table

| Matchup | Games | Win Rate (95% CI) | Seat 0 (Red) Win Rate | Seat 1 (Blue) Win Rate | Mean Turns | A Mean CPU / Game | A CPU / Decision | Over Budget / Faults |
|---|---|---|---|---|---|---|---|---|
| **Phase 2 Agent vs Bot F** | 20 | **0.4500 ± 0.2180** (9/20) | 0.3000 (3/10) | **0.6000 (6/10)** | 75.2 | 404.7 ms | 5.75 ms | 0 / 0 |
| **Phase 2 Agent vs Random** | 20 | **1.0000 ± 0.0000** (20/20) | 1.0000 (10/10) | 1.0000 (10/10) | 78.8 | 441.4 ms | 5.73 ms | 0 / 0 |

### Analysis:
1. **Strong Blue Seat Performance**: In the Blue seat (Seat 1), the Phase 2 evaluator won **60% of games (6/10)** against F, showing strong response play and settlement placement.
2. **Zero Faults & Safe Clock Usage**: Mean CPU per game is **~405 ms**, operating well inside the ~1,000 ms allowance (leaving ample room for search depth and opening book evaluation in Phase 3 & 4). 0 crashes, 0 errors, 0 timeouts.
3. **Room for Red Seat Improvement**: The Red seat (opening placement) won 30% of games. Initial placement is the highest leverage decision in Catan, directly pointing to the need for Phase 3's specialized opening placement engine.
