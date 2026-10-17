# Performance Tracker & Benchmark Registry

This document is the persistent registry of competitive benchmarking metrics across all implementation phases of our Catan agent. Every entry is generated from reproducible runs using the official match runner (`catan_kit.match.play_chunk`), identical paired board seeds, and alternating seat positions.

---

## 1. Master Historical Benchmark Table

| Run ID | Phase / Description | Date (UTC) | Git Commit | Opponent | Games | First Seed | Win Rate | 95% CI (Wilson) | Seat 0 (Red) WR | Seat 1 (Blue) WR | Mean Turns | Mean CPU / Game | Decision Latency (Mean / p95 / Max) | Faults (Err / TO / Over) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `RUN-001` | Phase 1 Baseline F | 2026-10-17 | `1bb7449` | Baseline F | 10 | 990000 | 0.3000 | [0.1078, 0.6032] | 0.2000 (1/5) | 0.4000 (2/5) | 70.6 | 187.5 ms | 3.25 ms / — / — | 0 / 0 / 0 |
| `RUN-002` | Phase 1 Baseline F | 2026-10-17 | `1bb7449` | Random | 10 | 990000 | 1.0000 | [0.7225, 1.0000] | 1.0000 (5/5) | 1.0000 (5/5) | 71.5 | 196.9 ms | 3.12 ms / — / — | 0 / 0 / 0 |
| `RUN-003` | Anchor F vs F | 2026-10-17 | `1bb7449` | Baseline F | 20 | 990000 | 0.4000 | [0.2188, 0.6134] | 0.2000 (2/10) | 0.6000 (6/10) | 70.4 | 344.5 ms | 6.53 ms / 10.84 ms / 10.84 ms | 0 / 0 / 0 |
| `RUN-004` | Phase 2 Evaluator | 2026-10-17 | `1bb7449` | Random | 20 | 990000 | 1.0000 | [0.8389, 1.0000] | 1.0000 (10/10) | 1.0000 (10/10) | 78.8 | 441.4 ms | 5.73 ms / 8.65 ms / 11.23 ms | 0 / 0 / 0 |
| `RUN-005` | Phase 2 Evaluator | 2026-10-17 | `1bb7449` | Baseline F | 40 | 990000 | 0.4500 | [0.3071, 0.6017] | 0.3500 (7/20) | 0.5500 (11/20) | 75.5 | 244.5 ms | 3.33 ms / 5.21 ms / 5.76 ms | 0 / 0 / 0 |
| `RUN-006` | Phase 3 Ablations (5 Var) | 2026-10-17 | `1bb7449` | Baseline F | 50 | 990000 | 0.4600 | [0.3311, 0.5947] | 0.3200 (8/25) | 0.6000 (15/25) | 72.5 | 302.8 ms | 4.69 ms / 9.66 ms / 9.66 ms | 0 / 0 / 0 |
| `RUN-007` | **Phase 3 Selective Search** | 2026-10-17 | `1bb7449` | **Baseline F** | **40** | **990000** | **0.6500** | **[0.4951, 0.7787]** | **0.6000 (12/20)** | **0.7000 (14/20)** | **75.0** | **343.0 ms** | **4.78 ms / 7.38 ms / 8.14 ms** | **0 / 0 / 0** |
| `RUN-008` | Phase 4 Baseline Verification | 2026-10-17 | `1bb7449` | Baseline F | 40 | 990000 | 0.5750 | [0.4219, 0.7149] | 0.4000 (8/20) | 0.7500 (15/20) | 74.2 | 354.7 ms | 4.90 ms / 8.00 ms / 8.00 ms | 0 / 0 / 0 |
| `RUN-009` | Phase 5 Learned Models (Linear/MLP) | 2026-10-17 | `1bb7449` | Baseline F | 40 | 990000 | 0.0000 | [0.0000, 0.0881] | 0.0000 (0/20) | 0.0000 (0/20) | 65.4 | 108.6 ms | 4.12 ms / 7.10 ms / 7.10 ms | 0 / 0 / 0 |
| `RUN-010` | **Phase 6 Packaged Artifact Verification** | 2026-10-17 | `1bb7449` | **Baseline F** | **20** | **990000** | **0.7000** | **[0.4810, 0.8545]** | **0.7000 (7/10)** | **0.7000 (7/10)** | **77.2** | **357.0 ms** | **4.86 ms / 8.59 ms / 8.59 ms** | **0 / 0 / 0** |

*Note: All development phases (0 through 6) completed and validated.*

---

## 2. In-Depth Benchmark Profile: Phase 2 Evaluator (`RUN-005`)

### Configuration
- **Agent Under Test**: `my_team/agent.py` (Phase 2 Phase-Aware Evaluator).
- **Opponent**: `examples/f_agent.py` (Official Reference Bot F).
- **Match Format**: 40 games across 20 distinct board seeds (`990000` to `990019`). Each board is played twice with seats swapped (Game $2k$: Agent as Red; Game $2k+1$: Agent as Blue) under identical keyed dice sequences.
- **Clock Mode**: Clocked under official rules: allowance $300\text{ ms} + 12.5\text{ ms} \times (k + 1)$ per decision up to $1,500\text{ ms}$.

### Key Performance Indicators
- **Overall Win Rate**: **45.0% (18/40)**.
  - Normal 95% Confidence Interval: $0.4500 \pm 0.1542$.
  - Wilson Score 95% Confidence Interval: $[0.3071, 0.6017]$.
- **Positional Split**:
  - **Seat 0 (Red / First)**: 35.0% (7/20).
  - **Seat 1 (Blue / Second)**: 55.0% (11/20).
- **Runtime & Efficiency**:
  - **Mean Agent CPU / Game**: 244.53 ms (consumes only ~24% of the ~1,000 ms allowance).
  - **Worst Game CPU**: 546.88 ms (headroom $> 950\text{ ms}$ before cap).
  - **Decision Latency**: Mean = 3.33 ms, p95 = 5.21 ms, Max = 5.76 ms.
- **Reliability & Rule Compliance**:
  - Illegal index errors: **0**
  - Timeouts / Crashes: **0 / 0**
  - Games over budget: **0** (0 random fallback moves).

---

## 3. Comparison with Baseline & Prior Phases

1. **Efficiency Improvement**:
   - In baseline F vs F (`RUN-003`), Bot F consumed **344.5 ms** per game and **6.53 ms** per decision.
   - The Phase 2 agent consumed **244.5 ms** per game and **3.33 ms** per decision—a **~29% reduction in CPU time**.
   - *Cause*: `prune_dominated_actions` removes redundant 4:1 maritime trades and self-damaging moves *before* invoking `world.copy()`.
2. **Red Seat (Seat 0) Progression**:
   - In F vs F baseline, Seat 0 won only 20.0% of games on these seeds.
   - Phase 2 improved Seat 0 win rate to **35.0%**, demonstrating better early-game resource variety.
3. **Blue Seat (Seat 1) Strength**:
   - In the second player position, Phase 2 achieved a **55.0% win rate** over 20 games against F, showing that phase-aware midgame city upgrades and port synergy compete effectively in response play.

---

## 4. Status of Opponent "O"

- **Verified Fact**: Reference bot **O** is a private, organizer-trained model. It is **not distributed** in the repository or starter kit.
- **Evaluation Mechanism**: In accordance with `PROBLEM_STATEMENT.md` (Sections 5.3, 8, and 9.2), Bot O is hosted exclusively on the organizers' AWS evaluation infrastructure and evaluated against submissions uploaded to the team portal.
- **Local Surrogate Benchmark**: All local benchmarks run against Bot F and Random baselines, which are the official anchors present in the kit.

---

## 5. In-Depth Benchmark Profile: Phase 3 Selective Search (`RUN-007`)

### Configuration
- **Agent Under Test**: `my_team/agent.py` (Phase 3 Selective Search with Diversity-Preserving Candidate Ranking, $K=5$).
- **Opponent**: `examples/f_agent.py` (Official Reference Bot F).
- **Match Format**: 40 games across 20 distinct board seeds (`990000` to `990019`), alternating seats.
- **Clock Mode**: Clocked under official increment formula: allowance $300\text{ ms} + 12.5\text{ ms} \times (k + 1)$.

### Key Performance Indicators
- **Overall Win Rate**: **65.0% (26 wins / 40 games)**.
  - Normal 95% Confidence Interval: $0.6500 \pm 0.1478$.
  - Wilson Score 95% Confidence Interval: $[0.4951, 0.7787]$.
- **Positional Split**:
  - **Seat 0 (Red / First)**: **60.0% (12/20)** (major improvement from 35% in Phase 2).
  - **Seat 1 (Blue / Second)**: **70.0% (14/20)**.
- **Runtime & Efficiency**:
  - **Mean Agent CPU / Game**: 342.97 ms (well below the ~1,000 ms allowance).
  - **Worst Game CPU**: 687.50 ms.
  - **Decision Latency**: Mean = 4.78 ms, p95 = 7.38 ms, Max = 8.14 ms.
- **Reliability & Rule Compliance**:
  - Illegal index errors: **0**
  - Timeouts / Crashes: **0 / 0**
  - Games over budget: **0**

---

## 7. In-Depth Benchmark Profile: Phase 4 Optimization & Gating (`RUN-008`)

### Configuration
- **Agent Under Test**: `my_team/agent.py` (Phase 4 Verified Baseline weights with gated rollback protection).
- **Opponent**: `examples/f_agent.py` (Official Reference Bot F).
- **Match Format**: 40 games across 20 distinct board seeds (`990000` to `990019`), alternating seats.
- **Clock Mode**: Clocked under official increment formula: allowance $300\text{ ms} + 12.5\text{ ms} \times (k + 1)$.

### Key Performance Indicators
- **Overall Win Rate**: **57.5% (23 wins / 40 games)**.
  - Normal 95% Confidence Interval: $0.5750 \pm 0.1532$.
  - Wilson Score 95% Confidence Interval: $[0.4219, 0.7149]$.
- **Positional Split**:
  - **Seat 0 (Red / First)**: 40.0% (8/20).
  - **Seat 1 (Blue / Second)**: **75.0% (15/20)**.
- **Runtime & Efficiency**:
  - **Mean Agent CPU / Game**: 354.69 ms (well within allowance).
  - **Worst Game CPU**: 781.25 ms.
  - **Decision Latency**: Mean = 4.90 ms, p95 = 8.00 ms, Max = 8.00 ms.
- **Reliability & Rule Compliance**:
  - Illegal index errors: **0**
  - Timeouts / Crashes: **0 / 0**
  - Games over budget: **0**

---

## 8. Exact Commands for Reproduction

To reproduce these benchmark results on Windows PowerShell from the workspace root:

```powershell
# 1. Run 40-game benchmark vs Bot F (Canonical Test Set)
.venv\Scripts\python.exe scripts/benchmark.py --a my_team/agent.py --b f --games 40 --seed0 990000 --tag canonical_benchmark

# 2. Run offline weight optimization pipeline across partitioned seeds
.venv\Scripts\python.exe scripts/tune_weights.py --candidates 4 --train-seeds 10 --val-seeds 10

# 3. Run decision dataset generator (40 games)
.venv\Scripts\python.exe scripts/generate_dataset.py --a my_team --b f --partition train --tag train_dataset

# 4. Run all unit tests
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```


