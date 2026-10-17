# Catan Agent Repository Audit

## 1. Executive Summary & Verification Context

This document provides a comprehensive systems and algorithmic audit of the Catan Agent Hackathon starter kit. All statements and specifications herein have been verified directly against the active repository source code, the pinned `catanatron` engine (commit `d3f4ad05bb78d8b2309631d6d3cfa8fcb6fda816`), and test executions in the Python 3.11 environment.

### Verification Status
- **Repository Commit**: `1bb7449e4da4ec666882d16c4c4c26c3da9955e4` (`origin/main`).
- **Python Runtime**: Python 3.11.15 running on Windows (`.venv`). Reference scoring machine: AWS `c7a.48xlarge` (AMD EPYC 9R14), Linux, Python 3.11.
- **Engine Build**: `catanatron==3.3.0` installed from GitHub commit `d3f4ad05bb78d8b2309631d6d3cfa8fcb6fda816`.
- **Pre-flight Checks**: `python -m catan_kit.selfcheck` executed and **PASSED** (100% checks green: environment, engine action set, determinism across seeds, example agent metrics).
- **Baseline Evaluation**: `python -m catan_kit.evaluate --a examples/f_agent.py --b examples/random_agent.py --games 5` executed and **PASSED** (F win rate 1.0000, 0 errors, 0 timeouts).

---

## 2. Codebase Structure & File Inventory

```
hyde_ai_hackathon/
├── .git/                     # Git metadata (branch: main)
├── .gitignore                # Basic ignore file
├── .venv/                    # Python 3.11.15 virtual environment (uv managed)
├── catan_kit/                # Competition SDK and execution harness
│   ├── __init__.py           # Applies determinism patch at import
│   ├── _patch.py             # Fixes maritime trade sorting & feature ordering RNG leak
│   ├── evaluate.py           # Official scoring match runner and CLI
│   ├── host.py               # Child process agent host, IPC protocol, and JSON serialization
│   ├── match.py              # OS process isolation, chess clock, cgroups, seat management
│   ├── obs.py                # Observation view (Obs) and world-sampling (masked_copy)
│   ├── sandbox.py            # Linux cgroup v2 containerization and core-pinning
│   └── selfcheck.py          # Pre-flight environment and determinism validator
├── examples/                 # Baseline agent implementations
│   ├── f_agent.py            # Reference bot F (1-ply heuristic search with 13 features)
│   └── random_agent.py       # Uniform random legal move selector
├── docs/                     # Engineering blueprints and technical contracts
├── PROBLEM_STATEMENT.md      # Official hackathon specification, scoring rules, constraints
├── README.md                 # Setup, submission and team machine instructions
└── requirements.txt          # Pinned dependency manifest
```

---

## 3. Detailed Component Inspections

### 3.1 Competition Entry Point & Agent Interface
- **Required File**: `agent.py` in the root of the submitted folder.
- **Required Class**: `class Agent:`.
- **Lifecycle Methods**:
  1. `__init__(self)`:
     - Called once when the agent child process starts.
     - Timeout: `INIT_TIMEOUT_S = 60.0` seconds.
     - **Not charged against any game clock**.
     - Used for loading weights, initializing tables, caching static graphs, and warming up PyTorch/NumPy.
  2. `decide(self, obs: Obs, playable_actions: List[Action]) -> int`:
     - Called for each decision where `len(playable_actions) > 1`.
     - Input: `obs` (read-only `Obs` instance) and `playable_actions` (list of legal `Action` namedtuples).
     - Output: Must return an `int` (or object supporting `operator.index()`, like `numpy.int64`, strictly non-boolean) corresponding to an index in `playable_actions`: `0 <= return_value < len(playable_actions)`.
     - Wall-clock hard timeout: `CALL_TIMEOUT_S = 3.0` seconds per decision.
     - Global wall slack: `WALL_SLACK_S = 3.0` seconds beyond CPU allowance per game.

### 3.2 Observation Model & Information Partitioning
Defined in `catan_kit/obs.py`. Information is rigorously partitioned:
- **Public & Symmetrical**:
  - `obs.board`: `catanatron.models.board.Board` containing:
    - `board.buildings`: `{node_id: (color, building_type)}` (all settlements and cities).
    - `board.roads`: `{(node_u, node_v): color}` (all built roads).
    - `board.robber_coordinate`: current coordinate `(x, y, z)` of the robber.
    - `board.map`: static geometry, 19 tiles, numbers, dice probabilities, 9 ports.
  - `obs.turn`: Current turn number (`st.num_turns`).
  - `obs.bank`: `{resource_name: count}` for each of `WOOD`, `BRICK`, `SHEEP`, `WHEAT`, `ORE`.
  - `obs.my_public_vps`: Visible victory points of self.
  - `obs.opp_public_vps`: Visible victory points of opponent.
  - `obs.played_devs`: Played development cards for both players:
    - `obs.played_devs["me"]`: `{dev_type: count}`
    - `obs.played_devs["opp"]`: `{dev_type: count}`
  - `obs.cpu_left`: Float seconds of CPU remaining in the current game allowance (or `None` if unclocked).
  - `obs.me`: Agent's seat color (`Color.RED` or `Color.BLUE`).
- **Private but Derivable**:
  - `obs.opp_resources`: Exact resource cards in opponent hand.
    - *Derivation*: There are exactly 19 cards per resource type. Total cards = 19.
    - `opp_count = 19 - bank_count - my_count`. Because trading between players is disabled, the bank and own hand uniquely determine the opponent's resource hand.
- **Truly Hidden Information**:
  - `obs.opp_dev_count`: Agent only sees the *total count* of unplayed development cards held by the opponent (`sum(ps[f"P{j}_{d}_IN_HAND"])`).
  - Agent **does not see** which development cards the opponent holds unplayed (specifically hidden VP cards).
  - Agent **does not see** the remaining development deck order.
- **World Sampling Mechanism**:
  - `obs.sample_world(rng)`: Calls `catan_kit.obs.masked_copy(game, me, rng)`.
  - Collects unplayed opponent dev cards and the remaining deck into a single multiset `unseen`.
  - Sorts `unseen` deterministically (to prevent seed inversion attacks).
  - Shuffles `unseen` with `rng` and partitions into `n_opp` cards for opponent hand and the remainder for the deck.
  - Masks `action_records` so opponent's historical `BUY_DEVELOPMENT_CARD` entries have `result=None`.
  - Returns an independent `catanatron.game.Game` instance that is legal, complete, and fully simulatable.

### 3.3 Action Space & Representation
Actions are instances of `catanatron.models.enums.Action = namedtuple("Action", ["color", "action_type", "value"])`.
- Legal action categories:
  1. `ROLL`: `value = None`.
  2. `END_TURN`: `value = None`.
  3. `BUILD_ROAD`: `value = (node_u, node_v)` where `node_u < node_v` are integer node IDs (0..53).
  4. `BUILD_SETTLEMENT`: `value = node_id` (integer 0..53).
  5. `BUILD_CITY`: `value = node_id` (integer 0..53).
  6. `BUY_DEVELOPMENT_CARD`: `value = None`.
  7. `PLAY_KNIGHT_CARD`: `value = None`.
  8. `PLAY_ROAD_BUILDING`: `value = None`.
  9. `PLAY_YEAR_OF_PLENTY`: `value = (resource1, resource2)`.
  10. `PLAY_MONOPOLY`: `value = resource` (e.g. `"WHEAT"`).
  11. `MOVE_ROBBER`: `value = (coordinate, victim_color_or_None)`.
  12. `DISCARD_RESOURCE`: `value = resource` (e.g. `"SHEEP"`).
  13. `MARITIME_TRADE`: `value = (give1, give2, give3_or_None, give4_or_None, receive)`.
- **Forced Moves**:
  - In `catan_kit/match.py` (`Seat.decide`), if `len(playable_actions) == 1`, the move is executed immediately by the harness without IPC and without calling the agent.
  - Forced moves account for ~47% of all game decisions (e.g. single roll, single road building sub-action, forced single-resource discard).
  - They cost 0 agent CPU and do not decrement the clock.

### 3.4 Official Rules, Mechanics & Differences from Tabletop Catan
- **2-Player Format**: Standard Catan board (19 land tiles, spiral letters A..R, official robber placement on Desert).
- **Target VPs**: 10 victory points to win.
- **Domestic Trading**: Strictly disabled. `OFFER_TRADE`, `ACCEPT_TRADE`, etc., are never generated.
- **Bank & Port Trades**:
  - 4:1 trade default with the bank.
  - 3:1 generic port trade.
  - 2:1 resource-specific port trade.
- **Development Cards**:
  - 25 cards total: 14 Knights, 5 Victory Points, 2 Road Building, 2 Year of Plenty, 2 Monopoly.
  - A card bought this turn cannot be played on the same turn (unless it is a VP card triggered at game end).
  - At most one development card played per turn.
- **Longest Road & Largest Army**:
  - Longest Road: >= 5 continuous edge segments, worth 2 VPs.
  - Largest Army: >= 3 Knight cards played, worth 2 VPs.
- **Resource Depletion**:
  - Bank has 19 cards per resource type. If resource yields exceed bank supply during production, catanatron handles allocation per standard rules.
- **Game Termination**:
  - Evaluated turn limit: 1,000 turns (`catan_kit/evaluate.py`). Games reaching 1,000 turns without a winner count as a loss for both players.

### 3.5 Execution Harness & Time Budgeting Mechanics
- **Process Isolation**:
  - Agent executes in a separate process spawned via `multiprocessing.get_context("spawn")`.
  - Child runs `catan_kit.host.serve`, redirecting `stdout`/`stderr` to `/dev/null` on the official machine.
  - Memory limited to 2 GB (`MEMORY_BYTES = 2 * 1024**3`). On Linux, enforced via cgroup v2 (`memory.max`) or `resource.setrlimit`.
  - Single CPU core pinned (`cpuset.cpus`). Worker threads in Python/PyTorch share this single core.
- **Chess Clock Allowance Formula**:
  - Base allowance: `CPU_BASE_S = 0.300` s (300 ms).
  - Increment per decision: `CPU_PER_DECISION_S = 0.0125` s (12.5 ms).
  - Cap: `CPU_MAX_S = 1.5` s (1,500 ms).
  - Formula after $k$ decisions:
    $$\text{allowance}(k) = \min(0.300 + 0.0125 \times (k + 1), 1.5)\text{ seconds}$$
  - Average game length: ~56 decisions per agent $\implies \approx 1.0$ s total compute budget per game.
  - The opening placement (which has the largest branching factor, ~30–54 settlement locations and ~54–100 road directions) can draw from the initial 300 ms pool.
- **Timeout & Over-Budget Consequences**:
  - If agent CPU exceeds allowance, `Seat.decide` marks `self.over = "cpu"` and plays **uniform random moves** for the remainder of that game.
  - If a single decision takes > 3.0 s wall time, agent is terminated, restarted, and finished with random moves.
  - More than 2% illegal indices or exceptions leads to submission disqualification.

### 3.6 Determinism & Reproducibility Guarantees
`catan_kit` addresses three major reproducibility vulnerabilities (`catan_kit/_patch.py`):
1. **Trade Offer Memory Address Hash**: `inner_maritime_trade_possibilities` originally used sets containing `None`. Python $\le$ 3.11 computes `hash(None)` based on memory address. Patched in `_patch.py` to sort offers deterministically.
2. **String Hash Randomization**: String iteration in sets randomized across processes. Controlled by forcing `PYTHONHASHSEED=0` via self-re-execution.
3. **Feature Ordering RNG Poisoning**: `catanatron.features.get_feature_ordering` initialized a dummy unseeded `Game`, poisoning `random.getstate()`. Patched with `_protect_feature_ordering` to save and restore RNG state.
4. **Dice Decoupling**: Dice rolls are generated from a keyed HMAC (`dice_seed(board_seed, pair_index, key)`), preventing agents from predicting dice from board seeds.

---

## 4. Current Baseline Agents

### 4.1 Random Agent (`examples/random_agent.py`)
- Selects uniformly from `playable_actions`.
- Win rate vs F: 0.001 (~0.0%).
- CPU per game: ~11 ms.

### 4.2 Bot F (`examples/f_agent.py`)
- Represents catanatron's `ValueFunctionPlayer`.
- **Strategy**:
  1. Calls `obs.sample_world(self.rng)` **once** to obtain a plausible `Game` instance.
  2. For every legal action in `playable_actions`, creates a full copy of the world: `successor = world.copy()`.
  3. Executes action on the copy: `successor.execute(action)`.
  4. Evaluates the resulting state using `VALUE_FN = get_value_fn("base_fn", DEFAULT_WEIGHTS)`.
  5. Selects the action yielding maximum heuristic value.
- **Features Used (13 Heuristic Values)**:
  1. `public_vps` ($3 \times 10^{14}$): Dominates all other features for game-winning states.
  2. `production` ($10^8$): Sum of production probabilities across 5 resources + variety bonus ($4 \times 2.778\%$).
  3. `enemy_production` ($-10^8$): Negative sum of enemy production probabilities.
  4. `reachable_production_0` ($0$): Direct road-reachable production at distance 0.
  5. `reachable_production_1` ($10^4$): Road-reachable production at distance 1.
  6. `buildable_nodes` ($10^3$): Count of available buildable settlement nodes.
  7. `hand_synergy` ($10^2$): Distance metric towards City (Wheat/Ore) and Settlement (Wood/Brick/Sheep/Wheat).
  8. `longest_road` ($10$): Weighted if buildable nodes == 0.
  9. `hand_devs` ($10$): Count of unplayed dev cards in hand.
  10. `army_size` ($10.1$): Played knights.
  11. `num_tiles` ($1$): Blockability / number of unique adjacent tiles.
  12. `hand_resources` ($1$): Number of resources held.
  13. `discard_penalty` ($-5$): Penalty applied if holding $> 7$ resources.
- **Weaknesses of F**:
  - **Single World Sample**: F samples exactly one world, ignoring dev card uncertainty.
  - **1-Ply Horizon**: Blind to multi-step planning (e.g. road-settlement combos, dev card chaining).
  - **No Expectimax / Chance Node Search**: Does not evaluate probabilistic outcomes of dice rolls or robber steals.
  - **Heavy Deep Copies**: `world.copy()` copies entire board, maps, dictionaries, and graph subgraphs for every legal move (up to 40+ moves), burning CPU on Python object allocation rather than intelligent search.

### 4.3 Reference Bot O
- The organizer's trained neural / heuristic agent that decisively beats F.
- Serves as the high-tier anchor on both public and private boards.

---

## 5. Official Verification Results

All tests were executed on the active workspace using the newly created Python 3.11 virtual environment:

1. **`python -m catan_kit.selfcheck`**:
   - Python version: 3.11.15 [OK]
   - Catanatron version: 3.3.0 [OK]
   - Networkx version: 3.6.1 [OK]
   - Determinism patch: Applied [OK]
   - PYTHONHASHSEED: 0 [OK]
   - Pinned Action Set: 18 actions matching [OK]
   - Determinism repeatability: Seed 4256 produced ('Color.RED', 111) x4 [OK]
   - Agent benchmark: `f_agent` beat `random_agent` (win rate 1.00, 0 errors, 455 ms CPU) [OK]

2. **Mini-Benchmark (`catan_kit.evaluate` 5 games)**:
   - Command: `.venv\Scripts\python.exe -m catan_kit.evaluate --a examples/f_agent.py --b examples/random_agent.py --games 5`
   - A (F) win rate: 1.0000 (5/5).
   - Mean CPU per game: 418.8 ms.
   - Games over budget: 0. Errors: 0. Timeouts: 0. Load time: 0.00 s.
