"""Game-Data and Strategic Decision Dataset Generator.

Generates reproducible decision-level and game-level datasets using the official
game mechanics, masked observations, decoupled dice seeding (HMAC), and agent interfaces.

Supports:
  - Current agent vs Baseline F
  - Current agent vs Random
  - Current agent self-play
  - Baseline F vs Baseline F
  - Multiple configurable seeds (train, val, test, or custom)
  - Alternating player seats for unbiased position balance
  - Decision-level feature extraction, action representations, and final game outcomes
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from catan_kit._patch import ensure_hash_seed
from catan_kit.match import DEFAULT_KEY, dice_seed
from catan_kit.obs import Obs, masked_copy
from catanatron.game import Game
from catanatron.models.enums import Action, ActionType
from catanatron.models.player import Color, RandomPlayer
from catanatron.state_functions import player_key

from my_team.evaluation.evaluator import detect_game_phase
from my_team.evaluation.features import extract_strategic_features


SCHEMA_VERSION = "1.0.0"


def load_seeds_from_partition(partition_name: str, config_path: str = "configs/seed_partitions.json") -> List[int]:
    """Load seeds from partitioned configuration file."""
    p = Path(config_path)
    if not p.is_file():
        # Fallback default seeds
        return [100000 + i for i in range(10)]
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get(f"{partition_name}_seeds", [100000 + i for i in range(10)])


def get_agent_instance(agent_name: str, color: Color):
    """Instantiate agent instance given identifier."""
    name = agent_name.lower()
    if name in ("my_team", "agent", "current"):
        from my_team.agent import Agent
        return Agent()
    elif name in ("f", "baseline_f"):
        from examples.f_agent import Agent
        return Agent()
    elif name in ("random", "rand"):
        from examples.random_agent import Agent
        return Agent()
    else:
        raise ValueError(f"Unknown agent type: {agent_name}")


class DecisionCollectorPlayer(RandomPlayer):
    """Player proxy that captures decisions, features, and timing."""

    def __init__(self, color: Color, agent_instance, player_label: str):
        super().__init__(color)
        self.agent = agent_instance
        self.player_label = player_label
        self.decisions_log: List[Dict[str, Any]] = []
        self.total_decisions = 0
        self.total_cpu_time = 0.0

    def begin_game(self, seed: int):
        self.mask_rng = random.Random(seed ^ 0x5EED)
        self.agent_rng = random.Random(seed)
        if hasattr(self.agent, "rng"):
            self.agent.rng = self.agent_rng
        self.decisions_log = []
        self.total_decisions = 0
        self.total_cpu_time = 0.0

    def decide(self, game: Game, playable_actions: List[Action]) -> Action:
        actions = list(playable_actions)
        if len(actions) == 1:
            return actions[0]

        # Generate masked observation matching competition environment
        view = masked_copy(game, self.color, self.mask_rng)
        view.state.players = [RandomPlayer(c) for c in view.state.colors]
        view.seed = 0
        obs = Obs(view, self.color)

        # Extract features and state context
        phase = detect_game_phase(view, self.color)
        features = extract_strategic_features(view, self.color)

        t_start = time.perf_counter()
        try:
            chosen_idx = self.agent.decide(obs, actions)
            if not isinstance(chosen_idx, int) or not (0 <= chosen_idx < len(actions)):
                chosen_idx = 0
        except Exception:
            chosen_idx = 0
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        chosen_action = actions[chosen_idx]
        self.total_decisions += 1
        self.total_cpu_time += latency_ms / 1000.0

        # Record decision record
        rec = {
            "turn": int(game.state.num_turns),
            "phase": phase.value,
            "player_label": self.player_label,
            "color": self.color.name,
            "num_actions": len(actions),
            "action_type": chosen_action.action_type.name,
            "action_value": str(chosen_action.value),
            "chosen_index": chosen_idx,
            "features": features,
            "latency_ms": round(latency_ms, 3),
        }
        self.decisions_log.append(rec)
        return chosen_action


def run_dataset_generation(
    agent_a_name: str,
    agent_b_name: str,
    seeds: List[int],
    output_dir: str = "results/datasets",
    tag: str = "dataset",
    vps: int = 10,
) -> str:
    """Generate game dataset across specified seeds and matchups."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(output_dir, f"decisions_{tag}_{timestamp}.jsonl")

    print(f"\n=======================================================")
    print(f" Catan Dataset Generation: {tag}")
    print(f" Matchup: {agent_a_name} (A) vs {agent_b_name} (B)")
    print(f" Board Seeds: {len(seeds)} | Games: {len(seeds) * 2}")
    print(f" Target File: {out_file}")
    print(f"=======================================================\n")

    total_decisions_written = 0
    total_games = len(seeds) * 2
    games_completed = 0
    wins_a, wins_b, draws = 0, 0, 0

    with open(out_file, "w", encoding="utf-8") as f_out:
        # Write metadata header as first line
        header = {
            "schema_version": SCHEMA_VERSION,
            "timestamp": timestamp,
            "agent_a": agent_a_name,
            "agent_b": agent_b_name,
            "vps_to_win": vps,
            "total_games": total_games,
            "seeds": seeds,
        }
        f_out.write(json.dumps({"_metadata": header}) + "\n")

        for s_idx, board_seed in enumerate(seeds):
            for seat_order in [0, 1]:
                # seat_order == 0: A is Red (First), B is Blue (Second)
                # seat_order == 1: B is Red (First), A is Blue (Second)
                game_idx = s_idx * 2 + seat_order
                a_first = (seat_order == 0)

                color_a = Color.RED if a_first else Color.BLUE
                color_b = Color.BLUE if a_first else Color.RED

                player_a = DecisionCollectorPlayer(color_a, get_agent_instance(agent_a_name, color_a), "A")
                player_b = DecisionCollectorPlayer(color_b, get_agent_instance(agent_b_name, color_b), "B")

                order = [player_a, player_b] if a_first else [player_b, player_a]

                # Decoupled dice seeding matching official harness
                random.seed(board_seed)
                game = Game(order, seed=board_seed, vps_to_win=vps)
                random.seed(dice_seed(board_seed, s_idx, DEFAULT_KEY))

                s_game = (board_seed * 1_000_003 + game_idx) & 0xFFFFFFFF
                player_a.begin_game(s_game)
                player_b.begin_game(s_game ^ 0xABCDEF)

                # Play game to completion
                winning_color = game.play()
                games_completed += 1

                # Extract final scores
                p0_key = player_key(game.state, Color.RED)
                p1_key = player_key(game.state, Color.BLUE)
                vps_red = int(game.state.player_state.get(f"{p0_key}_ACTUAL_VICTORY_POINTS", 0))
                vps_blue = int(game.state.player_state.get(f"{p1_key}_ACTUAL_VICTORY_POINTS", 0))

                vps_a = vps_red if a_first else vps_blue
                vps_b = vps_blue if a_first else vps_red

                if winning_color == color_a:
                    winner_label = "A"
                    wins_a += 1
                elif winning_color == color_b:
                    winner_label = "B"
                    wins_b += 1
                else:
                    winner_label = "DRAW"
                    draws += 1

                # Backpropagate final outcomes into decision records
                for d in player_a.decisions_log:
                    d["game_id"] = game_idx
                    d["board_seed"] = board_seed
                    d["a_first"] = a_first
                    d["seat"] = 0 if a_first else 1
                    d["final_vps_me"] = vps_a
                    d["final_vps_opp"] = vps_b
                    d["vps_delta"] = vps_a - vps_b
                    d["outcome"] = "WIN" if winner_label == "A" else ("LOSS" if winner_label == "B" else "DRAW")
                    d["is_win"] = 1 if winner_label == "A" else 0
                    d["total_game_turns"] = int(game.state.num_turns)
                    f_out.write(json.dumps(d) + "\n")
                    total_decisions_written += 1

                for d in player_b.decisions_log:
                    d["game_id"] = game_idx
                    d["board_seed"] = board_seed
                    d["a_first"] = a_first
                    d["seat"] = 1 if a_first else 0
                    d["final_vps_me"] = vps_b
                    d["final_vps_opp"] = vps_a
                    d["vps_delta"] = vps_b - vps_a
                    d["outcome"] = "WIN" if winner_label == "B" else ("LOSS" if winner_label == "A" else "DRAW")
                    d["is_win"] = 1 if winner_label == "B" else 0
                    d["total_game_turns"] = int(game.state.num_turns)
                    f_out.write(json.dumps(d) + "\n")
                    total_decisions_written += 1

            if (s_idx + 1) % 5 == 0 or (s_idx + 1) == len(seeds):
                print(f" Completed {s_idx + 1}/{len(seeds)} board seeds ({games_completed} games). Wins A: {wins_a}, Wins B: {wins_b}, Decisions: {total_decisions_written}")

    print(f"\n Dataset generation finished:")
    print(f" Total Games: {games_completed} (A wins: {wins_a}, B wins: {wins_b}, Draws: {draws})")
    print(f" Total Decisions Stored: {total_decisions_written}")
    print(f" Output: {out_file}\n")
    return out_file


def main():
    parser = argparse.ArgumentParser(description="Generate Catan game dataset.")
    parser.add_argument("--a", type=str, default="my_team", help="Agent A ('my_team', 'f', 'random')")
    parser.add_argument("--b", type=str, default="f", help="Agent B ('my_team', 'f', 'random')")
    parser.add_argument("--partition", type=str, default="train", choices=["train", "val", "test", "stress"], help="Seed partition")
    parser.add_argument("--seeds", type=int, default=None, help="Override number of seeds to run from partition")
    parser.add_argument("--tag", type=str, default="dataset", help="Dataset tag")
    parser.add_argument("--outdir", type=str, default="results/datasets", help="Output directory")

    args = parser.parse_args()
    seed_list = load_seeds_from_partition(args.partition)
    if args.seeds is not None:
        seed_list = seed_list[:args.seeds]

    run_dataset_generation(
        agent_a_name=args.a,
        agent_b_name=args.b,
        seeds=seed_list,
        output_dir=args.outdir,
        tag=f"{args.tag}_{args.partition}",
    )


if __name__ == "__main__":
    main()
