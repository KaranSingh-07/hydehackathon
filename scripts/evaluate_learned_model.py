"""Evaluate Learned Value Model Agent in Competitive Match Play.

Compares the Linear and MLP learned value evaluators against Bot F across 40 games
under official rules, seed decoupling, and clock constraints.
"""
import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict

# Ensure project root is in sys.path
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from catan_kit.match import DEFAULT_KEY, dice_seed
from catan_kit.obs import Obs, masked_copy
from catanatron.game import Game
from catanatron.models.enums import Action
from catanatron.models.player import Color, RandomPlayer
from catanatron.state_functions import player_key

from my_team.agent import Agent
from my_team.evaluation.learned_evaluator import LearnedValueEvaluator
from my_team.search.selective_search import SelectiveSearchEngine
from my_team.search.budget_manager import BudgetManager, SearchConfig
from examples.f_agent import Agent as BotFAgent


def evaluate_learned_agent(
    model_type: str = "linear",
    num_seeds: int = 20,
    seed0: int = 990000,
    vps: int = 10,
) -> Dict:
    """Run 2 * num_seeds games of Learned Agent vs Bot F."""
    learned_evaluator = LearnedValueEvaluator(
        model_path="my_team/evaluation/learned_model.json",
        model_type=model_type,
    )
    search_engine = SelectiveSearchEngine(evaluator=learned_evaluator)
    agent = Agent(evaluator=learned_evaluator, search_engine=search_engine)

    seeds = [seed0 + i for i in range(num_seeds)]
    total_games = len(seeds) * 2
    wins_a, wins_b = 0, 0
    wins_red, wins_blue = 0, 0
    games_red, games_blue = 0, 0
    cpu_times = []
    latencies = []

    print(f"\n=======================================================")
    print(f" Competitive Evaluation: Learned Value Model ({model_type.upper()})")
    print(f" Opponent: Bot F | Games: {total_games} | First Seed: {seed0}")
    print(f" Model Loaded: {learned_evaluator.is_loaded}")
    print(f"=======================================================\n")

    for s_idx, board_seed in enumerate(seeds):
        for seat_order in [0, 1]:
            game_idx = s_idx * 2 + seat_order
            agent_first = (seat_order == 0)

            color_a = Color.RED if agent_first else Color.BLUE
            color_b = Color.BLUE if agent_first else Color.RED

            # Fast evaluation seats
            class AgentSeat(RandomPlayer):
                def __init__(self, color, ag):
                    super().__init__(color)
                    self.ag = ag
                    self.cpu_s = 0.0
                def decide(self, g, actions):
                    actions = list(actions)
                    if len(actions) == 1:
                        return actions[0]
                    view = masked_copy(g, self.color, self.mask_rng)
                    view.state.players = [RandomPlayer(c) for c in view.state.colors]
                    view.seed = 0
                    obs = Obs(view, self.color)
                    t0 = time.perf_counter()
                    idx = self.ag.decide(obs, actions)
                    lat = (time.perf_counter() - t0) * 1000.0
                    self.cpu_s += lat / 1000.0
                    latencies.append(lat)
                    return actions[idx]

            class FSeat(RandomPlayer):
                def __init__(self, color, bot):
                    super().__init__(color)
                    self.bot = bot
                def decide(self, g, actions):
                    actions = list(actions)
                    if len(actions) == 1:
                        return actions[0]
                    view = masked_copy(g, self.color, self.mask_rng)
                    view.state.players = [RandomPlayer(c) for c in view.state.colors]
                    view.seed = 0
                    obs = Obs(view, self.color)
                    idx = self.bot.decide(obs, actions)
                    return actions[idx]

            import random
            seat_a = AgentSeat(color_a, agent)
            seat_b = FSeat(color_b, BotFAgent())

            order = [seat_a, seat_b] if agent_first else [seat_b, seat_a]

            random.seed(board_seed)
            game = Game(order, seed=board_seed, vps_to_win=vps)
            random.seed(dice_seed(board_seed, s_idx, DEFAULT_KEY))

            s_game = (board_seed * 1_000_003 + game_idx) & 0xFFFFFFFF
            seat_a.mask_rng = random.Random(s_game ^ 0x5EED)
            seat_b.mask_rng = random.Random(s_game ^ 0xABCDEF)

            winner = game.play()
            cpu_times.append(seat_a.cpu_s * 1000.0)

            if agent_first:
                games_red += 1
                if winner == color_a:
                    wins_a += 1
                    wins_red += 1
                else:
                    wins_b += 1
            else:
                games_blue += 1
                if winner == color_a:
                    wins_a += 1
                    wins_blue += 1
                else:
                    wins_b += 1

    win_rate = wins_a / total_games
    wr_red = wins_red / games_red if games_red else 0.0
    wr_blue = wins_blue / games_blue if games_blue else 0.0
    mean_cpu = sum(cpu_times) / len(cpu_times) if cpu_times else 0.0
    latencies.sort()
    p95_lat = latencies[int(0.95 * len(latencies))] if latencies else 0.0

    print(f"\n Evaluation Results ({model_type.upper()}):")
    print(f" Win Rate: {win_rate*100:.1f}% ({wins_a}/{total_games})")
    print(f" Seat 0 (Red / First): {wr_red*100:.1f}% ({wins_red}/{games_red})")
    print(f" Seat 1 (Blue / Second): {wr_blue*100:.1f}% ({wins_blue}/{games_blue})")
    print(f" Mean CPU / Game: {mean_cpu:.1f} ms | p95 Latency: {p95_lat:.2f} ms\n")

    return {
        "model_type": model_type,
        "win_rate": round(win_rate, 4),
        "wins": wins_a,
        "games": total_games,
        "wr_red": round(wr_red, 4),
        "wr_blue": round(wr_blue, 4),
        "mean_cpu_ms": round(mean_cpu, 2),
        "p95_lat_ms": round(p95_lat, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate learned value model.")
    parser.add_argument("--model", type=str, default="linear", choices=["linear", "mlp"], help="Model type to evaluate")
    parser.add_argument("--seeds", type=int, default=20, help="Number of seeds (2x games)")
    args = parser.parse_args()

    evaluate_learned_agent(model_type=args.model, num_seeds=args.seeds)


if __name__ == "__main__":
    main()
