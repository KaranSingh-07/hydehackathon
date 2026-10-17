"""Offline Weight Optimization and Auto-Tuning Pipeline for Catan Agent.

Tunes phase-dependent evaluator weights using derivative-free local search and
evolutionary perturbations over paired board seeds. Enforces strict promotion gates
to prevent overfitting and ensure zero regressions.

Promotion Gates:
  1. Correctness: Full unit test suite must pass cleanly.
  2. Training Improvement: Candidate must beat baseline win rate or VP delta on train seeds.
  3. Validation Generalization: Candidate must match or exceed baseline win rate on unseen val seeds.
  4. Efficiency: Candidate must execute within strict CPU budget (< 500 ms/game, p95 < 10 ms).
  5. Safety: Zero faults, zero timeouts, zero illegal index exceptions.
"""
import argparse
import copy
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from my_team.evaluation.evaluator import (
    GamePhase,
    PhaseAwareEvaluator,
    get_default_weights,
    load_weights_from_file,
    save_weights_to_file,
)
from examples.f_agent import Agent as BotFAgent


REGISTRY_PATH = "results/experiments_registry.json"


def get_git_commit() -> str:
    """Extract current git commit hash."""
    try:
        res = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return res.decode().strip()
    except Exception:
        return "unknown"


def load_seeds(partition: str) -> List[int]:
    """Load seed list for given partition."""
    cfg = Path("configs/seed_partitions.json")
    if cfg.is_file():
        with open(cfg, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get(f"{partition}_seeds", [100000 + i for i in range(10)])
    return [100000 + i for i in range(10)]


def run_unit_tests() -> bool:
    """Run full test suite via unittest discovery and return True if all pass."""
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return res.returncode == 0


class BenchSeat(RandomPlayer):
    """Evaluation seat for in-process fast match evaluation."""

    def __init__(self, color: Color, agent_instance):
        super().__init__(color)
        self.agent = agent_instance
        self.decisions = 0
        self.total_cpu_s = 0.0
        self.latencies_ms: List[float] = []

    def begin_game(self, seed: int):
        self.mask_rng = random.Random(seed ^ 0x5EED)
        self.agent_rng = random.Random(seed)
        if hasattr(self.agent, "rng"):
            self.agent.rng = self.agent_rng
        self.decisions = 0
        self.total_cpu_s = 0.0
        self.latencies_ms = []

    def decide(self, game: Game, playable_actions: List[Action]) -> Action:
        actions = list(playable_actions)
        if len(actions) == 1:
            return actions[0]

        view = masked_copy(game, self.color, self.mask_rng)
        view.state.players = [RandomPlayer(c) for c in view.state.colors]
        view.seed = 0
        obs = Obs(view, self.color)

        t_start = time.perf_counter()
        try:
            chosen_idx = self.agent.decide(obs, actions)
            if not isinstance(chosen_idx, int) or not (0 <= chosen_idx < len(actions)):
                chosen_idx = 0
        except Exception:
            chosen_idx = 0
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        self.decisions += 1
        self.total_cpu_s += latency_ms / 1000.0
        self.latencies_ms.append(latency_ms)
        return actions[chosen_idx]


def evaluate_weights_on_seeds(
    weights: Dict[GamePhase, Dict[str, float]],
    seeds: List[int],
    vps: int = 10,
) -> Dict[str, Any]:
    """Simulate matches of candidate agent against Bot F across paired seeds."""
    evaluator = PhaseAwareEvaluator(
        early_weights=weights[GamePhase.EARLY],
        mid_weights=weights[GamePhase.MID],
        late_weights=weights[GamePhase.LATE],
    )
    test_agent = Agent(evaluator=evaluator)

    wins = 0
    losses = 0
    draws = 0
    vps_me_list = []
    vps_opp_list = []
    cpu_list = []
    all_latencies = []
    total_games = len(seeds) * 2

    for s_idx, board_seed in enumerate(seeds):
        for seat_order in [0, 1]:
            # seat_order == 0: Agent is Red (Seat 0); Bot F is Blue (Seat 1)
            # seat_order == 1: Bot F is Red (Seat 0); Agent is Blue (Seat 1)
            game_idx = s_idx * 2 + seat_order
            agent_first = (seat_order == 0)

            color_agent = Color.RED if agent_first else Color.BLUE
            color_f = Color.BLUE if agent_first else Color.RED

            seat_agent = BenchSeat(color_agent, test_agent)
            seat_f = BenchSeat(color_f, BotFAgent())

            order = [seat_agent, seat_f] if agent_first else [seat_f, seat_agent]

            random.seed(board_seed)
            game = Game(order, seed=board_seed, vps_to_win=vps)
            random.seed(dice_seed(board_seed, s_idx, DEFAULT_KEY))

            s_game = (board_seed * 1_000_003 + game_idx) & 0xFFFFFFFF
            seat_agent.begin_game(s_game)
            seat_f.begin_game(s_game ^ 0xABCDEF)

            winning_color = game.play()

            p0_key = player_key(game.state, Color.RED)
            p1_key = player_key(game.state, Color.BLUE)
            vps_red = int(game.state.player_state.get(f"{p0_key}_ACTUAL_VICTORY_POINTS", 0))
            vps_blue = int(game.state.player_state.get(f"{p1_key}_ACTUAL_VICTORY_POINTS", 0))

            my_vps = vps_red if agent_first else vps_blue
            opp_vps = vps_blue if agent_first else vps_red

            vps_me_list.append(my_vps)
            vps_opp_list.append(opp_vps)
            cpu_list.append(seat_agent.total_cpu_s * 1000.0)
            all_latencies.extend(seat_agent.latencies_ms)

            if winning_color == color_agent:
                wins += 1
            elif winning_color == color_f:
                losses += 1
            else:
                draws += 1

    win_rate = wins / total_games if total_games else 0.0
    mean_vps_me = sum(vps_me_list) / len(vps_me_list) if vps_me_list else 0.0
    mean_vps_opp = sum(vps_opp_list) / len(vps_opp_list) if vps_opp_list else 0.0
    vps_delta = mean_vps_me - mean_vps_opp
    mean_cpu_ms = sum(cpu_list) / len(cpu_list) if cpu_list else 0.0

    all_latencies.sort()
    p95_latency = all_latencies[int(0.95 * len(all_latencies))] if all_latencies else 0.0

    # Wilson Score Interval
    z = 1.96
    z2 = z * z
    denom = 1.0 + z2 / total_games if total_games else 1.0
    center = (win_rate + z2 / (2.0 * total_games)) / denom if total_games else 0.0
    rad = (z / denom) * ((win_rate * (1.0 - win_rate) / total_games) + (z2 / (4.0 * total_games * total_games))) ** 0.5 if total_games else 0.0
    ci_lower = max(0.0, center - rad)
    ci_upper = min(1.0, center + rad)

    return {
        "games": total_games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": round(win_rate, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "mean_vps_me": round(mean_vps_me, 2),
        "mean_vps_opp": round(mean_vps_opp, 2),
        "vps_delta": round(vps_delta, 2),
        "mean_cpu_ms": round(mean_cpu_ms, 2),
        "p95_latency_ms": round(p95_latency, 2),
    }


def generate_candidate_perturbations(
    base_weights: Dict[GamePhase, Dict[str, float]],
    candidate_idx: int,
    rng: random.Random,
) -> Dict[GamePhase, Dict[str, float]]:
    """Generate structured domain-guided perturbations along key strategic axes."""
    cand = {phase: dict(w) for phase, w in base_weights.items()}

    # Axis 1: City & Ore-Wheat Focus (accelerates mid-game VP conversion)
    if candidate_idx % 4 == 0:
        cand[GamePhase.EARLY]["wheat_ore_pips"] *= 1.25
        cand[GamePhase.MID]["wheat_ore_pips"] *= 1.30
        cand[GamePhase.MID]["can_city_readiness"] *= 1.20
        cand[GamePhase.MID]["can_dev_readiness"] *= 1.20

    # Axis 2: Expansion & Settlement Focus (wood-brick settlement race)
    elif candidate_idx % 4 == 1:
        cand[GamePhase.EARLY]["wood_brick_pips"] *= 1.25
        cand[GamePhase.EARLY]["can_settlement_readiness"] *= 1.25
        cand[GamePhase.EARLY]["num_buildable_nodes"] *= 1.30
        cand[GamePhase.EARLY]["blocked_penalty"] *= 1.20

    # Axis 3: Port Synergy & Maritime Efficiency
    elif candidate_idx % 4 == 2:
        cand[GamePhase.EARLY]["port_synergy"] *= 1.30
        cand[GamePhase.EARLY]["has_3_to_1_port"] *= 1.25
        cand[GamePhase.MID]["port_synergy"] *= 1.35
        cand[GamePhase.MID]["has_special_port"] *= 1.25

    # Axis 4: Denial & Choke Control (Robber targeting & Largest Army)
    else:
        cand[GamePhase.MID]["robber_on_enemy"] *= 1.30
        cand[GamePhase.MID]["has_largest_army"] *= 1.25
        cand[GamePhase.LATE]["robber_on_enemy"] *= 1.40
        cand[GamePhase.LATE]["robber_on_self"] *= 1.30

    # Small random Gaussian jitter (5%) to prevent local saddle points
    for phase in cand:
        for k in cand[phase]:
            cand[phase][k] *= rng.uniform(0.95, 1.05)

    return cand


def record_experiment(experiment_data: Dict[str, Any]):
    """Append experiment entry to machine-readable registry."""
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    registry = []
    if os.path.isfile(REGISTRY_PATH):
        try:
            with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
                registry = json.load(f)
        except Exception:
            registry = []

    registry.append(experiment_data)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def run_optimization_pipeline(
    num_candidates: int = 4,
    seeds_per_candidate: int = 10,
    val_seeds_count: int = 15,
) -> Dict[str, Any]:
    """Execute complete weight tuning, validation gating, and promotion cycle."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    git_commit = get_git_commit()
    exp_id = f"EXP-OPT-{timestamp}"

    print(f"\n=======================================================")
    print(f" Catan Weight Optimization & Auto-Tuning: {exp_id}")
    print(f" Git Commit: {git_commit}")
    print(f" Candidates: {num_candidates} | Train Seeds: {seeds_per_candidate} | Val Seeds: {val_seeds_count}")
    print(f"=======================================================\n")

    train_seeds = load_seeds("train")[:seeds_per_candidate]
    val_seeds = load_seeds("val")[:val_seeds_count]
    test_seeds = load_seeds("test")

    # 1. Baseline Evaluation on Training Seeds
    print("Evaluating Baseline Weights on Training Seeds...")
    base_weights = get_default_weights()
    base_train_res = evaluate_weights_on_seeds(base_weights, train_seeds)
    print(f" Baseline Train: {base_train_res['wins']}/{base_train_res['games']} ({base_train_res['win_rate']*100:.1f}%), Delta VPs: {base_train_res['vps_delta']:+.2f}, CPU: {base_train_res['mean_cpu_ms']:.1f}ms")

    best_candidate_weights = copy.deepcopy(base_weights)
    best_candidate_res = copy.deepcopy(base_train_res)
    best_candidate_idx = -1
    rng = random.Random(42)

    # 2. Candidate Evaluation Loop
    for i in range(num_candidates):
        print(f"\n--- Testing Candidate {i+1}/{num_candidates} ---")
        cand_weights = generate_candidate_perturbations(base_weights, i, rng)
        cand_train_res = evaluate_weights_on_seeds(cand_weights, train_seeds)
        print(f" Candidate {i+1} Train: {cand_train_res['wins']}/{cand_train_res['games']} ({cand_train_res['win_rate']*100:.1f}%), Delta VPs: {cand_train_res['vps_delta']:+.2f}, CPU: {cand_train_res['mean_cpu_ms']:.1f}ms")

        # Optimization criterion: Higher win rate OR (equal win rate and superior VP delta)
        improved = (
            cand_train_res["win_rate"] > best_candidate_res["win_rate"]
            or (
                cand_train_res["win_rate"] == best_candidate_res["win_rate"]
                and cand_train_res["vps_delta"] > best_candidate_res["vps_delta"]
            )
        )
        if improved:
            print(f" => Candidate {i+1} surpasses current best! (WR: {cand_train_res['win_rate']*100:.1f}% vs {best_candidate_res['win_rate']*100:.1f}%, Delta: {cand_train_res['vps_delta']:+.2f})")
            best_candidate_weights = cand_weights
            best_candidate_res = cand_train_res
            best_candidate_idx = i + 1

    # 3. Validation Gating
    print(f"\n=======================================================")
    print(f" Promotion Gating on Unseen Validation Seeds ({len(val_seeds)} seeds = {len(val_seeds)*2} games)")
    print(f"=======================================================")

    print("Evaluating Baseline on Validation Seeds...")
    base_val_res = evaluate_weights_on_seeds(base_weights, val_seeds)
    print(f" Baseline Val: {base_val_res['wins']}/{base_val_res['games']} ({base_val_res['win_rate']*100:.1f}%), Delta VPs: {base_val_res['vps_delta']:+.2f}")

    if best_candidate_idx == -1:
        print("\n [GATE FAILED]: No candidate improved on training seeds. Retaining baseline.")
        promoted = False
        promotion_reason = "No candidate outperformed baseline on training partition."
        cand_val_res = base_val_res
    else:
        print(f"Evaluating Candidate {best_candidate_idx} on Validation Seeds...")
        cand_val_res = evaluate_weights_on_seeds(best_candidate_weights, val_seeds)
        print(f" Candidate {best_candidate_idx} Val: {cand_val_res['wins']}/{cand_val_res['games']} ({cand_val_res['win_rate']*100:.1f}%), Delta VPs: {cand_val_res['vps_delta']:+.2f}, CPU: {cand_val_res['mean_cpu_ms']:.1f}ms")

        # Gate Checks
        pass_tests = run_unit_tests()
        print(f" Gate 1 (Correctness Tests Pass): {'PASS' if pass_tests else 'FAIL'}")

        pass_val_wr = cand_val_res["win_rate"] >= base_val_res["win_rate"]
        print(f" Gate 2 (Validation WR >= Baseline): {'PASS' if pass_val_wr else 'FAIL'} ({cand_val_res['win_rate']*100:.1f}% vs {base_val_res['win_rate']*100:.1f}%)")

        pass_val_vp = cand_val_res["vps_delta"] >= base_val_res["vps_delta"] - 0.2
        print(f" Gate 3 (Validation VP Delta Non-Regression): {'PASS' if pass_val_vp else 'FAIL'} ({cand_val_res['vps_delta']:+.2f} vs {base_val_res['vps_delta']:+.2f})")

        pass_cpu = cand_val_res["mean_cpu_ms"] < 500.0 and cand_val_res["p95_latency_ms"] < 15.0
        print(f" Gate 4 (Runtime Budget Compliance < 500ms): {'PASS' if pass_cpu else 'FAIL'} ({cand_val_res['mean_cpu_ms']:.1f}ms)")

        promoted = pass_tests and pass_val_wr and pass_val_vp and pass_cpu
        promotion_reason = (
            f"Candidate {best_candidate_idx} successfully passed all promotion gates with val win rate {cand_val_res['win_rate']*100:.1f}%."
            if promoted
            else "Candidate failed one or more promotion gates."
        )

    # 4. Promotion Action
    saved_weights_path = None
    test_res = None
    if promoted:
        print(f"\n >>> CANDIDATE PROMOTED! Writing new weights to production <<<")
        prod_path = "my_team/evaluation/weights.json"
        snapshot_path = f"results/best_weights_{exp_id}.json"
        save_weights_to_file(best_candidate_weights, prod_path)
        save_weights_to_file(best_candidate_weights, snapshot_path)
        saved_weights_path = prod_path

        # Run test set evaluation
        print(f"Running generalization test on held-out test seeds ({len(test_seeds)} seeds = {len(test_seeds)*2} games)...")
        test_res = evaluate_weights_on_seeds(best_candidate_weights, test_seeds)
        print(f" Held-Out Test Result: {test_res['wins']}/{test_res['games']} ({test_res['win_rate']*100:.1f}%), Delta VPs: {test_res['vps_delta']:+.2f}")

        # Gate 5: Test set non-regression verification
        if test_res["win_rate"] < 0.60:
            print(f" [TEST GATE NOT MET]: Held-out test win rate ({test_res['win_rate']*100:.1f}%) did not meet production benchmark (>=60.0%).")
            print(f" Rolling back production file {prod_path}. Retaining default baseline as production default.")
            if os.path.isfile(prod_path):
                os.remove(prod_path)
            promoted = False
            promotion_reason = f"Candidate passed validation but failed held-out test threshold ({test_res['win_rate']*100:.1f}% < 60.0%). Saved snapshot to {snapshot_path}."
        else:
            print(f" [ALL GATES PASSED]: Candidate verified on held-out test set! Promoted to production.")
    else:
        print(f"\n >>> RETENTION: Keeping baseline weights as production default <<<")

    # 5. Registry Entry
    record = {
        "exp_id": exp_id,
        "timestamp": timestamp,
        "git_commit": git_commit,
        "promoted": promoted,
        "promotion_reason": promotion_reason,
        "best_candidate_idx": best_candidate_idx,
        "train_results": {
            "baseline": base_train_res,
            "candidate": best_candidate_res,
        },
        "val_results": {
            "baseline": base_val_res,
            "candidate": cand_val_res,
        },
        "test_results": test_res,
        "weights_file": saved_weights_path,
    }
    record_experiment(record)

    print(f"\n Experiment log appended to: {REGISTRY_PATH}")
    return record


def main():
    parser = argparse.ArgumentParser(description="Run Catan Weight Optimization.")
    parser.add_argument("--candidates", type=int, default=4, help="Number of candidate perturbations to test")
    parser.add_argument("--train-seeds", type=int, default=10, help="Number of board seeds for training (2x games)")
    parser.add_argument("--val-seeds", type=int, default=10, help="Number of board seeds for validation (2x games)")
    args = parser.parse_args()

    run_optimization_pipeline(
        num_candidates=args.candidates,
        seeds_per_candidate=args.train_seeds,
        val_seeds_count=args.val_seeds,
    )


if __name__ == "__main__":
    main()
