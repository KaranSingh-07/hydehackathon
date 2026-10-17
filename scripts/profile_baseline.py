"""Runtime Profiler for Baseline Catan Agent.

Measures fine-grained timing breakdowns across:
1. World sampling (`obs.sample_world`).
2. State deep-copying (`world.copy()`).
3. Action execution (`successor.execute(action)`).
4. Feature evaluation (`VALUE_FN(successor)` breakdown):
   - `production_features`
   - `reachability_features`
   - `resource_hand_features`
   - `longest_road`
   - `buildable_nodes`
Saves results to `results/profile_baseline.json`.
"""
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

# Ensure project root is in sys.path
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from catan_kit.obs import Obs
from catanatron.game import Game
from catanatron.models.player import Color, RandomPlayer
from catanatron.players.value import DEFAULT_WEIGHTS, get_value_fn
from catanatron.state_functions import (
    get_longest_road_length,
    player_key,
    player_num_dev_cards,
    player_num_resource_cards,
)
from catanatron.features import (
    build_production_features,
    reachability_features,
    resource_hand_features,
)


def profile_decision_components(num_trials: int = 50, seed: int = 4256):
    """Profile isolated operations across multiple realistic board states."""
    random.seed(seed)
    game = Game(
        [RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)],
        seed=seed,
        vps_to_win=10,
    )

    VALUE_FN = get_value_fn("base_fn", DEFAULT_WEIGHTS)
    rng = random.Random(0)

    # Collect measurements
    timings = defaultdict(list)
    action_counts = []

    # Advance game to various stages to gather realistic samples
    step = 0
    sampled_decisions = 0

    while game.winning_color() is None and sampled_decisions < num_trials:
        actions = list(game.playable_actions)
        color = game.state.current_color()

        if len(actions) > 1:
            sampled_decisions += 1
            action_counts.append(len(actions))
            obs = Obs(game, color)

            # 1. Profile sample_world
            t0 = time.perf_counter()
            world = obs.sample_world(rng)
            t1 = time.perf_counter()
            timings["sample_world_ms"].append((t1 - t0) * 1000)

            # Profile per-action components for up to 10 actions per decision
            sub_actions = actions[:10]
            copy_times = []
            exec_times = []
            eval_times = []
            prod_times = []
            reach_times = []
            hand_times = []
            nodes_times = []

            for action in sub_actions:
                # 2. Profile world.copy
                t_c0 = time.perf_counter()
                successor = world.copy()
                t_c1 = time.perf_counter()
                copy_times.append((t_c1 - t_c0) * 1000)

                # 3. Profile successor.execute
                t_e0 = time.perf_counter()
                try:
                    successor.execute(action)
                except Exception:
                    continue
                t_e1 = time.perf_counter()
                exec_times.append((t_e1 - t_e0) * 1000)

                # 4. Profile full VALUE_FN
                t_v0 = time.perf_counter()
                _ = VALUE_FN(successor, color)
                t_v1 = time.perf_counter()
                eval_times.append((t_v1 - t_v0) * 1000)

                # 5. Profile sub-features inside VALUE_FN
                # 5a. Production
                t_p0 = time.perf_counter()
                prod_fn = build_production_features(True)
                _ = prod_fn(successor, color)
                t_p1 = time.perf_counter()
                prod_times.append((t_p1 - t_p0) * 1000)

                # 5b. Reachability
                t_r0 = time.perf_counter()
                _ = reachability_features(successor, color, 2)
                t_r1 = time.perf_counter()
                reach_times.append((t_r1 - t_r0) * 1000)

                # 5c. Hand features
                t_h0 = time.perf_counter()
                _ = resource_hand_features(successor, color)
                t_h1 = time.perf_counter()
                hand_times.append((t_h1 - t_h0) * 1000)

                # 5d. Buildable nodes
                t_n0 = time.perf_counter()
                _ = len(successor.state.board.buildable_node_ids(color))
                t_n1 = time.perf_counter()
                nodes_times.append((t_n1 - t_n0) * 1000)

            if copy_times:
                timings["world_copy_ms_per_action"].append(sum(copy_times) / len(copy_times))
            if exec_times:
                timings["execute_ms_per_action"].append(sum(exec_times) / len(exec_times))
            if eval_times:
                timings["value_fn_ms_per_action"].append(sum(eval_times) / len(eval_times))
            if prod_times:
                timings["sub_production_ms"].append(sum(prod_times) / len(prod_times))
            if reach_times:
                timings["sub_reachability_ms"].append(sum(reach_times) / len(reach_times))
            if hand_times:
                timings["sub_hand_ms"].append(sum(hand_times) / len(hand_times))
            if nodes_times:
                timings["sub_buildable_nodes_ms"].append(sum(nodes_times) / len(nodes_times))

        chosen = random.choice(actions)
        game.execute(chosen)
        step += 1

    summary = {
        "trials_sampled": sampled_decisions,
        "mean_candidate_actions": round(sum(action_counts) / len(action_counts), 2),
        "metrics_ms": {
            "sample_world_mean": round(sum(timings["sample_world_ms"]) / len(timings["sample_world_ms"]), 4),
            "world_copy_mean_per_action": round(sum(timings["world_copy_ms_per_action"]) / len(timings["world_copy_ms_per_action"]), 4),
            "execute_mean_per_action": round(sum(timings["execute_ms_per_action"]) / len(timings["execute_ms_per_action"]), 4),
            "value_fn_total_mean_per_action": round(sum(timings["value_fn_ms_per_action"]) / len(timings["value_fn_ms_per_action"]), 4),
            "breakdown_value_fn": {
                "reachability_features_ms": round(sum(timings["sub_reachability_ms"]) / len(timings["sub_reachability_ms"]), 4),
                "production_features_ms": round(sum(timings["sub_production_ms"]) / len(timings["sub_production_ms"]), 4),
                "buildable_nodes_calc_ms": round(sum(timings["sub_buildable_nodes_ms"]) / len(timings["sub_buildable_nodes_ms"]), 4),
                "hand_features_ms": round(sum(timings["sub_hand_ms"]) / len(timings["sub_hand_ms"]), 4),
            },
        },
    }

    # Calculate percentage of decision time spent on copying vs evaluating
    copy_cost = summary["metrics_ms"]["world_copy_mean_per_action"]
    exec_cost = summary["metrics_ms"]["execute_mean_per_action"]
    eval_cost = summary["metrics_ms"]["value_fn_total_mean_per_action"]
    total_per_action = copy_cost + exec_cost + eval_cost

    summary["per_action_distribution_pct"] = {
        "world_copy_pct": round(copy_cost / total_per_action * 100, 2),
        "execute_pct": round(exec_cost / total_per_action * 100, 2),
        "value_eval_pct": round(eval_cost / total_per_action * 100, 2),
    }

    return summary


def main():
    print("Running baseline runtime profiling...")
    summary = profile_decision_components(num_trials=40)
    print("\n=== Baseline Profiling Results ===")
    print(json.dumps(summary, indent=2))

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "profile_baseline.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved profiling report to: {out_path}")


if __name__ == "__main__":
    main()
