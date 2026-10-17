"""Match Benchmark Harness for Catan Agent.

Reuses the official competition runner (`catan_kit.match.play_chunk`) without altering rules.
Supports:
  - Agent vs Baseline F (examples/f_agent.py or baselines/baseline_f.py)
  - Agent vs Random (examples/random_agent.py)
  - Agent Self-Play (my_team/agent.py vs my_team/agent.py)
  - Baseline F vs Random
  - Custom agent paths
Features:
  - Multiple configurable random seeds (seed0 + paired board evaluations)
  - Both player positions (swapped seats alternating per game)
  - Detailed positional splits (Win rate as Player 1 / Red vs Player 2 / Blue)
  - Per-decision runtime, game length (turns), CPU statistics
  - Fault tracking (timeouts, crashes, illegal indices, games over budget)
  - Machine-readable JSON and CSV report outputs saved to `results/`
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is in sys.path
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from catan_kit._patch import ensure_hash_seed
from catan_kit import match


def get_git_commit():
    """Extract current git commit hash."""
    try:
        res = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return res.decode().strip()
    except Exception:
        return "unknown"


def resolve_opponent_path(opp_name: str) -> str:
    """Resolve standard opponent shortcuts to filepaths."""
    root = Path(__file__).resolve().parent.parent
    mapping = {
        "f": str(root / "examples" / "f_agent.py"),
        "baseline_f": str(root / "baselines" / "baseline_f.py"),
        "random": str(root / "examples" / "random_agent.py"),
        "self": str(root / "my_team" / "agent.py"),
        "agent": str(root / "my_team" / "agent.py"),
    }
    return mapping.get(opp_name.lower(), opp_name)


def run_benchmark(
    path_a: str,
    path_b: str,
    num_games: int = 20,
    seed0: int = 990000,
    vps: int = 10,
    results_dir: str = "results",
    tag: str = "benchmark",
) -> dict:
    """Execute match benchmark using official catan_kit match runner."""
    path_a = str(Path(path_a).resolve())
    path_b = str(Path(path_b).resolve())

    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    git_commit = get_git_commit()

    print(f"\n=======================================================")
    print(f" Catan Match Benchmark: {tag}")
    print(f" Agent A: {path_a}")
    print(f" Agent B: {path_b}")
    print(f" Games: {num_games} | First Seed: {seed0} | VPs: {vps}")
    print(f" Git Commit: {git_commit}")
    print(f"=======================================================\n")

    t_start = time.time()
    seeds = [seed0 + i // 2 for i in range(num_games)]
    game_indices = list(range(num_games))

    records, info = match.play_chunk(
        path_a,
        path_b,
        seeds,
        game_indices,
        key=match.DEFAULT_KEY,
        vps=vps,
    )
    wall_duration = time.time() - t_start

    # Metrics computation
    total_games = len(records)
    wins_a = sum(1 for r in records if r["winner"] == "a")
    wins_b = sum(1 for r in records if r["winner"] == "b")
    draws = sum(1 for r in records if r["winner"] is None)

    win_rate_a = wins_a / total_games if total_games else 0.0
    ci95_normal = 1.96 * (win_rate_a * (1.0 - win_rate_a) / total_games) ** 0.5 if total_games else 0.0

    # Wilson Score Interval (robust for small samples and near 0/1 proportions)
    z = 1.96
    z2 = z * z
    denom = 1.0 + z2 / total_games if total_games else 1.0
    center = (win_rate_a + z2 / (2.0 * total_games)) / denom if total_games else 0.0
    rad = (z / denom) * ((win_rate_a * (1.0 - win_rate_a) / total_games) + (z2 / (4.0 * total_games * total_games))) ** 0.5 if total_games else 0.0
    wilson_lower = max(0.0, center - rad)
    wilson_upper = min(1.0, center + rad)

    # Positional breakdown
    games_a_first = [r for r in records if r["a_first"]]
    games_b_first = [r for r in records if not r["a_first"]]

    wins_a_as_first = sum(1 for r in games_a_first if r["winner"] == "a")
    wins_a_as_second = sum(1 for r in games_b_first if r["winner"] == "a")

    wr_a_first = wins_a_as_first / len(games_a_first) if games_a_first else 0.0
    wr_a_second = wins_a_as_second / len(games_b_first) if games_b_first else 0.0

    # Agent A CPU metrics
    a_cpus = [r["a"]["cpu_s"] for r in records]
    a_decisions = [r["a"]["decisions"] for r in records]
    mean_cpu_a = sum(a_cpus) / total_games if total_games else 0.0
    worst_cpu_a = max(a_cpus) if a_cpus else 0.0
    total_decisions_a = sum(a_decisions)
    cpu_per_decision_a = (sum(a_cpus) / total_decisions_a * 1000) if total_decisions_a else 0.0

    # Per-game decision latency distribution (p95 and max)
    per_game_latencies = [(r["a"]["cpu_s"] / max(1, r["a"]["decisions"])) * 1000 for r in records]
    per_game_latencies.sort()
    idx_p95 = min(len(per_game_latencies) - 1, int(round(0.95 * len(per_game_latencies))))
    p95_decision_ms = per_game_latencies[idx_p95] if per_game_latencies else 0.0
    max_decision_ms = max(per_game_latencies) if per_game_latencies else 0.0

    over_budget_a = sum(1 for r in records if r["a"]["over"] is not None)
    random_moves_a = sum(r["a"]["random_moves"] for r in records)
    errors_a = sum(r["a"]["errors"] for r in records)
    timeouts_a = sum(r["a"]["timeouts"] for r in records)
    crashes_a = sum(r["a"]["crashes"] for r in records)

    # Turns
    turn_counts = [r["turns"] for r in records]
    mean_turns = sum(turn_counts) / total_games if total_games else 0.0

    summary = {
        "timestamp_utc": timestamp,
        "git_commit": git_commit,
        "tag": tag,
        "path_a": path_a,
        "path_b": path_b,
        "total_games": total_games,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "draws": draws,
        "win_rate_a": round(win_rate_a, 4),
        "ci95_normal": round(ci95_normal, 4),
        "ci95_wilson_lower": round(wilson_lower, 4),
        "ci95_wilson_upper": round(wilson_upper, 4),
        "win_rate_a_as_seat_0": round(wr_a_first, 4),
        "win_rate_a_as_seat_1": round(wr_a_second, 4),
        "mean_turns": round(mean_turns, 2),
        "agent_a_mean_cpu_ms": round(mean_cpu_a * 1000, 2),
        "agent_a_worst_cpu_ms": round(worst_cpu_a * 1000, 2),
        "agent_a_cpu_per_decision_ms": round(cpu_per_decision_a, 3),
        "agent_a_p95_decision_ms": round(p95_decision_ms, 3),
        "agent_a_max_decision_ms": round(max_decision_ms, 3),
        "agent_a_total_decisions": total_decisions_a,
        "agent_a_games_over_budget": over_budget_a,
        "agent_a_random_moves_after_cap": random_moves_a,
        "agent_a_errors": errors_a,
        "agent_a_timeouts": timeouts_a,
        "agent_a_crashes": crashes_a,
        "agent_a_load_time_s": info.get("a_init_s", 0.0),
        "agent_b_load_time_s": info.get("b_init_s", 0.0),
        "total_wall_clock_s": round(wall_duration, 2),
    }

    # Print Report
    print(f"Results Summary:")
    print(f"  Win Rate A:        {summary['win_rate_a']:.4f} +- {summary['ci95_normal']:.4f} ({wins_a}/{total_games})")
    print(f"  95% CI (Wilson):   [{summary['ci95_wilson_lower']:.4f}, {summary['ci95_wilson_upper']:.4f}]")
    print(f"  Win Rate A (Red):  {summary['win_rate_a_as_seat_0']:.4f} ({wins_a_as_first}/{len(games_a_first)})")
    print(f"  Win Rate A (Blue): {summary['win_rate_a_as_seat_1']:.4f} ({wins_a_as_second}/{len(games_b_first)})")
    print(f"  Mean Game Turns:   {summary['mean_turns']}")
    print(f"  A Mean CPU / Game: {summary['agent_a_mean_cpu_ms']} ms")
    print(f"  A Worst CPU / Game:{summary['agent_a_worst_cpu_ms']} ms")
    print(f"  A CPU / Decision:  {summary['agent_a_cpu_per_decision_ms']} ms (p95: {summary['agent_a_p95_decision_ms']} ms, max: {summary['agent_a_max_decision_ms']} ms)")
    print(f"  A Games Over Cap:  {summary['agent_a_games_over_budget']}")
    print(f"  A Faults (Err/TO): {summary['agent_a_errors']} / {summary['agent_a_timeouts']}")
    print(f"  Wall Time:         {summary['total_wall_clock_s']} s")
    print(f"-------------------------------------------------------\n")

    # Save to JSON
    json_path = os.path.join(results_dir, f"{tag}_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump({"summary": summary, "game_records": records}, f, indent=2)
    print(f"Detailed results saved to: {json_path}")

    # Append to aggregate CSV
    csv_path = os.path.join(results_dir, "benchmark_history.csv")
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(summary)
    print(f"Aggregate record appended to: {csv_path}\n")

    return summary


def main():
    ensure_hash_seed()
    parser = argparse.ArgumentParser(description="Catan Match Benchmark Runner")
    parser.add_argument(
        "--a",
        default="my_team/agent.py",
        help="Path or alias for Agent A (default: my_team/agent.py)",
    )
    parser.add_argument(
        "--b",
        default="f",
        help="Path or alias for Agent B ('f', 'random', 'self', or file path; default: f)",
    )
    parser.add_argument("--games", type=int, default=20, help="Number of games to run (default: 20)")
    parser.add_argument("--seed0", type=int, default=990000, help="Starting board seed (default: 990000)")
    parser.add_argument("--vps", type=int, default=10, help="Victory points to win (default: 10)")
    parser.add_argument("--results-dir", default="results", help="Directory for output files")
    parser.add_argument("--tag", default="eval", help="Tag for this benchmark run")

    args = parser.parse_args()
    path_a = resolve_opponent_path(args.a)
    path_b = resolve_opponent_path(args.b)

    run_benchmark(
        path_a=path_a,
        path_b=path_b,
        num_games=args.games,
        seed0=args.seed0,
        vps=args.vps,
        results_dir=args.results_dir,
        tag=args.tag,
    )


if __name__ == "__main__":
    main()
