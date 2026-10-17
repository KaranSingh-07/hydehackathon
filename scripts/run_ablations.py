"""Ablation Experiment Runner for Catan Agent Search Layer.

Evaluates 5 variants against Bot F across identical fixed seeds and alternating seats:
  Variant A: Strategic Evaluator only (1-ply, 1 world, no candidate ranking)
  Variant B: Evaluator + Candidate Ranking (K=5, 1 world, no compound search)
  Variant C: Evaluator + Compound Search (K=5, same-turn trade/dev lookahead)
  Variant D: Evaluator + Multi-World Sampling (N=3 worlds, 1 candidate)
  Variant E: Full Adaptive Combination (K=5, N=3 when opp has devs, compound lookahead, anytime clock guard)

Saves summary results to `results/ablation_summary.json` and updates `docs/PERFORMANCE_TRACKER.md`.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

# Add project root to sys.path
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.benchmark import run_benchmark, resolve_opponent_path


VARIANTS = [
    ("Variant_A_eval_only", "eval_only", "Evaluator Only (1-ply, N=1, K=1)"),
    ("Variant_B_ranking", "ranking", "Evaluator + Candidate Ranking (K=5, N=1)"),
    ("Variant_C_compound", "compound", "Evaluator + Compound Search (K=5, Same-Turn Lookahead)"),
    ("Variant_D_multi_world", "multi_world", "Evaluator + Multi-World Sampling (N=3, K=1)"),
    ("Variant_E_full_adaptive", "full", "Full Adaptive Search (K=5, N=3, Compound, Clock Guard)"),
]


def run_ablations(games_per_variant: int = 20, seed0: int = 990000):
    """Run all ablation variants and collect comparative results."""
    agent_path = str(Path("my_team/agent.py").resolve())
    opp_path = resolve_opponent_path("f")
    ablation_results = []

    print("\n=======================================================")
    print(" Catan Phase 3 Search Ablation Suite")
    print(f" Opponent: Bot F ({opp_path})")
    print(f" Games per Variant: {games_per_variant} | First Seed: {seed0}")
    print("=======================================================\n")

    for tag, mode, desc in VARIANTS:
        print(f"\n--- Testing {tag}: {desc} ---")
        os.environ["CATAN_SEARCH_MODE"] = mode

        res = run_benchmark(
            path_a=agent_path,
            path_b=opp_path,
            num_games=games_per_variant,
            seed0=seed0,
            vps=10,
            results_dir="results",
            tag=f"ablation_{tag}",
        )
        res["variant_tag"] = tag
        res["variant_mode"] = mode
        res["variant_description"] = desc
        ablation_results.append(res)

    # Save summary
    out_path = Path("results/ablation_summary.json")
    with open(out_path, "w") as f:
        json.dump(ablation_results, f, indent=2)

    print("\n=======================================================")
    print(" ABLATION COMPARISON SUMMARY")
    print("=======================================================")
    print(f"{'Variant':<28} | {'Win Rate':<10} | {'95% CI':<16} | {'Red WR':<8} | {'Blue WR':<8} | {'Mean CPU':<10}")
    print("-" * 90)
    for r in ablation_results:
        ci_str = f"[{r['ci95_wilson_lower']:.2f}, {r['ci95_wilson_upper']:.2f}]"
        print(f"{r['variant_tag']:<28} | {r['win_rate_a']*100:6.1f}%   | {ci_str:<16} | {r['win_rate_a_as_seat_0']*100:6.1f}% | {r['win_rate_a_as_seat_1']*100:6.1f}% | {r['agent_a_mean_cpu_ms']:6.1f} ms")
    print("=======================================================\n")


if __name__ == "__main__":
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    run_ablations(games_per_variant=games)
