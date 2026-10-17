"""Time and Compute Budget Manager for Selective Search.

Dynamically regulates candidate branch limits and multi-world sampling count
based on remaining clock allowance (`obs.cpu_left`).
Enforces anytime deadlines to prevent clock overages.
"""
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class SearchConfig:
    """Configuration parameters for selective search."""
    max_candidates: int = 5
    num_worlds: int = 3
    allow_compound_search: bool = True
    time_limit_s: float = 0.015  # 15 ms per decision default target


class BudgetManager:
    """Allocates search depth, candidate limits, and time bounds per decision."""

    def __init__(self, target_cpu_per_game_s: float = 0.500):
        # Target consuming at most ~500 ms per game (safe 2x buffer below 1,000 ms allowance)
        self.target_cpu_per_game_s = target_cpu_per_game_s

    def get_search_config(
        self,
        cpu_left: Optional[float],
        turn: int,
        is_opening: bool,
        opp_dev_count: int,
    ) -> SearchConfig:
        """Derive search depth, candidate limit, and world count for current state."""
        # 1. Critical Emergency Clock Tier (< 150 ms left): Force 1-ply immediate return
        if cpu_left is not None and cpu_left < 0.150:
            return SearchConfig(
                max_candidates=1,
                num_worlds=1,
                allow_compound_search=False,
                time_limit_s=0.002,
            )

        # 2. Conservative Clock Tier (< 350 ms left): Reduce candidates
        if cpu_left is not None and cpu_left < 0.350:
            return SearchConfig(
                max_candidates=3,
                num_worlds=1,
                allow_compound_search=False,
                time_limit_s=0.006,
            )

        # 3. Explicit mode override for ablations (if set)
        import os
        mode = os.environ.get("CATAN_SEARCH_MODE")
        if mode:
            mode = mode.lower()
            if mode == "eval_only":
                return SearchConfig(max_candidates=1, num_worlds=1, allow_compound_search=False, time_limit_s=0.005)
            elif mode == "ranking":
                return SearchConfig(max_candidates=5, num_worlds=1, allow_compound_search=False, time_limit_s=0.020)
            elif mode == "compound":
                return SearchConfig(max_candidates=5, num_worlds=1, allow_compound_search=True, time_limit_s=0.020)
            elif mode == "multi_world":
                return SearchConfig(max_candidates=1, num_worlds=3, allow_compound_search=False, time_limit_s=0.020)

        # Case 1: Unclocked environment (e.g. testing)
        if cpu_left is None:
            return SearchConfig(
                max_candidates=6,
                num_worlds=3 if opp_dev_count > 0 else 1,
                allow_compound_search=True,
                time_limit_s=0.030,
            )

        # Case 4: Opening Phase (Turns 0-1)
        if is_opening:
            return SearchConfig(
                max_candidates=8,
                num_worlds=1,
                allow_compound_search=False,
                time_limit_s=0.040,
            )

        # Case 5: Default Production Mode (Variant B proven setting with dev card awareness)
        return SearchConfig(
            max_candidates=5,
            num_worlds=3 if opp_dev_count > 0 else 1,
            allow_compound_search=False,
            time_limit_s=0.020,
        )



class DeadlineTimer:
    """Monotonic deadline checker for cooperative anytime search."""

    def __init__(self, timeout_s: float):
        self.t_start = time.perf_counter()
        self.timeout_s = timeout_s
        self.t_deadline = self.t_start + timeout_s

    def is_expired(self) -> bool:
        """Return True if the deadline has passed."""
        return time.perf_counter() >= self.t_deadline

    def elapsed_ms(self) -> float:
        """Return milliseconds elapsed since timer initialization."""
        return (time.perf_counter() - self.t_start) * 1000.0
