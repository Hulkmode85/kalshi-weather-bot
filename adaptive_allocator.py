"""
adaptive_allocator.py — Thompson sampling bandit for adaptive capital allocation across strategies.

Uses Beta distributions to model each strategy's win probability, then samples
to produce allocation weights that explore-exploit automatically.

Usage:
    allocator = AdaptiveAllocator(["momentum", "mean_revert", "breakout"])
    allocator.update("momentum", won=True)
    weights = allocator.get_allocation(total_capital=5000)
    stats = allocator.get_posterior_stats()
"""

import json
import math
import os
import random
import time
from typing import Optional


class AdaptiveAllocator:
    """Thompson sampling multi-armed bandit for capital allocation."""

    SHADOW_LOG = "allocator_shadow.jsonl"

    def __init__(self, strategies: list[str], log_dir: str = ".",
                 min_allocation_pct: float = 0.05, decay: float = 1.0):
        """
        Args:
            strategies: list of strategy names.
            log_dir: directory for shadow JSONL log.
            min_allocation_pct: minimum allocation per strategy (prevents starving).
            decay: discount factor for older observations (1.0 = no decay).
        """
        self.strategies = list(strategies)
        self.alpha = {s: 1.0 for s in strategies}  # Beta prior successes
        self.beta_param = {s: 1.0 for s in strategies}  # Beta prior failures
        self.min_allocation_pct = min_allocation_pct
        self.decay = decay
        self.trade_count = {s: 0 for s in strategies}
        self.total_pnl = {s: 0.0 for s in strategies}
        self.log_path = os.path.join(log_dir, self.SHADOW_LOG)

    # --------------------------------------------------- Thompson Sampling
    def sample(self) -> dict[str, float]:
        """Thompson sampling: draw from Beta(alpha, beta) for each strategy.

        Returns dict mapping strategy -> sampled value in [0, 1].
        Higher sampled value = more capital.
        """
        samples = {}
        for s in self.strategies:
            samples[s] = random.betavariate(
                max(self.alpha[s], 0.01),
                max(self.beta_param[s], 0.01)
            )
        return samples

    def update(self, strategy: str, won: bool, pnl: float = 0.0):
        """Update Beta distribution params after trade result.

        Args:
            strategy: strategy that traded.
            won: True if trade was profitable.
            pnl: actual PnL (used for tracking, not for Beta update).
        """
        if strategy not in self.alpha:
            self._add_strategy(strategy)

        # Apply decay to all strategies (recency weighting)
        if self.decay < 1.0:
            for s in self.strategies:
                self.alpha[s] = max(1.0, self.alpha[s] * self.decay)
                self.beta_param[s] = max(1.0, self.beta_param[s] * self.decay)

        if won:
            self.alpha[strategy] += 1.0
        else:
            self.beta_param[strategy] += 1.0

        self.trade_count[strategy] = self.trade_count.get(strategy, 0) + 1
        self.total_pnl[strategy] = self.total_pnl.get(strategy, 0.0) + pnl

        self._shadow_log("update", {
            "strategy": strategy, "won": won, "pnl": pnl,
            "alpha": self.alpha[strategy],
            "beta": self.beta_param[strategy],
        })

    # --------------------------------------------------- Allocation
    def get_allocation(self, total_capital: float, n_samples: int = 1000) -> dict[str, float]:
        """Return recommended capital allocation per strategy.

        Uses averaged Thompson sampling (multiple draws) for smoother allocations.
        Enforces minimum allocation per strategy.

        Args:
            total_capital: total capital to allocate.
            n_samples: number of Thompson samples to average.

        Returns:
            dict mapping strategy -> dollar allocation.
        """
        if not self.strategies:
            return {}

        # Average multiple Thompson samples for stability
        avg_samples = {s: 0.0 for s in self.strategies}
        for _ in range(n_samples):
            draw = self.sample()
            for s in self.strategies:
                avg_samples[s] += draw[s]

        for s in self.strategies:
            avg_samples[s] /= n_samples

        # Normalize to weights
        total_sample = sum(avg_samples.values())
        if total_sample <= 0:
            # Uniform fallback
            weight = 1.0 / len(self.strategies)
            return {s: round(total_capital * weight, 2) for s in self.strategies}

        weights = {s: avg_samples[s] / total_sample for s in self.strategies}

        # Enforce minimum allocation
        min_alloc = self.min_allocation_pct
        n = len(self.strategies)
        if min_alloc * n < 1.0:  # only if feasible
            for s in self.strategies:
                if weights[s] < min_alloc:
                    deficit = min_alloc - weights[s]
                    weights[s] = min_alloc
                    # Redistribute deficit proportionally from others
                    others = [o for o in self.strategies if o != s and weights[o] > min_alloc]
                    if others:
                        other_total = sum(weights[o] for o in others)
                        for o in others:
                            weights[o] -= deficit * (weights[o] / max(other_total, 1e-9))

        # Renormalize
        w_total = sum(weights.values())
        allocations = {s: round(total_capital * weights[s] / max(w_total, 1e-9), 2)
                       for s in self.strategies}

        self._shadow_log("allocation", {
            "total_capital": total_capital,
            "allocations": allocations,
        })

        return allocations

    # --------------------------------------------------- Posterior Stats
    def get_posterior_stats(self) -> dict[str, dict]:
        """Return mean, variance, confidence interval per strategy.

        Returns:
            {strategy: {mean, variance, ci_low, ci_high, trades, total_pnl}}
        """
        stats = {}
        for s in self.strategies:
            a = self.alpha[s]
            b = self.beta_param[s]
            mean = a / (a + b)
            var = (a * b) / ((a + b) ** 2 * (a + b + 1))
            std = math.sqrt(var)

            # 95% credible interval (normal approximation for Beta)
            ci_low = max(0, mean - 1.96 * std)
            ci_high = min(1, mean + 1.96 * std)

            stats[s] = {
                "mean_win_rate": round(mean, 4),
                "variance": round(var, 6),
                "ci_95_low": round(ci_low, 4),
                "ci_95_high": round(ci_high, 4),
                "alpha": round(a, 2),
                "beta": round(b, 2),
                "trades": self.trade_count.get(s, 0),
                "total_pnl": round(self.total_pnl.get(s, 0.0), 4),
            }

        return stats

    # --------------------------------------------------- Strategy Management
    def add_strategy(self, strategy: str):
        """Add a new strategy to the allocator."""
        self._add_strategy(strategy)

    def remove_strategy(self, strategy: str):
        """Remove a strategy (e.g., if it's been retired)."""
        if strategy in self.strategies:
            self.strategies.remove(strategy)
            self.alpha.pop(strategy, None)
            self.beta_param.pop(strategy, None)
            self.trade_count.pop(strategy, None)
            self.total_pnl.pop(strategy, None)

    def _add_strategy(self, strategy: str):
        if strategy not in self.strategies:
            self.strategies.append(strategy)
        self.alpha.setdefault(strategy, 1.0)
        self.beta_param.setdefault(strategy, 1.0)
        self.trade_count.setdefault(strategy, 0)
        self.total_pnl.setdefault(strategy, 0.0)

    # --------------------------------------------------- Serialization
    def save_state(self, filepath: Optional[str] = None) -> str:
        """Save allocator state to JSON."""
        path = filepath or os.path.join(os.path.dirname(self.log_path), "allocator_state.json")
        state = {
            "strategies": self.strategies,
            "alpha": self.alpha,
            "beta": self.beta_param,
            "trade_count": self.trade_count,
            "total_pnl": self.total_pnl,
            "min_allocation_pct": self.min_allocation_pct,
            "decay": self.decay,
            "saved_at": time.time(),
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        return path

    @classmethod
    def load_state(cls, filepath: str) -> "AdaptiveAllocator":
        """Load allocator from saved state."""
        with open(filepath, "r") as f:
            state = json.load(f)

        alloc = cls(
            strategies=state["strategies"],
            min_allocation_pct=state.get("min_allocation_pct", 0.05),
            decay=state.get("decay", 1.0),
        )
        alloc.alpha = state["alpha"]
        alloc.beta_param = state["beta"]
        alloc.trade_count = state.get("trade_count", {s: 0 for s in state["strategies"]})
        alloc.total_pnl = state.get("total_pnl", {s: 0.0 for s in state["strategies"]})
        return alloc

    # --------------------------------------------------- Report
    def generate_report(self) -> str:
        """Markdown report of current allocations and posteriors."""
        stats = self.get_posterior_stats()
        alloc = self.get_allocation(5000)  # sample with $5K

        lines = [
            "# Adaptive Allocator Report",
            f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n",
            "| Strategy | Win Rate | 95% CI | Trades | PnL | $5K Alloc |",
            "|----------|----------|--------|--------|-----|-----------|",
        ]

        for s in sorted(self.strategies, key=lambda x: stats[x]["mean_win_rate"], reverse=True):
            st = stats[s]
            lines.append(
                f"| {s} | {st['mean_win_rate']*100:.1f}% | "
                f"[{st['ci_95_low']*100:.1f}%, {st['ci_95_high']*100:.1f}%] | "
                f"{st['trades']} | ${st['total_pnl']:.2f} | ${alloc.get(s, 0):.2f} |"
            )

        return "\n".join(lines)

    # --------------------------------------------------- Shadow log
    def _shadow_log(self, event: str, data: dict):
        try:
            entry = {"ts": time.time(), "event": event, **data}
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception:
            pass
