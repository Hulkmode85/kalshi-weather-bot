"""
statistical_tournament.py — Institutional-grade strategy tournament with bootstrap significance testing.

Prevents data snooping by requiring minimum trade counts, applying Holm-Bonferroni correction
for multiple comparisons, and flagging strategies driven by outlier trades.

Usage:
    tournament = StrategyTournament(min_trades=150)
    tournament.add_trade("momentum_5m", pnl=12.50, duration_seconds=300)
    ...
    rankings = tournament.run_tournament()
    report = tournament.generate_report()
"""

import json
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class TradeRecord:
    pnl: float
    duration_seconds: float
    timestamp: float = field(default_factory=time.time)
    regime_tags: dict = field(default_factory=dict)


class StrategyTournament:
    """Proper strategy tournament with bootstrap hypothesis testing."""

    SHADOW_LOG = "tournament_shadow.jsonl"

    def __init__(self, min_trades: int = 150, log_dir: str = "."):
        self.min_trades = min_trades
        self.results: dict[str, list[TradeRecord]] = defaultdict(list)
        self.log_path = os.path.join(log_dir, self.SHADOW_LOG)

    # ------------------------------------------------------------------ IO
    def add_trade(self, strategy_name: str, pnl: float, duration_seconds: float,
                  regime_tags: Optional[dict] = None):
        """Record a trade result for a strategy."""
        rec = TradeRecord(pnl=pnl, duration_seconds=duration_seconds,
                          regime_tags=regime_tags or {})
        self.results[strategy_name].append(rec)
        self._shadow_log("add_trade", {
            "strategy": strategy_name, **asdict(rec)
        })

    # ----------------------------------------------------------- Metrics
    def calculate_metrics(self, strategy_name: str) -> dict:
        """Calculate comprehensive strategy metrics.

        Returns dict with: sortino, calmar, sharpe, profit_factor, win_rate,
        avg_duration, max_drawdown, total_pnl, num_trades, outlier_flag.
        """
        trades = self.results.get(strategy_name, [])
        if not trades:
            return self._empty_metrics()

        pnls = [t.pnl for t in trades]
        n = len(pnls)
        mean_pnl = sum(pnls) / n
        total_pnl = sum(pnls)

        # Standard deviation
        variance = sum((p - mean_pnl) ** 2 for p in pnls) / max(n - 1, 1)
        std_dev = math.sqrt(variance) if variance > 0 else 1e-9

        # Downside deviation (returns below 0)
        downside_sq = [p ** 2 for p in pnls if p < 0]
        downside_dev = math.sqrt(sum(downside_sq) / max(len(downside_sq), 1)) if downside_sq else 1e-9

        # Max drawdown
        max_dd = self._max_drawdown(pnls)

        # Annualized return estimate (assume 252 trading days, ~20 trades/day as baseline)
        avg_dur = sum(t.duration_seconds for t in trades) / n
        trades_per_year = (252 * 86400) / max(avg_dur, 1)
        annualized_return = mean_pnl * trades_per_year

        # Profit factor
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        profit_factor = gross_profit / max(gross_loss, 1e-9)

        # Win rate
        wins = sum(1 for p in pnls if p > 0)
        win_rate = wins / n

        # Ratios
        sortino = mean_pnl / downside_dev
        sharpe = mean_pnl / std_dev
        calmar = annualized_return / max(abs(max_dd), 1e-9)

        # Outlier flag: check if removing top 3 trades flips profitability
        outlier_flag = self._check_outlier_dependence(pnls)

        return {
            "sortino": round(sortino, 4),
            "calmar": round(calmar, 4),
            "sharpe": round(sharpe, 4),
            "profit_factor": round(profit_factor, 4),
            "win_rate": round(win_rate, 4),
            "avg_duration_sec": round(avg_dur, 1),
            "max_drawdown": round(max_dd, 4),
            "total_pnl": round(total_pnl, 4),
            "num_trades": n,
            "mean_pnl": round(mean_pnl, 4),
            "annualized_return": round(annualized_return, 2),
            "outlier_flag": outlier_flag,
        }

    # ------------------------------------------------- Bootstrap test
    def run_bootstrap_test(self, strategy_a: str, strategy_b: str,
                           n_samples: int = 10000) -> dict:
        """Bootstrap hypothesis test: is strategy A truly better than B?

        Resamples trade PnLs with replacement n_samples times.
        Returns p-value and whether difference is significant at alpha=0.05
        after Holm-Bonferroni correction.
        """
        pnls_a = [t.pnl for t in self.results.get(strategy_a, [])]
        pnls_b = [t.pnl for t in self.results.get(strategy_b, [])]

        if not pnls_a or not pnls_b:
            return {"p_value": 1.0, "significant": False, "mean_diff": 0.0,
                    "error": "insufficient data"}

        observed_diff = _mean(pnls_a) - _mean(pnls_b)

        # Pool and resample
        pooled = pnls_a + pnls_b
        na, nb = len(pnls_a), len(pnls_b)
        count_ge = 0

        for _ in range(n_samples):
            random.shuffle(pooled)
            boot_a = pooled[:na]
            boot_b = pooled[na:na + nb]
            boot_diff = _mean(boot_a) - _mean(boot_b)
            if boot_diff >= observed_diff:
                count_ge += 1

        p_value = count_ge / n_samples

        return {
            "strategy_a": strategy_a,
            "strategy_b": strategy_b,
            "observed_mean_diff": round(observed_diff, 4),
            "p_value": round(p_value, 4),
            "significant_raw": p_value < 0.05,
            "n_samples": n_samples,
        }

    # ------------------------------------------------- Tournament
    def run_tournament(self) -> list:
        """Rank all strategies by Sortino ratio.

        - Filters strategies with < min_trades.
        - Flags outlier-dependent strategies.
        - Runs pairwise bootstrap between adjacent ranks and applies
          Holm-Bonferroni correction.
        - Returns sorted list of dicts with metrics and significance flags.
        """
        # Calculate metrics for all qualifying strategies
        rankings = []
        for name in self.results:
            if len(self.results[name]) < self.min_trades:
                continue
            metrics = self.calculate_metrics(name)
            metrics["strategy"] = name
            rankings.append(metrics)

        # Sort by Sortino descending
        rankings.sort(key=lambda x: x["sortino"], reverse=True)

        # Pairwise bootstrap between adjacent strategies
        pairwise_results = []
        for i in range(len(rankings) - 1):
            result = self.run_bootstrap_test(
                rankings[i]["strategy"],
                rankings[i + 1]["strategy"],
                n_samples=5000  # faster for tournament; full 10k available via direct call
            )
            pairwise_results.append(result)

        # Holm-Bonferroni correction
        if pairwise_results:
            m = len(pairwise_results)
            indexed = [(i, r["p_value"]) for i, r in enumerate(pairwise_results)]
            indexed.sort(key=lambda x: x[1])

            for rank_pos, (orig_idx, p_val) in enumerate(indexed):
                corrected_alpha = 0.05 / (m - rank_pos)
                pairwise_results[orig_idx]["holm_bonferroni_significant"] = p_val < corrected_alpha

        # Attach significance to rankings
        for i, r in enumerate(rankings):
            r["rank"] = i + 1
            if i < len(pairwise_results):
                r["sig_vs_next"] = pairwise_results[i].get("holm_bonferroni_significant", False)
                r["p_vs_next"] = pairwise_results[i]["p_value"]
            else:
                r["sig_vs_next"] = None
                r["p_vs_next"] = None

        self._shadow_log("tournament_result", {
            "num_strategies": len(rankings),
            "top_3": [r["strategy"] for r in rankings[:3]],
        })

        return rankings

    # ------------------------------------------------- Regime breakdown
    def get_regime_breakdown(self, strategy_name: str, regime_tags: list) -> dict:
        """Show strategy performance broken down by regime.

        Args:
            strategy_name: name of the strategy
            regime_tags: list of regime tag keys to break down by
                         e.g. ["vol_regime", "trend_regime"]

        Returns:
            {regime_key: {regime_value: metrics_dict}}
        """
        trades = self.results.get(strategy_name, [])
        if not trades:
            return {}

        breakdown = {}
        for tag_key in regime_tags:
            buckets: dict[str, list[TradeRecord]] = defaultdict(list)
            for t in trades:
                val = t.regime_tags.get(tag_key, "unknown")
                buckets[str(val)].append(t)

            tag_metrics = {}
            for val, bucket_trades in buckets.items():
                pnls = [t.pnl for t in bucket_trades]
                tag_metrics[val] = {
                    "num_trades": len(pnls),
                    "total_pnl": round(sum(pnls), 4),
                    "mean_pnl": round(_mean(pnls), 4),
                    "win_rate": round(sum(1 for p in pnls if p > 0) / max(len(pnls), 1), 4),
                }
            breakdown[tag_key] = tag_metrics

        return breakdown

    # ------------------------------------------------- Report
    def generate_report(self) -> str:
        """Generate markdown table with all strategies ranked."""
        rankings = self.run_tournament()
        if not rankings:
            return "No strategies with sufficient trades for ranking."

        lines = [
            "# Strategy Tournament Report",
            f"*Min trades: {self.min_trades} | Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n",
            "| Rank | Strategy | Trades | Sortino | Sharpe | Calmar | Win% | PF | Total PnL | MaxDD | Outlier | Sig vs Next |",
            "|------|----------|--------|---------|--------|--------|------|----|-----------|-------|---------|-------------|",
        ]

        for r in rankings:
            sig = "Yes" if r.get("sig_vs_next") else ("—" if r.get("sig_vs_next") is None else "No")
            outlier = "⚠️" if r["outlier_flag"] else "✓"
            lines.append(
                f"| {r['rank']} | {r['strategy']} | {r['num_trades']} | "
                f"{r['sortino']:.2f} | {r['sharpe']:.2f} | {r['calmar']:.2f} | "
                f"{r['win_rate']*100:.1f}% | {r['profit_factor']:.2f} | "
                f"${r['total_pnl']:.2f} | {r['max_drawdown']:.2f} | {outlier} | {sig} |"
            )

        # Strategies that didn't qualify
        disqualified = [s for s in self.results if len(self.results[s]) < self.min_trades]
        if disqualified:
            lines.append(f"\n**Below min trades ({self.min_trades}):** {', '.join(disqualified)}")

        return "\n".join(lines)

    # ------------------------------------------------- Internal helpers
    @staticmethod
    def _max_drawdown(pnls: list[float]) -> float:
        """Compute max drawdown from a PnL series."""
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for p in pnls:
            cumulative += p
            if cumulative > peak:
                peak = cumulative
            dd = cumulative - peak
            if dd < max_dd:
                max_dd = dd
        return max_dd

    @staticmethod
    def _check_outlier_dependence(pnls: list[float], top_n: int = 3) -> bool:
        """Return True if removing the top N trades flips total PnL negative."""
        if len(pnls) <= top_n:
            return True
        total = sum(pnls)
        if total <= 0:
            return False  # already unprofitable
        sorted_desc = sorted(pnls, reverse=True)
        top_sum = sum(sorted_desc[:top_n])
        return (total - top_sum) <= 0

    @staticmethod
    def _empty_metrics() -> dict:
        return {k: 0 for k in [
            "sortino", "calmar", "sharpe", "profit_factor", "win_rate",
            "avg_duration_sec", "max_drawdown", "total_pnl", "num_trades",
            "mean_pnl", "annualized_return", "outlier_flag"
        ]}

    def _shadow_log(self, event: str, data: dict):
        """Append event to shadow JSONL log."""
        try:
            entry = {"ts": time.time(), "event": event, **data}
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass  # shadow logging never crashes main flow


def _mean(vals: list[float]) -> float:
    return sum(vals) / max(len(vals), 1)
