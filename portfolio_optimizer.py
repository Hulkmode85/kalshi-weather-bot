"""
Portfolio Optimizer — Mean-variance optimization across all open positions.

Quant Concept:
    Mean-variance optimization (Markowitz, 1952) finds the portfolio allocation
    that maximizes expected return for a given level of risk, or equivalently
    minimizes risk for a given return target. The key insight: it's not just
    about picking the best individual trades, but about how trades interact.

    The optimization maximizes:
        E[return] - (risk_aversion / 2) * variance

    Subject to:
        - Total allocation <= available balance
        - No single position exceeds max_position_pct of balance
        - Minimum allocation threshold (avoid dust positions)

    This module runs in SHADOW MODE — it does not affect actual trading.
    It logs every evaluation to a JSONL file for future optimization.

Usage:
    from portfolio_optimizer import PortfolioOptimizer

    optimizer = PortfolioOptimizer()
    allocations = optimizer.optimize(
        opportunities=[
            {"id": "BTC-50K", "expected_return": 0.08, "volatility": 0.15},
            {"id": "ETH-3K", "expected_return": 0.05, "volatility": 0.20},
        ],
        current_positions=[],
        balance=5000.0
    )
"""

import json
import math
import os
import time
from pathlib import Path

LOG_DIR = os.environ.get("SHADOW_LOG_DIR", "/tmp/quant_shadow_logs")
LOG_FILE = os.path.join(LOG_DIR, "portfolio_optimizer.jsonl")


def shadow_log(entry: dict):
    """Append a JSON line to the shadow log file."""
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    entry["_module"] = "portfolio_optimizer"
    entry["_ts"] = time.time()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


class PortfolioOptimizer:
    """
    Mean-variance portfolio optimizer for event contract markets.

    Finds optimal capital allocation across multiple trading opportunities
    by balancing expected return against risk (variance).
    """

    def __init__(
        self,
        risk_aversion: float = 2.0,
        max_position_pct: float = 0.25,
        min_allocation_usd: float = 10.0,
        max_correlation: float = 0.8,
    ):
        """
        Args:
            risk_aversion: Higher = more conservative. Typical range 1-5.
                           1 = aggressive, 3 = moderate, 5 = conservative.
            max_position_pct: Maximum fraction of balance in any single position.
            min_allocation_usd: Minimum allocation to bother with (avoid dust).
            max_correlation: Maximum allowed correlation between any two positions.
        """
        self.risk_aversion = risk_aversion
        self.max_position_pct = max_position_pct
        self.min_allocation_usd = min_allocation_usd
        self.max_correlation = max_correlation

    def optimize(
        self,
        opportunities: list,
        current_positions: list,
        balance: float,
    ) -> list:
        """
        Calculate optimal allocation using mean-variance optimization.

        Uses an analytical approximation for independent assets (diagonal covariance),
        with iterative adjustment for position limits.

        Args:
            opportunities: List of dicts, each with:
                - id (str): Market/opportunity identifier
                - expected_return (float): Expected return as decimal (e.g., 0.08 = 8%)
                - volatility (float): Estimated volatility/risk as decimal
                - correlation_group (str, optional): Group ID for correlated assets
            current_positions: List of dicts with:
                - id (str): Position identifier
                - current_allocation (float): Current USD allocated
                - unrealized_pnl (float): Current P&L
            balance: Total available balance in USD.

        Returns:
            List of dicts with:
                - id (str): Opportunity identifier
                - optimal_allocation_usd (float): Recommended allocation
                - optimal_weight (float): Fraction of balance (0-1)
                - expected_return (float): Input expected return
                - risk_adjusted_return (float): Return after risk penalty
                - sharpe_proxy (float): Return / volatility
        """
        if not opportunities or balance <= 0:
            shadow_log({
                "action": "optimize",
                "num_opportunities": len(opportunities) if opportunities else 0,
                "balance": balance,
                "result": "no_opportunities_or_zero_balance",
            })
            return []

        # Step 1: Calculate risk-adjusted score for each opportunity
        scored = []
        for opp in opportunities:
            exp_ret = opp.get("expected_return", 0)
            vol = opp.get("volatility", 0.1)
            vol = max(vol, 0.001)  # prevent division by zero

            # Sharpe-like ratio
            sharpe = exp_ret / vol

            # Mean-variance utility: E[r] - (lambda/2) * var
            utility = exp_ret - (self.risk_aversion / 2.0) * (vol ** 2)

            scored.append({
                "id": opp["id"],
                "expected_return": exp_ret,
                "volatility": vol,
                "sharpe_proxy": round(sharpe, 4),
                "utility": utility,
                "correlation_group": opp.get("correlation_group", opp["id"]),
            })

        # Step 2: Filter out negative-utility opportunities
        viable = [s for s in scored if s["utility"] > 0]
        if not viable:
            # All opportunities have negative risk-adjusted return
            shadow_log({
                "action": "optimize",
                "num_opportunities": len(opportunities),
                "num_viable": 0,
                "balance": balance,
                "result": "no_positive_utility_opportunities",
                "all_utilities": {s["id"]: round(s["utility"], 6) for s in scored},
            })
            return [{
                "id": s["id"],
                "optimal_allocation_usd": 0,
                "optimal_weight": 0,
                "expected_return": s["expected_return"],
                "risk_adjusted_return": s["utility"],
                "sharpe_proxy": s["sharpe_proxy"],
            } for s in scored]

        # Step 3: Allocate proportional to utility (analytical solution for diagonal cov)
        # Optimal weight_i = (E[r_i]) / (lambda * sigma_i^2) for independent assets
        raw_weights = {}
        for v in viable:
            w = v["expected_return"] / (self.risk_aversion * v["volatility"] ** 2)
            raw_weights[v["id"]] = max(0, w)

        # Step 4: Normalize weights and apply constraints
        total_raw = sum(raw_weights.values())
        if total_raw <= 0:
            return []

        # Normalize to sum to 1.0 (or less, keeping cash)
        allocations = []
        total_allocated = 0

        # Check correlation groups — limit total allocation to correlated assets
        group_allocations = {}

        for v in viable:
            weight = raw_weights[v["id"]] / total_raw

            # Cap at max position size
            weight = min(weight, self.max_position_pct)

            # Check correlation group cap
            group = v["correlation_group"]
            group_total = group_allocations.get(group, 0)
            if group_total + weight > self.max_position_pct * 1.5:
                weight = max(0, self.max_position_pct * 1.5 - group_total)

            group_allocations[group] = group_allocations.get(group, 0) + weight

            alloc_usd = weight * balance

            # Skip dust positions
            if alloc_usd < self.min_allocation_usd:
                weight = 0
                alloc_usd = 0

            total_allocated += weight

            allocations.append({
                "id": v["id"],
                "optimal_allocation_usd": round(alloc_usd, 2),
                "optimal_weight": round(weight, 6),
                "expected_return": v["expected_return"],
                "risk_adjusted_return": round(v["utility"], 6),
                "sharpe_proxy": v["sharpe_proxy"],
            })

        # Add zero allocations for non-viable opportunities
        viable_ids = {v["id"] for v in viable}
        for s in scored:
            if s["id"] not in viable_ids:
                allocations.append({
                    "id": s["id"],
                    "optimal_allocation_usd": 0,
                    "optimal_weight": 0,
                    "expected_return": s["expected_return"],
                    "risk_adjusted_return": round(s["utility"], 6),
                    "sharpe_proxy": s["sharpe_proxy"],
                })

        # Step 5: Compare with current positions
        current_map = {p["id"]: p.get("current_allocation", 0) for p in current_positions}
        rebalance_needed = []
        for a in allocations:
            current = current_map.get(a["id"], 0)
            delta = a["optimal_allocation_usd"] - current
            if abs(delta) > self.min_allocation_usd:
                rebalance_needed.append({
                    "id": a["id"],
                    "current_usd": round(current, 2),
                    "optimal_usd": a["optimal_allocation_usd"],
                    "delta_usd": round(delta, 2),
                    "action": "increase" if delta > 0 else "decrease",
                })

        # Sort by allocation (highest first)
        allocations.sort(key=lambda x: x["optimal_allocation_usd"], reverse=True)

        shadow_log({
            "action": "optimize",
            "num_opportunities": len(opportunities),
            "num_viable": len(viable),
            "balance": balance,
            "risk_aversion": self.risk_aversion,
            "total_allocated_pct": round(total_allocated * 100, 2),
            "cash_reserve_pct": round((1.0 - total_allocated) * 100, 2),
            "allocations": allocations,
            "rebalance_suggestions": rebalance_needed,
            "portfolio_expected_return": round(
                sum(a["expected_return"] * a["optimal_weight"] for a in allocations), 6
            ),
        })

        return allocations

    def kelly_fraction(self, probability: float, odds: float) -> float:
        """
        Calculate Kelly Criterion fraction for a single bet.

        Kelly fraction = (p * (odds + 1) - 1) / odds

        Args:
            probability: Estimated win probability (0-1).
            odds: Decimal odds (e.g., 2.0 for even money).

        Returns:
            Optimal fraction of bankroll to bet (can be negative = don't bet).
        """
        if odds <= 0:
            return 0.0
        f = (probability * (odds + 1) - 1) / odds
        return max(0.0, f)

    def fractional_kelly(self, probability: float, odds: float, fraction: float = 0.75) -> float:
        """
        Fractional Kelly for more conservative sizing.

        Args:
            probability: Win probability.
            odds: Decimal odds.
            fraction: Kelly fraction multiplier (0.75 = 3/4 Kelly).

        Returns:
            Recommended bet size as fraction of bankroll.
        """
        full_kelly = self.kelly_fraction(probability, odds)
        return full_kelly * fraction
