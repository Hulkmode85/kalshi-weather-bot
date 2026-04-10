"""
Correlation Matrix — Track cross-market correlations, limit correlated exposure.

Quant Concept:
    Diversification only works when assets are uncorrelated. If you hold 10
    positions that are all 90% correlated with BTC, you effectively have 1
    position with 10x the size. This module tracks rolling correlations
    between all asset pairs and calculates portfolio-level correlation risk.

    High portfolio correlation = concentrated risk = one bad move wipes all positions.
    Low portfolio correlation = true diversification = losses in one offset by others.

    This module runs in SHADOW MODE — it does not affect actual trading.
    It logs every evaluation to a JSONL file for future optimization.

Usage:
    from correlation_matrix import CorrelationTracker

    tracker = CorrelationTracker()
    tracker.update("BTC", 50000)
    tracker.update("ETH", 3000)
    corr = tracker.get_correlation("BTC", "ETH")
    risk = tracker.get_portfolio_correlation_risk(["BTC", "ETH", "SOL"])
"""

import json
import math
import os
import time
from pathlib import Path

LOG_DIR = os.environ.get("SHADOW_LOG_DIR", "/tmp/quant_shadow_logs")
LOG_FILE = os.path.join(LOG_DIR, "correlation_matrix.jsonl")


def shadow_log(entry: dict):
    """Append a JSON line to the shadow log file."""
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    entry["_module"] = "correlation_matrix"
    entry["_ts"] = time.time()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


class CorrelationTracker:
    """Tracks rolling correlations between assets and calculates portfolio risk."""

    def __init__(self, window: int = 30, max_history: int = 500):
        """
        Args:
            window: Rolling window size for correlation calculation.
            max_history: Max price observations to keep per asset.
        """
        self.price_history = {}  # asset -> [(timestamp, price)]
        self.window = window
        self.max_history = max_history
        self._corr_cache = {}    # (asset_a, asset_b) -> (timestamp, correlation)
        self._cache_ttl = 10.0   # seconds

    def update(self, asset: str, price: float, timestamp: float = None):
        """
        Record a price observation for an asset.

        Args:
            asset: Asset identifier (e.g., "BTC", "ETH").
            price: Current price.
            timestamp: Unix timestamp. Defaults to now.
        """
        timestamp = timestamp or time.time()

        if asset not in self.price_history:
            self.price_history[asset] = []

        self.price_history[asset].append((timestamp, price))

        # Trim history
        if len(self.price_history[asset]) > self.max_history:
            self.price_history[asset] = self.price_history[asset][-self.window * 2:]

        # Invalidate cache for this asset
        keys_to_remove = [k for k in self._corr_cache if asset in k]
        for k in keys_to_remove:
            del self._corr_cache[k]

    def get_correlation(self, asset_a: str, asset_b: str) -> float:
        """
        Return rolling Pearson correlation between two assets' returns.

        Uses percentage returns (not raw prices) for proper correlation.

        Args:
            asset_a: First asset identifier.
            asset_b: Second asset identifier.

        Returns:
            Correlation coefficient (-1 to 1). Returns 0.0 if insufficient data.
        """
        cache_key = tuple(sorted([asset_a, asset_b]))
        now = time.time()

        # Check cache
        if cache_key in self._corr_cache:
            cached_ts, cached_corr = self._corr_cache[cache_key]
            if now - cached_ts < self._cache_ttl:
                return cached_corr

        returns_a = self._get_returns(asset_a)
        returns_b = self._get_returns(asset_b)

        if not returns_a or not returns_b:
            return 0.0

        # Align to same length (use min of both)
        n = min(len(returns_a), len(returns_b), self.window)
        if n < 5:
            return 0.0

        ra = returns_a[-n:]
        rb = returns_b[-n:]

        corr = self._pearson(ra, rb)

        self._corr_cache[cache_key] = (now, corr)
        return corr

    def get_correlation_matrix(self, assets: list = None) -> dict:
        """
        Build full correlation matrix for given assets (or all tracked).

        Returns:
            Dict of {(asset_a, asset_b): correlation, ...}
        """
        if assets is None:
            assets = list(self.price_history.keys())

        matrix = {}
        for i, a in enumerate(assets):
            for j, b in enumerate(assets):
                if i <= j:
                    if a == b:
                        matrix[(a, b)] = 1.0
                    else:
                        matrix[(a, b)] = self.get_correlation(a, b)
                        matrix[(b, a)] = matrix[(a, b)]

        return matrix

    def get_portfolio_correlation_risk(self, open_positions: list) -> float:
        """
        Calculate how correlated current open positions are.

        Risk score:
            0.0 = perfectly uncorrelated (ideal diversification)
            1.0 = perfectly correlated (no diversification, maximum risk)

        Args:
            open_positions: List of asset identifiers currently held.

        Returns:
            Portfolio correlation risk score (0-1).
        """
        if len(open_positions) <= 1:
            shadow_log({
                "action": "portfolio_risk",
                "positions": open_positions,
                "risk_score": 0.0,
                "reason": "single_or_no_position",
            })
            return 0.0

        # Calculate average absolute pairwise correlation
        correlations = []
        pair_details = {}
        for i in range(len(open_positions)):
            for j in range(i + 1, len(open_positions)):
                a, b = open_positions[i], open_positions[j]
                corr = self.get_correlation(a, b)
                correlations.append(abs(corr))
                pair_details[f"{a}-{b}"] = round(corr, 4)

        if not correlations:
            return 0.0

        avg_abs_corr = sum(correlations) / len(correlations)
        risk_score = min(1.0, max(0.0, avg_abs_corr))

        shadow_log({
            "action": "portfolio_risk",
            "positions": open_positions,
            "num_pairs": len(correlations),
            "pair_correlations": pair_details,
            "avg_abs_correlation": round(avg_abs_corr, 4),
            "risk_score": round(risk_score, 4),
            "recommendation": (
                "HIGH_RISK" if risk_score > 0.7
                else "MODERATE_RISK" if risk_score > 0.4
                else "LOW_RISK"
            ),
        })

        return risk_score

    def _get_returns(self, asset: str) -> list:
        """Calculate percentage returns from price history."""
        history = self.price_history.get(asset, [])
        if len(history) < 2:
            return []

        returns = []
        for i in range(1, len(history)):
            p0 = history[i - 1][1]
            p1 = history[i][1]
            if p0 != 0:
                returns.append((p1 - p0) / abs(p0))
        return returns

    @staticmethod
    def _pearson(x: list, y: list) -> float:
        """Compute Pearson correlation coefficient between two lists."""
        n = len(x)
        if n < 2:
            return 0.0

        mean_x = sum(x) / n
        mean_y = sum(y) / n

        cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        var_x = sum((xi - mean_x) ** 2 for xi in x)
        var_y = sum((yi - mean_y) ** 2 for yi in y)

        denom = math.sqrt(var_x * var_y)
        if denom == 0:
            return 0.0

        return cov / denom
