"""
adaptive_kelly.py — Dynamic Kelly Criterion Recalculation

Quant Concept:
    Fixed Kelly fractions leave money on the table when performance is strong
    and risk too much when performance is weak. This module adapts the Kelly
    fraction in real-time based on rolling Sharpe ratio vs a per-strategy
    target Sharpe. When the bot is crushing it, size up. When it's struggling,
    pull back automatically.

    Formula:
        kelly = base_kelly * min(1, (current_sharpe / target_sharpe) ** 0.5)
    Safety:
        If Sharpe drops below 60% of target, reduce to 0.3x baseline.
        Floor: never below 0.05 Kelly.

    This module runs in SHADOW MODE by default — it logs every evaluation
    to a JSONL file for analysis. In live mode, outputs drive actual sizing.

Usage:
    from adaptive_kelly import AdaptiveKelly

    ak = AdaptiveKelly()
    result = ak.evaluate(
        trades=[{"pnl": 12.5, "ts": 1712700000}, {"pnl": -3.2, "ts": 1712700060}],
        strategy_type="ARB"
    )
    print(result)
    # {'recommended_kelly': 0.62, 'sharpe_ratio': 1.85, 'regime': 'NORMAL', 'sizing_multiplier': 0.83}
"""

import json
import math
import os
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STRATEGY_TARGETS: Dict[str, float] = {
    "ARB": 2.0,
    "THETA": 1.5,
    "DIRECTIONAL": 1.2,
    "FLOW": 1.0,
    "MARKET_MAKER": 1.5,
}

STRATEGY_BASE_KELLY: Dict[str, float] = {
    "ARB": 0.75,
    "THETA": 0.50,
    "DIRECTIONAL": 0.25,
    "FLOW": 0.25,
    "MARKET_MAKER": 0.50,
}

KELLY_FLOOR = 0.05
CAUTIOUS_MULTIPLIER = 0.30
CAUTIOUS_THRESHOLD = 0.60  # Sharpe < 60% of target triggers cautious mode

# ---------------------------------------------------------------------------
# Shadow logging
# ---------------------------------------------------------------------------

LOG_DIR = os.environ.get("SHADOW_LOG_DIR", "/tmp/quant_shadow_logs")
LOG_FILE = os.path.join(LOG_DIR, "adaptive_kelly.jsonl")


def _shadow_log(entry: dict):
    """Append a JSON line to the shadow log file."""
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    entry["_module"] = "adaptive_kelly"
    entry["_ts"] = time.time()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class AdaptiveKelly:
    """Dynamically adjusts Kelly fraction based on rolling Sharpe performance."""

    def __init__(
        self,
        window: int = 50,
        mode: str = "paper",
        custom_targets: Optional[Dict[str, float]] = None,
        custom_base_kelly: Optional[Dict[str, float]] = None,
        log_dir: Optional[str] = None,
    ):
        """
        Args:
            window: Number of recent trades for rolling Sharpe calculation.
            mode: 'paper' or 'live'. In paper mode, all evaluations are shadow-logged.
            custom_targets: Override default target Sharpe per strategy.
            custom_base_kelly: Override default base Kelly per strategy.
            log_dir: Override shadow log directory.
        """
        self.window = window
        self.mode = mode
        self.targets = {**STRATEGY_TARGETS, **(custom_targets or {})}
        self.base_kelly = {**STRATEGY_BASE_KELLY, **(custom_base_kelly or {})}

        if log_dir:
            global LOG_DIR, LOG_FILE
            LOG_DIR = log_dir
            LOG_FILE = os.path.join(LOG_DIR, "adaptive_kelly.jsonl")

        # Rolling trade cache per strategy
        self._trade_cache: Dict[str, deque] = {}

    # ---------------------------------------------------------------- Public
    def evaluate(
        self,
        trades: List[Dict],
        strategy_type: str,
    ) -> Dict:
        """
        Compute adaptive Kelly fraction from recent trade results.

        Args:
            trades: List of dicts, each with at least 'pnl' (float) and
                    optionally 'ts' (unix timestamp). Most recent last.
            strategy_type: One of ARB, THETA, DIRECTIONAL, FLOW, MARKET_MAKER.

        Returns:
            dict with keys:
                recommended_kelly (float): The adaptive Kelly fraction.
                sharpe_ratio (float): Rolling annualized Sharpe.
                regime (str): NORMAL / CAUTIOUS / AGGRESSIVE.
                sizing_multiplier (float): Multiplier vs base Kelly (0-1+).
        """
        strategy_type = strategy_type.upper()
        base = self.base_kelly.get(strategy_type, 0.25)
        target = self.targets.get(strategy_type, 1.5)

        # Extract PnL series
        pnls = [t["pnl"] for t in trades if "pnl" in t]
        pnls = pnls[-self.window:]  # rolling window

        if len(pnls) < 3:
            # Not enough data — use base Kelly
            result = {
                "recommended_kelly": base,
                "sharpe_ratio": 0.0,
                "regime": "NORMAL",
                "sizing_multiplier": 1.0,
                "trades_used": len(pnls),
                "strategy": strategy_type,
                "note": "insufficient_data",
            }
            _shadow_log({"action": "evaluate", **result})
            return result

        sharpe = self._compute_sharpe(pnls)
        kelly, regime, multiplier = self._compute_kelly(sharpe, base, target)

        result = {
            "recommended_kelly": round(kelly, 4),
            "sharpe_ratio": round(sharpe, 4),
            "regime": regime,
            "sizing_multiplier": round(multiplier, 4),
            "trades_used": len(pnls),
            "strategy": strategy_type,
            "base_kelly": base,
            "target_sharpe": target,
        }
        _shadow_log({"action": "evaluate", **result})
        return result

    def add_trade(self, strategy_type: str, pnl: float, ts: Optional[float] = None):
        """Append a single trade to the rolling cache for a strategy."""
        strategy_type = strategy_type.upper()
        if strategy_type not in self._trade_cache:
            self._trade_cache[strategy_type] = deque(maxlen=self.window)
        self._trade_cache[strategy_type].append({
            "pnl": pnl,
            "ts": ts or time.time(),
        })

    def evaluate_cached(self, strategy_type: str) -> Dict:
        """Evaluate using the internal rolling trade cache."""
        strategy_type = strategy_type.upper()
        trades = list(self._trade_cache.get(strategy_type, []))
        return self.evaluate(trades, strategy_type)

    # --------------------------------------------------------------- Internal
    @staticmethod
    def _compute_sharpe(pnls: List[float], annualize_factor: float = 252.0) -> float:
        """
        Compute Sharpe ratio from a list of trade PnLs.

        Uses per-trade mean/std, annualized by sqrt(annualize_factor).
        If std is zero, returns a large positive Sharpe if mean > 0.
        """
        n = len(pnls)
        if n < 2:
            return 0.0
        mean_pnl = sum(pnls) / n
        var = sum((p - mean_pnl) ** 2 for p in pnls) / (n - 1)
        std = math.sqrt(var) if var > 0 else 0.0

        if std == 0.0:
            return 10.0 if mean_pnl > 0 else 0.0

        # Annualize: assume each trade is roughly one "period"
        sharpe = (mean_pnl / std) * math.sqrt(min(annualize_factor, n))
        return max(sharpe, 0.0)

    @staticmethod
    def _compute_kelly(
        sharpe: float,
        base: float,
        target: float,
    ) -> Tuple[float, str, float]:
        """
        Compute adaptive Kelly, regime label, and sizing multiplier.

        Returns:
            (kelly_fraction, regime, multiplier)
        """
        if target <= 0:
            target = 1.0

        ratio = sharpe / target if target > 0 else 0.0

        # Determine regime
        if ratio < CAUTIOUS_THRESHOLD:
            regime = "CAUTIOUS"
            kelly = base * CAUTIOUS_MULTIPLIER
            multiplier = CAUTIOUS_MULTIPLIER
        elif ratio >= 1.0:
            regime = "AGGRESSIVE"
            # Allow up to 1.0x base (cap at base, don't exceed)
            multiplier = min(1.0, ratio ** 0.5)
            kelly = base * multiplier
        else:
            regime = "NORMAL"
            multiplier = ratio ** 0.5
            kelly = base * multiplier

        # Floor
        kelly = max(kelly, KELLY_FLOOR)

        return kelly, regime, multiplier

    def get_log_path(self) -> str:
        """Return path to the shadow log file."""
        return LOG_FILE
