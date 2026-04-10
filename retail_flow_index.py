"""
retail_flow_index.py — Retail Participation Index for prediction markets.

Quant Concept:
    Retail traders exhibit distinctive patterns: small order sizes, round-number
    prices, market orders, clustering at psychological levels (25c, 50c, 75c, 90c).
    Institutional/smart money tends to use limit orders, odd sizes, and trades
    during off-hours. By tracking the ratio of retail vs smart-money flow, we can
    detect when retail is piling in (potential trap) or when smart money is quietly
    accumulating (follow signal).

    Signals:
    - retail_ratio >0.7 AND directional flow -> smart money likely to follow (FOLLOW_RETAIL)
    - retail_ratio >0.8 AND price near extreme (>90c or <10c) -> retail trap (FADE_RETAIL)
    - Otherwise -> NEUTRAL

    This module runs in SHADOW MODE by default — logs every evaluation to JSONL.

Usage:
    from retail_flow_index import RetailFlowIndex

    rfi = RetailFlowIndex()
    rfi.add_trade(price=0.65, size=10, side="buy", order_type="market", ts=1712700000)
    rfi.add_trade(price=0.66, size=500, side="buy", order_type="limit", ts=1712700300)
    result = rfi.evaluate()
    print(result)
    # {'retail_index': 0.72, 'smart_money_index': 0.28, 'signal': 'FOLLOW_RETAIL', ...}
"""

import json
import math
import os
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Psychological price levels in prediction markets (0-1 scale)
PSYCH_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
PSYCH_TOLERANCE = 0.02  # Within 2 cents of a round number

# Size thresholds (in contracts or dollars)
RETAIL_SIZE_MAX = 50     # Orders <= this are likely retail
INSTITUTIONAL_SIZE_MIN = 200  # Orders >= this are likely institutional

# Time-of-day weights (UTC hours). Higher = more retail activity expected.
# Retail peaks during US market hours (14-21 UTC = 9am-4pm ET)
RETAIL_HOUR_WEIGHTS = {
    h: 1.2 if 14 <= h <= 21 else (0.6 if 2 <= h <= 8 else 0.9)
    for h in range(24)
}

# ---------------------------------------------------------------------------
# Shadow logging
# ---------------------------------------------------------------------------

LOG_DIR = os.environ.get("SHADOW_LOG_DIR", "/tmp/quant_shadow_logs")
LOG_FILE = os.path.join(LOG_DIR, "retail_flow_index.jsonl")


def _shadow_log(entry: dict):
    """Append a JSON line to the shadow log file."""
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    entry["_module"] = "retail_flow_index"
    entry["_ts"] = time.time()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class RetailFlowIndex:
    """Tracks retail vs institutional order flow and generates trading signals."""

    def __init__(
        self,
        window: int = 200,
        mode: str = "paper",
        retail_size_max: int = RETAIL_SIZE_MAX,
        institutional_size_min: int = INSTITUTIONAL_SIZE_MIN,
        log_dir: Optional[str] = None,
    ):
        """
        Args:
            window: Max number of trades in rolling buffer.
            mode: 'paper' or 'live'.
            retail_size_max: Orders at or below this size score as retail.
            institutional_size_min: Orders at or above this size score as institutional.
            log_dir: Override shadow log directory.
        """
        self.window = window
        self.mode = mode
        self.retail_size_max = retail_size_max
        self.institutional_size_min = institutional_size_min
        self._trades: deque = deque(maxlen=window)

        if log_dir:
            global LOG_DIR, LOG_FILE
            LOG_DIR = log_dir
            LOG_FILE = os.path.join(LOG_DIR, "retail_flow_index.jsonl")

    # ------------------------------------------------------------ Add trades
    def add_trade(
        self,
        price: float,
        size: float,
        side: str = "buy",
        order_type: str = "market",
        ts: Optional[float] = None,
    ):
        """
        Record a single trade observation.

        Args:
            price: Trade price (0-1 for prediction markets).
            size: Order size in contracts or dollars.
            side: 'buy' or 'sell'.
            order_type: 'market' or 'limit'.
            ts: Unix timestamp. Defaults to now.
        """
        self._trades.append({
            "price": price,
            "size": size,
            "side": side.lower(),
            "order_type": order_type.lower(),
            "ts": ts or time.time(),
        })

    def add_trades(self, trades: List[Dict]):
        """Bulk-add trades. Each dict needs at least price, size, side, order_type."""
        for t in trades:
            self.add_trade(
                price=t["price"],
                size=t["size"],
                side=t.get("side", "buy"),
                order_type=t.get("order_type", "market"),
                ts=t.get("ts", t.get("timestamp")),
            )

    # ------------------------------------------------------------ Evaluate
    def evaluate(self, current_price: Optional[float] = None) -> Dict:
        """
        Compute retail flow index and generate signal.

        Args:
            current_price: Current market price (0-1). If None, uses latest trade price.

        Returns:
            dict with keys:
                retail_index (float 0-1): Estimated retail participation.
                smart_money_index (float 0-1): Estimated institutional participation.
                signal (str): FOLLOW_RETAIL / FADE_RETAIL / NEUTRAL.
                confidence (float 0-1): Signal confidence.
                flow_divergence (float): Directional imbalance (-1 to 1).
                retail_buy_ratio (float): Fraction of retail flow that is buying.
                trades_analyzed (int): Number of trades in window.
        """
        trades = list(self._trades)
        if len(trades) < 5:
            result = {
                "retail_index": 0.5,
                "smart_money_index": 0.5,
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "flow_divergence": 0.0,
                "retail_buy_ratio": 0.5,
                "trades_analyzed": len(trades),
                "note": "insufficient_data",
            }
            _shadow_log({"action": "evaluate", **result})
            return result

        if current_price is None:
            current_price = trades[-1]["price"]

        # Score each trade
        retail_scores = []
        retail_volume_buy = 0.0
        retail_volume_sell = 0.0
        smart_volume_buy = 0.0
        smart_volume_sell = 0.0

        for t in trades:
            score = self._retail_score(t)
            retail_scores.append(score)

            if score > 0.5:
                if t["side"] == "buy":
                    retail_volume_buy += t["size"]
                else:
                    retail_volume_sell += t["size"]
            else:
                if t["side"] == "buy":
                    smart_volume_buy += t["size"]
                else:
                    smart_volume_sell += t["size"]

        # Aggregate
        retail_index = sum(retail_scores) / len(retail_scores)
        smart_money_index = 1.0 - retail_index

        # Flow divergence: net buy pressure normalized (-1 = all sell, +1 = all buy)
        total_vol = retail_volume_buy + retail_volume_sell + smart_volume_buy + smart_volume_sell
        if total_vol > 0:
            net_buy = (retail_volume_buy + smart_volume_buy) - (retail_volume_sell + smart_volume_sell)
            flow_divergence = max(-1.0, min(1.0, net_buy / total_vol))
        else:
            flow_divergence = 0.0

        # Retail buy ratio
        retail_total = retail_volume_buy + retail_volume_sell
        retail_buy_ratio = retail_volume_buy / retail_total if retail_total > 0 else 0.5

        # Generate signal
        signal, confidence = self._generate_signal(
            retail_index, flow_divergence, current_price, retail_buy_ratio
        )

        result = {
            "retail_index": round(retail_index, 4),
            "smart_money_index": round(smart_money_index, 4),
            "signal": signal,
            "confidence": round(confidence, 4),
            "flow_divergence": round(flow_divergence, 4),
            "retail_buy_ratio": round(retail_buy_ratio, 4),
            "trades_analyzed": len(trades),
            "current_price": current_price,
        }
        _shadow_log({"action": "evaluate", **result})
        return result

    # -------------------------------------------------------------- Internal
    def _retail_score(self, trade: Dict) -> float:
        """
        Score a single trade as retail (1.0) or institutional (0.0).

        Factors:
        - Small size -> retail
        - Market order -> retail
        - Price near psychological level -> retail
        - Time of day adjustment
        """
        score = 0.0
        weight_total = 0.0

        # Size factor (weight: 3)
        w = 3.0
        weight_total += w
        if trade["size"] <= self.retail_size_max:
            score += w * 1.0
        elif trade["size"] >= self.institutional_size_min:
            score += w * 0.0
        else:
            # Linear interpolation
            frac = (trade["size"] - self.retail_size_max) / (self.institutional_size_min - self.retail_size_max)
            score += w * (1.0 - frac)

        # Order type factor (weight: 2)
        w = 2.0
        weight_total += w
        if trade["order_type"] == "market":
            score += w * 0.85  # Market orders strongly retail
        else:
            score += w * 0.30  # Limit orders lean institutional

        # Psychological price factor (weight: 1.5)
        w = 1.5
        weight_total += w
        near_psych = any(
            abs(trade["price"] - level) <= PSYCH_TOLERANCE
            for level in PSYCH_LEVELS
        )
        score += w * (0.75 if near_psych else 0.35)

        # Round size factor (weight: 1) — retail loves round numbers
        w = 1.0
        weight_total += w
        is_round = (trade["size"] % 5 == 0) and trade["size"] > 0
        score += w * (0.80 if is_round else 0.40)

        # Time-of-day adjustment (weight: 0.5)
        w = 0.5
        weight_total += w
        if trade.get("ts"):
            hour = int((trade["ts"] % 86400) / 3600)  # UTC hour
            tod_weight = RETAIL_HOUR_WEIGHTS.get(hour, 0.9)
            score += w * min(1.0, tod_weight)
        else:
            score += w * 0.5

        return score / weight_total if weight_total > 0 else 0.5

    @staticmethod
    def _generate_signal(
        retail_index: float,
        flow_divergence: float,
        current_price: float,
        retail_buy_ratio: float,
    ) -> tuple:
        """
        Generate trading signal from retail flow analysis.

        Returns:
            (signal, confidence) tuple.
        """
        is_extreme = current_price > 0.90 or current_price < 0.10
        is_directional = abs(flow_divergence) > 0.3

        # Retail trap detection
        if retail_index > 0.80 and is_extreme:
            confidence = min(1.0, retail_index * abs(flow_divergence) * 1.2)
            return "FADE_RETAIL", round(confidence, 4)

        # Follow retail signal
        if retail_index > 0.70 and is_directional:
            confidence = min(1.0, retail_index * abs(flow_divergence))
            return "FOLLOW_RETAIL", round(confidence, 4)

        # Weak signal zone
        confidence = max(0.0, (retail_index - 0.5) * abs(flow_divergence))
        return "NEUTRAL", round(confidence, 4)

    def clear(self):
        """Clear the trade buffer."""
        self._trades.clear()

    def get_log_path(self) -> str:
        """Return path to the shadow log file."""
        return LOG_FILE
