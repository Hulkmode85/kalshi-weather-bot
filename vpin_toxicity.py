"""
VPIN Toxicity — Volume-weighted Probability of Informed Trading.

Quant Concept:
    VPIN measures the imbalance between buy-initiated and sell-initiated volume.
    When informed traders (smart money) enter a market, they create directional
    volume pressure. High VPIN means one side is dominating — someone knows
    something. Market makers widen spreads when VPIN is high; we should too.

    VPIN is calculated by:
    1. Grouping trades into fixed-volume buckets
    2. Classifying each trade as buy or sell (tick rule or bulk classification)
    3. Computing volume imbalance across recent buckets
    4. Normalizing to 0-1 scale

    Score interpretation:
        0.0-0.3: Normal flow, safe to trade
        0.3-0.5: Elevated, proceed with caution
        0.5-0.7: High toxicity, widen spreads / reduce size
        0.7-1.0: Extreme toxicity, consider sitting out

    This module runs in SHADOW MODE — it does not affect actual trading.
    It logs every evaluation to a JSONL file for future optimization.

Usage:
    from vpin_toxicity import VPINTracker

    vpin = VPINTracker(bucket_size=50, num_buckets=50)
    vpin.add_trade(volume=10, direction="buy", price=0.65)
    score = vpin.get_vpin()
"""

import json
import os
import time
from pathlib import Path

LOG_DIR = os.environ.get("SHADOW_LOG_DIR", "/tmp/quant_shadow_logs")
LOG_FILE = os.path.join(LOG_DIR, "vpin_toxicity.jsonl")


def shadow_log(entry: dict):
    """Append a JSON line to the shadow log file."""
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    entry["_module"] = "vpin_toxicity"
    entry["_ts"] = time.time()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


class VPINTracker:
    """
    Tracks Volume-weighted Probability of Informed Trading (VPIN).

    VPIN groups trades into fixed-volume buckets, measures buy/sell imbalance
    in each bucket, then averages across recent buckets to detect informed flow.
    """

    def __init__(self, bucket_size: float = 50.0, num_buckets: int = 50):
        """
        Args:
            bucket_size: Volume threshold for each bucket (in contracts or USD).
            num_buckets: Number of recent buckets to use for VPIN calculation.
        """
        self.bucket_size = bucket_size
        self.num_buckets = num_buckets

        # Completed buckets: list of (buy_volume, sell_volume)
        self.buckets = []

        # Current (incomplete) bucket accumulator
        self._current_buy = 0.0
        self._current_sell = 0.0
        self._current_total = 0.0

        # Raw trade log for analysis
        self._trades = []
        self._max_trades = 5000

        # Last price for tick-rule classification
        self._last_price = None

    def add_trade(self, volume: float, direction: str = None, price: float = None):
        """
        Add a trade to VPIN calculation.

        Args:
            volume: Trade volume (contracts or USD).
            direction: "buy" or "sell". If None, uses tick rule based on price.
            price: Trade price. Required if direction is None.
        """
        if volume <= 0:
            return

        # Classify direction
        if direction is None and price is not None:
            direction = self._tick_classify(price)
        elif direction is None:
            direction = "buy"  # default if no info

        direction = direction.lower()

        # Record trade
        self._trades.append({
            "volume": volume,
            "direction": direction,
            "price": price,
            "ts": time.time(),
        })
        if len(self._trades) > self._max_trades:
            self._trades = self._trades[-self._max_trades // 2:]

        if price is not None:
            self._last_price = price

        # Fill buckets
        remaining = volume
        while remaining > 0:
            space_in_bucket = self.bucket_size - self._current_total
            fill = min(remaining, space_in_bucket)

            if direction == "buy":
                self._current_buy += fill
            else:
                self._current_sell += fill

            self._current_total += fill
            remaining -= fill

            # Bucket full? Finalize it
            if self._current_total >= self.bucket_size:
                self.buckets.append((self._current_buy, self._current_sell))
                self._current_buy = 0.0
                self._current_sell = 0.0
                self._current_total = 0.0

                # Keep bounded
                if len(self.buckets) > self.num_buckets * 3:
                    self.buckets = self.buckets[-self.num_buckets * 2:]

    def get_vpin(self) -> float:
        """
        Calculate current VPIN score.

        VPIN = average(|buy_vol - sell_vol| / bucket_size) across recent buckets.

        Returns:
            VPIN score 0-1. Higher = more toxic (informed) flow.
        """
        if len(self.buckets) < 2:
            shadow_log({
                "action": "get_vpin",
                "vpin_score": 0.0,
                "num_buckets": len(self.buckets),
                "reason": "insufficient_buckets",
                "recommendation": "INSUFFICIENT_DATA",
            })
            return 0.0

        recent = self.buckets[-self.num_buckets:]

        imbalances = []
        for buy_vol, sell_vol in recent:
            total = buy_vol + sell_vol
            if total > 0:
                imbalance = abs(buy_vol - sell_vol) / total
                imbalances.append(imbalance)

        if not imbalances:
            return 0.0

        vpin = sum(imbalances) / len(imbalances)
        vpin = max(0.0, min(1.0, vpin))

        # Determine recommendation
        if vpin >= 0.7:
            recommendation = "EXTREME_TOXICITY_SIT_OUT"
        elif vpin >= 0.5:
            recommendation = "HIGH_TOXICITY_REDUCE_SIZE"
        elif vpin >= 0.3:
            recommendation = "ELEVATED_CAUTION"
        else:
            recommendation = "NORMAL_FLOW"

        # Calculate bucket-level details
        recent_buys = sum(b for b, s in recent)
        recent_sells = sum(s for b, s in recent)
        total_vol = recent_buys + recent_sells

        shadow_log({
            "action": "get_vpin",
            "vpin_score": round(vpin, 4),
            "num_buckets_used": len(recent),
            "bucket_imbalances": [round(x, 4) for x in imbalances[-10:]],
            "total_buy_volume": round(recent_buys, 2),
            "total_sell_volume": round(recent_sells, 2),
            "buy_pct": round(recent_buys / total_vol * 100, 2) if total_vol > 0 else 0,
            "recommendation": recommendation,
        })

        return vpin

    def get_flow_direction(self) -> str:
        """
        Determine the dominant flow direction from recent buckets.

        Returns:
            "BUY_PRESSURE", "SELL_PRESSURE", or "BALANCED"
        """
        if len(self.buckets) < 3:
            return "BALANCED"

        recent = self.buckets[-self.num_buckets:]
        total_buy = sum(b for b, s in recent)
        total_sell = sum(s for b, s in recent)
        total = total_buy + total_sell

        if total == 0:
            return "BALANCED"

        buy_pct = total_buy / total
        if buy_pct > 0.6:
            return "BUY_PRESSURE"
        elif buy_pct < 0.4:
            return "SELL_PRESSURE"
        return "BALANCED"

    def _tick_classify(self, price: float) -> str:
        """Classify trade direction using tick rule: uptick = buy, downtick = sell."""
        if self._last_price is None:
            return "buy"
        if price > self._last_price:
            return "buy"
        elif price < self._last_price:
            return "sell"
        return "buy"  # unchanged = assume continuation

    def reset(self):
        """Clear all state."""
        self.buckets = []
        self._current_buy = 0.0
        self._current_sell = 0.0
        self._current_total = 0.0
        self._trades = []
        self._last_price = None
