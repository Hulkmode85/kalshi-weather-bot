"""
cross_platform_vol_arb.py — Cross-Platform Volatility Arbitrage.

Quant Concept:
    Different prediction market platforms often price the same event
    differently. When implied volatility (measured as price dispersion)
    diverges across platforms, an arbitrage opportunity exists: buy the
    cheap vol, sell the expensive vol.

    This module creates a synthetic volatility index by measuring
    probability range divergence across platforms (Kalshi, Thales, Buffer,
    Azuro). It adjusts for platform-specific fees and tracks vol regime
    transitions.

    Vol regime behavior:
        - Vol typically spikes 12-48 hours before events
        - Vol crashes within 1 hour after event resolution
        - Mean reversion is the dominant dynamic between events

    Entry: when std dev of mid prices across platforms > 0.15 (15 pp)
    Exit: when spread collapses below 0.05 or event resolves

    Platform fees (deducted from profit):
        Kalshi: 1.75%
        Thales: 2.00%
        Buffer: 2.50%
        Azuro:  3.00%

Usage:
    from cross_platform_vol_arb import CrossPlatformVolArb

    arb = CrossPlatformVolArb()
    arb.update_platform("kalshi", market_id="BTC-50K", bid=0.58, ask=0.62,
                        volume=5000, last_trade_time=time.time())
    arb.update_platform("thales", market_id="BTC-50K-TH", bid=0.50, ask=0.55,
                        volume=2000, last_trade_time=time.time())
    result = arb.evaluate()
    # result = {
    #     "vol_spread": 0.175,
    #     "signal": "BUY_LOW_VOL",
    #     "confidence": 0.78,
    #     "recommended_size": 85.0,
    #     "buy_platform": "thales",
    #     "sell_platform": "kalshi",
    #     ...
    # }
"""

import json
import math
import os
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --- Shadow logging ---
LOG_DIR = os.environ.get("SHADOW_LOG_DIR", "/tmp/quant_shadow_logs")
LOG_FILE = os.path.join(LOG_DIR, "cross_platform_vol_arb.jsonl")


def shadow_log(entry: dict):
    """Append a JSON line to the shadow log file."""
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    entry["_module"] = "cross_platform_vol_arb"
    entry["_ts"] = time.time()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# Platform fee schedule (as decimals)
PLATFORM_FEES = {
    "kalshi": 0.0175,
    "thales": 0.0200,
    "buffer": 0.0250,
    "azuro": 0.0300,
}


class CrossPlatformVolArb:
    """
    Cross-platform volatility arbitrage detector.

    Compares implied probability/price across prediction market platforms,
    identifies divergences, and generates arb signals adjusted for fees
    and vol regime.
    """

    def __init__(
        self,
        entry_threshold: float = 0.15,
        exit_threshold: float = 0.05,
        max_history: int = 500,
        vol_spike_window_hours: Tuple[float, float] = (12.0, 48.0),
        vol_crash_window_hours: float = 1.0,
        base_position_size: float = 100.0,
        max_position_size: float = 500.0,
        mode: str = "paper",
        log_dir: Optional[str] = None,
    ):
        """
        Args:
            entry_threshold: min std dev of prices across platforms to enter.
            exit_threshold: std dev below which to exit.
            max_history: max snapshots per platform to retain.
            vol_spike_window_hours: (min, max) hours before event where vol spikes.
            vol_crash_window_hours: hours after event where vol crashes.
            base_position_size: base dollar size for positions.
            max_position_size: max dollar size cap.
            mode: 'paper' or 'live'.
            log_dir: override shadow log directory.
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.max_history = max_history
        self.vol_spike_window = vol_spike_window_hours
        self.vol_crash_window = vol_crash_window_hours
        self.base_position_size = base_position_size
        self.max_position_size = max_position_size
        self.mode = mode

        if log_dir:
            global LOG_DIR, LOG_FILE
            LOG_DIR = log_dir
            LOG_FILE = os.path.join(LOG_DIR, "cross_platform_vol_arb.jsonl")

        # Current state per platform: {platform_name: {market_id, bid, ask, volume, ...}}
        self.platforms: Dict[str, Dict] = {}

        # Historical snapshots for vol regime tracking
        self.vol_history: deque = deque(maxlen=max_history)

        # Event tracking
        self.event_time: Optional[float] = None  # expected event resolution time

        # Tracking
        self._eval_count: int = 0

    # --------------------------------------------------------- Input
    def update_platform(
        self,
        platform: str,
        market_id: str,
        bid: float,
        ask: float,
        volume: float = 0.0,
        last_trade_time: Optional[float] = None,
    ):
        """Update price data for a platform.

        Args:
            platform: platform name (kalshi, thales, buffer, azuro).
            market_id: market identifier on that platform.
            bid: best bid price (probability 0-1).
            ask: best ask price (probability 0-1).
            volume: recent volume.
            last_trade_time: epoch seconds of last trade.
        """
        platform = platform.lower()
        mid = (bid + ask) / 2.0
        spread = ask - bid
        fee = PLATFORM_FEES.get(platform, 0.02)

        self.platforms[platform] = {
            "market_id": market_id,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "spread": spread,
            "volume": volume,
            "last_trade_time": last_trade_time or time.time(),
            "fee": fee,
            "updated_at": time.time(),
        }

    def set_event_time(self, event_time: float):
        """Set the expected event resolution time (epoch seconds)."""
        self.event_time = event_time

    # -------------------------------------------- Core vol computation
    def _compute_vol_metrics(self) -> Dict:
        """Compute cross-platform volatility metrics.

        Returns dict with vol_spread, mid_prices, cheapest/most expensive
        platforms, and fee-adjusted spreads.
        """
        if len(self.platforms) < 2:
            return {
                "vol_spread": 0.0,
                "platform_count": len(self.platforms),
                "sufficient": False,
            }

        mids = {p: d["mid"] for p, d in self.platforms.items()}
        mid_values = list(mids.values())

        # Std dev of mid prices across platforms
        mean_mid = sum(mid_values) / len(mid_values)
        variance = sum((m - mean_mid) ** 2 for m in mid_values) / len(mid_values)
        std_dev = math.sqrt(variance)

        # Range
        price_range = max(mid_values) - min(mid_values)

        # Cheapest and most expensive
        sorted_platforms = sorted(mids.items(), key=lambda x: x[1])
        cheapest = sorted_platforms[0]
        most_expensive = sorted_platforms[-1]

        # Fee-adjusted spread (raw spread minus fees on both sides)
        buy_fee = PLATFORM_FEES.get(cheapest[0], 0.02)
        sell_fee = PLATFORM_FEES.get(most_expensive[0], 0.02)
        raw_spread = most_expensive[1] - cheapest[1]
        fee_adjusted_spread = raw_spread - buy_fee - sell_fee

        return {
            "vol_spread": round(std_dev, 4),
            "price_range": round(price_range, 4),
            "mean_mid": round(mean_mid, 4),
            "raw_spread": round(raw_spread, 4),
            "fee_adjusted_spread": round(fee_adjusted_spread, 4),
            "cheapest_platform": cheapest[0],
            "cheapest_mid": round(cheapest[1], 4),
            "expensive_platform": most_expensive[0],
            "expensive_mid": round(most_expensive[1], 4),
            "all_mids": {p: round(m, 4) for p, m in mids.items()},
            "platform_count": len(self.platforms),
            "sufficient": True,
        }

    # ------------------------------------------------ Vol regime
    def _detect_vol_regime(self, vol_spread: float) -> Dict:
        """Detect current volatility regime based on history and event timing.

        Returns dict with regime, hours_to_event, regime_confidence.
        """
        now = time.time()

        # Record this snapshot
        self.vol_history.append({
            "vol_spread": vol_spread,
            "timestamp": now,
        })

        # Event-based regime
        if self.event_time is not None:
            hours_to_event = (self.event_time - now) / 3600.0
            hours_since_event = -hours_to_event  # positive if event passed

            if hours_since_event > 0 and hours_since_event <= self.vol_crash_window:
                return {
                    "regime": "POST_EVENT_CRASH",
                    "hours_to_event": round(hours_to_event, 2),
                    "regime_confidence": 0.9,
                    "description": "Vol crashing post-event, close positions",
                }
            elif self.vol_spike_window[0] <= hours_to_event <= self.vol_spike_window[1]:
                return {
                    "regime": "PRE_EVENT_SPIKE",
                    "hours_to_event": round(hours_to_event, 2),
                    "regime_confidence": 0.7,
                    "description": "Vol likely to spike, opportunity forming",
                }
            elif 0 < hours_to_event < self.vol_spike_window[0]:
                return {
                    "regime": "IMMINENT_EVENT",
                    "hours_to_event": round(hours_to_event, 2),
                    "regime_confidence": 0.8,
                    "description": "Event imminent, vol at peak, prepare to exit",
                }

        # History-based regime
        if len(self.vol_history) >= 5:
            recent_vols = [h["vol_spread"] for h in list(self.vol_history)[-10:]]
            avg_vol = sum(recent_vols) / len(recent_vols)
            trend = recent_vols[-1] - recent_vols[0]

            if avg_vol > self.entry_threshold and trend > 0:
                return {
                    "regime": "VOL_EXPANDING",
                    "hours_to_event": None,
                    "regime_confidence": 0.6,
                    "description": "Vol expanding, arb opportunities growing",
                }
            elif avg_vol > self.entry_threshold and trend <= 0:
                return {
                    "regime": "VOL_CONTRACTING",
                    "hours_to_event": None,
                    "regime_confidence": 0.6,
                    "description": "Vol high but contracting, be cautious",
                }

        return {
            "regime": "NORMAL",
            "hours_to_event": None,
            "regime_confidence": 0.5,
            "description": "Normal vol environment",
        }

    # ------------------------------------------------ Position sizing
    def _compute_size(self, fee_adjusted_spread: float, confidence: float,
                      regime: str) -> float:
        """Compute recommended position size.

        Args:
            fee_adjusted_spread: spread after fees.
            confidence: signal confidence.
            regime: current vol regime.

        Returns dollar amount for position.
        """
        if fee_adjusted_spread <= 0:
            return 0.0

        # Base size scaled by spread magnitude and confidence
        spread_factor = min(fee_adjusted_spread / self.entry_threshold, 2.0)
        size = self.base_position_size * spread_factor * confidence

        # Regime adjustments
        regime_multipliers = {
            "PRE_EVENT_SPIKE": 1.3,
            "VOL_EXPANDING": 1.2,
            "NORMAL": 1.0,
            "VOL_CONTRACTING": 0.7,
            "IMMINENT_EVENT": 0.5,
            "POST_EVENT_CRASH": 0.0,
        }
        size *= regime_multipliers.get(regime, 1.0)

        # Paper mode: push to max for data collection
        if self.mode == "paper":
            size = max(size, self.base_position_size)

        return round(min(size, self.max_position_size), 2)

    # ------------------------------------------------ Evaluate
    def evaluate(self) -> Dict:
        """Run cross-platform volatility arbitrage evaluation.

        Returns:
            dict with vol_spread, signal, confidence, recommended_size,
            buy_platform, sell_platform, vol_regime, fee details, and mode.
        """
        self._eval_count += 1

        vol_metrics = self._compute_vol_metrics()

        if not vol_metrics["sufficient"]:
            result = {
                "vol_spread": 0.0,
                "signal": "NEUTRAL",
                "confidence": 0.0,
                "recommended_size": 0.0,
                "buy_platform": None,
                "sell_platform": None,
                "vol_regime": "UNKNOWN",
                "platform_count": vol_metrics["platform_count"],
                "mode": self.mode,
                "eval_count": self._eval_count,
                "reason": f"need 2+ platforms, have {vol_metrics['platform_count']}",
            }
            shadow_log({"event": "evaluate", "result": result})
            return result

        vol_regime = self._detect_vol_regime(vol_metrics["vol_spread"])

        # Determine signal
        signal, confidence, reason = self._determine_signal(vol_metrics, vol_regime)

        # Position sizing
        recommended_size = self._compute_size(
            vol_metrics["fee_adjusted_spread"],
            confidence,
            vol_regime["regime"],
        )

        # Staleness check
        stale_platforms = self._check_staleness()

        result = {
            "vol_spread": vol_metrics["vol_spread"],
            "price_range": vol_metrics["price_range"],
            "raw_spread": vol_metrics["raw_spread"],
            "fee_adjusted_spread": vol_metrics["fee_adjusted_spread"],
            "signal": signal,
            "confidence": round(confidence, 4),
            "recommended_size": recommended_size,
            "buy_platform": vol_metrics["cheapest_platform"] if signal == "BUY_LOW_VOL" else None,
            "sell_platform": vol_metrics["expensive_platform"] if signal == "BUY_LOW_VOL" else None,
            "buy_price": vol_metrics["cheapest_mid"] if signal == "BUY_LOW_VOL" else None,
            "sell_price": vol_metrics["expensive_mid"] if signal == "BUY_LOW_VOL" else None,
            "vol_regime": vol_regime["regime"],
            "vol_regime_detail": vol_regime["description"],
            "all_mids": vol_metrics["all_mids"],
            "stale_platforms": stale_platforms,
            "platform_count": vol_metrics["platform_count"],
            "mode": self.mode,
            "eval_count": self._eval_count,
            "reason": reason,
        }

        shadow_log({"event": "evaluate", "result": result})
        return result

    def _determine_signal(self, vol_metrics: Dict, vol_regime: Dict) -> Tuple[str, float, str]:
        """Determine arb signal from vol metrics and regime.

        Returns (signal, confidence, reason).
        """
        vol_spread = vol_metrics["vol_spread"]
        fee_adj = vol_metrics["fee_adjusted_spread"]
        regime = vol_regime["regime"]

        # No trade if post-event crash
        if regime == "POST_EVENT_CRASH":
            return "NEUTRAL", 0.1, "post_event_vol_crash"

        # No trade if fee-adjusted spread is negative (fees eat the profit)
        if fee_adj <= 0:
            return "NEUTRAL", 0.2, "fees_exceed_spread"

        # Entry: vol spread exceeds threshold
        if vol_spread >= self.entry_threshold:
            # Confidence based on:
            # 1. How far above threshold
            # 2. Fee-adjusted profitability
            # 3. Regime favorability
            threshold_excess = min((vol_spread - self.entry_threshold) / self.entry_threshold, 1.0)
            profit_factor = min(fee_adj / 0.05, 1.0)
            regime_conf = vol_regime["regime_confidence"]

            confidence = 0.3 * threshold_excess + 0.4 * profit_factor + 0.3 * regime_conf
            confidence = min(confidence, 1.0)

            # High vol with expanding regime = sell high vol (mean reversion)
            if regime in ("VOL_EXPANDING", "IMMINENT_EVENT"):
                return "SELL_HIGH_VOL", confidence * 0.9, f"vol_above_threshold_regime_{regime}"

            return "BUY_LOW_VOL", confidence, f"vol_above_threshold_{vol_spread:.3f}"

        # Below exit threshold
        if vol_spread < self.exit_threshold:
            return "NEUTRAL", 0.1, "vol_below_exit_threshold"

        # Between exit and entry: hold if already in, no new entry
        return "NEUTRAL", 0.3, "vol_in_between_thresholds"

    def _check_staleness(self, max_age_sec: float = 300.0) -> List[str]:
        """Check for stale platform data (>5min old by default)."""
        now = time.time()
        stale = []
        for platform, data in self.platforms.items():
            age = now - data["updated_at"]
            if age > max_age_sec:
                stale.append(platform)
        return stale

    # ------------------------------------------------ Batch input
    def update_all(self, platform_data: Dict[str, Dict]):
        """Bulk update all platforms at once.

        Args:
            platform_data: {platform_name: {market_id, bid, ask, volume, last_trade_time}}
        """
        for platform, data in platform_data.items():
            self.update_platform(
                platform=platform,
                market_id=data.get("market_id", ""),
                bid=data.get("bid", 0.0),
                ask=data.get("ask", 0.0),
                volume=data.get("volume", 0.0),
                last_trade_time=data.get("last_trade_time"),
            )

    # ------------------------------------------------ Utilities
    def get_fee_schedule(self) -> Dict[str, float]:
        """Return platform fee schedule."""
        return dict(PLATFORM_FEES)

    def reset(self):
        """Reset all state."""
        self.platforms.clear()
        self.vol_history.clear()
        self.event_time = None
        self._eval_count = 0

    def get_stats(self) -> Dict:
        """Return summary statistics."""
        platform_summaries = {}
        for p, d in self.platforms.items():
            platform_summaries[p] = {
                "mid": round(d["mid"], 4),
                "spread": round(d["spread"], 4),
                "volume": d["volume"],
                "fee": d["fee"],
                "age_sec": round(time.time() - d["updated_at"], 1),
            }
        return {
            "platform_count": len(self.platforms),
            "platforms": platform_summaries,
            "vol_snapshots": len(self.vol_history),
            "event_time_set": self.event_time is not None,
            "mode": self.mode,
        }
