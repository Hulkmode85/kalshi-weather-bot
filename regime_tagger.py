"""
regime_tagger.py — Tags market opportunities with regime metadata for per-regime analysis.

Tracks rolling price/volume/spread data and computes volatility regime, trend regime,
liquidity regime, time regime, ATR, and ADX.

Usage:
    tagger = RegimeTagger(window=100)
    tagger.update("BTC", price=67500, volume=1200, spread=0.05)
    tags = tagger.get_regime_tags("BTC")
"""

import json
import math
import os
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Optional


class RegimeTagger:
    """Tags every opportunity with the current market regime."""

    SHADOW_LOG = "regime_shadow.jsonl"

    def __init__(self, window: int = 100, log_dir: str = "."):
        """
        Args:
            window: max rolling history length for regime calculations.
            log_dir: directory for shadow JSONL log.
        """
        self.window = window
        self.price_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self.volume_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self.spread_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self.high_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self.low_history: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self.log_path = os.path.join(log_dir, self.SHADOW_LOG)

    # --------------------------------------------------------- Update
    def update(self, asset: str, price: float, volume: float = 0.0,
               spread: float = 0.0, high: Optional[float] = None,
               low: Optional[float] = None):
        """Update rolling data for regime calculation.

        Args:
            asset: asset identifier (e.g. "BTC", "ETH", market_id)
            price: current/close price
            volume: current period volume
            spread: bid-ask spread
            high: period high (defaults to price if not provided)
            low: period low (defaults to price if not provided)
        """
        self.price_history[asset].append(price)
        self.volume_history[asset].append(volume)
        self.spread_history[asset].append(spread)
        self.high_history[asset].append(high if high is not None else price)
        self.low_history[asset].append(low if low is not None else price)

    # --------------------------------------------------- Regime tags
    def get_regime_tags(self, asset: str) -> dict:
        """Return regime tags for current market state.

        Returns dict with keys: vol_regime, trend_regime, liquidity_regime,
        time_regime, day_regime, vol_percentile, atr_14, adx.
        """
        prices = list(self.price_history.get(asset, []))
        volumes = list(self.volume_history.get(asset, []))
        spreads = list(self.spread_history.get(asset, []))
        highs = list(self.high_history.get(asset, []))
        lows = list(self.low_history.get(asset, []))

        # Volatility regime — 20-period realized vol percentile
        vol_pct = self._vol_percentile(prices, period=20)
        if vol_pct >= 0.67:
            vol_regime = "high"
        elif vol_pct >= 0.33:
            vol_regime = "medium"
        else:
            vol_regime = "low"

        # Trend regime — 50-period linear regression slope
        trend_regime = self._trend_regime(prices, period=min(50, len(prices)))

        # Liquidity regime — spread percentile
        liq_regime = self._liquidity_regime(spreads)

        # Time regimes
        now = datetime.now(timezone.utc)
        time_regime = self._time_regime(now)
        day_regime = "weekend" if now.weekday() >= 5 else "weekday"

        # ATR-14
        atr = self._atr(highs, lows, prices, period=14)

        # ADX
        adx = self._adx(highs, lows, prices, period=14)

        tags = {
            "vol_regime": vol_regime,
            "trend_regime": trend_regime,
            "liquidity_regime": liq_regime,
            "time_regime": time_regime,
            "day_regime": day_regime,
            "vol_percentile": round(vol_pct, 4),
            "atr_14": round(atr, 6),
            "adx": round(adx, 2),
        }

        self._shadow_log("regime_tags", {"asset": asset, **tags})
        return tags

    # --------------------------------------------------- Volatility
    def _vol_percentile(self, prices: list[float], period: int = 20) -> float:
        """Compute realized vol percentile over rolling windows."""
        if len(prices) < period + 1:
            return 0.5  # neutral default

        # Compute log returns
        returns = [math.log(prices[i] / prices[i - 1])
                    for i in range(1, len(prices))
                    if prices[i - 1] > 0 and prices[i] > 0]

        if len(returns) < period:
            return 0.5

        # Rolling realized vol
        vols = []
        for i in range(period, len(returns) + 1):
            window = returns[i - period:i]
            mean_r = sum(window) / period
            var = sum((r - mean_r) ** 2 for r in window) / max(period - 1, 1)
            vols.append(math.sqrt(var))

        if not vols:
            return 0.5

        current_vol = vols[-1]
        rank = sum(1 for v in vols if v <= current_vol)
        return rank / len(vols)

    # --------------------------------------------------- Trend
    def _trend_regime(self, prices: list[float], period: int = 50) -> str:
        """Classify trend via linear regression slope on last `period` prices."""
        if len(prices) < max(period, 5):
            return "sideways"

        window = prices[-period:]
        n = len(window)
        x_mean = (n - 1) / 2.0
        y_mean = sum(window) / n

        numerator = sum((i - x_mean) * (window[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return "sideways"

        slope = numerator / denominator
        # Normalize slope by mean price to get relative slope
        rel_slope = slope / max(abs(y_mean), 1e-9)

        if rel_slope > 0.001:
            return "uptrend"
        elif rel_slope < -0.001:
            return "downtrend"
        else:
            return "sideways"

    # --------------------------------------------------- Liquidity
    def _liquidity_regime(self, spreads: list[float]) -> str:
        """Classify liquidity by spread percentile."""
        if not spreads or all(s == 0 for s in spreads):
            return "normal"

        current = spreads[-1]
        if current == 0:
            return "deep"

        rank = sum(1 for s in spreads if s <= current)
        pct = rank / len(spreads)

        # Lower spread = deeper liquidity
        if pct <= 0.33:
            return "deep"
        elif pct <= 0.67:
            return "normal"
        else:
            return "thin"

    # --------------------------------------------------- Time
    @staticmethod
    def _time_regime(now: datetime) -> str:
        """Classify time-of-day (US Eastern approximation, UTC-based)."""
        hour_utc = now.hour
        # Approximate US Eastern = UTC-4 (EDT) or UTC-5 (EST)
        hour_et = (hour_utc - 4) % 24

        if 6 <= hour_et < 10:
            return "morning"
        elif 10 <= hour_et < 13:
            return "midday"
        elif 13 <= hour_et < 16:
            return "afternoon"
        elif 16 <= hour_et < 20:
            return "evening"
        else:
            return "overnight"

    # --------------------------------------------------- ATR
    def _atr(self, highs: list[float], lows: list[float],
             closes: list[float], period: int = 14) -> float:
        """Average True Range over `period` bars."""
        n = min(len(highs), len(lows), len(closes))
        if n < 2:
            return 0.0

        true_ranges = []
        for i in range(1, n):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1])
            )
            true_ranges.append(tr)

        if not true_ranges:
            return 0.0

        # Exponential smoothing (Wilder's method)
        if len(true_ranges) < period:
            return sum(true_ranges) / len(true_ranges)

        atr_val = sum(true_ranges[:period]) / period
        for i in range(period, len(true_ranges)):
            atr_val = (atr_val * (period - 1) + true_ranges[i]) / period
        return atr_val

    # --------------------------------------------------- ADX
    def _adx(self, highs: list[float], lows: list[float],
             closes: list[float], period: int = 14) -> float:
        """Average Directional Index — measures trend strength (0-100)."""
        n = min(len(highs), len(lows), len(closes))
        if n < period + 1:
            return 0.0

        plus_dm = []
        minus_dm = []
        tr_list = []

        for i in range(1, n):
            up_move = highs[i] - highs[i - 1]
            down_move = lows[i - 1] - lows[i]

            pdm = up_move if (up_move > down_move and up_move > 0) else 0.0
            mdm = down_move if (down_move > up_move and down_move > 0) else 0.0
            plus_dm.append(pdm)
            minus_dm.append(mdm)

            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1])
            )
            tr_list.append(tr)

        if len(tr_list) < period:
            return 0.0

        # Wilder smoothing
        def wilder_smooth(data: list[float], p: int) -> list[float]:
            smoothed = [sum(data[:p])]
            for i in range(p, len(data)):
                smoothed.append(smoothed[-1] - smoothed[-1] / p + data[i])
            return smoothed

        sm_tr = wilder_smooth(tr_list, period)
        sm_pdm = wilder_smooth(plus_dm, period)
        sm_mdm = wilder_smooth(minus_dm, period)

        length = min(len(sm_tr), len(sm_pdm), len(sm_mdm))
        if length == 0:
            return 0.0

        dx_list = []
        for i in range(length):
            if sm_tr[i] == 0:
                continue
            plus_di = 100 * sm_pdm[i] / sm_tr[i]
            minus_di = 100 * sm_mdm[i] / sm_tr[i]
            di_sum = plus_di + minus_di
            if di_sum == 0:
                continue
            dx = 100 * abs(plus_di - minus_di) / di_sum
            dx_list.append(dx)

        if len(dx_list) < period:
            return sum(dx_list) / max(len(dx_list), 1)

        # Smooth DX into ADX
        adx_val = sum(dx_list[:period]) / period
        for i in range(period, len(dx_list)):
            adx_val = (adx_val * (period - 1) + dx_list[i]) / period

        return adx_val

    # --------------------------------------------------- Shadow log
    def _shadow_log(self, event: str, data: dict):
        try:
            entry = {"ts": time.time(), "event": event, **data}
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass
