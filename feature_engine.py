"""
Feature Engine — Extract advanced features from raw market data.

Quant Concept:
    Raw market data (price, volume, time) is noisy. Feature engineering transforms
    raw data into structured signals that ML models and decision rules can use.
    Features include cyclical time encoding (hour-of-day, day-of-week), momentum
    at multiple timeframes, volatility regime classification, volume acceleration,
    order book imbalance, and liquidity scoring.

    Every feature extraction is logged as a row of ML training data. Over time,
    this builds a dataset that can be used to train supervised models predicting
    market outcomes.

    This module runs in SHADOW MODE — it does not affect actual trading.
    It logs every evaluation to a JSONL file for future optimization.

Usage:
    from feature_engine import FeatureEngine

    engine = FeatureEngine()
    features = engine.extract(market_data)
    # features = {"hour_sin": 0.87, "momentum_5": 0.02, "vol_regime": "high", ...}
"""

import json
import math
import os
import time
from pathlib import Path

LOG_DIR = os.environ.get("SHADOW_LOG_DIR", "/tmp/quant_shadow_logs")
LOG_FILE = os.path.join(LOG_DIR, "feature_engine.jsonl")


def shadow_log(entry: dict):
    """Append a JSON line to the shadow log file."""
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    entry["_module"] = "feature_engine"
    entry["_ts"] = time.time()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


class FeatureEngine:
    """
    Extracts quant features from raw market data for ML training data collection.

    All features are designed to be:
    - Stationary (no trend contamination)
    - Bounded (suitable for ML without normalization)
    - Interpretable (each has a clear market meaning)
    """

    def extract(self, market_data: dict) -> dict:
        """
        Extract quant features from market data.

        Args:
            market_data: Dict with optional keys:
                - timestamp (float): Unix timestamp. Default: now.
                - current_price (float): Current price 0-1.
                - price_history (list[float]): Recent prices, newest last.
                - volume (float): Current period volume.
                - volume_history (list[float]): Recent volumes, newest last.
                - bid (float): Best bid price.
                - ask (float): Best ask price.
                - bid_size (float): Volume at best bid.
                - ask_size (float): Volume at best ask.
                - minutes_to_expiry (float): Minutes until expiry.
                - total_window_minutes (float): Total market window.
                - market_id (str): Market identifier for logging.

        Returns:
            Dict of feature_name -> value. All numeric except vol_regime (str).
        """
        features = {}
        ts = market_data.get("timestamp", time.time())
        price = market_data.get("current_price", 0.5)
        prices = market_data.get("price_history", [price])
        vol = market_data.get("volume", 0)
        volumes = market_data.get("volume_history", [vol])
        bid = market_data.get("bid", None)
        ask = market_data.get("ask", None)
        bid_size = market_data.get("bid_size", None)
        ask_size = market_data.get("ask_size", None)
        mins_to_expiry = market_data.get("minutes_to_expiry", None)
        total_window = market_data.get("total_window_minutes", None)

        # --- Time-of-day pattern (cyclical encoding) ---
        # Hour as sin/cos to capture cyclical nature (23:00 is close to 00:00)
        from datetime import datetime
        dt = datetime.fromtimestamp(ts)
        hour_frac = dt.hour + dt.minute / 60.0
        features["hour_sin"] = round(math.sin(2 * math.pi * hour_frac / 24), 6)
        features["hour_cos"] = round(math.cos(2 * math.pi * hour_frac / 24), 6)

        # --- Day-of-week effect ---
        # Monday=0, Sunday=6, encoded cyclically
        dow = dt.weekday()
        features["dow_sin"] = round(math.sin(2 * math.pi * dow / 7), 6)
        features["dow_cos"] = round(math.cos(2 * math.pi * dow / 7), 6)
        features["is_weekend"] = 1 if dow >= 5 else 0

        # --- Price momentum at multiple timeframes ---
        features["momentum_5"] = self._momentum(prices, 5)
        features["momentum_10"] = self._momentum(prices, 10)
        features["momentum_20"] = self._momentum(prices, 20)

        # --- Price acceleration (rate of change of momentum) ---
        if len(prices) >= 10:
            mom_recent = self._momentum(prices, 5)
            mom_prior = self._momentum(prices[:-5], 5) if len(prices) > 10 else 0
            features["momentum_acceleration"] = round(mom_recent - mom_prior, 6)
        else:
            features["momentum_acceleration"] = 0.0

        # --- Volume acceleration (rate of volume change) ---
        features["volume_acceleration"] = self._volume_acceleration(volumes)

        # --- Volume ratio (current vs average) ---
        if volumes and len(volumes) >= 2:
            avg_vol = sum(volumes) / len(volumes)
            features["volume_ratio"] = round(vol / avg_vol, 4) if avg_vol > 0 else 1.0
        else:
            features["volume_ratio"] = 1.0

        # --- Volatility regime ---
        vol_data = self._volatility_features(prices)
        features["rolling_std_10"] = vol_data["std_10"]
        features["rolling_std_20"] = vol_data["std_20"]
        features["vol_regime"] = vol_data["regime"]
        features["vol_regime_numeric"] = {"low": 0, "medium": 1, "high": 2}.get(vol_data["regime"], 1)

        # --- Order book depth imbalance ---
        if bid_size is not None and ask_size is not None:
            total_depth = bid_size + ask_size
            if total_depth > 0:
                # Positive = more buying pressure, negative = more selling
                features["book_imbalance"] = round((bid_size - ask_size) / total_depth, 4)
            else:
                features["book_imbalance"] = 0.0
        else:
            features["book_imbalance"] = 0.0

        # --- Spread as % of price (liquidity score) ---
        if bid is not None and ask is not None and price > 0:
            spread = ask - bid
            features["spread_pct"] = round(spread / price, 6) if price > 0 else 0
            features["spread_cents"] = round(spread * 100, 4)
        else:
            features["spread_pct"] = 0.0
            features["spread_cents"] = 0.0

        # --- Time features ---
        if mins_to_expiry is not None and total_window is not None and total_window > 0:
            time_frac = max(0, min(1, mins_to_expiry / total_window))
            features["time_fraction_remaining"] = round(time_frac, 4)
            features["time_urgency"] = round(1.0 - time_frac, 4)
            # Sqrt transform for non-linear time perception
            features["time_sqrt"] = round(math.sqrt(time_frac), 4)
        else:
            features["time_fraction_remaining"] = None
            features["time_urgency"] = None
            features["time_sqrt"] = None

        # --- Distance from 0.5 (how decided the market is) ---
        features["distance_from_half"] = round(abs(price - 0.5), 4)
        features["price_side"] = 1 if price >= 0.5 else 0

        # --- Mean reversion signal ---
        if len(prices) >= 10:
            mean_10 = sum(prices[-10:]) / 10
            features["reversion_signal_10"] = round(mean_10 - price, 6)
        else:
            features["reversion_signal_10"] = 0.0

        # Log everything for ML training data
        shadow_log({
            "action": "extract",
            "market_id": market_data.get("market_id", "unknown"),
            "features": {k: v for k, v in features.items()},
            "raw_price": price,
            "raw_volume": vol,
        })

        return features

    def _momentum(self, prices: list, period: int) -> float:
        """Calculate price momentum over N periods (percentage change)."""
        if len(prices) < period + 1:
            return 0.0
        old = prices[-period - 1]
        new = prices[-1]
        if old == 0:
            return 0.0
        return round((new - old) / abs(old), 6)

    def _volume_acceleration(self, volumes: list) -> float:
        """Calculate rate of change of volume."""
        if len(volumes) < 4:
            return 0.0
        recent_avg = sum(volumes[-2:]) / 2
        prior_avg = sum(volumes[-4:-2]) / 2
        if prior_avg == 0:
            return 0.0
        return round((recent_avg - prior_avg) / prior_avg, 4)

    def _volatility_features(self, prices: list) -> dict:
        """Calculate rolling volatility and classify regime."""
        result = {"std_10": 0.0, "std_20": 0.0, "regime": "medium"}

        if len(prices) >= 10:
            subset = prices[-10:]
            mean = sum(subset) / len(subset)
            var = sum((p - mean) ** 2 for p in subset) / len(subset)
            result["std_10"] = round(math.sqrt(var), 6)

        if len(prices) >= 20:
            subset = prices[-20:]
            mean = sum(subset) / len(subset)
            var = sum((p - mean) ** 2 for p in subset) / len(subset)
            result["std_20"] = round(math.sqrt(var), 6)

        # Regime classification based on 10-period vol
        std = result["std_10"]
        if std < 0.01:
            result["regime"] = "low"
        elif std > 0.03:
            result["regime"] = "high"
        else:
            result["regime"] = "medium"

        return result
