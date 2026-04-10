"""
order_flow_forecaster.py — Order Flow Imbalance Forecasting.

Quant Concept:
    Order flow imbalance measures the directional pressure in a market.
    When buy volume consistently exceeds sell volume (or vice versa), the
    market is likely to move in that direction. This module forecasts
    2-step-ahead imbalance using exponential weighted moving averages
    of signed volume, its momentum, and its acceleration.

    Unlike neural approaches, EWMA-based forecasting is:
    - Interpretable (you know exactly why it predicted what it did)
    - Fast (O(1) per update)
    - Robust (no overfitting to training data)

    Output signals:
        FOLLOW   — imbalance is persistent and strong, trade with it
        FADE     — imbalance flip predicted, counter-trend opportunity
        NEUTRAL  — no clear signal, sit tight

    When high imbalance persistence is detected, Kelly sizing should be
    increased (the edge is more reliable). When a flip is predicted,
    entry bands should be widened for counter-trend trades.

Usage:
    from order_flow_forecaster import OrderFlowForecaster

    ofc = OrderFlowForecaster()
    ofc.add_volume(buy_volume=100, sell_volume=40)
    ofc.add_volume(buy_volume=120, sell_volume=35)
    result = ofc.forecast()
    # result = {
    #     "predicted_imbalance": 0.72,
    #     "confidence": 0.85,
    #     "signal": "FOLLOW",
    #     "persistence_score": 0.91,
    #     "kelly_adjustment": 1.15,
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
LOG_FILE = os.path.join(LOG_DIR, "order_flow_forecaster.jsonl")


def shadow_log(entry: dict):
    """Append a JSON line to the shadow log file."""
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    entry["_module"] = "order_flow_forecaster"
    entry["_ts"] = time.time()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


class OrderFlowForecaster:
    """
    Forecasts 2-step-ahead order flow imbalance using EWMA of signed volume.

    Tracks buy/sell volume, computes net imbalance, momentum, and acceleration,
    then generates FOLLOW/FADE/NEUTRAL signals with confidence.
    """

    def __init__(
        self,
        fast_alpha: float = 0.3,
        slow_alpha: float = 0.1,
        max_history: int = 1000,
        persistence_window: int = 20,
        flip_threshold: float = 0.15,
        strong_imbalance_threshold: float = 0.4,
        mode: str = "paper",
        log_dir: Optional[str] = None,
    ):
        """
        Args:
            fast_alpha: EWMA decay for fast signal (higher = more reactive).
            slow_alpha: EWMA decay for slow signal (lower = smoother).
            max_history: max observations to retain.
            persistence_window: lookback for persistence score calculation.
            flip_threshold: momentum threshold to predict direction flip.
            strong_imbalance_threshold: imbalance above this = strong signal.
            mode: 'paper' or 'live'.
            log_dir: override shadow log directory.
        """
        self.fast_alpha = fast_alpha
        self.slow_alpha = slow_alpha
        self.max_history = max_history
        self.persistence_window = persistence_window
        self.flip_threshold = flip_threshold
        self.strong_imbalance_threshold = strong_imbalance_threshold
        self.mode = mode

        if log_dir:
            global LOG_DIR, LOG_FILE
            LOG_DIR = log_dir
            LOG_FILE = os.path.join(LOG_DIR, "order_flow_forecaster.jsonl")

        # Raw history
        self.observations: deque = deque(maxlen=max_history)

        # EWMA state
        self._ewma_fast: float = 0.0
        self._ewma_slow: float = 0.0
        self._ewma_momentum: float = 0.0
        self._ewma_accel: float = 0.0
        self._prev_momentum: float = 0.0

        # Tracking
        self._update_count: int = 0
        self._eval_count: int = 0

    # --------------------------------------------------------- Input
    def add_volume(
        self,
        buy_volume: float,
        sell_volume: float,
        timestamp: Optional[float] = None,
    ):
        """Record a volume observation.

        Args:
            buy_volume: volume on the buy side (positive).
            sell_volume: volume on the sell side (positive).
            timestamp: epoch seconds (defaults to now).
        """
        ts = timestamp or time.time()
        total = buy_volume + sell_volume

        # Net imbalance normalized to [-1, 1]
        if total > 0:
            imbalance = (buy_volume - sell_volume) / total
        else:
            imbalance = 0.0

        self.observations.append({
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "net_imbalance": imbalance,
            "total_volume": total,
            "timestamp": ts,
        })

        self._update_count += 1

        # Update EWMA signals
        self._update_ewma(imbalance)

    def _update_ewma(self, imbalance: float):
        """Update exponential weighted moving averages."""
        if self._update_count == 1:
            # Initialize
            self._ewma_fast = imbalance
            self._ewma_slow = imbalance
            self._ewma_momentum = 0.0
            self._ewma_accel = 0.0
            return

        # Fast and slow EWMA of imbalance
        self._ewma_fast = self.fast_alpha * imbalance + (1 - self.fast_alpha) * self._ewma_fast
        self._ewma_slow = self.slow_alpha * imbalance + (1 - self.slow_alpha) * self._ewma_slow

        # Momentum = fast - slow (crossover signal)
        new_momentum = self._ewma_fast - self._ewma_slow

        # Acceleration = change in momentum
        self._ewma_accel = new_momentum - self._ewma_momentum

        self._prev_momentum = self._ewma_momentum
        self._ewma_momentum = new_momentum

    # ------------------------------------------------ Persistence score
    def _compute_persistence(self) -> float:
        """Compute persistence score: how long imbalance has been directional.

        Returns value in [0, 1]. High = imbalance has been consistently
        in one direction for many periods.
        """
        if len(self.observations) < 3:
            return 0.0

        recent = list(self.observations)[-self.persistence_window:]
        imbalances = [o["net_imbalance"] for o in recent]

        if not imbalances:
            return 0.0

        # Count consecutive same-sign observations from the end
        last_sign = 1 if imbalances[-1] >= 0 else -1
        streak = 0
        for imb in reversed(imbalances):
            if (imb >= 0 and last_sign > 0) or (imb < 0 and last_sign < 0):
                streak += 1
            else:
                break

        # Persistence = streak / window, capped at 1
        persistence = min(streak / self.persistence_window, 1.0)

        # Weight by average magnitude (strong imbalance = more persistent signal)
        avg_magnitude = sum(abs(i) for i in imbalances[-streak:]) / max(streak, 1)
        weighted = persistence * (0.5 + 0.5 * min(avg_magnitude / self.strong_imbalance_threshold, 1.0))

        return round(min(weighted, 1.0), 4)

    # ------------------------------------------------ Forecast
    def forecast(self) -> Dict:
        """Generate 2-step-ahead imbalance forecast.

        Returns:
            dict with predicted_imbalance, confidence, signal, persistence_score,
            kelly_adjustment, current metrics, and mode.
        """
        self._eval_count += 1

        if self._update_count < 3:
            result = {
                "predicted_imbalance": 0.0,
                "confidence": 0.0,
                "signal": "NEUTRAL",
                "persistence_score": 0.0,
                "kelly_adjustment": 1.0,
                "imbalance_momentum": 0.0,
                "imbalance_acceleration": 0.0,
                "ewma_fast": 0.0,
                "ewma_slow": 0.0,
                "observations": self._update_count,
                "mode": self.mode,
                "eval_count": self._eval_count,
                "reason": "insufficient_data",
            }
            shadow_log({"event": "forecast", "result": result})
            return result

        # 2-step-ahead prediction using momentum extrapolation
        # predicted = current_ewma_fast + 2 * momentum + acceleration
        predicted = self._ewma_fast + 2 * self._ewma_momentum + self._ewma_accel
        predicted = max(-1.0, min(1.0, predicted))

        persistence = self._compute_persistence()

        # Confidence based on:
        # 1. Agreement between fast and slow (less divergence = more confident)
        # 2. Persistence (longer streak = more confident)
        # 3. Data quantity
        agreement = 1.0 - min(abs(self._ewma_fast - self._ewma_slow) * 2, 1.0)
        data_factor = min(self._update_count / 20, 1.0)
        confidence = (0.3 * agreement + 0.4 * persistence + 0.3 * data_factor)
        confidence = round(min(confidence, 1.0), 4)

        # Signal determination
        signal, reason = self._determine_signal(predicted, persistence)

        # Kelly adjustment
        kelly_adj = self._kelly_adjustment(persistence, confidence, signal)

        result = {
            "predicted_imbalance": round(predicted, 4),
            "confidence": confidence,
            "signal": signal,
            "persistence_score": persistence,
            "kelly_adjustment": round(kelly_adj, 4),
            "imbalance_momentum": round(self._ewma_momentum, 4),
            "imbalance_acceleration": round(self._ewma_accel, 4),
            "ewma_fast": round(self._ewma_fast, 4),
            "ewma_slow": round(self._ewma_slow, 4),
            "current_imbalance": round(self.observations[-1]["net_imbalance"], 4) if self.observations else 0.0,
            "observations": self._update_count,
            "mode": self.mode,
            "eval_count": self._eval_count,
            "reason": reason,
        }

        shadow_log({"event": "forecast", "result": result})
        return result

    def _determine_signal(self, predicted: float, persistence: float) -> Tuple[str, str]:
        """Determine trading signal from forecast.

        Returns (signal, reason) tuple.
        """
        abs_predicted = abs(predicted)
        abs_momentum = abs(self._ewma_momentum)

        # Flip detection: momentum reversing against current imbalance
        current_sign = 1 if self._ewma_fast >= 0 else -1
        momentum_sign = 1 if self._ewma_momentum >= 0 else -1

        if (current_sign != momentum_sign and
                abs_momentum > self.flip_threshold and
                abs_predicted < self.strong_imbalance_threshold):
            return "FADE", "momentum_reversal_detected"

        # Acceleration check: if acceleration opposes current direction
        accel_sign = 1 if self._ewma_accel >= 0 else -1
        if (current_sign != accel_sign and
                abs(self._ewma_accel) > self.flip_threshold * 0.5 and
                persistence < 0.3):
            return "FADE", "acceleration_reversal"

        # Strong persistent imbalance: FOLLOW
        if abs_predicted >= self.strong_imbalance_threshold and persistence >= 0.4:
            return "FOLLOW", "strong_persistent_imbalance"

        # Moderate imbalance with high persistence
        if abs_predicted >= 0.2 and persistence >= 0.6:
            return "FOLLOW", "moderate_imbalance_high_persistence"

        return "NEUTRAL", "no_clear_signal"

    def _kelly_adjustment(self, persistence: float, confidence: float, signal: str) -> float:
        """Compute Kelly sizing adjustment factor.

        Returns multiplier: >1 means increase size, <1 means decrease.
        """
        if signal == "FOLLOW" and persistence > 0.5:
            # Persistent flow = more reliable edge, increase sizing
            return 1.0 + 0.3 * persistence * confidence
        elif signal == "FADE":
            # Counter-trend = riskier, reduce sizing
            return max(0.5, 1.0 - 0.3 * (1.0 - persistence))
        return 1.0

    # ------------------------------------------------ Utilities
    def get_entry_band_adjustment(self) -> Dict:
        """Suggest entry band widening/tightening based on forecast.

        When a flip is predicted, widen entry bands for counter-trend trades.
        When strong persistence, tighten bands (confident in direction).
        """
        if self._update_count < 5:
            return {"adjustment": 0.0, "reason": "insufficient_data"}

        forecast = self.forecast()
        self._eval_count -= 1  # Don't double-count

        if forecast["signal"] == "FADE":
            # Widen bands: flip expected, need more room
            width = 0.05 + 0.10 * (1.0 - forecast["persistence_score"])
            return {
                "adjustment": round(width, 4),
                "direction": "widen",
                "reason": "flip_predicted",
            }
        elif forecast["signal"] == "FOLLOW" and forecast["persistence_score"] > 0.6:
            # Tighten bands: strong conviction
            tighten = -0.03 * forecast["persistence_score"]
            return {
                "adjustment": round(tighten, 4),
                "direction": "tighten",
                "reason": "high_persistence",
            }
        return {"adjustment": 0.0, "direction": "none", "reason": "neutral"}

    def reset(self):
        """Reset forecaster state."""
        self.observations.clear()
        self._ewma_fast = 0.0
        self._ewma_slow = 0.0
        self._ewma_momentum = 0.0
        self._ewma_accel = 0.0
        self._prev_momentum = 0.0
        self._update_count = 0
        self._eval_count = 0

    def get_stats(self) -> Dict:
        """Return summary statistics."""
        if not self.observations:
            return {"observation_count": 0}
        obs = list(self.observations)
        buy_total = sum(o["buy_volume"] for o in obs)
        sell_total = sum(o["sell_volume"] for o in obs)
        imbalances = [o["net_imbalance"] for o in obs]
        return {
            "observation_count": len(obs),
            "total_buy_volume": round(buy_total, 2),
            "total_sell_volume": round(sell_total, 2),
            "avg_imbalance": round(sum(imbalances) / len(imbalances), 4),
            "current_ewma_fast": round(self._ewma_fast, 4),
            "current_ewma_slow": round(self._ewma_slow, 4),
            "current_momentum": round(self._ewma_momentum, 4),
            "mode": self.mode,
        }
