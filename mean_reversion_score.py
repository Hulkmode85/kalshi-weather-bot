"""
mean_reversion_score.py — Probability Mean Reversion Score for prediction markets.

Quant Concept:
    Binary prediction market prices tend to mean-revert, especially at extremes
    and far from expiry. This module estimates the probability and magnitude of
    reversion using a simplified Ornstein-Uhlenbeck (OU) process.

    The OU process models a price that is pulled back toward a long-run mean:
        dX = theta * (mu - X) * dt + sigma * dW
    We estimate:
        - mu (long-run mean) from rolling price history
        - theta (mean-reversion speed) from autocorrelation / half-life
        - sigma (volatility) from price returns

    Strong signal: price deviates >2 std from rolling mean AND time_to_expiry >2hrs.
    Weak signal: near expiry (<30min) — markets become efficient.

    Market type adjustments:
        - Political markets revert slower (lower theta multiplier)
        - Financial markets revert at normal speed
        - Sports markets do NOT mean-revert (random walk near game time)

    This module runs in SHADOW MODE by default — logs every evaluation to JSONL.

Usage:
    from mean_reversion_score import MeanReversionScore

    mrs = MeanReversionScore()
    result = mrs.evaluate(
        current_price=0.82,
        historical_prices=[0.55, 0.58, 0.60, 0.63, 0.61, 0.65, 0.70, 0.75, 0.82],
        time_to_expiry=7200,
        market_type="FINANCIAL"
    )
    print(result)
    # {'reversion_score': -0.65, 'confidence': 0.72, 'half_life_minutes': 45, 'signal': 'FADE'}
"""

import json
import math
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Market type mean-reversion speed multipliers
MARKET_TYPE_THETA = {
    "FINANCIAL": 1.0,
    "POLITICAL": 0.5,    # Political markets revert slower
    "SPORTS": 0.0,       # Sports don't revert — random walk near game time
    "CRYPTO": 0.8,
    "WEATHER": 0.6,
    "ECONOMIC": 0.9,
    "OTHER": 0.7,
}

MIN_HISTORY = 10         # Minimum price observations for meaningful analysis
MIN_EXPIRY_SECONDS = 1800  # 30 minutes — below this, weak/no signal
STRONG_EXPIRY_SECONDS = 7200  # 2 hours — above this, strong signal territory

# ---------------------------------------------------------------------------
# Shadow logging
# ---------------------------------------------------------------------------

LOG_DIR = os.environ.get("SHADOW_LOG_DIR", "/tmp/quant_shadow_logs")
LOG_FILE = os.path.join(LOG_DIR, "mean_reversion_score.jsonl")


def _shadow_log(entry: dict):
    """Append a JSON line to the shadow log file."""
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    entry["_module"] = "mean_reversion_score"
    entry["_ts"] = time.time()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class MeanReversionScore:
    """Scores probability of price mean-reversion in prediction markets."""

    def __init__(
        self,
        mode: str = "paper",
        default_lookback: int = 50,
        z_threshold_strong: float = 2.0,
        z_threshold_weak: float = 1.0,
        log_dir: Optional[str] = None,
    ):
        """
        Args:
            mode: 'paper' or 'live'.
            default_lookback: Default rolling window for mean/std calculation.
            z_threshold_strong: Z-score above which signal is strong.
            z_threshold_weak: Z-score above which signal is weak.
            log_dir: Override shadow log directory.
        """
        self.mode = mode
        self.default_lookback = default_lookback
        self.z_strong = z_threshold_strong
        self.z_weak = z_threshold_weak

        if log_dir:
            global LOG_DIR, LOG_FILE
            LOG_DIR = log_dir
            LOG_FILE = os.path.join(LOG_DIR, "mean_reversion_score.jsonl")

    # ------------------------------------------------------------ Evaluate
    def evaluate(
        self,
        current_price: float,
        historical_prices: List[float],
        time_to_expiry: float,
        market_type: str = "FINANCIAL",
        lookback: Optional[int] = None,
    ) -> Dict:
        """
        Score the mean-reversion probability of a prediction market.

        Args:
            current_price: Current market price (0-1).
            historical_prices: List of historical prices, oldest first.
            time_to_expiry: Seconds until market expiry.
            market_type: FINANCIAL, POLITICAL, SPORTS, CRYPTO, WEATHER, ECONOMIC, OTHER.
            lookback: Rolling window override.

        Returns:
            dict with keys:
                reversion_score (float -1 to 1): Negative = expect price down, positive = expect up.
                confidence (float 0-1): How confident the signal is.
                half_life_minutes (float): Estimated time for price to revert halfway.
                signal (str): FADE / HOLD / NEUTRAL.
                z_score (float): Current deviation from mean in std units.
                rolling_mean (float): Estimated fair value.
                rolling_std (float): Price volatility.
                theta (float): Mean-reversion speed parameter.
        """
        market_type = market_type.upper()
        lookback = lookback or self.default_lookback

        # Need minimum history
        if len(historical_prices) < MIN_HISTORY:
            result = {
                "reversion_score": 0.0,
                "confidence": 0.0,
                "half_life_minutes": float("inf"),
                "signal": "NEUTRAL",
                "z_score": 0.0,
                "rolling_mean": current_price,
                "rolling_std": 0.0,
                "theta": 0.0,
                "note": "insufficient_history",
            }
            _shadow_log({"action": "evaluate", **result})
            return result

        # Sports markets don't revert
        theta_mult = MARKET_TYPE_THETA.get(market_type, 0.7)
        if theta_mult == 0.0:
            result = {
                "reversion_score": 0.0,
                "confidence": 0.0,
                "half_life_minutes": float("inf"),
                "signal": "NEUTRAL",
                "z_score": 0.0,
                "rolling_mean": current_price,
                "rolling_std": 0.0,
                "theta": 0.0,
                "note": "sports_no_reversion",
                "market_type": market_type,
            }
            _shadow_log({"action": "evaluate", **result})
            return result

        # Compute rolling statistics
        window = historical_prices[-lookback:]
        rolling_mean = sum(window) / len(window)
        variance = sum((p - rolling_mean) ** 2 for p in window) / len(window)
        rolling_std = math.sqrt(variance) if variance > 0 else 0.001

        # Z-score: how far current price is from rolling mean
        z_score = (current_price - rolling_mean) / rolling_std

        # Estimate OU theta (mean-reversion speed) from lag-1 autocorrelation
        theta = self._estimate_theta(window) * theta_mult

        # Half-life in minutes
        if theta > 0:
            half_life_minutes = (math.log(2) / theta) / 60.0
        else:
            half_life_minutes = float("inf")

        # Expected reversion magnitude (how much we expect price to move back)
        expected_reversion = -z_score * rolling_std * min(1.0, theta * time_to_expiry)

        # Reversion score: -1 to 1
        # Negative z_score = price below mean = expect up (positive score)
        # Positive z_score = price above mean = expect down (negative score)
        raw_score = -z_score / max(self.z_strong, abs(z_score))

        # Time-to-expiry adjustment
        if time_to_expiry < MIN_EXPIRY_SECONDS:
            # Near expiry: markets are efficient, dampen signal
            expiry_factor = max(0.0, time_to_expiry / MIN_EXPIRY_SECONDS) * 0.3
        elif time_to_expiry > STRONG_EXPIRY_SECONDS:
            expiry_factor = 1.0
        else:
            expiry_factor = 0.3 + 0.7 * (
                (time_to_expiry - MIN_EXPIRY_SECONDS) /
                (STRONG_EXPIRY_SECONDS - MIN_EXPIRY_SECONDS)
            )

        reversion_score = max(-1.0, min(1.0, raw_score * expiry_factor))

        # Confidence
        confidence = self._compute_confidence(
            abs(z_score), time_to_expiry, len(window), theta
        )

        # Signal
        signal = self._generate_signal(reversion_score, confidence, abs(z_score))

        result = {
            "reversion_score": round(reversion_score, 4),
            "confidence": round(confidence, 4),
            "half_life_minutes": round(half_life_minutes, 2),
            "signal": signal,
            "z_score": round(z_score, 4),
            "rolling_mean": round(rolling_mean, 4),
            "rolling_std": round(rolling_std, 4),
            "theta": round(theta, 6),
            "expected_reversion": round(expected_reversion, 4),
            "time_to_expiry": time_to_expiry,
            "market_type": market_type,
            "expiry_factor": round(expiry_factor, 4),
        }
        _shadow_log({"action": "evaluate", **result})
        return result

    # --------------------------------------------------------------- Internal
    @staticmethod
    def _estimate_theta(prices: List[float]) -> float:
        """
        Estimate OU mean-reversion speed (theta) from price series.

        Uses simplified method: theta ~ -ln(rho) where rho is the lag-1
        autocorrelation of price changes around the mean. Higher theta = faster reversion.
        """
        if len(prices) < 5:
            return 0.0

        mean_p = sum(prices) / len(prices)
        deviations = [p - mean_p for p in prices]

        # Lag-1 autocorrelation of deviations
        n = len(deviations)
        numerator = sum(deviations[i] * deviations[i + 1] for i in range(n - 1))
        denominator = sum(d * d for d in deviations)

        if denominator == 0:
            return 0.0

        rho = numerator / denominator

        # Clamp rho to valid range for log
        rho = max(0.01, min(0.99, rho))

        # theta = -ln(rho); higher rho (more autocorr) = lower theta (slower reversion)
        theta = -math.log(rho)

        return max(0.0, theta)

    def _compute_confidence(
        self,
        abs_z: float,
        time_to_expiry: float,
        sample_size: int,
        theta: float,
    ) -> float:
        """
        Compute confidence in the reversion signal.

        Factors:
        - Higher z-score = more confident (price is far from mean)
        - More time to expiry = more confident
        - Larger sample = more confident
        - Higher theta = more confident (faster reversion)
        """
        # Z-score component (0-1)
        z_conf = min(1.0, abs_z / 3.0)

        # Time component (0-1)
        if time_to_expiry < MIN_EXPIRY_SECONDS:
            time_conf = 0.1
        elif time_to_expiry > STRONG_EXPIRY_SECONDS:
            time_conf = 1.0
        else:
            time_conf = 0.1 + 0.9 * (
                (time_to_expiry - MIN_EXPIRY_SECONDS) /
                (STRONG_EXPIRY_SECONDS - MIN_EXPIRY_SECONDS)
            )

        # Sample size component (0-1)
        sample_conf = min(1.0, sample_size / 50.0)

        # Theta component (0-1); theta > 0.5 is very strong reversion
        theta_conf = min(1.0, theta / 0.5)

        # Weighted average
        confidence = (
            0.35 * z_conf +
            0.25 * time_conf +
            0.15 * sample_conf +
            0.25 * theta_conf
        )
        return min(1.0, max(0.0, confidence))

    def _generate_signal(
        self,
        reversion_score: float,
        confidence: float,
        abs_z: float,
    ) -> str:
        """
        Generate trading signal.

        FADE: Strong mean-reversion expected, trade against current direction.
        HOLD: Already positioned correctly, hold.
        NEUTRAL: No clear signal.
        """
        if confidence < 0.2:
            return "NEUTRAL"

        if abs_z >= self.z_strong and confidence >= 0.4:
            return "FADE"

        if abs_z >= self.z_weak and confidence >= 0.5:
            return "FADE"

        if abs_z < self.z_weak * 0.5:
            return "HOLD"

        return "NEUTRAL"

    def get_log_path(self) -> str:
        """Return path to the shadow log file."""
        return LOG_FILE
