"""
Ensemble Model — Run multiple models, only trade when majority agree.

Quant Concept:
    No single model captures all market dynamics. An ensemble runs 5 independent
    models (lognormal, momentum, mean reversion, volume flow, time decay), each
    producing a probability estimate. The consensus probability is the weighted
    average, and agreement_pct measures how aligned the models are. High agreement
    (>80%) signals strong conviction; low agreement signals uncertainty.

    This module runs in SHADOW MODE — it does not affect actual trading.
    It logs every evaluation to a JSONL file for future optimization.

Usage:
    from ensemble_model import EnsembleModel

    ensemble = EnsembleModel()
    result = ensemble.evaluate(market_data)
    # result = {"models": {...}, "consensus": 0.62, "agreement_pct": 0.85}
"""

import json
import math
import os
import time
from pathlib import Path

LOG_DIR = os.environ.get("SHADOW_LOG_DIR", "/tmp/quant_shadow_logs")
LOG_FILE = os.path.join(LOG_DIR, "ensemble_model.jsonl")


def shadow_log(entry: dict):
    """Append a JSON line to the shadow log file."""
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    entry["_module"] = "ensemble_model"
    entry["_ts"] = time.time()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


class EnsembleModel:
    """Runs 5 independent probability models and aggregates their estimates."""

    # Model weights (tunable via shadow log analysis later)
    DEFAULT_WEIGHTS = {
        "lognormal": 0.20,
        "momentum": 0.25,
        "mean_reversion": 0.20,
        "volume_flow": 0.15,
        "time_decay": 0.20,
    }

    def __init__(self, weights: dict = None):
        """
        Args:
            weights: Dict of model_name -> weight. Must sum to 1.0.
                     Defaults to equal-ish weighting with slight momentum bias.
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()

    def evaluate(self, market_data: dict) -> dict:
        """
        Run all 5 models on market_data and return consensus.

        Args:
            market_data: Dict with keys:
                - current_price (float): Current market price 0-1
                - price_history (list[float]): Recent prices, newest last
                - volume (float): Current period volume
                - volume_history (list[float]): Recent volumes, newest last
                - minutes_to_expiry (float): Minutes until market expires
                - total_window_minutes (float): Total market window in minutes

        Returns:
            Dict with:
                - models: {model_name: probability, ...}
                - consensus: Weighted average probability
                - agreement_pct: How aligned models are (0-1)
        """
        current_price = market_data.get("current_price", 0.5)
        price_history = market_data.get("price_history", [current_price])
        volume = market_data.get("volume", 0)
        volume_history = market_data.get("volume_history", [volume])
        minutes_to_expiry = market_data.get("minutes_to_expiry", 60)
        total_window = market_data.get("total_window_minutes", 120)

        models = {}
        models["lognormal"] = self._lognormal_model(current_price, price_history)
        models["momentum"] = self._momentum_model(current_price, price_history)
        models["mean_reversion"] = self._mean_reversion_model(current_price, price_history)
        models["volume_flow"] = self._volume_flow_model(current_price, volume, volume_history)
        models["time_decay"] = self._time_decay_model(current_price, minutes_to_expiry, total_window)

        # Weighted consensus
        consensus = sum(
            models[name] * self.weights[name] for name in models
        )
        consensus = max(0.01, min(0.99, consensus))

        # Agreement: 1 - normalized standard deviation of model outputs
        probs = list(models.values())
        mean_p = sum(probs) / len(probs)
        variance = sum((p - mean_p) ** 2 for p in probs) / len(probs)
        std_dev = math.sqrt(variance)
        # Max possible std_dev for 5 values in [0,1] is ~0.5
        agreement_pct = max(0.0, 1.0 - (std_dev / 0.25))
        agreement_pct = min(1.0, agreement_pct)

        result = {
            "models": {k: round(v, 6) for k, v in models.items()},
            "consensus": round(consensus, 6),
            "agreement_pct": round(agreement_pct, 4),
        }

        shadow_log({
            "action": "evaluate",
            "market_data_summary": {
                "current_price": current_price,
                "price_history_len": len(price_history),
                "volume": volume,
                "minutes_to_expiry": minutes_to_expiry,
            },
            "models": result["models"],
            "consensus": result["consensus"],
            "agreement_pct": result["agreement_pct"],
        })

        return result

    def _lognormal_model(self, current_price: float, price_history: list) -> float:
        """
        Lognormal diffusion model.
        Assumes price follows geometric Brownian motion.
        Estimates probability based on drift and volatility of log-returns.
        """
        if len(price_history) < 3:
            return current_price

        # Compute log returns
        log_returns = []
        for i in range(1, len(price_history)):
            p0 = max(0.01, price_history[i - 1])
            p1 = max(0.01, price_history[i])
            log_returns.append(math.log(p1 / p0))

        if not log_returns:
            return current_price

        drift = sum(log_returns) / len(log_returns)
        variance = sum((r - drift) ** 2 for r in log_returns) / max(1, len(log_returns) - 1)
        vol = math.sqrt(variance) if variance > 0 else 0.01

        # Project forward: current price * exp(drift)
        projected = current_price * math.exp(drift)
        projected = max(0.01, min(0.99, projected))
        return projected

    def _momentum_model(self, current_price: float, price_history: list) -> float:
        """
        Momentum model.
        Recent trend continues. Weighted linear regression on recent prices.
        """
        if len(price_history) < 3:
            return current_price

        # Use last 10 prices max
        recent = price_history[-10:]
        n = len(recent)

        # Simple linear regression slope
        x_mean = (n - 1) / 2.0
        y_mean = sum(recent) / n
        numerator = sum((i - x_mean) * (recent[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return current_price

        slope = numerator / denominator
        # Project 1 step forward
        projected = current_price + slope
        return max(0.01, min(0.99, projected))

    def _mean_reversion_model(self, current_price: float, price_history: list) -> float:
        """
        Mean reversion model.
        Prices tend to revert to their moving average.
        The further from the mean, the stronger the pull back.
        """
        if len(price_history) < 5:
            return current_price

        # 20-period mean (or full history if shorter)
        lookback = price_history[-20:]
        mean_price = sum(lookback) / len(lookback)

        # Reversion strength: 30% pull toward mean per step
        reversion_rate = 0.3
        projected = current_price + reversion_rate * (mean_price - current_price)
        return max(0.01, min(0.99, projected))

    def _volume_flow_model(self, current_price: float, volume: float, volume_history: list) -> float:
        """
        Volume flow model.
        High volume with price increase = bullish signal.
        High volume with price decrease = bearish signal.
        Low volume moves are noise.
        """
        if not volume_history or len(volume_history) < 2:
            return current_price

        avg_volume = sum(volume_history) / len(volume_history)
        if avg_volume == 0:
            return current_price

        volume_ratio = volume / avg_volume

        # Volume-weighted adjustment: high volume amplifies recent move direction
        if len(volume_history) >= 2:
            recent_direction = 1 if volume_history[-1] >= volume_history[-2] else -1
        else:
            recent_direction = 0

        # Scale effect by how unusual current volume is
        volume_signal = min(volume_ratio - 1.0, 1.0) * 0.05 * recent_direction
        projected = current_price + volume_signal
        return max(0.01, min(0.99, projected))

    def _time_decay_model(self, current_price: float, minutes_to_expiry: float, total_window: float) -> float:
        """
        Time decay model.
        As expiry approaches, probability polarizes toward 0 or 1.
        Prices above 0.5 drift higher; below 0.5 drift lower.
        """
        if total_window <= 0:
            return current_price

        time_fraction_remaining = max(0.0, min(1.0, minutes_to_expiry / total_window))

        # As time runs out, probability polarizes
        # Strength of polarization increases as time_fraction_remaining -> 0
        polarization = 1.0 - time_fraction_remaining
        direction = 1 if current_price >= 0.5 else -1
        magnitude = abs(current_price - 0.5)

        # Amplify distance from 0.5
        adjustment = direction * magnitude * polarization * 0.3
        projected = current_price + adjustment
        return max(0.01, min(0.99, projected))
