"""
Bayesian Updater — Real-time probability revision as new data arrives.

Quant Concept:
    Bayesian updating treats probability as a belief that gets revised when
    new evidence appears. Each price movement is evidence: if a market moves
    from 60c to 65c, that's evidence the true probability is higher than our
    prior. We use the magnitude and direction of price changes to update our
    probability estimate, weighting recent evidence more heavily.

    This module runs in SHADOW MODE — it does not affect actual trading.
    It logs every evaluation to a JSONL file for future optimization.

Usage:
    from bayesian_updater import BayesianUpdater

    updater = BayesianUpdater()
    updater.update("BTC-2026-04-09-50000", new_price=0.65, timestamp=time.time())
    posterior = updater.get_posterior("BTC-2026-04-09-50000")
"""

import json
import math
import os
import time
from pathlib import Path

LOG_DIR = os.environ.get("SHADOW_LOG_DIR", "/tmp/quant_shadow_logs")
LOG_FILE = os.path.join(LOG_DIR, "bayesian_updater.jsonl")


def shadow_log(entry: dict):
    """Append a JSON line to the shadow log file."""
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    entry["_module"] = "bayesian_updater"
    entry["_ts"] = time.time()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


class BayesianUpdater:
    """Maintains and updates probability estimates for markets using Bayesian inference."""

    def __init__(self, learning_rate: float = 0.15, decay_factor: float = 0.95):
        """
        Args:
            learning_rate: How strongly new evidence shifts the posterior (0-1).
                           Higher = more reactive. Default 0.15.
            decay_factor:  Weight decay for older observations (0-1).
                           Lower = forget faster. Default 0.95.
        """
        self.prior = {}          # market_id -> current probability estimate
        self.history = {}        # market_id -> list of (timestamp, price, posterior)
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor

    def update(self, market_id: str, new_evidence_price: float, timestamp: float = None):
        """
        Update probability estimate using Bayes' theorem with price movement as evidence.

        The market price IS the crowd's probability estimate. We treat the delta between
        our prior and the new price as a likelihood ratio, then compute a weighted posterior.

        Args:
            market_id: Unique market identifier.
            new_evidence_price: Latest market price (0-1 range, e.g. 0.65 = 65 cents).
            timestamp: Unix timestamp of the observation. Defaults to now.
        """
        timestamp = timestamp or time.time()
        new_evidence_price = max(0.01, min(0.99, new_evidence_price))

        if market_id not in self.prior:
            # First observation: set prior to market price
            self.prior[market_id] = new_evidence_price
            self.history[market_id] = [(timestamp, new_evidence_price, new_evidence_price)]
            shadow_log({
                "action": "initialize",
                "market_id": market_id,
                "initial_prior": new_evidence_price,
                "timestamp": timestamp,
            })
            return

        old_prior = self.prior[market_id]

        # Likelihood ratio from price movement
        # If price moved toward 1, evidence supports higher probability
        # Convert to log-odds for proper Bayesian update
        prior_logodds = math.log(old_prior / (1.0 - old_prior))
        evidence_logodds = math.log(new_evidence_price / (1.0 - new_evidence_price))

        # Weighted update in log-odds space
        updated_logodds = (
            (1.0 - self.learning_rate) * prior_logodds
            + self.learning_rate * evidence_logodds
        )

        # Apply time-weighted decay toward evidence if history is long
        history = self.history.get(market_id, [])
        if len(history) > 5:
            # More observations = trust recent evidence more
            effective_lr = min(self.learning_rate * 1.5, 0.4)
            updated_logodds = (
                (1.0 - effective_lr) * prior_logodds
                + effective_lr * evidence_logodds
            )

        # Convert back to probability
        posterior = 1.0 / (1.0 + math.exp(-updated_logodds))
        posterior = max(0.01, min(0.99, posterior))

        delta = posterior - old_prior

        self.prior[market_id] = posterior
        if market_id not in self.history:
            self.history[market_id] = []
        self.history[market_id].append((timestamp, new_evidence_price, posterior))

        # Keep history bounded
        if len(self.history[market_id]) > 500:
            self.history[market_id] = self.history[market_id][-250:]

        shadow_log({
            "action": "update",
            "market_id": market_id,
            "original_estimate": round(old_prior, 6),
            "evidence_price": round(new_evidence_price, 6),
            "updated_estimate": round(posterior, 6),
            "delta": round(delta, 6),
            "num_observations": len(self.history[market_id]),
            "timestamp": timestamp,
        })

    def get_posterior(self, market_id: str) -> float:
        """
        Get current best probability estimate for a market.

        Returns:
            Float probability (0-1), or 0.5 if no data available.
        """
        return self.prior.get(market_id, 0.5)

    def get_confidence(self, market_id: str) -> float:
        """
        Get confidence in our estimate based on number of observations.

        Returns:
            Float 0-1, where more observations = higher confidence.
        """
        n = len(self.history.get(market_id, []))
        # Saturating confidence: 50 observations -> ~0.95 confidence
        return 1.0 - math.exp(-n / 20.0)

    def get_trend(self, market_id: str, lookback: int = 10) -> float:
        """
        Get recent trend direction and magnitude.

        Returns:
            Positive = trending up, negative = trending down.
            Magnitude indicates strength.
        """
        history = self.history.get(market_id, [])
        if len(history) < 2:
            return 0.0
        recent = history[-lookback:]
        if len(recent) < 2:
            return 0.0
        return recent[-1][2] - recent[0][2]
