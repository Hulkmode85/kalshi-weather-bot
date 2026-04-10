"""
Meta Learner — Track agent accuracy and adjust weights via reinforcement learning.

Quant Concept:
    After each prediction market resolves, we compare each agent's probability
    estimate to the actual binary outcome (0 or 1). We score each agent using
    the Brier score (lower = better) and maintain a rolling accuracy profile.

    Over time, agents that consistently predict better get higher weights in the
    Supervisor's synthesis, creating a self-improving feedback loop:

        Agent predicts -> Market resolves -> Score agent -> Update weight ->
        Supervisor uses updated weights -> Better predictions

    This module runs in SHADOW MODE — it does not affect actual trading.
    It logs every resolution and weight update to JSONL files.

Scoring:
    Brier Score = (probability - outcome)^2
        Perfect prediction: 0.0
        Worst prediction:   1.0
        Random baseline:    0.25

    Agent weight = 1 / (rolling_brier_score + epsilon)
    Weights are normalized so they sum to 1.0 across all agents.

Usage:
    from meta_learner import MetaLearner

    learner = MetaLearner()

    # After a market resolves:
    learner.record_resolution(
        market_id="btc-100k-2026-04-30",
        outcome=1,  # 1=YES, 0=NO
        agent_predictions={
            "researcher": None,       # Researcher doesn't give probabilities
            "base_rate": 0.65,
            "narrative": 0.58,
            "quant": 0.72,
            "risk": None,             # Risk doesn't give probabilities
            "supervisor": 0.68,
        }
    )

    # Get current weights for the orchestrator
    weights = learner.get_agent_weights()
    # {"base_rate": 0.28, "narrative": 0.22, "quant": 0.30, "supervisor": 0.20}

Env Vars:
    SHADOW_LOG_DIR — log directory (default "/tmp/agent_shadow_logs")
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

log = logging.getLogger("meta_learner")

SHADOW_LOG_DIR = os.getenv("SHADOW_LOG_DIR", "/tmp/agent_shadow_logs")
META_LOG_FILE = os.path.join(SHADOW_LOG_DIR, "meta_learner.jsonl")
WEIGHTS_FILE = os.path.join(SHADOW_LOG_DIR, "agent_weights.json")

# Agents that produce probability estimates (others are excluded from scoring)
SCORED_AGENTS = ["base_rate", "narrative", "quant", "supervisor"]

# Rolling window for Brier score calculation
ROLLING_WINDOW = int(os.getenv("META_LEARNER_WINDOW", "100"))

# Epsilon to prevent division by zero when computing weights
EPSILON = 0.01

# Minimum observations before an agent's weight is adjusted (avoid premature shifts)
MIN_OBSERVATIONS = 5

# Default equal weight
DEFAULT_WEIGHT = 1.0 / len(SCORED_AGENTS)


# ---------------------------------------------------------------------------
# Shadow Logger
# ---------------------------------------------------------------------------


def shadow_log(entry: dict):
    """Append a JSON line to the meta learner log file."""
    Path(SHADOW_LOG_DIR).mkdir(parents=True, exist_ok=True)
    entry["_module"] = "meta_learner"
    entry["_ts"] = time.time()
    try:
        with open(META_LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        log.warning("Meta learner log write failed: %s", e)


# ---------------------------------------------------------------------------
# Brier Score Utilities
# ---------------------------------------------------------------------------


def brier_score(probability: float, outcome: int) -> float:
    """Calculate Brier score for a single prediction.

    Args:
        probability: Agent's predicted probability (0.0 to 1.0).
        outcome: Actual outcome (0 or 1).

    Returns:
        Brier score (0.0 = perfect, 1.0 = worst, 0.25 = random).
    """
    return (probability - outcome) ** 2


def calibration_bucket(probability: float, outcome: int, buckets: int = 10) -> int:
    """Return the calibration bucket index for a prediction.

    Used for calibration curve analysis: group predictions by decile
    and compare average predicted probability to actual frequency.
    """
    return min(int(probability * buckets), buckets - 1)


# ---------------------------------------------------------------------------
# Meta Learner
# ---------------------------------------------------------------------------


class MetaLearner:
    """Tracks per-agent accuracy and computes adaptive weights.

    Stores all resolution data in JSONL format for analysis.
    Maintains a rolling Brier score per agent and converts to weights.
    """

    def __init__(self):
        """Initialize the meta learner, loading any existing history."""
        self.history: list[dict] = []  # All resolution records
        self.agent_scores: dict[str, list[float]] = {
            agent: [] for agent in SCORED_AGENTS
        }
        self.weights: dict[str, float] = {
            agent: DEFAULT_WEIGHT for agent in SCORED_AGENTS
        }
        self._load_history()

    def _load_history(self):
        """Load existing resolution history from the log file."""
        if not os.path.exists(META_LOG_FILE):
            log.info("No existing meta learner history found.")
            return

        loaded = 0
        try:
            with open(META_LOG_FILE, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if entry.get("event") == "resolution":
                            self.history.append(entry)
                            # Rebuild per-agent score arrays
                            for agent in SCORED_AGENTS:
                                score = entry.get("scores", {}).get(agent)
                                if score is not None:
                                    self.agent_scores[agent].append(score)
                            loaded += 1
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            log.warning("Failed to load meta learner history: %s", e)
            return

        if loaded > 0:
            log.info("Loaded %d resolution records from history.", loaded)
            self._recompute_weights()

    # -------------------------------------------------------------------
    # Record Resolution
    # -------------------------------------------------------------------

    def record_resolution(
        self,
        market_id: str,
        outcome: int,
        agent_predictions: dict[str, Optional[float]],
        market_price: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Record a market resolution and score each agent.

        Args:
            market_id: Unique identifier for the market.
            outcome: Actual outcome (1 = YES, 0 = NO).
            agent_predictions: Dict mapping agent name to predicted probability.
                Use None for agents that didn't produce a probability.
            market_price: The market price at time of prediction (for comparison).
            metadata: Any additional context to log.

        Returns:
            Dict with per-agent Brier scores and updated weights.
        """
        if outcome not in (0, 1):
            raise ValueError(f"Outcome must be 0 or 1, got {outcome}")

        scores = {}
        for agent in SCORED_AGENTS:
            pred = agent_predictions.get(agent)
            if pred is not None:
                if not (0.0 <= pred <= 1.0):
                    log.warning(
                        "Agent '%s' prediction %.4f out of [0,1] range for %s",
                        agent, pred, market_id,
                    )
                    pred = max(0.0, min(1.0, pred))
                score = brier_score(pred, outcome)
                scores[agent] = round(score, 6)
                self.agent_scores[agent].append(score)

        # Also score the market price as a baseline
        market_brier = None
        if market_price is not None:
            market_prob = market_price / 100.0 if market_price > 1 else market_price
            market_brier = round(brier_score(market_prob, outcome), 6)

        # Recompute weights
        self._recompute_weights()

        record = {
            "event": "resolution",
            "market_id": market_id,
            "outcome": outcome,
            "predictions": {
                agent: agent_predictions.get(agent) for agent in SCORED_AGENTS
            },
            "scores": scores,
            "market_price": market_price,
            "market_brier": market_brier,
            "updated_weights": dict(self.weights),
            "rolling_briers": self._rolling_briers(),
            "total_resolutions": len(self.history) + 1,
            **(metadata or {}),
        }

        self.history.append(record)
        shadow_log(record)
        self._save_weights()

        log.info(
            "Resolution: %s outcome=%d | scores=%s | weights=%s",
            market_id,
            outcome,
            {k: f"{v:.4f}" for k, v in scores.items()},
            {k: f"{v:.3f}" for k, v in self.weights.items()},
        )

        return record

    # -------------------------------------------------------------------
    # Weight Computation
    # -------------------------------------------------------------------

    def _rolling_briers(self) -> dict[str, Optional[float]]:
        """Compute rolling average Brier score per agent."""
        result = {}
        for agent in SCORED_AGENTS:
            scores = self.agent_scores[agent]
            if not scores:
                result[agent] = None
                continue
            window = scores[-ROLLING_WINDOW:]
            result[agent] = round(sum(window) / len(window), 6)
        return result

    def _recompute_weights(self):
        """Recompute agent weights from rolling Brier scores.

        Weight formula: w_i = 1 / (brier_i + epsilon)
        Then normalize so all weights sum to 1.0.

        Agents with fewer than MIN_OBSERVATIONS keep the default weight.
        """
        rolling = self._rolling_briers()
        raw_weights = {}

        for agent in SCORED_AGENTS:
            avg_brier = rolling[agent]
            if avg_brier is None or len(self.agent_scores[agent]) < MIN_OBSERVATIONS:
                raw_weights[agent] = 1.0 / (0.25 + EPSILON)  # Assume random baseline
            else:
                raw_weights[agent] = 1.0 / (avg_brier + EPSILON)

        # Normalize
        total = sum(raw_weights.values())
        if total > 0:
            self.weights = {
                agent: round(w / total, 6) for agent, w in raw_weights.items()
            }
        else:
            self.weights = {agent: DEFAULT_WEIGHT for agent in SCORED_AGENTS}

    def _save_weights(self):
        """Persist current weights to a JSON file for the orchestrator to read."""
        Path(SHADOW_LOG_DIR).mkdir(parents=True, exist_ok=True)
        try:
            payload = {
                "weights": self.weights,
                "rolling_briers": self._rolling_briers(),
                "total_resolutions": len(self.history),
                "updated_at": time.time(),
            }
            with open(WEIGHTS_FILE, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            log.warning("Failed to save weights: %s", e)

    # -------------------------------------------------------------------
    # Public Interface
    # -------------------------------------------------------------------

    def get_agent_weights(self) -> dict[str, float]:
        """Return current agent weights for the orchestrator.

        Returns:
            Dict mapping agent name to weight (sums to ~1.0).
        """
        return dict(self.weights)

    def get_agent_stats(self) -> dict:
        """Return detailed per-agent statistics.

        Returns:
            Dict with rolling Brier scores, total predictions, weights,
            and calibration data per agent.
        """
        stats = {}
        for agent in SCORED_AGENTS:
            scores = self.agent_scores[agent]
            if not scores:
                stats[agent] = {
                    "total_predictions": 0,
                    "rolling_brier": None,
                    "all_time_brier": None,
                    "weight": self.weights[agent],
                    "best_brier": None,
                    "worst_brier": None,
                }
                continue

            window = scores[-ROLLING_WINDOW:]
            stats[agent] = {
                "total_predictions": len(scores),
                "rolling_brier": round(sum(window) / len(window), 6),
                "all_time_brier": round(sum(scores) / len(scores), 6),
                "weight": self.weights[agent],
                "best_brier": round(min(scores), 6),
                "worst_brier": round(max(scores), 6),
                "recent_trend": self._recent_trend(scores),
            }

        # Add market baseline if available
        market_scores = [
            r["market_brier"]
            for r in self.history
            if r.get("market_brier") is not None
        ]
        if market_scores:
            stats["_market_baseline"] = {
                "total": len(market_scores),
                "avg_brier": round(sum(market_scores) / len(market_scores), 6),
            }

        return stats

    def _recent_trend(self, scores: list[float], window: int = 10) -> Optional[str]:
        """Determine if an agent is improving, declining, or stable.

        Compares last `window` scores to the previous `window` scores.
        """
        if len(scores) < window * 2:
            return "insufficient_data"

        recent = sum(scores[-window:]) / window
        prior = sum(scores[-2 * window : -window]) / window

        diff = recent - prior
        if diff < -0.02:
            return "improving"
        elif diff > 0.02:
            return "declining"
        else:
            return "stable"

    def get_leaderboard(self) -> list[dict]:
        """Return agents ranked by rolling Brier score (best first).

        Returns:
            List of dicts with agent name, brier score, and weight.
        """
        rolling = self._rolling_briers()
        entries = []
        for agent in SCORED_AGENTS:
            b = rolling[agent]
            entries.append({
                "agent": agent,
                "rolling_brier": b,
                "weight": self.weights[agent],
                "predictions": len(self.agent_scores[agent]),
            })
        # Sort by brier score (lower is better), None goes last
        entries.sort(key=lambda x: x["rolling_brier"] if x["rolling_brier"] is not None else 999)
        return entries

    def should_promote_agent(self, agent: str) -> bool:
        """Check if an agent is consistently beating the market baseline.

        Returns True if the agent's rolling Brier is better (lower) than
        the market baseline by at least 0.02.
        """
        rolling = self._rolling_briers()
        agent_brier = rolling.get(agent)
        if agent_brier is None:
            return False

        market_scores = [
            r["market_brier"]
            for r in self.history
            if r.get("market_brier") is not None
        ]
        if not market_scores:
            return False

        market_avg = sum(market_scores) / len(market_scores)
        return agent_brier < (market_avg - 0.02)

    def status(self) -> dict:
        """Return meta learner status for health checks."""
        return {
            "total_resolutions": len(self.history),
            "agents_tracked": SCORED_AGENTS,
            "rolling_window": ROLLING_WINDOW,
            "min_observations": MIN_OBSERVATIONS,
            "current_weights": self.weights,
            "rolling_briers": self._rolling_briers(),
            "weights_file": WEIGHTS_FILE,
        }


# ---------------------------------------------------------------------------
# Load Weights Helper (for orchestrator integration)
# ---------------------------------------------------------------------------


def load_saved_weights() -> Optional[dict[str, float]]:
    """Load the most recently saved agent weights from disk.

    This is the integration point: the MultiAgentOrchestrator calls this
    at init time to get the latest weights from the MetaLearner.

    Returns:
        Dict of agent weights, or None if no weights file exists.
    """
    if not os.path.exists(WEIGHTS_FILE):
        return None
    try:
        with open(WEIGHTS_FILE, "r") as f:
            data = json.load(f)
        weights = data.get("weights")
        if weights:
            log.info(
                "Loaded saved agent weights (from %d resolutions)",
                data.get("total_resolutions", 0),
            )
        return weights
    except Exception as e:
        log.warning("Failed to load saved weights: %s", e)
        return None


# ---------------------------------------------------------------------------
# Convenience: module-level singleton
# ---------------------------------------------------------------------------

_learner: Optional[MetaLearner] = None


def get_learner() -> MetaLearner:
    """Get or create the module-level MetaLearner singleton."""
    global _learner
    if _learner is None:
        _learner = MetaLearner()
    return _learner
