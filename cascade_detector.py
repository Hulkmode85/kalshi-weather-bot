"""
cascade_detector.py — Information Cascade Detection using Hidden Markov Model.

Quant Concept:
    Information cascades occur when traders ignore their private signals and
    follow the crowd. A sequence of same-direction trades with accelerating
    volume and tightening spreads is a strong cascade signature. Cascades are
    profitable to ride early but lethal to join late — the exhaustion phase
    reverses fast.

    Hidden states:
        RANDOM              — no directional pattern, normal market noise
        CASCADE_INITIATED   — 3+ same-direction trades detected
        CASCADE_ACCELERATING — volume accelerating + spread tightening
        CASCADE_EXHAUSTED   — sell/pause detected or 30s without continuation

    This module uses a simplified Viterbi-like forward algorithm with
    hand-tuned transition and emission matrices. No numpy required.

    Score interpretation:
        cascade_probability 0.0-0.25: noise, ignore
        cascade_probability 0.25-0.50: possible cascade forming
        cascade_probability 0.50-0.75: likely cascade, watch closely
        cascade_probability 0.75-1.00: HIGH_CONVICTION — act now or exit

Usage:
    from cascade_detector import CascadeDetector

    detector = CascadeDetector()
    detector.add_trade(direction="buy", volume=10, price=0.62, timestamp=time.time())
    detector.add_trade(direction="buy", volume=15, price=0.63)
    result = detector.evaluate()
    # result = {
    #     "cascade_probability": 0.82,
    #     "cascade_state": "CASCADE_ACCELERATING",
    #     "recommended_action": "HIGH_CONVICTION",
    #     "confirmation_signals": 3,
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
LOG_FILE = os.path.join(LOG_DIR, "cascade_detector.jsonl")


def shadow_log(entry: dict):
    """Append a JSON line to the shadow log file."""
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    entry["_module"] = "cascade_detector"
    entry["_ts"] = time.time()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


# --- Constants ---
STATES = ["RANDOM", "CASCADE_INITIATED", "CASCADE_ACCELERATING", "CASCADE_EXHAUSTED"]
STATE_IDX = {s: i for i, s in enumerate(STATES)}

# Hand-tuned transition matrix P(next_state | current_state)
# Rows = from, Cols = to  [RANDOM, INITIATED, ACCELERATING, EXHAUSTED]
TRANSITION = [
    [0.70, 0.25, 0.03, 0.02],  # from RANDOM
    [0.15, 0.40, 0.40, 0.05],  # from INITIATED
    [0.05, 0.05, 0.70, 0.20],  # from ACCELERATING
    [0.60, 0.10, 0.05, 0.25],  # from EXHAUSTED
]

# Actions mapped to states and probability thresholds
ACTIONS = {
    "RANDOM": "NO_ACTION",
    "CASCADE_INITIATED": "MONITOR",
    "CASCADE_ACCELERATING": "FOLLOW",
    "CASCADE_EXHAUSTED": "EXIT",
}


class CascadeDetector:
    """
    Detects information cascades in trade sequences using a simplified HMM.

    Tracks consecutive same-direction trades, volume acceleration, and spread
    tightening to identify cascade formation and exhaustion.
    """

    def __init__(
        self,
        max_trades: int = 500,
        cascade_min_streak: int = 3,
        volume_accel_threshold: float = 1.15,
        spread_tighten_threshold: float = 0.95,
        exhaustion_timeout_sec: float = 30.0,
        high_conviction_threshold: float = 0.75,
        min_confirmation_signals: int = 2,
        mode: str = "paper",
        log_dir: Optional[str] = None,
    ):
        """
        Args:
            max_trades: max trade history to retain.
            cascade_min_streak: min same-direction trades to trigger INITIATED.
            volume_accel_threshold: ratio above which volume is 'accelerating'.
            spread_tighten_threshold: ratio below which spread is 'tightening'.
            exhaustion_timeout_sec: seconds without continuation to trigger EXHAUSTED.
            high_conviction_threshold: cascade_probability above this = HIGH_CONVICTION.
            min_confirmation_signals: signals needed alongside high probability.
            mode: 'paper' or 'live'.
            log_dir: override shadow log directory.
        """
        self.max_trades = max_trades
        self.cascade_min_streak = cascade_min_streak
        self.volume_accel_threshold = volume_accel_threshold
        self.spread_tighten_threshold = spread_tighten_threshold
        self.exhaustion_timeout_sec = exhaustion_timeout_sec
        self.high_conviction_threshold = high_conviction_threshold
        self.min_confirmation_signals = min_confirmation_signals
        self.mode = mode

        if log_dir:
            global LOG_DIR, LOG_FILE
            LOG_DIR = log_dir
            LOG_FILE = os.path.join(LOG_DIR, "cascade_detector.jsonl")

        # Trade history: list of dicts
        self.trades: deque = deque(maxlen=max_trades)

        # HMM state probabilities (belief vector)
        self.state_probs: List[float] = [0.85, 0.10, 0.04, 0.01]

        # Tracking
        self._eval_count = 0

    # --------------------------------------------------------- Input
    def add_trade(
        self,
        direction: str,
        volume: float,
        price: float,
        timestamp: Optional[float] = None,
        spread: Optional[float] = None,
    ):
        """Record a trade observation.

        Args:
            direction: 'buy' or 'sell'.
            volume: trade volume (contracts or USD).
            price: execution price (probability 0-1 or dollar amount).
            timestamp: epoch seconds (defaults to now).
            spread: bid-ask spread at time of trade (optional).
        """
        ts = timestamp or time.time()
        self.trades.append({
            "direction": direction.lower(),
            "volume": volume,
            "price": price,
            "timestamp": ts,
            "spread": spread,
        })

    # --------------------------------------------------------- Core HMM
    def _compute_emission_scores(self) -> List[float]:
        """Compute emission likelihood for each hidden state given recent trades.

        Returns list of 4 scores (one per state), each in [0, 1].
        """
        if len(self.trades) < 2:
            return [0.9, 0.05, 0.03, 0.02]

        trades = list(self.trades)
        recent = trades[-min(10, len(trades)):]

        # --- Feature 1: Direction streak ---
        streak = 1
        last_dir = recent[-1]["direction"]
        for t in reversed(recent[:-1]):
            if t["direction"] == last_dir:
                streak += 1
            else:
                break
        streak_score = min(streak / max(self.cascade_min_streak, 1), 1.0)

        # --- Feature 2: Volume acceleration ---
        volumes = [t["volume"] for t in recent if t["volume"] > 0]
        vol_accel = 0.0
        if len(volumes) >= 3:
            recent_avg = sum(volumes[-3:]) / 3
            older_avg = sum(volumes[:-3]) / max(len(volumes) - 3, 1)
            if older_avg > 0:
                vol_accel = min((recent_avg / older_avg) - 1.0, 1.0)
                vol_accel = max(vol_accel, 0.0)

        # --- Feature 3: Spread tightening ---
        spreads = [t["spread"] for t in recent if t["spread"] is not None and t["spread"] > 0]
        spread_tighten = 0.0
        if len(spreads) >= 3:
            recent_spread = sum(spreads[-3:]) / 3
            older_spread = sum(spreads[:-3]) / max(len(spreads) - 3, 1)
            if older_spread > 0:
                ratio = recent_spread / older_spread
                spread_tighten = max(1.0 - ratio, 0.0)  # tightening = positive

        # --- Feature 4: Time gap (exhaustion) ---
        time_gap = trades[-1]["timestamp"] - trades[-2]["timestamp"] if len(trades) >= 2 else 0
        exhaustion_signal = min(time_gap / self.exhaustion_timeout_sec, 1.0)

        # --- Feature 5: Direction flip ---
        direction_flip = 0.0
        if len(trades) >= 2:
            if trades[-1]["direction"] != trades[-2]["direction"]:
                direction_flip = 1.0

        # Emission scores per state
        e_random = 0.3 + 0.5 * (1.0 - streak_score) + 0.2 * direction_flip
        e_initiated = 0.2 + 0.6 * streak_score + 0.2 * (1.0 - direction_flip)
        e_accelerating = (
            0.1
            + 0.3 * streak_score
            + 0.3 * vol_accel
            + 0.2 * spread_tighten
            + 0.1 * (1.0 - exhaustion_signal)
        )
        e_exhausted = (
            0.1
            + 0.4 * exhaustion_signal
            + 0.3 * direction_flip
            + 0.2 * (1.0 - streak_score)
        )

        # Normalize
        total = e_random + e_initiated + e_accelerating + e_exhausted
        if total > 0:
            return [e / total for e in [e_random, e_initiated, e_accelerating, e_exhausted]]
        return [0.25, 0.25, 0.25, 0.25]

    def _forward_step(self):
        """Run one forward step of the HMM (simplified Viterbi-like update)."""
        emissions = self._compute_emission_scores()

        # Predict: multiply current belief by transition matrix
        predicted = [0.0] * 4
        for j in range(4):
            for i in range(4):
                predicted[j] += self.state_probs[i] * TRANSITION[i][j]

        # Update: multiply predicted by emissions
        updated = [predicted[i] * emissions[i] for i in range(4)]

        # Normalize
        total = sum(updated)
        if total > 0:
            self.state_probs = [u / total for u in updated]
        else:
            self.state_probs = [0.25, 0.25, 0.25, 0.25]

    # ------------------------------------------------ Confirmation signals
    def _count_confirmation_signals(self) -> Tuple[int, List[str]]:
        """Count how many independent confirmation signals are active."""
        if len(self.trades) < 3:
            return 0, []

        signals = []
        trades = list(self.trades)
        recent = trades[-min(10, len(trades)):]

        # Signal 1: 3+ consecutive same-direction trades
        streak = 1
        last_dir = recent[-1]["direction"]
        for t in reversed(recent[:-1]):
            if t["direction"] == last_dir:
                streak += 1
            else:
                break
        if streak >= self.cascade_min_streak:
            signals.append(f"direction_streak_{streak}")

        # Signal 2: Volume acceleration
        volumes = [t["volume"] for t in recent if t["volume"] > 0]
        if len(volumes) >= 4:
            recent_avg = sum(volumes[-2:]) / 2
            older_avg = sum(volumes[:-2]) / max(len(volumes) - 2, 1)
            if older_avg > 0 and (recent_avg / older_avg) >= self.volume_accel_threshold:
                signals.append("volume_accelerating")

        # Signal 3: Spread tightening
        spreads = [t["spread"] for t in recent if t["spread"] is not None and t["spread"] > 0]
        if len(spreads) >= 4:
            recent_spread = sum(spreads[-2:]) / 2
            older_spread = sum(spreads[:-2]) / max(len(spreads) - 2, 1)
            if older_spread > 0 and (recent_spread / older_spread) <= self.spread_tighten_threshold:
                signals.append("spread_tightening")

        # Signal 4: Price trending in cascade direction
        prices = [t["price"] for t in recent]
        if len(prices) >= 3:
            if last_dir == "buy" and all(prices[i] <= prices[i + 1] for i in range(len(prices) - 3, len(prices) - 1)):
                signals.append("price_trending_up")
            elif last_dir == "sell" and all(prices[i] >= prices[i + 1] for i in range(len(prices) - 3, len(prices) - 1)):
                signals.append("price_trending_down")

        # Signal 5: Increasing trade frequency
        timestamps = [t["timestamp"] for t in recent]
        if len(timestamps) >= 4:
            recent_gaps = [timestamps[i] - timestamps[i - 1] for i in range(-2, 0)]
            older_gaps = [timestamps[i] - timestamps[i - 1] for i in range(1, min(4, len(timestamps)))]
            if older_gaps and recent_gaps:
                avg_recent = sum(recent_gaps) / len(recent_gaps)
                avg_older = sum(older_gaps) / len(older_gaps)
                if avg_older > 0 and avg_recent < avg_older * 0.7:
                    signals.append("frequency_increasing")

        return len(signals), signals

    # ------------------------------------------------ Evaluate
    def evaluate(self) -> Dict:
        """Run cascade detection on current trade sequence.

        Returns:
            dict with cascade_probability, cascade_state, recommended_action,
            confirmation_signals, confirmation_details, state_probs, mode.
        """
        self._eval_count += 1

        # Run HMM forward step
        self._forward_step()

        # Most likely state
        max_prob = max(self.state_probs)
        max_idx = self.state_probs.index(max_prob)
        cascade_state = STATES[max_idx]

        # Cascade probability = P(INITIATED) + P(ACCELERATING)
        cascade_probability = self.state_probs[STATE_IDX["CASCADE_INITIATED"]] + \
                              self.state_probs[STATE_IDX["CASCADE_ACCELERATING"]]
        cascade_probability = min(cascade_probability, 1.0)

        # Confirmation signals
        num_signals, signal_details = self._count_confirmation_signals()

        # Recommended action
        if (cascade_probability >= self.high_conviction_threshold and
                num_signals >= self.min_confirmation_signals):
            recommended_action = "HIGH_CONVICTION"
        elif cascade_state == "CASCADE_EXHAUSTED" and self.state_probs[STATE_IDX["CASCADE_EXHAUSTED"]] > 0.4:
            recommended_action = "EXIT"
        else:
            recommended_action = ACTIONS.get(cascade_state, "NO_ACTION")

        # Check exhaustion via timeout
        if len(self.trades) >= 2:
            gap = time.time() - self.trades[-1]["timestamp"]
            if gap > self.exhaustion_timeout_sec:
                recommended_action = "EXIT"

        # Check exhaustion via direction flip
        if len(self.trades) >= 2:
            last_two = list(self.trades)[-2:]
            dominant_dir = self._dominant_direction()
            if last_two[-1]["direction"] != dominant_dir and dominant_dir is not None:
                if recommended_action == "HIGH_CONVICTION":
                    recommended_action = "EXIT"

        result = {
            "cascade_probability": round(cascade_probability, 4),
            "cascade_state": cascade_state,
            "recommended_action": recommended_action,
            "confirmation_signals": num_signals,
            "confirmation_details": signal_details,
            "state_probs": {s: round(p, 4) for s, p in zip(STATES, self.state_probs)},
            "trade_count": len(self.trades),
            "mode": self.mode,
            "eval_count": self._eval_count,
        }

        # Shadow log every evaluation
        shadow_log({
            "event": "evaluate",
            "result": result,
        })

        return result

    # ------------------------------------------------ Helpers
    def _dominant_direction(self) -> Optional[str]:
        """Return the dominant direction in recent trades, or None."""
        if len(self.trades) < self.cascade_min_streak:
            return None
        recent = list(self.trades)[-self.cascade_min_streak:]
        buys = sum(1 for t in recent if t["direction"] == "buy")
        sells = len(recent) - buys
        if buys > sells:
            return "buy"
        elif sells > buys:
            return "sell"
        return None

    def reset(self):
        """Reset detector state for a new market/event."""
        self.trades.clear()
        self.state_probs = [0.85, 0.10, 0.04, 0.01]
        self._eval_count = 0

    def get_stats(self) -> Dict:
        """Return summary statistics."""
        if not self.trades:
            return {"trade_count": 0}
        trades = list(self.trades)
        buys = sum(1 for t in trades if t["direction"] == "buy")
        sells = len(trades) - buys
        volumes = [t["volume"] for t in trades]
        return {
            "trade_count": len(trades),
            "buy_count": buys,
            "sell_count": sells,
            "total_volume": round(sum(volumes), 2),
            "avg_volume": round(sum(volumes) / len(volumes), 4) if volumes else 0,
            "time_span_sec": round(trades[-1]["timestamp"] - trades[0]["timestamp"], 2),
            "mode": self.mode,
        }
