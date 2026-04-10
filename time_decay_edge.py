"""
Time Decay Edge — Edge weighted by time remaining.

Quant Concept:
    A 10% edge with 15 minutes to expiry is far more valuable than a 10% edge
    with 2 minutes left. Why? More time means more opportunity for the market
    to correct toward our predicted value. As expiry approaches, there's less
    time for the edge to be realized, and execution risk increases.

    This module applies a time-decay multiplier to raw edge calculations.
    The decay follows a square-root curve — edge decays slowly at first,
    then accelerates as expiry nears (similar to options theta decay).

    This module runs in SHADOW MODE — it does not affect actual trading.
    It logs every evaluation to a JSONL file for future optimization.

Usage:
    from time_decay_edge import calculate_time_weighted_edge

    weighted = calculate_time_weighted_edge(
        raw_edge=0.10,          # 10% raw edge
        minutes_to_expiry=8.0,  # 8 minutes left
        total_window_minutes=15 # 15-minute market
    )
"""

import json
import math
import os
import time
from pathlib import Path

LOG_DIR = os.environ.get("SHADOW_LOG_DIR", "/tmp/quant_shadow_logs")
LOG_FILE = os.path.join(LOG_DIR, "time_decay_edge.jsonl")


def shadow_log(entry: dict):
    """Append a JSON line to the shadow log file."""
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    entry["_module"] = "time_decay_edge"
    entry["_ts"] = time.time()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def calculate_time_weighted_edge(
    raw_edge: float,
    minutes_to_expiry: float,
    total_window_minutes: float,
    min_time_fraction: float = 0.05,
) -> float:
    """
    Apply time-decay weighting to a raw edge estimate.

    Edge decays as expiry approaches using a square-root curve:
        time_weight = sqrt(time_remaining / total_window)

    This means:
        - At 100% time remaining: weight = 1.0 (full edge)
        - At 50% time remaining:  weight = 0.71
        - At 25% time remaining:  weight = 0.50
        - At 10% time remaining:  weight = 0.32
        - At 1% time remaining:   weight = 0.10

    Args:
        raw_edge: The raw edge as a decimal (e.g., 0.10 for 10%).
        minutes_to_expiry: Minutes until market expires.
        total_window_minutes: Total market window in minutes.
        min_time_fraction: Below this fraction, edge is zeroed out
                           (too close to expiry to act). Default 5%.

    Returns:
        Time-weighted edge as a decimal.
    """
    if total_window_minutes <= 0:
        shadow_log({
            "action": "evaluate",
            "raw_edge": raw_edge,
            "minutes_to_expiry": minutes_to_expiry,
            "total_window_minutes": total_window_minutes,
            "time_weighted_edge": 0.0,
            "reason": "invalid_total_window",
        })
        return 0.0

    time_fraction = max(0.0, min(1.0, minutes_to_expiry / total_window_minutes))

    # Below minimum time threshold, edge is worthless
    if time_fraction < min_time_fraction:
        shadow_log({
            "action": "evaluate",
            "raw_edge": round(raw_edge, 6),
            "minutes_to_expiry": round(minutes_to_expiry, 2),
            "total_window_minutes": total_window_minutes,
            "time_fraction": round(time_fraction, 4),
            "time_weighted_edge": 0.0,
            "reason": "below_min_time_threshold",
        })
        return 0.0

    # Square-root decay: preserves more edge early, drops fast near expiry
    time_weight = math.sqrt(time_fraction)

    time_weighted_edge = raw_edge * time_weight

    shadow_log({
        "action": "evaluate",
        "raw_edge": round(raw_edge, 6),
        "minutes_to_expiry": round(minutes_to_expiry, 2),
        "total_window_minutes": total_window_minutes,
        "time_fraction": round(time_fraction, 4),
        "time_weight": round(time_weight, 4),
        "time_weighted_edge": round(time_weighted_edge, 6),
        "edge_reduction_pct": round((1.0 - time_weight) * 100, 2),
    })

    return time_weighted_edge


def calculate_optimal_entry_time(
    raw_edge: float,
    total_window_minutes: float,
    edge_threshold: float = 0.02,
) -> float:
    """
    Calculate the latest entry point where time-weighted edge still exceeds threshold.

    Useful for knowing "how late can I enter and still have a good trade?"

    Args:
        raw_edge: Raw edge decimal.
        total_window_minutes: Total market window.
        edge_threshold: Minimum acceptable time-weighted edge.

    Returns:
        Latest entry time in minutes before expiry.
    """
    if raw_edge <= edge_threshold:
        return total_window_minutes  # Edge is never good enough, need full time

    # Solve: raw_edge * sqrt(t/T) = threshold
    # t/T = (threshold/raw_edge)^2
    # t = T * (threshold/raw_edge)^2
    min_time_fraction = (edge_threshold / raw_edge) ** 2
    latest_entry_minutes = min_time_fraction * total_window_minutes

    shadow_log({
        "action": "optimal_entry",
        "raw_edge": round(raw_edge, 6),
        "total_window_minutes": total_window_minutes,
        "edge_threshold": edge_threshold,
        "latest_entry_minutes_before_expiry": round(latest_entry_minutes, 2),
    })

    return latest_entry_minutes
