"""
Market Impact — Estimate how our orders move the market.

Quant Concept:
    Every order moves the market. Large orders relative to daily volume cause
    more slippage. The square-root impact model is the industry standard:

        impact = spread * sqrt(order_size / daily_volume)

    This captures the empirical fact that impact grows sub-linearly with size:
    doubling your order doesn't double your slippage, it increases it by ~41%.

    Why it matters: If your edge is 5 cents but your estimated impact is 4 cents,
    your net edge is only 1 cent. Many "profitable" strategies become unprofitable
    once market impact is properly accounted for.

    This module runs in SHADOW MODE — it does not affect actual trading.
    It logs every evaluation to a JSONL file for future optimization.

Usage:
    from market_impact import estimate_market_impact, net_edge_after_impact

    impact = estimate_market_impact(
        order_size_usd=100,
        avg_daily_volume=50000,
        bid_ask_spread=0.02
    )
    net = net_edge_after_impact(raw_edge_cents=5.0, impact_cents=impact)
"""

import json
import math
import os
import time
from pathlib import Path

LOG_DIR = os.environ.get("SHADOW_LOG_DIR", "/tmp/quant_shadow_logs")
LOG_FILE = os.path.join(LOG_DIR, "market_impact.jsonl")


def shadow_log(entry: dict):
    """Append a JSON line to the shadow log file."""
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    entry["_module"] = "market_impact"
    entry["_ts"] = time.time()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def estimate_market_impact(
    order_size_usd: float,
    avg_daily_volume: float,
    bid_ask_spread: float,
    volatility: float = None,
) -> float:
    """
    Estimate price impact of our order using the square-root impact model.

    Model: impact = spread * sqrt(order_size / daily_volume)

    For low-liquidity markets (like Kalshi event contracts), we add a
    liquidity premium that increases impact when ADV is very low.

    Args:
        order_size_usd: Order size in USD.
        avg_daily_volume: Average daily volume in USD.
        bid_ask_spread: Current bid-ask spread in dollars (e.g., 0.02 for 2 cents).
        volatility: Optional intraday volatility. If provided, impact scales with vol.

    Returns:
        Estimated price impact in cents (e.g., 1.5 means 1.5 cents slippage).
    """
    if avg_daily_volume <= 0 or order_size_usd <= 0:
        shadow_log({
            "action": "estimate",
            "order_size_usd": order_size_usd,
            "avg_daily_volume": avg_daily_volume,
            "bid_ask_spread": bid_ask_spread,
            "estimated_impact_cents": 0.0,
            "reason": "zero_volume_or_size",
        })
        return 0.0

    # Participation rate: what fraction of daily volume is our order
    participation_rate = order_size_usd / avg_daily_volume

    # Square-root impact model
    # Standard: impact = spread * sqrt(participation_rate)
    base_impact = bid_ask_spread * math.sqrt(participation_rate)

    # Volatility scaling: higher vol = more impact
    vol_multiplier = 1.0
    if volatility is not None and volatility > 0:
        # Normalize against typical daily vol of ~2%
        vol_multiplier = max(0.5, min(3.0, volatility / 0.02))

    # Low-liquidity premium: when ADV < $10K, impact increases sharply
    liquidity_premium = 1.0
    if avg_daily_volume < 10000:
        liquidity_premium = 1.0 + (10000 - avg_daily_volume) / 10000
        liquidity_premium = min(3.0, liquidity_premium)

    total_impact = base_impact * vol_multiplier * liquidity_premium

    # Convert to cents
    impact_cents = total_impact * 100

    shadow_log({
        "action": "estimate",
        "order_size_usd": round(order_size_usd, 2),
        "avg_daily_volume": round(avg_daily_volume, 2),
        "bid_ask_spread": round(bid_ask_spread, 4),
        "participation_rate": round(participation_rate, 6),
        "base_impact_dollars": round(base_impact, 6),
        "vol_multiplier": round(vol_multiplier, 4),
        "liquidity_premium": round(liquidity_premium, 4),
        "estimated_impact_cents": round(impact_cents, 4),
        "volatility": volatility,
    })

    return impact_cents


def net_edge_after_impact(raw_edge_cents: float, impact_cents: float) -> float:
    """
    Calculate net edge after subtracting estimated market impact.

    Args:
        raw_edge_cents: Raw edge in cents before impact.
        impact_cents: Estimated market impact in cents.

    Returns:
        Net edge in cents. Negative means the trade is unprofitable after impact.
    """
    net = raw_edge_cents - impact_cents

    shadow_log({
        "action": "net_edge",
        "raw_edge_cents": round(raw_edge_cents, 4),
        "impact_cents": round(impact_cents, 4),
        "net_edge_cents": round(net, 4),
        "profitable_after_impact": net > 0,
        "impact_as_pct_of_edge": (
            round(impact_cents / raw_edge_cents * 100, 2)
            if raw_edge_cents != 0 else 0
        ),
    })

    return net


def optimal_order_size(
    target_edge_cents: float,
    avg_daily_volume: float,
    bid_ask_spread: float,
    min_net_edge_cents: float = 1.0,
) -> float:
    """
    Calculate the largest order size that preserves at least min_net_edge_cents.

    Solves: target_edge - spread * sqrt(size / adv) * 100 >= min_net_edge
    => size <= adv * ((target_edge - min_net_edge) / (spread * 100))^2

    Args:
        target_edge_cents: Expected edge in cents.
        avg_daily_volume: Average daily volume in USD.
        bid_ask_spread: Bid-ask spread in dollars.
        min_net_edge_cents: Minimum acceptable net edge after impact.

    Returns:
        Maximum order size in USD that preserves the minimum net edge.
    """
    if target_edge_cents <= min_net_edge_cents:
        return 0.0

    if bid_ask_spread <= 0:
        return float("inf")

    edge_budget = (target_edge_cents - min_net_edge_cents) / 100.0  # to dollars
    max_size = avg_daily_volume * (edge_budget / bid_ask_spread) ** 2

    shadow_log({
        "action": "optimal_size",
        "target_edge_cents": round(target_edge_cents, 4),
        "avg_daily_volume": round(avg_daily_volume, 2),
        "bid_ask_spread": round(bid_ask_spread, 4),
        "min_net_edge_cents": min_net_edge_cents,
        "max_order_size_usd": round(max_size, 2),
    })

    return max_size
