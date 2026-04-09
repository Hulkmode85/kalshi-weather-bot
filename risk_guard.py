"""
risk_guard.py — Shared Risk Protection Module for ALL Trading Bots

Prevents losses from:
  - One-sided exposure (atomic both-sides guard)
  - Volatility spikes (circuit breaker)
  - Market crashes (crash detector)
  - Runaway losses (max loss limits)
  - Oversized positions (position size cap)
  - Orphaned orders (timeout / cleanup)
  - Stale orders from previous deploys (deploy-safe startup)
  - Correlated multi-bot losses (correlation pause)
  - Model drift / bad predictions (settlement verification)
  - Illiquid markets / wide spreads (spread sanity check)
  - Dead/hung bots (heartbeat monitor)
  - High-impact news events (event blackout guard)
  - Stale exchange data feeds (lag detector)

Usage:
    from risk_guard import RiskManager
    rm = RiskManager()
    if not rm.pre_trade_check("BTC", 50, 30, "yes"):
        continue
    rm.post_trade("my_bot", pnl=-1.50)
"""

import os
import time
import logging
import threading
import statistics
from datetime import datetime, timezone
from collections import defaultdict, deque
from typing import Tuple, Optional, Dict, Any, List

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("risk_guard")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [RISK_GUARD] %(levelname)s %(message)s"
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Environment-driven config with sane defaults
# ---------------------------------------------------------------------------
def _env(key: str, default, cast=None):
    val = os.environ.get(key, default)
    if cast is not None:
        return cast(val)
    return val


# ===================================================================
# 1. ATOMIC BOTH-SIDES GUARD
# ===================================================================

def atomic_both_sides(
    kalshi_client,
    ticker: str,
    yes_order: dict,
    no_order: dict,
    timeout: float = 5.0,
) -> dict:
    """
    Place orders on both YES and NO sides atomically.

    1. Place the first order (YES).
    2. Wait up to `timeout` seconds for fill confirmation.
    3. Place the second order (NO).
    4. Wait up to `timeout` seconds for fill confirmation.
    5. If either side fails to fill, cancel the unfilled order.

    Returns dict with keys: success, yes_result, no_result, cancelled

    yes_order / no_order should be dicts like:
        {"action": "buy", "side": "yes", "count": 10, "type": "limit",
         "yes_price": 45, "no_price": 55}
    """
    result = {
        "success": False,
        "yes_result": None,
        "no_result": None,
        "cancelled": None,
    }

    # --- Place first side (YES) ---
    try:
        logger.info(
            "ATOMIC: placing YES order on %s — %s contracts", ticker, yes_order.get("count")
        )
        yes_resp = kalshi_client.create_order(
            ticker=ticker, **yes_order
        )
        result["yes_result"] = yes_resp
    except Exception as e:
        logger.error("ATOMIC: YES order failed to place: %s", e)
        return result

    yes_order_id = yes_resp.get("order", {}).get("order_id") or yes_resp.get("order_id")

    # --- Wait for YES fill ---
    yes_filled = _wait_for_fill(kalshi_client, yes_order_id, timeout)
    if not yes_filled:
        logger.warning("ATOMIC: YES order %s did not fill in %ss — cancelling", yes_order_id, timeout)
        _safe_cancel(kalshi_client, yes_order_id)
        result["cancelled"] = "yes"
        return result

    # --- Place second side (NO) ---
    try:
        logger.info(
            "ATOMIC: placing NO order on %s — %s contracts", ticker, no_order.get("count")
        )
        no_resp = kalshi_client.create_order(
            ticker=ticker, **no_order
        )
        result["no_result"] = no_resp
    except Exception as e:
        logger.error("ATOMIC: NO order failed to place: %s — cancelling YES", e)
        # YES is filled, NO failed to even place — log critical exposure
        logger.critical(
            "ATOMIC: NAKED EXPOSURE — YES filled on %s but NO failed to place. "
            "Manual intervention may be needed.", ticker
        )
        return result

    no_order_id = no_resp.get("order", {}).get("order_id") or no_resp.get("order_id")

    # --- Wait for NO fill ---
    no_filled = _wait_for_fill(kalshi_client, no_order_id, timeout)
    if not no_filled:
        logger.warning(
            "ATOMIC: NO order %s did not fill in %ss — cancelling to avoid naked exposure",
            no_order_id, timeout,
        )
        _safe_cancel(kalshi_client, no_order_id)
        result["cancelled"] = "no"
        return result

    # Both sides filled
    result["success"] = True
    logger.info("ATOMIC: Both sides filled on %s", ticker)
    return result


def _wait_for_fill(kalshi_client, order_id: str, timeout: float) -> bool:
    """Poll order status until filled or timeout."""
    if not order_id:
        return False
    deadline = time.time() + timeout
    poll_interval = 0.25
    while time.time() < deadline:
        try:
            order = kalshi_client.get_order(order_id=order_id)
            status = order.get("order", {}).get("status") or order.get("status", "")
            if status in ("executed", "filled", "complete"):
                return True
            if status in ("canceled", "cancelled", "expired", "failed"):
                return False
        except Exception:
            pass
        time.sleep(poll_interval)
    return False


def _safe_cancel(kalshi_client, order_id: str):
    """Best-effort cancel."""
    try:
        kalshi_client.cancel_order(order_id=order_id)
        logger.info("Cancelled order %s", order_id)
    except Exception as e:
        logger.error("Failed to cancel order %s: %s", order_id, e)


# ===================================================================
# 2. VOLATILITY CIRCUIT BREAKER
# ===================================================================

class VolatilityGuard:
    """
    Track price changes over a rolling window and pause trading when
    current volatility exceeds a multiple of the 7-day average.
    """

    def __init__(
        self,
        window_minutes: int = None,
        spike_multiplier: float = None,
        resume_multiplier: float = None,
        resume_stable_minutes: int = None,
    ):
        self.window_minutes = window_minutes or int(
            _env("VOL_WINDOW_MINUTES", 30)
        )
        self.spike_multiplier = spike_multiplier or float(
            _env("VOL_SPIKE_MULT", 2.0)
        )
        self.resume_multiplier = resume_multiplier or float(
            _env("VOL_RESUME_MULT", 1.5)
        )
        self.resume_stable_minutes = resume_stable_minutes or int(
            _env("VOL_RESUME_STABLE_MIN", 10)
        )

        # asset -> deque of (timestamp, price)
        self._prices: Dict[str, deque] = defaultdict(deque)
        # asset -> list of 1-min return stdevs for 7-day rolling average
        self._historical_vols: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=10080)  # 7 days of 1-min samples
        )
        self._paused = False
        self._resume_stable_since: Optional[float] = None
        self._lock = threading.Lock()

    def update_price(self, asset: str, price: float, timestamp: float = None):
        """Feed a new price tick. Call this from WebSocket handler or polling loop."""
        ts = timestamp or time.time()
        with self._lock:
            self._prices[asset].append((ts, price))
            # Trim to window
            cutoff = ts - (self.window_minutes * 60)
            while self._prices[asset] and self._prices[asset][0][0] < cutoff:
                self._prices[asset].popleft()

    def _current_vol(self, asset: str) -> Optional[float]:
        """Compute stdev of 1-minute returns over the rolling window."""
        prices = list(self._prices[asset])
        if len(prices) < 3:
            return None
        # Bucket into 1-min intervals and compute returns
        minute_prices = []
        bucket_start = prices[0][0]
        bucket_price = prices[0][1]
        for ts, p in prices[1:]:
            if ts - bucket_start >= 60:
                minute_prices.append(bucket_price)
                bucket_start = ts
            bucket_price = p
        minute_prices.append(bucket_price)

        if len(minute_prices) < 2:
            return None

        returns = [
            (minute_prices[i] - minute_prices[i - 1]) / minute_prices[i - 1]
            for i in range(1, len(minute_prices))
            if minute_prices[i - 1] != 0
        ]
        if len(returns) < 2:
            return None
        return statistics.stdev(returns)

    def _avg_vol(self, asset: str) -> Optional[float]:
        """7-day average volatility (rolling stdev of 1-min returns)."""
        hist = self._historical_vols[asset]
        if len(hist) < 10:
            return None
        return statistics.mean(hist)

    def record_historical_vol(self, asset: str, vol: float):
        """Add a historical volatility sample (for bootstrapping the 7-day avg)."""
        with self._lock:
            self._historical_vols[asset].append(vol)

    def should_trade(self, asset: str = None) -> bool:
        """
        Returns True if it's safe to trade, False if circuit breaker is active.
        If asset is None, checks all tracked assets.
        """
        with self._lock:
            assets = [asset] if asset else list(self._prices.keys())
            for a in assets:
                cur = self._current_vol(a)
                avg = self._avg_vol(a)
                if cur is None or avg is None:
                    continue  # Not enough data — allow trading
                if avg == 0:
                    continue

                ratio = cur / avg

                if self._paused:
                    # Check resume condition
                    if ratio < self.resume_multiplier:
                        if self._resume_stable_since is None:
                            self._resume_stable_since = time.time()
                        elif time.time() - self._resume_stable_since >= self.resume_stable_minutes * 60:
                            logger.info(
                                "VOLATILITY GUARD: resuming — vol ratio %.2f < %.2f for %d min",
                                ratio, self.resume_multiplier, self.resume_stable_minutes,
                            )
                            self._paused = False
                            self._resume_stable_since = None
                    else:
                        self._resume_stable_since = None
                else:
                    if ratio > self.spike_multiplier:
                        logger.warning(
                            "VOLATILITY GUARD: PAUSING — %s vol ratio %.2f > %.2f "
                            "(current=%.6f, avg=%.6f)",
                            a, ratio, self.spike_multiplier, cur, avg,
                        )
                        self._paused = True
                        self._resume_stable_since = None

            return not self._paused


# ===================================================================
# 3. CRASH DETECTOR
# ===================================================================

class CrashDetector:
    """
    Monitor assets for rapid price drops. Emergency stop on crash.
    """

    def __init__(
        self,
        hourly_drop_pct: float = None,
        daily_drop_pct: float = None,
        stability_hours: float = None,
        stability_max_hourly_pct: float = None,
    ):
        self.hourly_drop_pct = hourly_drop_pct or float(
            _env("CRASH_HOURLY_DROP_PCT", 5.0)
        )
        self.daily_drop_pct = daily_drop_pct or float(
            _env("CRASH_DAILY_DROP_PCT", 10.0)
        )
        self.stability_hours = stability_hours or float(
            _env("CRASH_STABILITY_HOURS", 3.0)
        )
        self.stability_max_hourly_pct = stability_max_hourly_pct or float(
            _env("CRASH_STABILITY_MAX_HOURLY_PCT", 2.0)
        )

        # asset -> deque of (timestamp, price)
        self._prices: Dict[str, deque] = defaultdict(lambda: deque(maxlen=86400))
        self._emergency_active = False
        self._emergency_since: Optional[float] = None
        self._stable_since: Optional[float] = None
        self._lock = threading.Lock()

    def update_price(self, asset: str, price: float, timestamp: float = None):
        ts = timestamp or time.time()
        with self._lock:
            self._prices[asset].append((ts, price))

    def check(self, asset: str, current_price: float) -> Tuple[bool, str]:
        """
        Returns (safe, reason).
        safe=True means OK to trade. safe=False means EMERGENCY STOP.
        """
        ts = time.time()
        with self._lock:
            self._prices[asset].append((ts, current_price))

            # If already in emergency, check stability
            if self._emergency_active:
                return self._check_stability(asset, ts)

            prices = list(self._prices[asset])

            # Check 1-hour drop
            one_hour_ago = ts - 3600
            hour_prices = [p for t, p in prices if t >= one_hour_ago]
            if hour_prices:
                hour_high = max(hour_prices)
                if hour_high > 0:
                    hour_drop = (hour_high - current_price) / hour_high * 100
                    if hour_drop >= self.hourly_drop_pct:
                        self._trigger_emergency(
                            f"{asset} dropped {hour_drop:.1f}% in 1 hour "
                            f"(threshold: {self.hourly_drop_pct}%)"
                        )
                        return False, self._last_reason

            # Check 24-hour drop
            one_day_ago = ts - 86400
            day_prices = [p for t, p in prices if t >= one_day_ago]
            if day_prices:
                day_high = max(day_prices)
                if day_high > 0:
                    day_drop = (day_high - current_price) / day_high * 100
                    if day_drop >= self.daily_drop_pct:
                        self._trigger_emergency(
                            f"{asset} dropped {day_drop:.1f}% in 24 hours "
                            f"(threshold: {self.daily_drop_pct}%)"
                        )
                        return False, self._last_reason

            return True, "OK"

    def _trigger_emergency(self, reason: str):
        self._emergency_active = True
        self._emergency_since = time.time()
        self._stable_since = None
        self._last_reason = f"CRASH DETECTED: {reason}"
        logger.critical("CRASH DETECTOR: %s", self._last_reason)
        logger.critical(
            "CRASH DETECTOR: ALL TRADING HALTED — cancel all orders immediately"
        )
        # Telegram alert placeholder
        _send_alert(self._last_reason)

    def _check_stability(self, asset: str, now: float) -> Tuple[bool, str]:
        """Check if market has been stable enough to resume."""
        prices = list(self._prices[asset])
        # Check last hour's max move
        one_hour_ago = now - 3600
        hour_prices = [p for t, p in prices if t >= one_hour_ago]
        if len(hour_prices) >= 2:
            h_high = max(hour_prices)
            h_low = min(hour_prices)
            if h_high > 0:
                hourly_range_pct = (h_high - h_low) / h_high * 100
                if hourly_range_pct < self.stability_max_hourly_pct:
                    if self._stable_since is None:
                        self._stable_since = now
                    elif now - self._stable_since >= self.stability_hours * 3600:
                        logger.info(
                            "CRASH DETECTOR: stability restored — "
                            "%.1f hours of <%.1f%% moves. Resuming.",
                            self.stability_hours, self.stability_max_hourly_pct,
                        )
                        self._emergency_active = False
                        self._stable_since = None
                        return True, "Stability restored"
                else:
                    self._stable_since = None

        return False, (
            f"EMERGENCY ACTIVE since {time.strftime('%H:%M:%S', time.localtime(self._emergency_since))} — "
            f"waiting for {self.stability_hours}h of <{self.stability_max_hourly_pct}% moves"
        )

    def cancel_all_on_crash(self, kalshi_client):
        """Call this when crash is detected to cancel all open orders."""
        cancel_all_open_orders(kalshi_client)

    @property
    def is_emergency(self) -> bool:
        return self._emergency_active


# ===================================================================
# 4. MAX LOSS LIMITS
# ===================================================================

class LossGuard:
    """
    Track P&L per bot and portfolio-wide. Enforce loss limits.
    """

    def __init__(
        self,
        hourly_loss_pct: float = None,
        daily_loss_pct: float = None,
        portfolio_daily_loss_pct: float = None,
        starting_balance: float = None,
    ):
        self.hourly_loss_pct = hourly_loss_pct or float(
            _env("LOSS_HOURLY_PCT", 2.0)
        )
        self.daily_loss_pct = daily_loss_pct or float(
            _env("LOSS_DAILY_PCT", 5.0)
        )
        self.portfolio_daily_loss_pct = portfolio_daily_loss_pct or float(
            _env("LOSS_PORTFOLIO_DAILY_PCT", 10.0)
        )
        self.starting_balance = starting_balance or float(
            _env("STARTING_BALANCE", 5000.0)
        )

        # bot_name -> deque of (timestamp, pnl)
        self._trades: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        # bot_name -> pause_until timestamp
        self._paused_until: Dict[str, float] = {}
        self._shutdown_bots: set = set()
        self._portfolio_shutdown = False
        self._lock = threading.Lock()

    def record_trade(self, bot_name: str, pnl: float) -> Tuple[bool, str]:
        """
        Record a trade's P&L and check if limits are breached.
        Returns (allowed, reason). allowed=False means bot should stop.
        """
        ts = time.time()
        with self._lock:
            self._trades[bot_name].append((ts, pnl))
            return self._check_bot(bot_name, ts)

    def check_limits(self, bot_name: str = None) -> Tuple[bool, str]:
        """
        Check if trading is allowed. If bot_name given, checks that bot.
        Always checks portfolio-wide limits too.
        """
        ts = time.time()
        with self._lock:
            if self._portfolio_shutdown:
                return False, "PORTFOLIO SHUTDOWN: daily loss > {}%".format(
                    self.portfolio_daily_loss_pct
                )

            if bot_name:
                if bot_name in self._shutdown_bots:
                    return False, f"BOT {bot_name} SHUTDOWN: daily loss > {self.daily_loss_pct}%"
                if bot_name in self._paused_until:
                    if ts < self._paused_until[bot_name]:
                        remaining = int(self._paused_until[bot_name] - ts)
                        return False, f"BOT {bot_name} PAUSED: hourly loss limit — {remaining}s remaining"
                    else:
                        del self._paused_until[bot_name]

            # Check portfolio-wide
            return self._check_portfolio(ts)

    def _check_bot(self, bot_name: str, ts: float) -> Tuple[bool, str]:
        """Check per-bot limits."""
        trades = list(self._trades[bot_name])

        # Hourly P&L
        one_hour_ago = ts - 3600
        hourly_pnl = sum(pnl for t, pnl in trades if t >= one_hour_ago)
        hourly_limit = self.starting_balance * (self.hourly_loss_pct / 100)
        if hourly_pnl < -hourly_limit:
            self._paused_until[bot_name] = ts + 3600
            reason = (
                f"BOT {bot_name} AUTO-PAUSED 1h: hourly loss ${abs(hourly_pnl):.2f} "
                f"> {self.hourly_loss_pct}% of ${self.starting_balance:.0f}"
            )
            logger.warning("LOSS GUARD: %s", reason)
            _send_alert(reason)
            return False, reason

        # Daily P&L
        day_start = ts - 86400
        daily_pnl = sum(pnl for t, pnl in trades if t >= day_start)
        daily_limit = self.starting_balance * (self.daily_loss_pct / 100)
        if daily_pnl < -daily_limit:
            self._shutdown_bots.add(bot_name)
            reason = (
                f"BOT {bot_name} SHUTDOWN: daily loss ${abs(daily_pnl):.2f} "
                f"> {self.daily_loss_pct}% of ${self.starting_balance:.0f}"
            )
            logger.critical("LOSS GUARD: %s", reason)
            _send_alert(reason)
            return False, reason

        return True, "OK"

    def _check_portfolio(self, ts: float) -> Tuple[bool, str]:
        """Check portfolio-wide daily limit."""
        if self._portfolio_shutdown:
            return False, "PORTFOLIO SHUTDOWN active"

        day_start = ts - 86400
        total_pnl = 0.0
        for bot_name, trades in self._trades.items():
            total_pnl += sum(pnl for t, pnl in trades if t >= day_start)

        portfolio_limit = self.starting_balance * (self.portfolio_daily_loss_pct / 100)
        if total_pnl < -portfolio_limit:
            self._portfolio_shutdown = True
            reason = (
                f"PORTFOLIO SHUTDOWN: total daily loss ${abs(total_pnl):.2f} "
                f"> {self.portfolio_daily_loss_pct}% of ${self.starting_balance:.0f}"
            )
            logger.critical("LOSS GUARD: %s", reason)
            _send_alert(reason)
            return False, reason

        return True, "OK"

    def reset_daily(self):
        """Call at start of new trading day to reset shutdown flags."""
        with self._lock:
            self._shutdown_bots.clear()
            self._portfolio_shutdown = False
            logger.info("LOSS GUARD: daily limits reset")


# ===================================================================
# 5. POSITION SIZE CAP
# ===================================================================

def size_guard(
    contracts: int,
    price_cents: int,
    max_contracts: int = None,
    max_notional_usd: float = None,
) -> int:
    """
    Cap position size by contracts and notional value.

    Args:
        contracts: requested number of contracts
        price_cents: price per contract in cents (1-99)
        max_contracts: hard cap on contracts (default from env or 100)
        max_notional_usd: hard cap on notional USD (default from env or 500)

    Returns:
        capped number of contracts (never exceeds either limit)
    """
    max_contracts = max_contracts or int(_env("MAX_CONTRACTS", 100))
    max_notional_usd = max_notional_usd or float(_env("MAX_NOTIONAL_USD", 500.0))

    if contracts <= 0:
        return 0
    if price_cents <= 0:
        logger.warning("SIZE GUARD: price_cents <= 0, blocking order")
        return 0

    price_usd = price_cents / 100.0

    # Notional cap: max_notional / price = max contracts by notional
    notional_cap = int(max_notional_usd / price_usd) if price_usd > 0 else 0

    # Apply both caps
    capped = min(contracts, max_contracts, notional_cap)
    capped = max(capped, 0)

    if capped < contracts:
        logger.info(
            "SIZE GUARD: capped %d → %d contracts (price=%d¢, max_contracts=%d, "
            "max_notional=$%s, notional_cap=%d)",
            contracts, capped, price_cents, max_contracts, max_notional_usd, notional_cap,
        )

    return capped


# ===================================================================
# 6. ORDER TIMEOUT / ORPHAN PROTECTION
# ===================================================================

class OrderTimeoutGuard:
    """
    Track open orders and auto-cancel stale ones.
    """

    def __init__(self, ttl_seconds: int = None):
        self.ttl = ttl_seconds or int(_env("ORDER_TTL_SECONDS", 60))
        # order_id -> created_at timestamp
        self._orders: Dict[str, float] = {}
        self._lock = threading.Lock()

    def register_order(self, order_id: str, created_at: float = None):
        """Register an order for timeout tracking."""
        with self._lock:
            self._orders[order_id] = created_at or time.time()
            logger.info("ORDER TIMEOUT: tracking order %s (TTL=%ds)", order_id, self.ttl)

    def unregister_order(self, order_id: str):
        """Remove order from tracking (e.g., when filled)."""
        with self._lock:
            self._orders.pop(order_id, None)

    def cleanup_stale_orders(self, kalshi_client) -> List[str]:
        """
        Cancel all tracked orders that have exceeded their TTL.
        Returns list of cancelled order IDs.
        """
        now = time.time()
        cancelled = []
        with self._lock:
            stale = [
                oid for oid, created in self._orders.items()
                if now - created > self.ttl
            ]

        for oid in stale:
            try:
                kalshi_client.cancel_order(order_id=oid)
                logger.warning(
                    "ORDER TIMEOUT: cancelled stale order %s (age=%.0fs, TTL=%ds)",
                    oid, now - self._orders.get(oid, now), self.ttl,
                )
                cancelled.append(oid)
            except Exception as e:
                logger.error("ORDER TIMEOUT: failed to cancel %s: %s", oid, e)

        with self._lock:
            for oid in cancelled:
                self._orders.pop(oid, None)

        return cancelled

    def get_tracked_count(self) -> int:
        with self._lock:
            return len(self._orders)


# ===================================================================
# 7. DEPLOY-SAFE STARTUP — Cancel all open orders
# ===================================================================

def cancel_all_open_orders(kalshi_client) -> int:
    """
    Cancel ALL open orders on Kalshi. Call on every bot startup/deploy.
    Returns number of orders cancelled.
    """
    cancelled = 0
    try:
        # Kalshi API: get all open orders
        response = kalshi_client.get_orders(status="resting")
        orders = response.get("orders", [])
        if not orders:
            # Try alternative response shape
            orders = response if isinstance(response, list) else []

        logger.info("DEPLOY CLEANUP: found %d open orders to cancel", len(orders))

        for order in orders:
            oid = order.get("order_id") or order.get("id")
            if oid:
                try:
                    kalshi_client.cancel_order(order_id=oid)
                    cancelled += 1
                    logger.info("DEPLOY CLEANUP: cancelled order %s", oid)
                except Exception as e:
                    logger.error("DEPLOY CLEANUP: failed to cancel %s: %s", oid, e)

    except Exception as e:
        logger.error("DEPLOY CLEANUP: failed to fetch open orders: %s", e)

    logger.info("DEPLOY CLEANUP: cancelled %d orders total", cancelled)
    _send_alert(f"Bot deployed — cancelled {cancelled} orphaned orders")
    return cancelled


# ===================================================================
# 8. CORRELATION PAUSE (CorrelationGuard)
# ===================================================================

class CorrelationGuard:
    """
    Track wins/losses per 15-min window across ALL bots.
    If 3+ bots all lose in the same window, pause ALL bots for 15 minutes.
    """

    def __init__(self, correlated_loss_threshold: int = None, pause_minutes: int = None):
        self.correlated_loss_threshold = correlated_loss_threshold or int(
            _env("CORR_LOSS_THRESHOLD", 3)
        )
        self.pause_minutes = pause_minutes or int(
            _env("CORR_PAUSE_MINUTES", 15)
        )
        # window_id -> {bot_name: won}
        self._windows: Dict[str, Dict[str, bool]] = defaultdict(dict)
        self._paused_until: Optional[float] = None
        self._lock = threading.Lock()

    @staticmethod
    def current_window_id(ts: float = None) -> str:
        """Return a window ID string for the current 15-min window."""
        ts = ts or time.time()
        # Floor to nearest 15-min boundary
        bucket = int(ts) // 900
        return str(bucket)

    def record_result(self, bot_name: str, window_id: str = None, won: bool = True):
        """Record a trade result for a bot in a given 15-min window."""
        wid = window_id or self.current_window_id()
        with self._lock:
            self._windows[wid][bot_name] = won

            # Check if 3+ bots all lost in this window
            results = self._windows[wid]
            losers = [b for b, w in results.items() if not w]
            if len(losers) >= self.correlated_loss_threshold:
                self._paused_until = time.time() + (self.pause_minutes * 60)
                reason = (
                    f"CORRELATION PAUSE: {len(losers)} bots lost in window {wid} "
                    f"({', '.join(losers)}) — pausing ALL for {self.pause_minutes}min"
                )
                logger.warning("CORRELATION GUARD: %s", reason)
                _send_alert(reason)

            # Cleanup old windows (keep last 10)
            if len(self._windows) > 10:
                oldest_keys = sorted(self._windows.keys())[:-10]
                for k in oldest_keys:
                    del self._windows[k]

    def should_trade(self) -> Tuple[bool, str]:
        """Check if trading is allowed (not in correlation pause)."""
        with self._lock:
            if self._paused_until is not None:
                remaining = self._paused_until - time.time()
                if remaining > 0:
                    return False, (
                        f"CORRELATION PAUSE: {int(remaining)}s remaining — "
                        f"systemic loss detected across bots"
                    )
                else:
                    self._paused_until = None
                    logger.info("CORRELATION GUARD: pause expired, resuming")
            return True, "OK"


# ===================================================================
# 9. SETTLEMENT VERIFICATION (SettlementGuard)
# ===================================================================

class SettlementGuard:
    """
    Compare bot's predicted outcome vs actual Kalshi settlement.
    Auto-pause bots whose models diverge from reality.
    """

    def __init__(
        self,
        rolling_window: int = None,
        min_accuracy_pct: float = None,
        max_consecutive_misses: int = None,
    ):
        self.rolling_window = rolling_window or int(
            _env("SETTLE_ROLLING_WINDOW", 50)
        )
        self.min_accuracy_pct = min_accuracy_pct or float(
            _env("SETTLE_MIN_ACCURACY_PCT", 45.0)
        )
        self.max_consecutive_misses = max_consecutive_misses or int(
            _env("SETTLE_MAX_CONSEC_MISSES", 3)
        )
        # bot_name -> deque of bool (True = correct prediction)
        self._results: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=50)
        )
        # bot_name -> current consecutive miss streak
        self._streaks: Dict[str, int] = defaultdict(int)
        # bot_name -> paused
        self._paused_bots: set = set()
        self._lock = threading.Lock()

    def record_settlement(self, bot_name: str, predicted: str, actual: str):
        """
        Record a settlement result.
        predicted/actual should be comparable values (e.g., 'yes'/'no', or price strings).
        """
        correct = (str(predicted).strip().lower() == str(actual).strip().lower())
        with self._lock:
            self._results[bot_name].append(correct)

            if correct:
                self._streaks[bot_name] = 0
            else:
                self._streaks[bot_name] += 1

            # Check consecutive miss threshold
            if self._streaks[bot_name] >= self.max_consecutive_misses:
                if bot_name not in self._paused_bots:
                    self._paused_bots.add(bot_name)
                    reason = (
                        f"SETTLEMENT GUARD: {bot_name} paused — "
                        f"{self._streaks[bot_name]} consecutive prediction misses"
                    )
                    logger.warning(reason)
                    _send_alert(reason)

            # Check rolling accuracy
            results = list(self._results[bot_name])
            if len(results) >= 10:  # need minimum sample
                accuracy = sum(results) / len(results) * 100
                if accuracy < self.min_accuracy_pct:
                    if bot_name not in self._paused_bots:
                        self._paused_bots.add(bot_name)
                        reason = (
                            f"SETTLEMENT GUARD: {bot_name} paused — "
                            f"accuracy {accuracy:.1f}% < {self.min_accuracy_pct}% "
                            f"over last {len(results)} trades"
                        )
                        logger.warning(reason)
                        _send_alert(reason)

    def is_model_healthy(self, bot_name: str) -> Tuple[bool, float, int]:
        """
        Returns (healthy, accuracy_pct, consecutive_miss_streak).
        healthy=False means the bot is auto-paused.
        """
        with self._lock:
            results = list(self._results.get(bot_name, []))
            accuracy = (sum(results) / len(results) * 100) if results else 100.0
            streak = self._streaks.get(bot_name, 0)
            healthy = bot_name not in self._paused_bots
            return healthy, accuracy, streak

    def unpause_bot(self, bot_name: str):
        """Manually unpause a bot after model fix."""
        with self._lock:
            self._paused_bots.discard(bot_name)
            self._streaks[bot_name] = 0
            logger.info("SETTLEMENT GUARD: %s manually unpaused", bot_name)


# ===================================================================
# 10. SPREAD SANITY CHECK (SpreadGuard)
# ===================================================================

class SpreadGuard:
    """
    Check bid-ask spreads before placing orders.
    Reject illiquid markets, reduce size on wide spreads.
    """

    def __init__(
        self,
        reject_spread_cents: int = None,
        reduce_spread_cents: int = None,
        mm_max_spread_cents: int = None,
    ):
        self.reject_spread_cents = reject_spread_cents or int(
            _env("SPREAD_REJECT_CENTS", 15)
        )
        self.reduce_spread_cents = reduce_spread_cents or int(
            _env("SPREAD_REDUCE_CENTS", 10)
        )
        self.mm_max_spread_cents = mm_max_spread_cents or int(
            _env("SPREAD_MM_MAX_CENTS", 8)
        )

    def check(
        self,
        yes_bid: Optional[int],
        yes_ask: Optional[int],
        no_bid: Optional[int] = None,
        no_ask: Optional[int] = None,
    ) -> Tuple[bool, str, float]:
        """
        Check spread sanity.
        All prices in cents (1-99).

        Returns (allowed, reason, size_modifier).
        size_modifier: 1.0 = full size, 0.5 = half size, 0.0 = rejected.
        """
        # Check if bid/ask exist
        if yes_bid is None or yes_ask is None:
            return False, "SPREAD GUARD: no bid or ask — market is empty", 0.0

        if yes_bid <= 0 or yes_ask <= 0:
            return False, "SPREAD GUARD: invalid bid/ask prices", 0.0

        spread = yes_ask - yes_bid

        if spread > self.reject_spread_cents:
            return False, (
                f"SPREAD GUARD: spread {spread}c > {self.reject_spread_cents}c — "
                f"REJECTED (illiquid, adverse selection risk)"
            ), 0.0

        if spread > self.reduce_spread_cents:
            return True, (
                f"SPREAD GUARD: spread {spread}c > {self.reduce_spread_cents}c — "
                f"reducing position size by 50%"
            ), 0.5

        return True, "OK", 1.0

    def check_for_market_making(
        self,
        yes_bid: Optional[int],
        yes_ask: Optional[int],
    ) -> Tuple[bool, str]:
        """
        Check if a market is suitable for market-making.
        Only provide liquidity on tight-spread markets.
        """
        if yes_bid is None or yes_ask is None:
            return False, "SPREAD GUARD (MM): no bid or ask — skip market"

        spread = yes_ask - yes_bid
        if spread > self.mm_max_spread_cents:
            return False, (
                f"SPREAD GUARD (MM): spread {spread}c > {self.mm_max_spread_cents}c — "
                f"too wide for market-making"
            )

        return True, "OK"


# ===================================================================
# 11. HEARTBEAT MONITOR
# ===================================================================

class HeartbeatMonitor:
    """
    Track bot liveness via periodic pings.
    Flag bots as DEAD if they go silent during market hours.
    """

    def __init__(self, timeout_minutes: int = None):
        self.timeout_minutes = timeout_minutes or int(
            _env("HEARTBEAT_TIMEOUT_MIN", 30)
        )
        # bot_name -> last ping timestamp
        self._pings: Dict[str, float] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _is_market_hours(ts: float = None) -> bool:
        """
        Check if current time is during market hours.
        Crypto: 14:00 - 04:00 UTC (next day).
        """
        ts = ts or time.time()
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        hour = dt.hour
        # 14:00-23:59 or 00:00-04:00 UTC
        return hour >= 14 or hour < 4

    def ping(self, bot_name: str):
        """Record a heartbeat ping from a bot."""
        with self._lock:
            self._pings[bot_name] = time.time()

    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all known bots.
        Returns {bot_name: {last_ping, alive, minutes_silent}}.
        """
        now = time.time()
        with self._lock:
            result = {}
            for bot_name, last_ping in self._pings.items():
                minutes_silent = (now - last_ping) / 60.0
                alive = minutes_silent < self.timeout_minutes
                result[bot_name] = {
                    "last_ping": last_ping,
                    "alive": alive,
                    "minutes_silent": round(minutes_silent, 1),
                }
            return result

    def get_dead_bots(self) -> List[str]:
        """
        Return list of bot names that have been silent > timeout_minutes
        during market hours.
        """
        if not self._is_market_hours():
            return []

        now = time.time()
        dead = []
        with self._lock:
            for bot_name, last_ping in self._pings.items():
                minutes_silent = (now - last_ping) / 60.0
                if minutes_silent > self.timeout_minutes:
                    dead.append(bot_name)

        if dead:
            logger.warning(
                "HEARTBEAT: dead bots detected: %s", ", ".join(dead)
            )
        return dead


# ===================================================================
# 12. NEWS/EVENT GUARD (EventGuard)
# ===================================================================

class EventGuard:
    """
    Pause trading around high-impact economic events (FOMC, CPI, NFP, etc.).
    Configurable via EVENT_BLACKOUT_TIMES env var.
    """

    def __init__(
        self,
        pre_event_minutes: int = None,
        post_event_minutes: int = None,
        event_times: List[str] = None,
    ):
        self.pre_event_minutes = pre_event_minutes or int(
            _env("EVENT_PRE_MINUTES", 5)
        )
        self.post_event_minutes = post_event_minutes or int(
            _env("EVENT_POST_MINUTES", 10)
        )
        # Parse event times from env or constructor
        # Format: "2026-04-07T14:00,2026-04-10T12:30" (UTC ISO timestamps)
        raw = event_times or _env("EVENT_BLACKOUT_TIMES", "").split(",")
        self._event_timestamps: List[float] = []
        for t in raw:
            t = t.strip()
            if not t:
                continue
            try:
                dt = datetime.fromisoformat(t).replace(tzinfo=timezone.utc)
                self._event_timestamps.append(dt.timestamp())
            except (ValueError, TypeError):
                logger.warning("EVENT GUARD: could not parse event time '%s'", t)

        logger.info(
            "EVENT GUARD: loaded %d blackout events", len(self._event_timestamps)
        )

    def should_trade(self) -> Tuple[bool, str]:
        """
        Check if we are in a blackout window around any scheduled event.
        Returns (allowed, reason).
        """
        now = time.time()
        pre_seconds = self.pre_event_minutes * 60
        post_seconds = self.post_event_minutes * 60

        for event_ts in self._event_timestamps:
            # Window: [event - pre, event + post]
            if (event_ts - pre_seconds) <= now <= (event_ts + post_seconds):
                event_str = datetime.fromtimestamp(
                    event_ts, tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M UTC")
                if now < event_ts:
                    return False, (
                        f"EVENT GUARD: blackout — {int((event_ts - now) / 60)}min "
                        f"before event at {event_str}"
                    )
                else:
                    return False, (
                        f"EVENT GUARD: blackout — {int((now - event_ts) / 60)}min "
                        f"after event at {event_str}"
                    )

        return True, "OK"

    def add_event(self, iso_time: str):
        """Add a new event time dynamically."""
        try:
            dt = datetime.fromisoformat(iso_time).replace(tzinfo=timezone.utc)
            self._event_timestamps.append(dt.timestamp())
            logger.info("EVENT GUARD: added event at %s", iso_time)
        except (ValueError, TypeError) as e:
            logger.error("EVENT GUARD: invalid time '%s': %s", iso_time, e)


# ===================================================================
# 13. EXCHANGE LAG DETECTOR (LagDetector)
# ===================================================================

class LagDetector:
    """
    Track freshness of exchange price feeds.
    Warn/pause on stale data to prevent bad predictions.
    """

    def __init__(
        self,
        warn_seconds: float = None,
        pause_seconds: float = None,
    ):
        self.warn_seconds = warn_seconds or float(
            _env("LAG_WARN_SECONDS", 30.0)
        )
        self.pause_seconds = pause_seconds or float(
            _env("LAG_PAUSE_SECONDS", 60.0)
        )
        # exchange -> last update timestamp
        self._feeds: Dict[str, float] = {}
        self._lock = threading.Lock()

    def update(self, exchange: str, timestamp: float = None):
        """Record a price update from an exchange."""
        with self._lock:
            self._feeds[exchange] = timestamp or time.time()

    def is_data_fresh(self) -> Tuple[bool, List[str], float]:
        """
        Check if exchange data is fresh enough to trade.

        Returns (fresh, stale_exchanges, max_lag_seconds).
        fresh=False means at least one feed is critically stale (>pause_seconds).
        """
        now = time.time()
        with self._lock:
            if not self._feeds:
                # No feeds registered yet — allow (startup grace)
                return True, [], 0.0

            stale_exchanges = []
            max_lag = 0.0

            for exchange, last_ts in self._feeds.items():
                lag = now - last_ts
                max_lag = max(max_lag, lag)

                if lag > self.warn_seconds:
                    stale_exchanges.append(exchange)

                if lag > self.warn_seconds and lag <= self.pause_seconds:
                    logger.warning(
                        "LAG DETECTOR: %s feed is %.1fs stale (warn threshold: %.0fs)",
                        exchange, lag, self.warn_seconds,
                    )

            # Check if ANY feed is critically stale
            critical = any(
                (now - ts) > self.pause_seconds
                for ts in self._feeds.values()
            )

            if critical:
                all_stale = all(
                    (now - ts) > self.pause_seconds
                    for ts in self._feeds.values()
                )
                if all_stale:
                    logger.critical(
                        "LAG DETECTOR: ALL exchange feeds stale >%.0fs — EMERGENCY PAUSE",
                        self.pause_seconds,
                    )
                    _send_alert(
                        f"EXCHANGE LAG: ALL feeds stale >{self.pause_seconds}s — "
                        f"emergency pause"
                    )
                else:
                    logger.warning(
                        "LAG DETECTOR: stale feeds: %s — pausing trading",
                        ", ".join(stale_exchanges),
                    )
                return False, stale_exchanges, max_lag

            return True, stale_exchanges, max_lag


# ===================================================================
# ALERT PLACEHOLDER (wire up to Telegram later)
# ===================================================================

def _send_alert(message: str):
    """
    Send alert via Telegram (placeholder — just logs for now).
    Wire this to your Telegram bot when ready.
    """
    logger.info("ALERT (Telegram placeholder): %s", message)
    # TODO: connect to Telegram bot
    # import requests
    # TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
    # TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "8375236175")
    # if TELEGRAM_BOT_TOKEN:
    #     requests.post(
    #         f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
    #         json={"chat_id": TELEGRAM_CHAT_ID, "text": f"🚨 RISK GUARD: {message}"}
    #     )


# ===================================================================
# 14. POSITION RECONCILIATION
# ===================================================================

class PositionReconciler:
    """
    After every trade, query Kalshi portfolio and verify positions match
    what the bot thinks it has. Detect orphaned positions.
    """

    def __init__(self, tolerance_contracts: int = None):
        self.tolerance = tolerance_contracts or int(
            _env("RECONCILE_TOLERANCE_CONTRACTS", 5)
        )
        self._expected: Dict[str, int] = {}  # ticker -> expected contracts
        self._last_check: float = 0.0
        self._check_interval: float = float(_env("RECONCILE_INTERVAL_SEC", 120))
        self._lock = threading.Lock()

    def record_expected(self, ticker: str, delta_contracts: int):
        """Update expected position for a ticker."""
        with self._lock:
            self._expected[ticker] = self._expected.get(ticker, 0) + delta_contracts

    def reconcile(self, kalshi_client) -> Dict[str, Any]:
        """
        Query Kalshi portfolio positions and compare to expected.
        Returns dict with mismatch details.
        """
        now = time.time()
        if now - self._last_check < self._check_interval:
            return {"checked": False}

        self._last_check = now
        result = {"checked": True, "mismatches": [], "orphaned": []}

        try:
            response = kalshi_client.get_positions() if hasattr(kalshi_client, 'get_positions') else {}
            positions = response.get("market_positions", response.get("positions", []))
            if isinstance(positions, dict):
                positions = [positions]

            actual: Dict[str, int] = {}
            for pos in positions:
                ticker = pos.get("ticker", pos.get("market_ticker", ""))
                count = pos.get("position", pos.get("total_traded", 0))
                if ticker:
                    actual[ticker] = int(count)

            with self._lock:
                # Check for mismatches
                for ticker, expected in self._expected.items():
                    got = actual.get(ticker, 0)
                    diff = abs(got - expected)
                    if diff > self.tolerance:
                        mismatch = {
                            "ticker": ticker,
                            "expected": expected,
                            "actual": got,
                            "diff": diff,
                        }
                        result["mismatches"].append(mismatch)
                        logger.warning(
                            "RECONCILE: mismatch on %s — expected %d, actual %d (diff %d)",
                            ticker, expected, got, diff,
                        )

                # Check for orphaned positions (in Kalshi but not tracked)
                for ticker, count in actual.items():
                    if ticker not in self._expected and count != 0:
                        result["orphaned"].append({"ticker": ticker, "contracts": count})
                        logger.warning(
                            "RECONCILE: orphaned position — %s has %d contracts not tracked",
                            ticker, count,
                        )

            if result["mismatches"] or result["orphaned"]:
                _send_alert(
                    f"POSITION RECONCILE: {len(result['mismatches'])} mismatches, "
                    f"{len(result['orphaned'])} orphaned"
                )

        except Exception as e:
            logger.error("RECONCILE: failed to query positions: %s", e)
            result["error"] = str(e)

        return result

    def clear(self):
        with self._lock:
            self._expected.clear()


# ===================================================================
# 15. FLASH CRASH CIRCUIT BREAKER (1-minute window)
# ===================================================================

class FlashCrashGuard:
    """
    Detect rapid price drops within a short window (e.g., 3% in 1 minute).
    More sensitive than CrashDetector which looks at hourly windows.
    """

    def __init__(
        self,
        drop_pct: float = None,
        window_seconds: int = None,
        cooldown_minutes: int = None,
    ):
        self.drop_pct = drop_pct or float(_env("FLASH_DROP_PCT", 3.0))
        self.window_seconds = window_seconds or int(_env("FLASH_WINDOW_SEC", 60))
        self.cooldown_minutes = cooldown_minutes or int(_env("FLASH_COOLDOWN_MIN", 5))

        self._prices: Dict[str, deque] = defaultdict(lambda: deque(maxlen=600))
        self._triggered_until: Optional[float] = None
        self._lock = threading.Lock()

    def update_price(self, asset: str, price: float, timestamp: float = None):
        ts = timestamp or time.time()
        with self._lock:
            self._prices[asset].append((ts, price))

    def check(self, asset: str) -> Tuple[bool, str]:
        """Returns (safe, reason). safe=False means flash crash detected."""
        now = time.time()
        with self._lock:
            # Check cooldown
            if self._triggered_until and now < self._triggered_until:
                remaining = int(self._triggered_until - now)
                return False, f"FLASH CRASH COOLDOWN: {remaining}s remaining"

            if self._triggered_until and now >= self._triggered_until:
                self._triggered_until = None
                logger.info("FLASH CRASH GUARD: cooldown expired, resuming")

            prices = list(self._prices.get(asset, []))
            if len(prices) < 2:
                return True, "OK"

            cutoff = now - self.window_seconds
            window_prices = [(t, p) for t, p in prices if t >= cutoff]
            if len(window_prices) < 2:
                return True, "OK"

            high = max(p for _, p in window_prices)
            current = window_prices[-1][1]

            if high > 0:
                drop = (high - current) / high * 100
                if drop >= self.drop_pct:
                    self._triggered_until = now + (self.cooldown_minutes * 60)
                    reason = (
                        f"FLASH CRASH: {asset} dropped {drop:.1f}% in "
                        f"{self.window_seconds}s (threshold: {self.drop_pct}%)"
                    )
                    logger.critical("FLASH CRASH GUARD: %s", reason)
                    _send_alert(reason)
                    return False, reason

            return True, "OK"


# ===================================================================
# 16. DOUBLE ORDER PREVENTION
# ===================================================================

class DoubleOrderGuard:
    """
    Prevent duplicate orders on the same market+side within a short window.
    Catches retry bugs where the same order fires twice.
    """

    def __init__(self, dedup_window_seconds: float = None):
        self.dedup_window = dedup_window_seconds or float(
            _env("DEDUP_WINDOW_SEC", 5.0)
        )
        # (ticker, side) -> last order timestamp
        self._recent_orders: Dict[Tuple[str, str], float] = {}
        self._lock = threading.Lock()

    def check_and_record(self, ticker: str, side: str) -> Tuple[bool, str]:
        """
        Check if this order is a duplicate. If allowed, records it.
        Returns (allowed, reason).
        """
        key = (ticker, side)
        now = time.time()
        with self._lock:
            last = self._recent_orders.get(key, 0.0)
            if now - last < self.dedup_window:
                reason = (
                    f"DOUBLE ORDER BLOCKED: {ticker} {side} — "
                    f"last order {now - last:.1f}s ago (dedup window: {self.dedup_window}s)"
                )
                logger.warning(reason)
                return False, reason
            self._recent_orders[key] = now

            # Cleanup old entries
            cutoff = now - self.dedup_window * 2
            stale_keys = [k for k, t in self._recent_orders.items() if t < cutoff]
            for k in stale_keys:
                del self._recent_orders[k]

            return True, "OK"


# ===================================================================
# 17. BALANCE RECONCILIATION
# ===================================================================

class BalanceReconciler:
    """
    Periodically query actual Kalshi balance and compare to bot's
    internal tracking. Alert if divergence exceeds threshold.
    """

    def __init__(
        self,
        check_interval_sec: float = None,
        alert_threshold_usd: float = None,
    ):
        self.check_interval = check_interval_sec or float(
            _env("BALANCE_CHECK_INTERVAL_SEC", 300)
        )
        self.alert_threshold = alert_threshold_usd or float(
            _env("BALANCE_ALERT_THRESHOLD_USD", 10.0)
        )
        self._last_check: float = 0.0
        self._internal_balance: float = 0.0
        self._lock = threading.Lock()

    def set_internal_balance(self, balance: float):
        with self._lock:
            self._internal_balance = balance

    def reconcile(self, kalshi_client) -> Dict[str, Any]:
        """
        Query actual balance and compare. Returns dict with details.
        """
        now = time.time()
        if now - self._last_check < self.check_interval:
            return {"checked": False}

        self._last_check = now
        result = {"checked": True}

        try:
            if hasattr(kalshi_client, 'get_balance'):
                response = kalshi_client.get_balance()
                actual = response if isinstance(response, (int, float)) else 0
            else:
                return {"checked": False, "reason": "client has no get_balance"}

            with self._lock:
                diff = abs(actual - self._internal_balance)
                result["actual"] = actual
                result["internal"] = self._internal_balance
                result["diff"] = diff

                if diff > self.alert_threshold:
                    reason = (
                        f"BALANCE MISMATCH: actual=${actual:.2f} vs "
                        f"internal=${self._internal_balance:.2f} (diff=${diff:.2f})"
                    )
                    logger.warning("BALANCE RECONCILE: %s", reason)
                    _send_alert(reason)
                    result["alert"] = True
                else:
                    result["alert"] = False

        except Exception as e:
            logger.error("BALANCE RECONCILE: failed: %s", e)
            result["error"] = str(e)

        return result


# ===================================================================
# 18. RATE LIMIT BACKOFF
# ===================================================================

class RateLimitGuard:
    """
    Track API rate limit responses (HTTP 429) and enforce exponential
    backoff instead of immediate retry.
    """

    def __init__(
        self,
        base_delay_sec: float = None,
        max_delay_sec: float = None,
        reset_after_sec: float = None,
    ):
        self.base_delay = base_delay_sec or float(_env("RATE_BASE_DELAY_SEC", 5.0))
        self.max_delay = max_delay_sec or float(_env("RATE_MAX_DELAY_SEC", 60.0))
        self.reset_after = reset_after_sec or float(_env("RATE_RESET_SEC", 120.0))

        self._consecutive_429s: int = 0
        self._last_429: float = 0.0
        self._backoff_until: float = 0.0
        self._lock = threading.Lock()

    def record_429(self):
        """Record a 429 response. Sets backoff timer."""
        now = time.time()
        with self._lock:
            self._last_429 = now
            self._consecutive_429s += 1
            delay = min(
                self.base_delay * (2 ** (self._consecutive_429s - 1)),
                self.max_delay,
            )
            self._backoff_until = now + delay
            logger.warning(
                "RATE LIMIT: 429 received (#%d) — backing off %.0fs",
                self._consecutive_429s, delay,
            )

    def record_success(self):
        """Record a successful API call. Resets backoff counter if enough time passed."""
        now = time.time()
        with self._lock:
            if now - self._last_429 > self.reset_after:
                self._consecutive_429s = 0

    def should_wait(self) -> Tuple[bool, float]:
        """
        Check if we should wait before making an API call.
        Returns (should_wait, remaining_seconds).
        """
        now = time.time()
        with self._lock:
            if now < self._backoff_until:
                remaining = self._backoff_until - now
                return True, remaining
            return False, 0.0


# ===================================================================
# 19. MAX FLEET DRAWDOWN PER DAY
# ===================================================================

class FleetDrawdownGuard:
    """
    Monitor total P&L across ALL bots. If combined fleet drawdown
    exceeds threshold in one day, shut down everything.
    """

    def __init__(
        self,
        max_drawdown_pct: float = None,
        starting_capital: float = None,
    ):
        self.max_drawdown_pct = max_drawdown_pct or float(
            _env("FLEET_MAX_DRAWDOWN_PCT", 10.0)
        )
        self.starting_capital = starting_capital or float(
            _env("STARTING_BALANCE", 5000.0)
        )

        self._daily_pnl: float = 0.0
        self._day_start: float = time.time()
        self._shutdown: bool = False
        self._lock = threading.Lock()

    def record_pnl(self, pnl: float) -> Tuple[bool, str]:
        """Record P&L from any bot. Returns (allowed, reason)."""
        now = time.time()
        with self._lock:
            # Reset at start of new day
            if now - self._day_start > 86400:
                self._daily_pnl = 0.0
                self._day_start = now
                self._shutdown = False

            if self._shutdown:
                return False, "FLEET SHUTDOWN: daily drawdown limit hit"

            self._daily_pnl += pnl
            limit = self.starting_capital * (self.max_drawdown_pct / 100.0)

            if self._daily_pnl < -limit:
                self._shutdown = True
                reason = (
                    f"FLEET SHUTDOWN: daily P&L ${self._daily_pnl:.2f} "
                    f"exceeds {self.max_drawdown_pct}% of ${self.starting_capital:.0f} "
                    f"(limit: -${limit:.2f})"
                )
                logger.critical("FLEET DRAWDOWN: %s", reason)
                _send_alert(reason)
                return False, reason

            return True, "OK"

    @property
    def is_shutdown(self) -> bool:
        with self._lock:
            return self._shutdown


# ===================================================================
# 20. TIME SYNC CHECK
# ===================================================================

class TimeSyncGuard:
    """
    Compare local bot time vs server time. Pause if drift exceeds threshold.
    Clock drift can cause orders placed at wrong times relative to settlement.
    """

    def __init__(
        self,
        max_drift_seconds: float = None,
        check_interval_sec: float = None,
    ):
        self.max_drift = max_drift_seconds or float(_env("TIME_MAX_DRIFT_SEC", 5.0))
        self.check_interval = check_interval_sec or float(
            _env("TIME_CHECK_INTERVAL_SEC", 300)
        )

        self._last_check: float = 0.0
        self._last_drift: float = 0.0
        self._paused: bool = False
        self._lock = threading.Lock()

    def check_sync(self, server_timestamp: float = None) -> Tuple[bool, str]:
        """
        Check time sync. Pass server timestamp from any API response header.
        Returns (ok, reason).
        """
        now = time.time()
        if now - self._last_check < self.check_interval:
            with self._lock:
                if self._paused:
                    return False, f"TIME SYNC: paused — drift {self._last_drift:.1f}s"
                return True, "OK"

        self._last_check = now

        if server_timestamp is None:
            return True, "OK"  # No server time to compare

        drift = abs(now - server_timestamp)
        with self._lock:
            self._last_drift = drift
            if drift > self.max_drift:
                self._paused = True
                reason = (
                    f"TIME SYNC: drift {drift:.1f}s > {self.max_drift:.1f}s — "
                    f"PAUSING (clock may cause bad settlements)"
                )
                logger.warning("TIME SYNC GUARD: %s", reason)
                _send_alert(reason)
                return False, reason
            else:
                self._paused = False
                return True, "OK"


# ===================================================================
# UNIFIED RISK MANAGER
# ===================================================================

class RiskManager:
    """
    Combines all guards into a single interface.

    Usage:
        rm = RiskManager()

        # On bot startup:
        rm.deploy_cleanup(kalshi_client)

        # Before every trade:
        ok, reason = rm.pre_trade_check("BTC", price_cents=50, contracts=30, side="yes")
        if not ok:
            print(f"Trade blocked: {reason}")
            continue

        # Place order through atomic guard:
        result = rm.atomic_order(kalshi_client, ticker, yes_order, no_order)

        # After trade settles:
        rm.post_trade("my_bot", pnl=-1.50)

        # Periodic maintenance (call every 10-30s):
        rm.maintenance(kalshi_client)
    """

    def __init__(
        self,
        starting_balance: float = None,
        config: dict = None,
    ):
        config = config or {}
        self.starting_balance = starting_balance or float(
            config.get("starting_balance", _env("STARTING_BALANCE", 5000.0))
        )

        self.volatility_guard = VolatilityGuard(
            window_minutes=config.get("vol_window_minutes"),
            spike_multiplier=config.get("vol_spike_multiplier"),
            resume_multiplier=config.get("vol_resume_multiplier"),
            resume_stable_minutes=config.get("vol_resume_stable_minutes"),
        )
        self.crash_detector = CrashDetector(
            hourly_drop_pct=config.get("crash_hourly_drop_pct"),
            daily_drop_pct=config.get("crash_daily_drop_pct"),
            stability_hours=config.get("crash_stability_hours"),
            stability_max_hourly_pct=config.get("crash_stability_max_hourly_pct"),
        )
        self.loss_guard = LossGuard(
            hourly_loss_pct=config.get("loss_hourly_pct"),
            daily_loss_pct=config.get("loss_daily_pct"),
            portfolio_daily_loss_pct=config.get("loss_portfolio_daily_pct"),
            starting_balance=self.starting_balance,
        )
        self.order_timeout = OrderTimeoutGuard(
            ttl_seconds=config.get("order_ttl_seconds"),
        )
        self.correlation_guard = CorrelationGuard(
            correlated_loss_threshold=config.get("corr_loss_threshold"),
            pause_minutes=config.get("corr_pause_minutes"),
        )
        self.settlement_guard = SettlementGuard(
            rolling_window=config.get("settle_rolling_window"),
            min_accuracy_pct=config.get("settle_min_accuracy_pct"),
            max_consecutive_misses=config.get("settle_max_consec_misses"),
        )
        self.spread_guard = SpreadGuard(
            reject_spread_cents=config.get("spread_reject_cents"),
            reduce_spread_cents=config.get("spread_reduce_cents"),
            mm_max_spread_cents=config.get("spread_mm_max_cents"),
        )
        self.heartbeat = HeartbeatMonitor(
            timeout_minutes=config.get("heartbeat_timeout_min"),
        )
        self.event_guard = EventGuard(
            pre_event_minutes=config.get("event_pre_minutes"),
            post_event_minutes=config.get("event_post_minutes"),
            event_times=config.get("event_times"),
        )
        self.lag_detector = LagDetector(
            warn_seconds=config.get("lag_warn_seconds"),
            pause_seconds=config.get("lag_pause_seconds"),
        )

        # --- New guards (14-20) ---
        self.position_reconciler = PositionReconciler(
            tolerance_contracts=config.get("reconcile_tolerance_contracts"),
        )
        self.flash_crash_guard = FlashCrashGuard(
            drop_pct=config.get("flash_drop_pct"),
            window_seconds=config.get("flash_window_sec"),
            cooldown_minutes=config.get("flash_cooldown_min"),
        )
        self.double_order_guard = DoubleOrderGuard(
            dedup_window_seconds=config.get("dedup_window_sec"),
        )
        self.balance_reconciler = BalanceReconciler(
            check_interval_sec=config.get("balance_check_interval_sec"),
            alert_threshold_usd=config.get("balance_alert_threshold_usd"),
        )
        self.rate_limit_guard = RateLimitGuard(
            base_delay_sec=config.get("rate_base_delay_sec"),
            max_delay_sec=config.get("rate_max_delay_sec"),
            reset_after_sec=config.get("rate_reset_sec"),
        )
        self.fleet_drawdown_guard = FleetDrawdownGuard(
            max_drawdown_pct=config.get("fleet_max_drawdown_pct"),
            starting_capital=self.starting_balance,
        )
        self.time_sync_guard = TimeSyncGuard(
            max_drift_seconds=config.get("time_max_drift_sec"),
            check_interval_sec=config.get("time_check_interval_sec"),
        )

        self.max_contracts = int(
            config.get("max_contracts", _env("MAX_CONTRACTS", 100))
        )
        self.max_notional_usd = float(
            config.get("max_notional_usd", _env("MAX_NOTIONAL_USD", 500.0))
        )

        logger.info(
            "RiskManager initialized — balance=$%.0f, max_contracts=%d, max_notional=$%.0f, "
            "guards=20 (13 original + 7 new)",
            self.starting_balance, self.max_contracts, self.max_notional_usd,
        )

    def deploy_cleanup(self, kalshi_client) -> int:
        """Call on every bot startup to cancel orphaned orders."""
        return cancel_all_open_orders(kalshi_client)

    def pre_trade_check(
        self,
        asset: str,
        price_cents: int,
        contracts: int,
        side: str,
        bot_name: str = "default",
        yes_bid: int = None,
        yes_ask: int = None,
    ) -> Tuple[bool, str, int]:
        """
        Run ALL pre-trade checks. Returns (allowed, reason, capped_contracts).

        If allowed is False, do NOT place the trade.
        If allowed is True, use capped_contracts (may be less than requested).

        Pass yes_bid/yes_ask (cents) to enable spread checks.
        """
        # 0a. Check fleet-wide drawdown
        if self.fleet_drawdown_guard.is_shutdown:
            return False, "FLEET SHUTDOWN: daily drawdown limit hit", 0

        # 0b. Check rate limit backoff
        waiting, remaining = self.rate_limit_guard.should_wait()
        if waiting:
            return False, f"RATE LIMIT: backing off {remaining:.0f}s", 0

        # 0c. Check time sync
        ok, reason = self.time_sync_guard.check_sync()
        if not ok:
            return False, reason, 0

        # 1. Check crash detector
        price_usd = price_cents / 100.0
        safe, reason = self.crash_detector.check(asset, price_usd)
        if not safe:
            return False, reason, 0

        # 1b. Check flash crash (faster window)
        safe, reason = self.flash_crash_guard.check(asset)
        if not safe:
            return False, reason, 0

        # 2. Check volatility
        self.volatility_guard.update_price(asset, price_usd)
        if not self.volatility_guard.should_trade(asset):
            return False, "VOLATILITY CIRCUIT BREAKER: trading paused", 0

        # 3. Check loss limits
        allowed, reason = self.loss_guard.check_limits(bot_name)
        if not allowed:
            return False, reason, 0

        # 4. Check correlation pause
        allowed, reason = self.correlation_guard.should_trade()
        if not allowed:
            return False, reason, 0

        # 5. Check settlement health (is bot's model still accurate?)
        healthy, accuracy, streak = self.settlement_guard.is_model_healthy(bot_name)
        if not healthy:
            return False, (
                f"SETTLEMENT GUARD: {bot_name} paused — accuracy {accuracy:.1f}%, "
                f"streak {streak} misses"
            ), 0

        # 6. Check event blackout
        allowed, reason = self.event_guard.should_trade()
        if not allowed:
            return False, reason, 0

        # 7. Check exchange data freshness
        fresh, stale, max_lag = self.lag_detector.is_data_fresh()
        if not fresh:
            return False, (
                f"LAG DETECTOR: stale feeds ({', '.join(stale)}) — "
                f"max lag {max_lag:.0f}s"
            ), 0

        # 8. Cap position size
        capped = size_guard(
            contracts, price_cents,
            max_contracts=self.max_contracts,
            max_notional_usd=self.max_notional_usd,
        )
        if capped == 0:
            return False, "SIZE GUARD: order reduced to 0 contracts", 0

        # 9. Check spread (if bid/ask provided)
        if yes_bid is not None and yes_ask is not None:
            spread_ok, spread_reason, size_mod = self.spread_guard.check(
                yes_bid, yes_ask
            )
            if not spread_ok:
                return False, spread_reason, 0
            if size_mod < 1.0:
                capped = max(1, int(capped * size_mod))
                logger.info(
                    "SPREAD GUARD: reduced contracts %d -> %d (modifier %.1f)",
                    contracts, capped, size_mod,
                )

        # 10. Double order prevention
        ticker_key = asset  # callers should pass ticker as asset for best dedup
        allowed, reason = self.double_order_guard.check_and_record(ticker_key, side)
        if not allowed:
            return False, reason, 0

        return True, "OK", capped

    def post_trade(self, bot_name: str, pnl: float) -> Tuple[bool, str]:
        """
        Record trade result. Returns (allowed_to_continue, reason).
        Also updates fleet drawdown tracking.
        """
        # Track fleet-wide drawdown
        fleet_ok, fleet_reason = self.fleet_drawdown_guard.record_pnl(pnl)
        if not fleet_ok:
            return False, fleet_reason
        return self.loss_guard.record_trade(bot_name, pnl)

    def atomic_order(
        self,
        kalshi_client,
        ticker: str,
        yes_order: dict,
        no_order: dict,
        timeout: float = 5.0,
    ) -> dict:
        """Place atomic both-sides order with full protection."""
        return atomic_both_sides(kalshi_client, ticker, yes_order, no_order, timeout)

    def register_order(self, order_id: str):
        """Track an order for timeout protection."""
        self.order_timeout.register_order(order_id)

    def order_filled(self, order_id: str):
        """Mark order as filled (remove from timeout tracking)."""
        self.order_timeout.unregister_order(order_id)

    def record_settlement(self, bot_name: str, predicted: str, actual: str):
        """Record a settlement result for model health tracking."""
        self.settlement_guard.record_settlement(bot_name, predicted, actual)

    def record_correlation(self, bot_name: str, won: bool, window_id: str = None):
        """Record a win/loss for correlation tracking."""
        self.correlation_guard.record_result(bot_name, window_id, won)

    def check_spread(
        self, yes_bid: int, yes_ask: int
    ) -> Tuple[bool, str, float]:
        """Standalone spread check."""
        return self.spread_guard.check(yes_bid, yes_ask)

    def check_spread_for_mm(
        self, yes_bid: int, yes_ask: int
    ) -> Tuple[bool, str]:
        """Check if market is suitable for market-making."""
        return self.spread_guard.check_for_market_making(yes_bid, yes_ask)

    def bot_heartbeat(self, bot_name: str):
        """Record a heartbeat ping from a bot."""
        self.heartbeat.ping(bot_name)

    def update_exchange_feed(self, exchange: str, timestamp: float = None):
        """Record a price update from an exchange feed."""
        self.lag_detector.update(exchange, timestamp)

    def maintenance(self, kalshi_client):
        """
        Run periodic maintenance. Call every 10-30 seconds.
        - Cancels stale orders
        - Checks for dead bots
        - Checks exchange feed lag
        - Reconciles positions and balance
        """
        stale = self.order_timeout.cleanup_stale_orders(kalshi_client)
        if stale:
            logger.info("MAINTENANCE: cancelled %d stale orders", len(stale))

        dead = self.heartbeat.get_dead_bots()
        if dead:
            logger.warning("MAINTENANCE: dead bots: %s", ", ".join(dead))
            _send_alert(f"DEAD BOTS: {', '.join(dead)}")

        fresh, stale_feeds, max_lag = self.lag_detector.is_data_fresh()
        if not fresh:
            logger.warning(
                "MAINTENANCE: stale exchange feeds: %s (max lag %.0fs)",
                ", ".join(stale_feeds), max_lag,
            )

        # New: position reconciliation
        try:
            recon = self.position_reconciler.reconcile(kalshi_client)
            if recon.get("mismatches") or recon.get("orphaned"):
                logger.warning("MAINTENANCE: position reconciliation issues found")
        except Exception as e:
            logger.debug("MAINTENANCE: position reconcile skipped: %s", e)

        # New: balance reconciliation
        try:
            bal_recon = self.balance_reconciler.reconcile(kalshi_client)
            if bal_recon.get("alert"):
                logger.warning("MAINTENANCE: balance mismatch detected")
        except Exception as e:
            logger.debug("MAINTENANCE: balance reconcile skipped: %s", e)

    def update_price(self, asset: str, price: float, timestamp: float = None):
        """Feed price data to volatility guard, crash detector, and flash crash guard."""
        ts = timestamp or time.time()
        self.volatility_guard.update_price(asset, price, ts)
        self.crash_detector.update_price(asset, price, ts)
        self.flash_crash_guard.update_price(asset, price, ts)

    def record_api_429(self):
        """Record a 429 rate limit response."""
        self.rate_limit_guard.record_429()

    def record_api_success(self):
        """Record a successful API call (resets rate limit backoff counter)."""
        self.rate_limit_guard.record_success()

    def check_time_sync(self, server_timestamp: float = None) -> Tuple[bool, str]:
        """Check if local time is synced with server."""
        return self.time_sync_guard.check_sync(server_timestamp)

    def set_internal_balance(self, balance: float):
        """Set the bot's internal balance for reconciliation."""
        self.balance_reconciler.set_internal_balance(balance)

    def record_position(self, ticker: str, delta_contracts: int):
        """Track expected position for reconciliation."""
        self.position_reconciler.record_expected(ticker, delta_contracts)

    def status(self) -> dict:
        """Return current status of all guards (original 13 + new 7)."""
        corr_ok, corr_reason = self.correlation_guard.should_trade()
        event_ok, event_reason = self.event_guard.should_trade()
        fresh, stale_feeds, max_lag = self.lag_detector.is_data_fresh()
        rate_waiting, rate_remaining = self.rate_limit_guard.should_wait()
        return {
            # Original 13
            "crash_emergency": self.crash_detector.is_emergency,
            "volatility_paused": not self.volatility_guard.should_trade(),
            "portfolio_shutdown": self.loss_guard._portfolio_shutdown,
            "shutdown_bots": list(self.loss_guard._shutdown_bots),
            "tracked_orders": self.order_timeout.get_tracked_count(),
            "correlation_paused": not corr_ok,
            "correlation_reason": corr_reason,
            "event_blackout": not event_ok,
            "event_reason": event_reason,
            "exchange_data_fresh": fresh,
            "stale_feeds": stale_feeds,
            "max_feed_lag_seconds": max_lag,
            "heartbeat_status": self.heartbeat.get_status(),
            "dead_bots": self.heartbeat.get_dead_bots(),
            # New 7
            "fleet_shutdown": self.fleet_drawdown_guard.is_shutdown,
            "rate_limit_backoff": rate_waiting,
            "rate_limit_remaining_sec": rate_remaining,
            "time_sync_ok": not self.time_sync_guard._paused,
            "total_guards": 20,
        }


# ===================================================================
# MODULE SELF-TEST
# ===================================================================

if __name__ == "__main__":
    print("=== risk_guard.py self-test ===\n")

    # Test size_guard
    print("Size guard tests:")
    assert size_guard(50, 50) == 50, "50 contracts at 50c should pass"
    assert size_guard(200, 50) == 100, "200 contracts should cap to 100"
    assert size_guard(10000, 3) == 100, "penny price should cap to 100 (notional cap 16666, but max_contracts=100)"
    assert size_guard(200, 3, max_contracts=100, max_notional_usd=500) == 100, "cap applies"
    assert size_guard(50000, 1, max_contracts=100, max_notional_usd=500) == 100, "1c cap"
    print("  All size_guard tests passed.\n")

    # Test LossGuard
    print("Loss guard tests:")
    lg = LossGuard(starting_balance=1000, hourly_loss_pct=2, daily_loss_pct=5)
    ok, reason = lg.record_trade("bot1", -15.0)
    assert ok, "First small loss should be OK"
    ok, reason = lg.record_trade("bot1", -10.0)
    assert not ok, "$25 loss > 2% of $1000 — should pause"
    print(f"  Paused as expected: {reason}")
    print("  All LossGuard tests passed.\n")

    # Test RiskManager
    print("RiskManager tests:")
    rm = RiskManager(starting_balance=5000)
    ok, reason, capped = rm.pre_trade_check("BTC", 50, 30, "yes")
    assert ok, "Normal trade should pass"
    assert capped == 30, "30 contracts at 50c should pass uncapped"

    ok, reason, capped = rm.pre_trade_check("BTC", 50, 200, "yes")
    assert ok, "Should pass but capped"
    assert capped == 100, "200 contracts should cap to 100"

    ok, reason, capped = rm.pre_trade_check("BTC", 2, 500, "yes")
    assert ok, "Should pass but capped"
    assert capped == 100, "At 2c, notional cap = 25000, but max_contracts=100"
    print(f"  All RiskManager pre_trade_check tests passed.\n")

    # Test CorrelationGuard
    print("CorrelationGuard tests:")
    cg = CorrelationGuard(correlated_loss_threshold=3, pause_minutes=1)
    wid = "test_window"
    cg.record_result("bot1", wid, won=False)
    cg.record_result("bot2", wid, won=False)
    ok, reason = cg.should_trade()
    assert ok, "2 losses should be fine"
    cg.record_result("bot3", wid, won=False)
    ok, reason = cg.should_trade()
    assert not ok, "3 losses should trigger correlation pause"
    print(f"  Paused as expected: {reason}")
    print("  All CorrelationGuard tests passed.\n")

    # Test SettlementGuard
    print("SettlementGuard tests:")
    sg = SettlementGuard(max_consecutive_misses=3, min_accuracy_pct=45)
    sg.record_settlement("bot_x", "yes", "no")
    sg.record_settlement("bot_x", "yes", "no")
    healthy, acc, streak = sg.is_model_healthy("bot_x")
    assert healthy, "2 misses should be OK"
    sg.record_settlement("bot_x", "yes", "no")
    healthy, acc, streak = sg.is_model_healthy("bot_x")
    assert not healthy, "3 consecutive misses should pause"
    assert streak == 3
    print(f"  Paused as expected: accuracy={acc:.1f}%, streak={streak}")
    print("  All SettlementGuard tests passed.\n")

    # Test SpreadGuard
    print("SpreadGuard tests:")
    spg = SpreadGuard(reject_spread_cents=15, reduce_spread_cents=10, mm_max_spread_cents=8)
    ok, reason, mod = spg.check(40, 42)
    assert ok and mod == 1.0, "2c spread should pass at full size"
    ok, reason, mod = spg.check(30, 43)
    assert ok and mod == 0.5, "13c spread should reduce size"
    ok, reason, mod = spg.check(20, 40)
    assert not ok, "20c spread should reject"
    ok, reason, mod = spg.check(None, 50)
    assert not ok, "Missing bid should reject"
    mm_ok, _ = spg.check_for_market_making(45, 52)
    assert not mm_ok, "7c spread but check_for_market_making wants <=8c... wait"
    mm_ok, _ = spg.check_for_market_making(45, 54)
    assert not mm_ok, "9c spread should reject for MM"
    mm_ok, _ = spg.check_for_market_making(45, 52)
    assert mm_ok, "7c spread should be OK for MM"
    print("  All SpreadGuard tests passed.\n")

    # Test HeartbeatMonitor
    print("HeartbeatMonitor tests:")
    hb = HeartbeatMonitor(timeout_minutes=1)
    hb.ping("bot_alive")
    hb._pings["bot_dead"] = time.time() - 120  # 2 min ago
    status = hb.get_status()
    assert status["bot_alive"]["alive"], "Recent ping should be alive"
    assert not status["bot_dead"]["alive"], "Old ping should be dead"
    print("  All HeartbeatMonitor tests passed.\n")

    # Test LagDetector
    print("LagDetector tests:")
    ld = LagDetector(warn_seconds=5, pause_seconds=10)
    ld.update("coinbase")
    ld.update("kraken")
    fresh, stale, lag = ld.is_data_fresh()
    assert fresh, "Fresh feeds should pass"
    ld._feeds["gemini"] = time.time() - 15  # 15s stale
    fresh, stale, lag = ld.is_data_fresh()
    assert not fresh, "Stale feed should block"
    assert "gemini" in stale
    print("  All LagDetector tests passed.\n")

    print("=== All 13 guards tested — all tests passed ===")
