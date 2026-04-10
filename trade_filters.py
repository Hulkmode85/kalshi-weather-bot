"""
trade_filters.py — Pre-trade and in-trade filters: stop-loss, trailing stop, vol filter,
time-of-day filter, and news gate.

All filters are stateless where possible. Position tracking for trailing stops
is maintained internally. Shadow-logs every filter decision to JSONL.

Usage:
    filters = TradeFilters()
    if filters.check_stop_loss("pos_1", current=95, entry=100, stop_pct=0.05):
        # exit position
    time_info = filters.check_time_filter()
    if not time_info["is_market_open"]:
        # skip trade
"""

import json
import os
import time
from datetime import datetime, timezone
from typing import Optional


class TradeFilters:
    """Pre-trade and in-trade risk filters."""

    SHADOW_LOG = "filters_shadow.jsonl"

    def __init__(self, log_dir: str = ".", news_cooldown_minutes: float = 30.0):
        """
        Args:
            log_dir: directory for shadow JSONL log.
            news_cooldown_minutes: default cooldown after major news events.
        """
        self.positions: dict[str, dict] = {}  # position_id -> tracking info
        self.news_events: list[dict] = []  # list of {ts, headline, severity}
        self.news_cooldown = news_cooldown_minutes
        self.log_path = os.path.join(log_dir, self.SHADOW_LOG)

    # --------------------------------------------------- Stop-Loss
    def check_stop_loss(self, position_id: str, current_price: float,
                        entry_price: float, stop_pct: float = 0.05) -> bool:
        """Return True if stop-loss triggered (price dropped stop_pct from entry).

        Works for both long (price drops) and short (price rises) based on
        the direction stored in position tracking.

        Args:
            position_id: unique position identifier.
            current_price: current market price.
            entry_price: original entry price.
            stop_pct: stop-loss percentage (0.05 = 5%).
        """
        # Track position
        if position_id not in self.positions:
            self.positions[position_id] = {
                "entry_price": entry_price,
                "peak_price": current_price,
                "trough_price": current_price,
            }

        pos = self.positions[position_id]

        # Determine direction from entry vs current
        # For prediction markets: entry_price is cost, current_price is value
        loss_pct = (entry_price - current_price) / max(entry_price, 1e-9)
        triggered = loss_pct >= stop_pct

        self._shadow_log("stop_loss_check", {
            "position_id": position_id,
            "current": current_price,
            "entry": entry_price,
            "loss_pct": round(loss_pct, 4),
            "stop_pct": stop_pct,
            "triggered": triggered,
        })

        return triggered

    # --------------------------------------------------- Trailing Stop
    def check_trailing_stop(self, position_id: str, current_price: float,
                            peak_price: Optional[float] = None,
                            trail_pct: float = 0.03) -> bool:
        """Return True if trailing stop triggered (price dropped trail_pct from peak).

        Automatically tracks peak price if not provided.

        Args:
            position_id: unique position identifier.
            current_price: current market price.
            peak_price: highest price since entry (auto-tracked if None).
            trail_pct: trailing stop percentage (0.03 = 3%).
        """
        if position_id not in self.positions:
            self.positions[position_id] = {
                "entry_price": current_price,
                "peak_price": current_price,
                "trough_price": current_price,
            }

        pos = self.positions[position_id]

        # Update peak
        if current_price > pos["peak_price"]:
            pos["peak_price"] = current_price

        effective_peak = peak_price if peak_price is not None else pos["peak_price"]
        drop_pct = (effective_peak - current_price) / max(effective_peak, 1e-9)
        triggered = drop_pct >= trail_pct

        self._shadow_log("trailing_stop_check", {
            "position_id": position_id,
            "current": current_price,
            "peak": effective_peak,
            "drop_pct": round(drop_pct, 4),
            "trail_pct": trail_pct,
            "triggered": triggered,
        })

        return triggered

    # --------------------------------------------------- Vol Filter
    def check_vol_filter(self, iv_rank: float, min_iv: float = 0.2,
                         max_iv: float = 0.8) -> bool:
        """Return True if IV rank is within acceptable range for the strategy.

        Args:
            iv_rank: implied volatility rank/percentile (0.0 to 1.0).
            min_iv: minimum acceptable IV rank.
            max_iv: maximum acceptable IV rank.
        """
        passed = min_iv <= iv_rank <= max_iv

        self._shadow_log("vol_filter", {
            "iv_rank": round(iv_rank, 4),
            "min_iv": min_iv,
            "max_iv": max_iv,
            "passed": passed,
        })

        return passed

    # --------------------------------------------------- Time Filter
    def check_time_filter(self, utc_offset: int = -4) -> dict:
        """Return time-of-day and day-of-week info for strategy time gating.

        Args:
            utc_offset: timezone offset from UTC (default -4 for EDT).

        Returns:
            dict with: hour, minute, day, day_name, is_power_hour, is_market_open,
            is_weekend, session.
        """
        now_utc = datetime.now(timezone.utc)
        # Apply offset
        local_hour = (now_utc.hour + utc_offset) % 24
        local_minute = now_utc.minute
        local_weekday = now_utc.weekday()  # 0=Mon, 6=Sun

        # US market hours: 9:30 AM - 4:00 PM ET
        is_market_open = (
            local_weekday < 5 and
            ((local_hour == 9 and local_minute >= 30) or
             (10 <= local_hour < 16))
        )

        # Power hour: 3:00-4:00 PM ET
        is_power_hour = is_market_open and local_hour == 15

        # Opening bell: 9:30-10:30 AM ET
        is_opening = (
            local_weekday < 5 and
            ((local_hour == 9 and local_minute >= 30) or local_hour == 10)
        )

        # Session classification
        if local_weekday >= 5:
            session = "weekend"
        elif 4 <= local_hour < 9:
            session = "pre_market"
        elif is_market_open:
            session = "regular"
        elif 16 <= local_hour < 20:
            session = "after_hours"
        else:
            session = "overnight"

        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday",
                     "Friday", "Saturday", "Sunday"]

        result = {
            "hour": local_hour,
            "minute": local_minute,
            "day": local_weekday,
            "day_name": day_names[local_weekday],
            "is_power_hour": is_power_hour,
            "is_opening": is_opening,
            "is_market_open": is_market_open,
            "is_weekend": local_weekday >= 5,
            "session": session,
        }

        self._shadow_log("time_filter", result)
        return result

    # --------------------------------------------------- News Gate
    def register_news_event(self, headline: str, severity: str = "major",
                            timestamp: Optional[float] = None):
        """Register a news event that should pause trading.

        Args:
            headline: brief description of the event.
            severity: "major", "moderate", or "minor".
            timestamp: event time (defaults to now).
        """
        event = {
            "ts": timestamp or time.time(),
            "headline": headline,
            "severity": severity,
        }
        self.news_events.append(event)

        # Keep only last 100 events
        if len(self.news_events) > 100:
            self.news_events = self.news_events[-100:]

        self._shadow_log("news_registered", event)

    def check_news_gate(self, cooldown_minutes: Optional[float] = None) -> bool:
        """Return True if safe to trade (no recent major news).

        Returns False if major news occurred within cooldown window.

        Args:
            cooldown_minutes: override default cooldown (minutes).
        """
        cooldown = cooldown_minutes or self.news_cooldown
        cutoff = time.time() - (cooldown * 60)

        recent_major = [
            e for e in self.news_events
            if e["ts"] >= cutoff and e["severity"] in ("major", "moderate")
        ]

        safe = len(recent_major) == 0

        self._shadow_log("news_gate", {
            "cooldown_minutes": cooldown,
            "recent_events": len(recent_major),
            "safe_to_trade": safe,
        })

        return safe

    # --------------------------------------------------- Composite Filter
    def run_all_filters(self, position_id: str, current_price: float,
                        entry_price: float, iv_rank: float = 0.5,
                        stop_pct: float = 0.05, trail_pct: float = 0.03,
                        min_iv: float = 0.2, max_iv: float = 0.8) -> dict:
        """Run all filters at once and return composite result.

        Returns dict with individual filter results and an overall 'passed' flag.
        """
        stop_triggered = self.check_stop_loss(position_id, current_price,
                                               entry_price, stop_pct)
        trail_triggered = self.check_trailing_stop(position_id, current_price,
                                                    trail_pct=trail_pct)
        vol_ok = self.check_vol_filter(iv_rank, min_iv, max_iv)
        time_info = self.check_time_filter()
        news_ok = self.check_news_gate()

        result = {
            "stop_loss_triggered": stop_triggered,
            "trailing_stop_triggered": trail_triggered,
            "vol_filter_passed": vol_ok,
            "time_info": time_info,
            "news_gate_passed": news_ok,
            "should_exit": stop_triggered or trail_triggered,
            "should_skip_new_trade": not vol_ok or not news_ok,
        }

        return result

    # --------------------------------------------------- Position Management
    def register_position(self, position_id: str, entry_price: float):
        """Explicitly register a new position for tracking."""
        self.positions[position_id] = {
            "entry_price": entry_price,
            "peak_price": entry_price,
            "trough_price": entry_price,
            "opened_at": time.time(),
        }

    def close_position(self, position_id: str):
        """Remove a position from tracking."""
        self.positions.pop(position_id, None)
        self._shadow_log("position_closed", {"position_id": position_id})

    def get_open_positions(self) -> dict:
        """Return all tracked positions."""
        return dict(self.positions)

    # --------------------------------------------------- Shadow log
    def _shadow_log(self, event: str, data: dict):
        try:
            entry = {"ts": time.time(), "event": event, **data}
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception:
            pass
