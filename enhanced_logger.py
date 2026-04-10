"""
enhanced_logger.py — Enriched trade logging with regime tags, features, and tournament integration.

Logs every opportunity evaluation to JSONL with full context: regime tags, feature vectors,
virtual portfolio results, and actual trade outcomes. On resolution, feeds results into
the StrategyTournament for significance-tested ranking.

Usage:
    logger = EnhancedLogger(log_file="enhanced_trades.jsonl")
    logger.log_opportunity(opportunity, virtual_results, regime_tags, features)
    logger.log_resolution(market_id, outcome=True, settlement_price=0.95)
    report = logger.generate_analysis()
"""

import json
import os
import time
from typing import Optional

from regime_tagger import RegimeTagger
from statistical_tournament import StrategyTournament


class EnhancedLogger:
    """Full-context opportunity and trade logger with built-in analysis."""

    def __init__(self, log_file: str = "enhanced_trades.jsonl",
                 tournament_min_trades: int = 150, log_dir: str = "."):
        self.log_file = os.path.join(log_dir, log_file)
        self.log_dir = log_dir
        self.regime_tagger = RegimeTagger(log_dir=log_dir)
        self.tournament = StrategyTournament(min_trades=tournament_min_trades, log_dir=log_dir)
        self._pending: dict[str, dict] = {}  # market_id -> entry (awaiting resolution)

        # Reload pending entries from existing log
        self._reload_pending()

    # ------------------------------------------------------- Logging
    def log_opportunity(self, opportunity: dict, virtual_results: list,
                        regime_tags: dict, features: dict):
        """Log a fully enriched opportunity evaluation.

        Args:
            opportunity: dict with at minimum {market_id, strategy, ...}
            virtual_results: list of dicts from virtual portfolio evaluation
            regime_tags: dict from RegimeTagger.get_regime_tags()
            features: dict from FeatureEngine or similar
        """
        market_id = opportunity.get("market_id", f"unknown_{time.time()}")

        entry = {
            "ts": time.time(),
            "type": "opportunity",
            "market_id": market_id,
            "opportunity": opportunity,
            "regime_tags": regime_tags,
            "features": features,
            "virtual_portfolios": virtual_results,
            "actual_trade": None,
            "resolution": None,
        }

        self._write_entry(entry)
        self._pending[market_id] = entry

    def log_trade_execution(self, market_id: str, trade_details: dict):
        """Called when an actual trade is placed. Links to the opportunity.

        Args:
            market_id: the market this trade belongs to
            trade_details: {strategy, side, size, price, ...}
        """
        entry = {
            "ts": time.time(),
            "type": "trade_execution",
            "market_id": market_id,
            "trade_details": trade_details,
        }
        self._write_entry(entry)

        # Update pending entry
        if market_id in self._pending:
            self._pending[market_id]["actual_trade"] = trade_details

    def log_resolution(self, market_id: str, outcome: bool,
                       settlement_price: float):
        """Called when a market resolves. Updates tournament with actual results.

        Args:
            market_id: resolved market identifier
            outcome: True if the bet won
            settlement_price: final settlement price
        """
        pending = self._pending.pop(market_id, None)

        resolution = {
            "ts": time.time(),
            "type": "resolution",
            "market_id": market_id,
            "outcome": outcome,
            "settlement_price": settlement_price,
        }

        # Calculate PnL if we have trade details
        pnl = 0.0
        duration = 0.0
        strategy = "unknown"
        regime_tags = {}

        if pending:
            trade = pending.get("actual_trade") or {}
            regime_tags = pending.get("regime_tags", {})
            strategy = (pending.get("opportunity", {}).get("strategy")
                        or trade.get("strategy", "unknown"))

            entry_price = trade.get("price", 0)
            size = trade.get("size", 1)
            side = trade.get("side", "yes")

            if entry_price > 0:
                if side == "yes":
                    pnl = (settlement_price - entry_price) * size
                else:
                    pnl = (entry_price - settlement_price) * size

            duration = time.time() - pending.get("ts", time.time())

            # Also feed virtual portfolio results into tournament
            for vr in pending.get("virtual_portfolios", []):
                vr_strategy = vr.get("strategy", strategy)
                vr_pnl = vr.get("pnl", 0)
                vr_dur = vr.get("duration", duration)
                self.tournament.add_trade(
                    f"virtual_{vr_strategy}", vr_pnl, vr_dur,
                    regime_tags=regime_tags
                )

        resolution["pnl"] = round(pnl, 4)
        resolution["duration_seconds"] = round(duration, 1)
        resolution["strategy"] = strategy
        self._write_entry(resolution)

        # Feed into tournament
        if pnl != 0 or pending:
            self.tournament.add_trade(strategy, pnl, duration,
                                      regime_tags=regime_tags)

    # ------------------------------------------------------- Analysis
    def generate_analysis(self) -> str:
        """Read all logged data, run tournament, output regime-by-regime breakdown."""
        lines = [
            "# Enhanced Trade Analysis",
            f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n",
        ]

        # Tournament report
        tournament_report = self.tournament.generate_report()
        lines.append(tournament_report)
        lines.append("")

        # Regime breakdown for each qualifying strategy
        strategies_with_trades = [
            s for s in self.tournament.results
            if len(self.tournament.results[s]) >= self.tournament.min_trades
        ]

        if strategies_with_trades:
            lines.append("\n## Regime Breakdown\n")
            regime_keys = ["vol_regime", "trend_regime", "liquidity_regime",
                           "time_regime", "day_regime"]

            for strategy in strategies_with_trades[:5]:  # top 5
                lines.append(f"### {strategy}\n")
                breakdown = self.tournament.get_regime_breakdown(strategy, regime_keys)

                for regime_key, values in breakdown.items():
                    lines.append(f"**{regime_key}:**")
                    for val, metrics in sorted(values.items()):
                        lines.append(
                            f"  - {val}: {metrics['num_trades']} trades, "
                            f"PnL=${metrics['total_pnl']:.2f}, "
                            f"win={metrics['win_rate']*100:.0f}%"
                        )
                    lines.append("")

        # Pending resolutions
        if self._pending:
            lines.append(f"\n## Pending Resolutions: {len(self._pending)}\n")
            for mid in list(self._pending.keys())[:10]:
                entry = self._pending[mid]
                age_hrs = (time.time() - entry["ts"]) / 3600
                lines.append(f"  - {mid} (age: {age_hrs:.1f}h)")

        report = "\n".join(lines)

        # Shadow log the analysis
        self._write_entry({
            "ts": time.time(),
            "type": "analysis_generated",
            "num_strategies": len(strategies_with_trades),
            "pending_count": len(self._pending),
        })

        return report

    # ------------------------------------------------------- Helpers
    def get_stats(self) -> dict:
        """Quick summary stats."""
        total_entries = 0
        try:
            with open(self.log_file, "r") as f:
                total_entries = sum(1 for _ in f)
        except FileNotFoundError:
            pass

        return {
            "log_file": self.log_file,
            "total_log_entries": total_entries,
            "pending_resolutions": len(self._pending),
            "tournament_strategies": len(self.tournament.results),
        }

    def _write_entry(self, entry: dict):
        """Append a JSON entry to the log file."""
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception:
            pass

    def _reload_pending(self):
        """Reload pending (unresolved) entries from existing log file."""
        try:
            if not os.path.exists(self.log_file):
                return

            resolved_ids = set()
            opportunities = {}

            with open(self.log_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    etype = entry.get("type", "")
                    mid = entry.get("market_id", "")

                    if etype == "opportunity" and mid:
                        opportunities[mid] = entry
                    elif etype == "resolution" and mid:
                        resolved_ids.add(mid)

            # Pending = opportunities without resolutions
            for mid, entry in opportunities.items():
                if mid not in resolved_ids:
                    self._pending[mid] = entry

        except Exception:
            pass
