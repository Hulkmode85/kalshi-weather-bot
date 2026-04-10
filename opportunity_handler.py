"""
Shared Opportunity Handler — Central router for all bots.

Every bot calls OpportunityHandler.process() with an opportunity dict.
The handler extracts features, tags regime, classifies the time horizon,
and dispatches to the correct evaluation tier. This prevents duplication
across 64+ bots.

Evaluation Tiers (STRICT GATING):
    quant_only   — Short (<=15 min) or arb/theta/market_maker bots. NO agents. Speed wins.
    quick_evaluate — Medium (15 min - 24 hr). 2-3 agent calls if enabled.
    full_swarm   — Long (>24 hr). Full multi-agent evaluation if enabled.

Usage:
    from opportunity_handler import OpportunityHandler

    handler = OpportunityHandler(bot_id="kalshi_btc_01", bot_type="directional")
    result = handler.process({
        "market_id": "BTC-50K-2026",
        "minutes_to_expiry": 10,
        "price": 52,
        "asset": "BTC",
    })
    # result = {"tier": "quant_only", "horizon": "short", "features": {...}, ...}
"""

import json
import os
import time
from pathlib import Path

LOG_DIR = os.environ.get("SHADOW_LOG_DIR", "/tmp/quant_shadow_logs")
LOG_FILE = os.path.join(LOG_DIR, "opportunity_handler.jsonl")


def shadow_log(entry: dict):
    """Append a JSON line to the shadow log file."""
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    entry["_module"] = "opportunity_handler"
    entry["_ts"] = time.time()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


class OpportunityHandler:
    def __init__(self, bot_id: str, bot_type: str = "directional"):
        """
        Args:
            bot_id: Unique identifier for the calling bot.
            bot_type: One of "arb", "theta", "directional", "flow", "market_maker".
        """
        self.bot_id = bot_id
        self.bot_type = bot_type

        # Import available modules (graceful degradation if not present)
        try:
            from feature_engine import FeatureEngine
            self.features = FeatureEngine()
        except ImportError:
            self.features = None

        try:
            from regime_tagger import RegimeTagger
            self.regime = RegimeTagger()
        except ImportError:
            self.regime = None

    def classify_horizon(self, minutes_to_expiry: float) -> str:
        """Classify opportunity time horizon."""
        if minutes_to_expiry <= 15:
            return "short"
        elif minutes_to_expiry <= 1440:  # 24 hours
            return "medium"
        else:
            return "long"

    def get_evaluation_tier(self, minutes_to_expiry: float) -> str:
        """STRICT GATING: determines which evaluation layers to activate.

        Short (<=15 min): QUANT ONLY. No agents. Speed wins.
        - Arb, theta, scalper bots ALWAYS use this regardless of horizon.
        Medium (15 min - 24 hr): QUICK EVALUATE (2-3 agent calls if enabled).
        Long (>24 hr): FULL SWARM (if enabled).
        """
        horizon = self.classify_horizon(minutes_to_expiry)

        # Arb, theta, scalper, market_maker ALWAYS quant-only
        if self.bot_type in ("arb", "theta", "market_maker"):
            return "quant_only"

        # Short horizon: always quant-only
        if horizon == "short":
            return "quant_only"
        elif horizon == "medium":
            return "quick_evaluate"
        else:
            return "full_swarm"

    def process(self, opportunity: dict) -> dict:
        """Main entry point. Routes opportunity through the right evaluation pipeline.

        Args:
            opportunity: Dict with at minimum 'minutes_to_expiry'. Optional:
                'asset', 'market_id', 'price', plus any bot-specific fields.

        Returns:
            Dict with: tier, horizon, features, regime_tags, bot_type, decision_time_ms
        """
        start = time.time()

        minutes_to_expiry = opportunity.get("minutes_to_expiry", 15)
        tier = self.get_evaluation_tier(minutes_to_expiry)
        horizon = self.classify_horizon(minutes_to_expiry)

        # Extract features if available
        features = {}
        if self.features:
            try:
                features = self.features.extract(opportunity)
            except Exception:
                features = {"error": "feature_extraction_failed"}

        # Get regime tags if available
        regime_tags = {}
        if self.regime:
            try:
                asset = opportunity.get("asset", opportunity.get("market_id", "unknown"))
                price = opportunity.get("price", 50)
                self.regime.update(asset, price)
                regime_tags = self.regime.get_regime_tags(asset)
            except Exception:
                regime_tags = {"error": "regime_tagging_failed"}

        decision_time_ms = (time.time() - start) * 1000

        result = {
            "tier": tier,
            "horizon": horizon,
            "features": features,
            "regime_tags": regime_tags,
            "bot_type": self.bot_type,
            "decision_time_ms": round(decision_time_ms, 2),
        }

        # Shadow log the tier decision for analysis
        shadow_log({
            "action": "process_opportunity",
            "bot_id": self.bot_id,
            "bot_type": self.bot_type,
            "market_id": opportunity.get("market_id", opportunity.get("asset", "unknown")),
            "minutes_to_expiry": minutes_to_expiry,
            "tier": tier,
            "horizon": horizon,
            "regime_tags": regime_tags,
            "decision_time_ms": round(decision_time_ms, 2),
        })

        return result
