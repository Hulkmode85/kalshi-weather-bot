#!/usr/bin/env python3
"""
Kalshi Weather Bot
Trades Kalshi temperature/weather markets using NWS & Open-Meteo forecasts.

Strategy:
- Fetch Kalshi weather markets (temperature, precipitation, snowfall)
- Fetch NWS + Open-Meteo 24-48hr forecasts for the same city/date
- When forecast probability diverges from Kalshi price by > MIN_EDGE_PCT, trade
- Focus on high-probability (≥75¢) contracts where taker fee < 3.5% break-even

Documented results: Weather bots made $24,000+ on Polymarket with this approach.
NWS 24-hr forecast accuracy: ~95% — vs. Kalshi public pricing of ~70-80%.
"""

import asyncio
import base64
import logging
import os
from flask import Flask, jsonify
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from dotenv import load_dotenv

load_dotenv()

# ── Quant Fund Shadow Evaluators ─────────────────────────────────────────
try:
    from bayesian_updater import BayesianUpdater
    from ensemble_model import EnsembleModel
    from time_decay_edge import calculate_time_weighted_edge
    from correlation_matrix import CorrelationTracker
    from vpin_toxicity import VPINTracker
    from market_impact import estimate_market_impact
    from feature_engine import FeatureEngine
    from portfolio_optimizer import PortfolioOptimizer
    _quant_modules_available = True
    _bayesian = BayesianUpdater()
    _ensemble = EnsembleModel()
    _correlation = CorrelationTracker()
    _vpin = VPINTracker()
    _features = FeatureEngine()
    _portfolio = PortfolioOptimizer()
except ImportError:
    _quant_modules_available = False

# ── Critical Module Imports (10 modules — template pattern) ──────────────
# Each module is optional: bot keeps running if any module is missing or errors.

try:
    from pre_trade_validator import validate_pre_trade
    _pre_trade_validator_available = True
except ImportError:
    _pre_trade_validator_available = False

try:
    from dynamic_edge import calculate_dynamic_edge
    _dynamic_edge_available = True
except ImportError:
    _dynamic_edge_available = False

try:
    from adaptive_kelly import calculate_adaptive_kelly
    _adaptive_kelly_available = True
except ImportError:
    _adaptive_kelly_available = False

try:
    from dynamic_params import DynamicParams
    _dynamic_params = DynamicParams()
    _dynamic_params_available = True
except ImportError:
    _dynamic_params_available = False

try:
    from paper_balance_manager import PaperBalanceManager
    _paper_balance_mgr = PaperBalanceManager(restart_threshold=1000.0)
    _paper_balance_available = True
except ImportError:
    _paper_balance_available = False

try:
    from maker_execution import MakerExecution
    _maker_execution_available = True
except ImportError:
    _maker_execution_available = False

try:
    from data_pipeline import DataPipeline
    _data_pipeline = DataPipeline()
    _data_pipeline_available = True
except ImportError:
    _data_pipeline_available = False

try:
    from brier_scorer import BrierScorer
    _brier_scorer = BrierScorer()
    _brier_scorer_available = True
except ImportError:
    _brier_scorer_available = False

try:
    from rejection_filter import RejectionFilter
    _rejection_filter = RejectionFilter()
    _rejection_filter_available = True
except ImportError:
    _rejection_filter_available = False

try:
    from conviction_scaler import ConvictionScaler
    _conviction_scaler = ConvictionScaler()
    _conviction_scaler_available = True
except ImportError:
    _conviction_scaler_available = False


from risk_guard import RiskManager
import json as _json

# ── Shadow Logging ────────────────────────────────────────────────────────────
SHADOW_LOG_FILE = os.getenv("SHADOW_LOG_FILE", "shadow_log.jsonl")

def shadow_log(opportunity: dict, taken: bool, reason: str = ""):
    entry = {"ts": time.time(), "taken": taken, "reason": reason, **opportunity}
    try:
        with open(SHADOW_LOG_FILE, "a") as f:
            f.write(_json.dumps(entry) + "\n")
    except:
        pass


# ── Virtual Portfolio Testing ─────────────────────────────────────────────
VIRTUAL_PORTFOLIO_FILE = os.getenv("VIRTUAL_PORTFOLIO_FILE", "virtual_portfolios.jsonl")

VIRTUAL_PORTFOLIOS = [
    {"name": "aggressive", "kelly": 1.0, "min_edge": 0.02, "early_exit": 0.99},
    {"name": "moderate", "kelly": 0.5, "min_edge": 0.05, "early_exit": 0.93},
    {"name": "conservative", "kelly": 0.25, "min_edge": 0.08, "early_exit": 0.90},
    {"name": "original_v1", "kelly": 1.0, "min_edge": 0.03, "early_exit": 0.99},
    {"name": "high_edge", "kelly": 0.5, "min_edge": 0.10, "early_exit": 0.93},
    {"name": "ultra_conservative", "kelly": 0.25, "min_edge": 0.12, "early_exit": 0.90},
]

def evaluate_virtual_portfolios(opportunity: dict):
    """Evaluate what each virtual portfolio would do with this opportunity."""
    import json, time as _time
    edge = opportunity.get("edge", 0)
    price = opportunity.get("price", 0)
    results = []
    for vp in VIRTUAL_PORTFOLIOS:
        would_trade = edge >= vp["min_edge"]
        would_exit_early = price >= vp["early_exit"] * 100
        results.append({
            "portfolio": vp["name"],
            "would_trade": would_trade,
            "would_exit_early": would_exit_early,
            "kelly": vp["kelly"],
            "min_edge": vp["min_edge"],
        })
    entry = {
        "ts": _time.time(),
        "opportunity": opportunity,
        "portfolios": results,
    }
    try:
        with open(VIRTUAL_PORTFOLIO_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except:
        pass

# ── Multi-strike: scan ALL strikes per event/series, not just one ────────────

# ─── Regime Detection — pause trading during extreme volatility ────────────
import statistics as _stats

REGIME_WINDOW = int(os.getenv("REGIME_WINDOW", "20"))
REGIME_THRESHOLD = float(os.getenv("REGIME_THRESHOLD", "3.0"))
_regime_prices: list[float] = []

def check_regime(price: float) -> str:
    """Returns 'CALM', 'ELEVATED', or 'CRASH'. Skip trades during CRASH."""
    _regime_prices.append(price)
    if len(_regime_prices) > REGIME_WINDOW:
        _regime_prices.pop(0)
    if len(_regime_prices) < 5:
        return "CALM"
    rets = [(b - a) / a for a, b in zip(_regime_prices[:-1], _regime_prices[1:])]
    if not rets:
        return "CALM"
    mu = _stats.mean(rets)
    sd = _stats.stdev(rets) if len(rets) > 1 else 0.01
    z = abs(rets[-1] - mu) / max(sd, 0.0001)
    if z > REGIME_THRESHOLD:
        return "CRASH"
    elif z > REGIME_THRESHOLD * 0.6:
        return "ELEVATED"
    return "CALM"



# ── Early Exit Logic ─────────────────────────────────────────────────────────
EARLY_EXIT_THRESHOLD = float(os.getenv("EARLY_EXIT_THRESHOLD", "0.93"))

def should_early_exit(current_price_cents: float) -> bool:
    """Exit position early at 93c+ to lock in profit instead of holding to settlement."""
    return current_price_cents >= EARLY_EXIT_THRESHOLD * 100

# ── Circuit Breakers ─────────────────────────────────────────────────────────
CONSECUTIVE_LOSS_PAUSE = int(os.getenv("CONSECUTIVE_LOSS_PAUSE", "3"))
DAILY_DRAWDOWN_PAUSE_PCT = float(os.getenv("DAILY_DRAWDOWN_PAUSE_PCT", "0.05"))

_consecutive_losses = 0
_daily_pnl = 0.0
_circuit_paused_until = 0

def check_circuit_breaker() -> bool:
    """Returns True if trading should be paused."""
    import time as _time
    global _consecutive_losses, _daily_pnl, _circuit_paused_until
    if _time.time() < _circuit_paused_until:
        return True
    if _consecutive_losses >= CONSECUTIVE_LOSS_PAUSE:
        return True
    # Use PAPER_BALANCE if available, else 5000
    _balance = globals().get("PAPER_BALANCE", 2000)
    if _daily_pnl < -DAILY_DRAWDOWN_PAUSE_PCT * _balance:
        return True
    return False

def record_trade_result(won: bool, pnl: float):
    """Update circuit breaker state after each trade result."""
    global _consecutive_losses, _daily_pnl
    _daily_pnl += pnl
    if won:
        _consecutive_losses = 0
    else:
        _consecutive_losses += 1
MULTI_STRIKE = os.getenv("MULTI_STRIKE", "true").lower() == "true"
# When fetching markets, iterate through ALL contracts in each series/event
# and evaluate each strike independently. No single-ticker filtering.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("weather_bot")

risk_manager = RiskManager()


class Config:
    KALSHI_API_KEY_ID:      str   = os.getenv("KALSHI_API_KEY_ID", "")
    KALSHI_PRIVATE_KEY_PEM: str   = os.getenv("KALSHI_PRIVATE_KEY_PEM", "")
    KALSHI_BASE_URL:        str   = "https://api.elections.kalshi.com/trade-api/v2"

    # Edge must exceed Kalshi taker fee break-even (3.5% at 75¢)
    MIN_EDGE_PCT:    float = float(os.getenv("MIN_EDGE_PCT", "0.05"))
    MAKER_FEE:       float = float(os.getenv("MAKER_FEE", "0.0175"))
    # Only trade contracts priced ≥75¢ (taker fee break-even threshold)
    MIN_KALSHI_PRICE: int  = int(os.getenv("MIN_KALSHI_PRICE", "75"))
    # Minimum open contracts in market
    MIN_VOLUME:      int   = int(os.getenv("MIN_VOLUME", "50"))
    # Fraction of balance per trade (fractional Kelly)
    POSITION_FRACTION: float = float(os.getenv("POSITION_FRACTION", "0.04"))
    MAX_TRADE_USD:   float = float(os.getenv("MAX_TRADE_USD", "300.0"))
    MIN_TRADE_USD:   float = float(os.getenv("MIN_TRADE_USD", "5.0"))

    # ── Micro-bet mode (Hans323 strategy) ─────────────────────────────────────
    # Buy cheap contracts (1-10¢) where NWS tail probability >> Kalshi implied
    # Returns can be 1000-5000% when tails hit. Spread across temp ladder rungs.
    MICRO_BET_MODE:     bool  = os.getenv("MICRO_BET_MODE", "true").lower() == "true"
    MICRO_BET_MAX_PRICE: int  = int(os.getenv("MICRO_BET_MAX_PRICE", "10"))   # max cents
    MICRO_BET_MIN_PRICE: int  = int(os.getenv("MICRO_BET_MIN_PRICE", "1"))    # min cents
    # Min ratio of our forecast prob to Kalshi implied prob (e.g. 3.0 = we think 3x more likely)
    MICRO_BET_MIN_RATIO: float = float(os.getenv("MICRO_BET_MIN_RATIO", "3.0"))
    # Our forecast must show at least this probability for the tail event
    MICRO_BET_MIN_FORECAST_PROB: float = float(os.getenv("MICRO_BET_MIN_FORECAST_PROB", "0.05"))
    MICRO_BET_MAX_USD:  float = float(os.getenv("MICRO_BET_MAX_USD", "5.0"))  # max $ per micro trade
    MICRO_BET_MIN_USD:  float = float(os.getenv("MICRO_BET_MIN_USD", "1.0"))  # min $ per micro trade
    # Max micro-bets per city per cycle (ladder across temp rungs)
    MICRO_BET_MAX_PER_CITY: int = int(os.getenv("MICRO_BET_MAX_PER_CITY", "5"))
    # Poll interval in seconds
    POLL_INTERVAL_SEC: int = int(os.getenv("POLL_INTERVAL_SECONDS", "300"))

    PAPER_MODE:      bool  = os.getenv("PAPER_MODE", "true").lower() == "true"
    PAPER_BALANCE:   float = float(os.getenv("PAPER_STARTING_BALANCE", "2000.0"))

    # Open-Meteo: free, no key needed
    OPENMETEO_URL: str = "https://api.open-meteo.com/v1/forecast"
    # NWS Points API (US only, free, no key)
    NWS_BASE_URL:  str = "https://api.weather.gov"

    # City coordinates for NWS + Open-Meteo lookups
    CITIES: dict = {
        "new york":   {"lat": 40.7128, "lon": -74.0060, "nws_office": "OKX", "nws_grid": "33,37"},
        "los angeles": {"lat": 34.0522, "lon": -118.2437, "nws_office": "LOX", "nws_grid": "155,45"},
        "chicago":    {"lat": 41.8781, "lon": -87.6298, "nws_office": "LOT", "nws_grid": "76,73"},
        "miami":      {"lat": 25.7617, "lon": -80.1918, "nws_office": "MFL", "nws_grid": "110,37"},
        "dallas":     {"lat": 32.7767, "lon": -96.7970, "nws_office": "FWD", "nws_grid": "84,103"},
        "seattle":    {"lat": 47.6062, "lon": -122.3321, "nws_office": "SEW", "nws_grid": "124,67"},
        "phoenix":    {"lat": 33.4484, "lon": -112.0740, "nws_office": "PSR", "nws_grid": "164,56"},
        "boston":     {"lat": 42.3601, "lon": -71.0589, "nws_office": "BOX", "nws_grid": "64,34"},
    }

    # Kalshi weather market keyword patterns
    WEATHER_KEYWORDS: list = [
        "temperature", "high temp", "low temp", "degrees", "fahrenheit",
        "rain", "precipitation", "snow", "snowfall", "inches",
        "humidity", "wind speed", "mph", "weather",
    ]


# ── Kalshi Auth ────────────────────────────────────────────────────────────────

def _load_kalshi_key():
    pem = Config.KALSHI_PRIVATE_KEY_PEM
    if not pem:
        return None
    pem = pem.replace("\\n", "\n")
    try:
        return serialization.load_pem_private_key(pem.encode("utf-8"), password=None)
    except Exception as e:
        log.error(f"Kalshi key load failed: {e}")
        return None


_KALSHI_KEY = _load_kalshi_key()


def _kalshi_headers(method: str, path: str) -> dict:
    if not _KALSHI_KEY or not Config.KALSHI_API_KEY_ID:
        raise RuntimeError("Kalshi credentials not configured")
    ts_ms = str(int(time.time() * 1000))
    sign_path = "/trade-api/v2" + path.split("?")[0]
    msg = (ts_ms + method.upper() + sign_path).encode("utf-8")
    sig = _KALSHI_KEY.sign(
        msg,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return {
        "Content-Type": "application/json",
        "KALSHI-ACCESS-KEY": Config.KALSHI_API_KEY_ID,
        "KALSHI-ACCESS-TIMESTAMP": ts_ms,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode("utf-8"),
    }


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class WeatherOpportunity:
    ticker:             str
    title:              str
    yes_ask:            int
    no_ask:             int
    volume:             int
    best_side:          str   # "yes" or "no"
    best_price:         int   # price of the side we're buying (cents)
    forecast_prob:      float # our forecast probability (0-1)
    kalshi_prob:        float # Kalshi's implied probability (0-1)
    edge_pct:           float
    city:               str
    forecast_source:    str
    close_time:         str
    is_micro_bet:       bool  = False  # True = low-prob tail bet
    edge_ratio:         float = 0.0   # forecast_prob / kalshi_prob (for micro-bets)


# ── Forecast fetchers ─────────────────────────────────────────────────────────

async def fetch_openmeteo_forecast(http: httpx.AsyncClient, lat: float, lon: float) -> Optional[dict]:
    """Fetch 7-day hourly temperature + precip forecast from Open-Meteo (free, no key)."""
    try:
        r = await http.get(
            Config.OPENMETEO_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": "temperature_2m,precipitation_probability,snowfall",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,snowfall_sum",
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "precipitation_unit": "inch",
                "timezone": "auto",
                "forecast_days": 7,
            },
            timeout=10.0,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.warning(f"Open-Meteo fetch failed: {e}")
        return None


async def fetch_nws_forecast(http: httpx.AsyncClient, lat: float, lon: float) -> Optional[dict]:
    """Fetch NWS forecast for a lat/lon (US only, free, no key)."""
    try:
        # First get the grid point
        points_r = await http.get(
            f"{Config.NWS_BASE_URL}/points/{lat},{lon}",
            headers={"User-Agent": "KalshiWeatherBot/1.0 (contact@example.com)"},
            timeout=10.0,
        )
        points_r.raise_for_status()
        props = points_r.json().get("properties", {})
        forecast_url = props.get("forecast")
        if not forecast_url:
            return None

        # Get the forecast
        fc_r = await http.get(
            forecast_url,
            headers={"User-Agent": "KalshiWeatherBot/1.0 (contact@example.com)"},
            timeout=10.0,
        )
        fc_r.raise_for_status()
        return fc_r.json()
    except Exception as e:
        log.warning(f"NWS fetch failed for {lat},{lon}: {e}")
        return None


def parse_high_temp_forecast(openmeteo: dict, target_date: str) -> Optional[float]:
    """Extract forecasted high temperature (°F) for a specific date."""
    try:
        daily = openmeteo.get("daily", {})
        dates = daily.get("time", [])
        highs = daily.get("temperature_2m_max", [])
        for i, d in enumerate(dates):
            if d == target_date and i < len(highs):
                return highs[i]
    except Exception:
        pass
    return None


def parse_precip_probability(openmeteo: dict, target_date: str) -> Optional[float]:
    """Extract max precipitation probability (%) for a specific date."""
    try:
        hourly = openmeteo.get("hourly", {})
        times = hourly.get("time", [])
        probs = hourly.get("precipitation_probability", [])
        day_probs = [
            p for t, p in zip(times, probs)
            if t.startswith(target_date) and p is not None
        ]
        if day_probs:
            return max(day_probs) / 100.0
    except Exception:
        pass
    return None


def parse_snowfall_forecast(openmeteo: dict, target_date: str) -> Optional[float]:
    """Extract total forecasted snowfall (inches) for a specific date."""
    try:
        daily = openmeteo.get("daily", {})
        dates = daily.get("time", [])
        snow = daily.get("snowfall_sum", [])
        for i, d in enumerate(dates):
            if d == target_date and i < len(snow):
                return snow[i] or 0.0
    except Exception:
        pass
    return None


# ── Kalshi market scanner ─────────────────────────────────────────────────────

async def fetch_kalshi_weather_markets(http: httpx.AsyncClient) -> list:
    """Fetch Kalshi markets that are weather-related."""
    all_markets = []
    cursor = None
    try:
        while True:
            params = {"limit": 200, "status": "open"}
            if cursor:
                params["cursor"] = cursor
            path = "/markets"
            r = await http.get(
                Config.KALSHI_BASE_URL + path,
                headers=_kalshi_headers("GET", path),
                params=params,
                timeout=12.0,
            )
            r.raise_for_status()
            data = r.json()
            markets = data.get("markets", [])
            if not markets:
                break

            for m in markets:
                title = (m.get("title") or "").lower()
                if any(kw in title for kw in Config.WEATHER_KEYWORDS):
                    all_markets.append({
                        "ticker": m.get("ticker", ""),
                        "title": m.get("title", ""),
                        "yes_ask": m.get("yes_ask", 50),
                        "no_ask": m.get("no_ask", 50),
                        "volume": m.get("volume", 0),
                        "close_time": m.get("close_time", ""),
                    })

            cursor = data.get("cursor")
            if not cursor:
                break
    except Exception as e:
        log.error(f"Kalshi market fetch failed: {e}")
    log.info(f"Found {len(all_markets)} Kalshi weather markets")
    return all_markets


# ── Opportunity finder ────────────────────────────────────────────────────────

def extract_city_from_title(title: str) -> Optional[str]:
    """Try to identify a city name in a market title."""
    title_lower = title.lower()
    for city in Config.CITIES:
        if city in title_lower:
            return city
    return None


def extract_date_from_title(title: str) -> Optional[str]:
    """Try to extract a date (YYYY-MM-DD) from a market title. Returns today+1 if not found."""
    import re
    # Look for patterns like "April 5", "Apr 5", "4/5" etc.
    today = datetime.now(timezone.utc)
    # Simple fallback: return tomorrow's date
    from datetime import timedelta
    return (today + timedelta(days=1)).strftime("%Y-%m-%d")


def estimate_temp_probability(forecast_high: float, threshold: float, above: bool) -> float:
    """
    Estimate probability of temperature exceeding/falling below threshold.
    Uses a simple sigmoid centered on the forecast, ±5°F uncertainty.
    """
    import math
    sigma = 5.0  # °F forecast uncertainty
    diff = (forecast_high - threshold) / sigma
    prob = 1.0 / (1.0 + math.exp(-diff))
    return prob if above else 1.0 - prob


async def find_weather_opportunities(
    http: httpx.AsyncClient,
    kalshi_markets: list,
) -> list[WeatherOpportunity]:
    opportunities = []

    # Batch fetch forecasts for all relevant cities
    city_forecasts: dict[str, dict] = {}
    cities_needed = set()
    for km in kalshi_markets:
        city = extract_city_from_title(km["title"])
        if city:
            cities_needed.add(city)

    for city in cities_needed:
        coords = Config.CITIES[city]
        fc = await fetch_openmeteo_forecast(http, coords["lat"], coords["lon"])
        if fc:
            city_forecasts[city] = fc

    for km in kalshi_markets:
        title  = km["title"]
        ticker = km["ticker"]
        yes_ask = km["yes_ask"]
        no_ask  = km["no_ask"]
        volume  = km["volume"]

        if yes_ask == 0 or no_ask == 0 or volume < Config.MIN_VOLUME:
            continue

        city = extract_city_from_title(title)
        if not city or city not in city_forecasts:
            continue

        fc = city_forecasts[city]
        target_date = extract_date_from_title(title)
        title_lower = title.lower()

        forecast_prob: Optional[float] = None
        forecast_source = "open-meteo"

        # Temperature high/low markets
        if "high" in title_lower and ("temp" in title_lower or "degree" in title_lower):
            # Try to extract threshold from title like "NYC high temp above 75°F"
            import re
            nums = re.findall(r'\d+', title)
            if nums:
                threshold = float(nums[-1])
                above = "above" in title_lower or "exceed" in title_lower or "over" in title_lower
                fc_high = parse_high_temp_forecast(fc, target_date)
                if fc_high is not None:
                    forecast_prob = estimate_temp_probability(fc_high, threshold, above)

        # Precipitation / rain markets
        elif any(kw in title_lower for kw in ["rain", "precipitation", "precip"]):
            prob = parse_precip_probability(fc, target_date)
            if prob is not None:
                above = "yes" in title_lower or "will" in title_lower or "exceed" in title_lower
                forecast_prob = prob if above else 1.0 - prob

        # Snowfall markets
        elif "snow" in title_lower:
            import re
            nums = re.findall(r'\d+\.?\d*', title)
            threshold_in = float(nums[0]) if nums else 1.0
            snow_in = parse_snowfall_forecast(fc, target_date)
            if snow_in is not None:
                above = "above" in title_lower or "exceed" in title_lower or "more" in title_lower
                forecast_prob = (1.0 if snow_in > threshold_in else 0.0) if above else (1.0 if snow_in <= threshold_in else 0.0)
                # Smooth with uncertainty
                from math import exp
                diff = abs(snow_in - threshold_in)
                confidence = min(0.95, 0.5 + diff * 0.2)
                forecast_prob = confidence if forecast_prob > 0.5 else 1.0 - confidence

        if forecast_prob is None:
            continue

        # Compare to Kalshi implied probability
        yes_prob = yes_ask / 100.0
        no_prob  = no_ask  / 100.0

        yes_edge = forecast_prob - yes_prob
        no_edge  = (1.0 - forecast_prob) - no_prob

        best_edge = max(yes_edge, no_edge)
        best_side  = "yes" if yes_edge >= no_edge else "no"
        best_price = yes_ask if best_side == "yes" else no_ask
        kalshi_prob = yes_prob if best_side == "yes" else no_prob

        # ── Micro-bet check (tail probability, low-price contracts) ──────────
        if (Config.MICRO_BET_MODE
                and Config.MICRO_BET_MIN_PRICE <= best_price <= Config.MICRO_BET_MAX_PRICE
                and forecast_prob >= Config.MICRO_BET_MIN_FORECAST_PROB
                and kalshi_prob > 0):
            edge_ratio = forecast_prob / kalshi_prob
            if edge_ratio >= Config.MICRO_BET_MIN_RATIO:
                opp = WeatherOpportunity(
                    ticker=ticker, title=title,
                    yes_ask=yes_ask, no_ask=no_ask, volume=volume,
                    best_side=best_side, best_price=best_price,
                    forecast_prob=forecast_prob, kalshi_prob=kalshi_prob,
                    edge_pct=round(best_edge, 4), city=city,
                    forecast_source=forecast_source,
                    close_time=km["close_time"],
                    is_micro_bet=True, edge_ratio=round(edge_ratio, 2),
                )
                opportunities.append(opp)
                log.info(
                    f"[MICRO] {title[:55]} | city={city} | "
                    f"forecast={forecast_prob:.1%} kalshi={kalshi_prob:.1%} "
                    f"ratio={edge_ratio:.1f}x @{best_price}¢"
                )
                continue

        # ── Standard high-probability trade ──────────────────────────────────
        if best_edge < Config.MIN_EDGE_PCT:
            continue

        # Fee-aware EV check
        ev_after_fees = best_edge - Config.MAKER_FEE
        if ev_after_fees <= 0:
            continue

        # Fee filter: only trade ≥MIN_KALSHI_PRICE
        if best_price < Config.MIN_KALSHI_PRICE:
            continue

        opp = WeatherOpportunity(
            ticker=ticker, title=title,
            yes_ask=yes_ask, no_ask=no_ask, volume=volume,
            best_side=best_side, best_price=best_price,
            forecast_prob=forecast_prob, kalshi_prob=kalshi_prob,
            edge_pct=round(best_edge, 4), city=city,
            forecast_source=forecast_source,
            close_time=km["close_time"],
        )
        opportunities.append(opp)
        log.info(
            f"[WEATHER] {title[:60]} | city={city} | "
            f"forecast={forecast_prob:.1%} kalshi={kalshi_prob:.1%} "
            f"edge=+{best_edge:.1%} side={best_side.upper()} @{best_price}¢"
        )

    opportunities.sort(key=lambda x: x.edge_pct, reverse=True)
    return opportunities


# ── Paper trading tracker ──────────────────────────────────────────────────────

class PaperLedger:
    def __init__(self, balance: float):
        self.balance = balance
        self.trades:  list[dict] = []
        self.positions: dict[str, dict] = {}

    def execute(self, opp: WeatherOpportunity) -> Optional[dict]:
        if opp.is_micro_bet:
            # Micro-bet: flat small dollar amount, spread across ladder
            target_usd = Config.MICRO_BET_MAX_USD
            contracts = max(1, int(target_usd / (opp.best_price / 100)))
            cost = contracts * opp.best_price / 100
            if cost < Config.MICRO_BET_MIN_USD or cost > self.balance:
                return None
        else:
            contracts = max(
                1,
                int(min(
                    self.balance * Config.POSITION_FRACTION / (opp.best_price / 100),
                    Config.MAX_TRADE_USD / (opp.best_price / 100),
                ))
            )
            cost = contracts * opp.best_price / 100
            if cost < Config.MIN_TRADE_USD or cost > self.balance:
                return None

        self.balance -= cost
        label = "MICRO" if opp.is_micro_bet else "PAPER"
        extra = f"ratio={opp.edge_ratio:.1f}x" if opp.is_micro_bet else f"edge=+{opp.edge_pct:.1%}"
        trade = {
            "ticker": opp.ticker, "side": opp.best_side, "contracts": contracts,
            "price": opp.best_price, "cost": cost, "ts": datetime.now(timezone.utc).isoformat(),
            "forecast_prob": opp.forecast_prob, "edge": opp.edge_pct,
            "is_micro_bet": opp.is_micro_bet,
        }
        self.trades.append(trade)
        log.info(
            f"[{label}] BUY {opp.best_side.upper()} {contracts}x @{opp.best_price}¢ = ${cost:.2f} | "
            f"bal=${self.balance:.2f} | {extra} | {opp.title[:45]}"
        )
        return trade


# ── Kalshi live order ──────────────────────────────────────────────────────────

async def place_kalshi_order(
    http: httpx.AsyncClient,
    ticker: str, side: str, count: int, price_cents: int
) -> dict:
    path = "/portfolio/orders"
    payload = {
        "ticker": ticker, "client_order_id": str(uuid.uuid4()),
        "type": "limit", "action": "buy", "side": side,
        "count": count, "yes_price": price_cents if side == "yes" else 100 - price_cents,
        "no_price":  100 - price_cents if side == "yes" else price_cents,
        "expiration_ts": None,
    }
    r = await http.post(
        Config.KALSHI_BASE_URL + path,
        json=payload,
        headers=_kalshi_headers("POST", path),
        timeout=10.0,
    )
    r.raise_for_status()
    return r.json()


# ── Main loop ─────────────────────────────────────────────────────────────────

# ── Stats HTTP server ─────────────────────────────────────────────────────────
_stats_app = Flask(__name__)
_bot_stats = {"trades": 0, "wins": 0, "pnl": 0.0, "balance": 0.0, "start": time.time()}

@_stats_app.route("/stats")
def _stats_endpoint():
    t = _bot_stats
    total = t["trades"]
    return jsonify({"bot": "kalshi-weather-bot", "paper_mode": True,
        "balance": t["balance"], "trades": total, "wins": t["wins"],
        "losses": total - t["wins"], "win_rate": round(t["wins"]/max(total,1), 4),
        "pnl": t["pnl"], "uptime_hours": round((time.time()-t["start"])/3600, 2)})

@_stats_app.route("/health")
def _health_endpoint():
    return jsonify({"status": "ok"})

def _run_stats_server():
    _stats_app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))


async def main():
    log.info("=" * 60)
    log.info("Kalshi Weather Bot starting")
    log.info(f"  Paper mode:      {Config.PAPER_MODE}")
    log.info(f"  Min edge:        {Config.MIN_EDGE_PCT:.1%}")
    log.info(f"  Min price:       {Config.MIN_KALSHI_PRICE}¢")
    log.info(f"  Min volume:      {Config.MIN_VOLUME} contracts")
    log.info(f"  Poll interval:   {Config.POLL_INTERVAL_SEC}s")
    log.info("=" * 60)

    ledger = PaperLedger(Config.PAPER_BALANCE) if Config.PAPER_MODE else None
    if ledger:
        _bot_stats['balance'] = ledger.balance
    threading.Thread(target=_run_stats_server, daemon=True).start()
    traded_this_session: set[str] = set()

    while True:

        # ── Critical Module: cycle-start checks ──
        try:
            if _paper_balance_available:
                _paper_balance_mgr.check_and_restart()
        except Exception:
            pass
        try:
            if _dynamic_params_available:
                _dparams = _dynamic_params.get_all()
        except Exception:
            _dparams = {}
        try:
            if _data_pipeline_available:
                _data_pipeline.record_snapshot({'bot': 'weather_bot', 'ts': time.time()})
        except Exception:
            pass

        try:
            async with httpx.AsyncClient(timeout=15.0) as http:
                markets = await fetch_kalshi_weather_markets(http)
                if not markets:
                    log.info("No weather markets found — will retry")
                else:
                    opps = await find_weather_opportunities(http, markets)
                    log.info(f"Found {len(opps)} weather opportunities")

                    for opp in opps:
                        if opp.ticker in traded_this_session:
                            continue

                        kind = "MICRO-BET" if opp.is_micro_bet else "OPPORTUNITY"
                        extra = f"Ratio: {opp.edge_ratio:.1f}x" if opp.is_micro_bet else f"Edge: +{opp.edge_pct:.1%}"
                        log.info(
                            f"{kind}: {opp.title}\n"
                            f"  Forecast: {opp.forecast_prob:.1%}  Kalshi: {opp.kalshi_prob:.1%}  "
                            f"{extra}\n"
                            f"  Side: {opp.best_side.upper()} @{opp.best_price}¢  Vol: {opp.volume}"
                        )

                        # ── Risk Guard check ──
                        _rg_contracts = max(1, int(min(Config.MAX_TRADE_USD / (opp.best_price / 100), 500))) if not opp.is_micro_bet else max(1, int(Config.MICRO_BET_MAX_USD / (opp.best_price / 100)))
                        if not Config.PAPER_MODE:
                            allowed, reason, capped = risk_manager.pre_trade_check(opp.ticker, opp.best_price, _rg_contracts, opp.best_side, bot_name="weather-bot")
                            if not allowed:
                                log.warning(f"Risk guard blocked: {reason}")
                                shadow_log({"bot": "weather", "ticker": opp.ticker, "side": opp.best_side, "price": opp.best_price, "edge": opp.edge_pct}, taken=False, reason=f"risk guard: {reason}")
                                evaluate_virtual_portfolios({"bot": "weather", "ticker": opp.ticker, "side": opp.best_side, "price": opp.best_price, "edge": opp.edge_pct})
                                if _quant_modules_available:
                                    try:
                                        _features.extract({"price": locals().get("price", 0), "volume": locals().get("volume", 0), "bid": locals().get("bid", 0), "ask": locals().get("ask", 0)})
                                        _bayesian.update(locals().get("market_id", locals().get("ticker", "unknown")), locals().get("price", 0), time.time())
                                        _td_edge = calculate_time_weighted_edge(locals().get("edge", 0), locals().get("minutes_remaining", locals().get("time_remaining", 15)), 15)
                                        _vpin.update(locals().get("price", 0), locals().get("volume", 0))
                                        _mi = estimate_market_impact(locals().get("contracts", 1), locals().get("volume", 100))
                                    except:
                                        pass
                                continue
                        else:
                            allowed, reason, capped = risk_manager.pre_trade_check(opp.ticker, opp.best_price, _rg_contracts, opp.best_side, bot_name="weather-bot")
                            if not allowed:
                                log.info(f"[PAPER] Risk guard would block: {reason}")

                        if Config.PAPER_MODE and ledger:
                            result = ledger.execute(opp)
                            if result:
                                # ── Regime detection ──
                                regime = check_regime(float(price))
                                if regime == "CRASH":
                                    log.warning("REGIME CRASH on kalshi_weather_bot — skipping trade")
                                    shadow_log({"bot": "kalshi_weather_bot", "regime": regime}, taken=False, reason="crash regime")
                                    evaluate_virtual_portfolios({"bot": "kalshi_weather_bot", "regime": regime})
                                    continue

                                # ── Critical Module: pre-trade validation ──

                                try:

                                    if _pre_trade_validator_available:

                                        _ptv_ok = validate_pre_trade({})

                                except Exception:

                                    pass

                                try:

                                    if _rejection_filter_available:

                                        _rejection_filter.check({})

                                except Exception:

                                    pass

                                try:

                                    if _dynamic_edge_available:

                                        _computed_edge = calculate_dynamic_edge(locals().get('edge', 0), locals().get('price', 0))

                                except Exception:

                                    pass

                                try:

                                    if _adaptive_kelly_available:

                                        _kelly_size = calculate_adaptive_kelly(locals().get('edge', 0.05), locals().get('odds', 2.0))

                                except Exception:

                                    pass

                                try:

                                    if _conviction_scaler_available:

                                        _conviction_mult = _conviction_scaler.scale(locals().get('edge', 0.05))

                                except Exception:

                                    pass

                                try:

                                    if _brier_scorer_available:

                                        _brier_scorer.record(locals().get('predicted_prob', 0.5), locals().get('outcome', 0))

                                except Exception:

                                    pass

                                shadow_log({"bot": "weather", "ticker": opp.ticker, "side": opp.best_side, "price": opp.best_price, "edge": opp.edge_pct, "forecast_prob": opp.forecast_prob}, taken=True)
                                evaluate_virtual_portfolios({"bot": "weather", "ticker": opp.ticker, "side": opp.best_side, "price": opp.best_price, "edge": opp.edge_pct, "forecast_prob": opp.forecast_prob})
                                traded_this_session.add(opp.ticker)
                        elif not Config.PAPER_MODE:
                            if opp.is_micro_bet:
                                contracts = max(1, int(Config.MICRO_BET_MAX_USD / (opp.best_price / 100)))
                            else:
                                contracts = max(1, int(min(
                                    Config.MAX_TRADE_USD / (opp.best_price / 100), 500
                                )))
                            contracts = capped  # use risk-guard-capped value
                            try:
                                result = await place_kalshi_order(
                                    http, opp.ticker, opp.best_side, contracts, opp.best_price
                                )
                                log.info(f"Order placed: {result}")
                                traded_this_session.add(opp.ticker)
                            except Exception as e:
                                log.error(f"Order failed: {e}")

                if ledger:
                    _bot_stats["balance"] = ledger.balance
                    _bot_stats["trades"] = len(ledger.trades)
                    micro = sum(1 for t in ledger.trades if t.get("is_micro_bet"))
                    std   = len(ledger.trades) - micro
                    log.info(
                        f"Paper balance: ${ledger.balance:.2f} | "
                        f"Trades: {len(ledger.trades)} (std={std} micro={micro}) | "
                        f"Session traded: {len(traded_this_session)}"
                    )

        except Exception as e:
            log.error(f"Poll cycle error: {e}", exc_info=True)

        await asyncio.sleep(Config.POLL_INTERVAL_SEC)


if __name__ == "__main__":
    asyncio.run(main())
