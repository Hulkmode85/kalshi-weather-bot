"""
Multi-Agent AI Orchestrator — 6 specialized Claude agents debate prediction market trades.

Quant Concept:
    This module sits ON TOP of the 8 quant modules (bayesian_updater, correlation_matrix,
    ensemble_model, feature_engine, market_impact, portfolio_optimizer, time_decay_edge,
    vpin_toxicity). It uses Claude API to run 6 specialized AI agents — each with a
    distinct analytical lens — that evaluate, debate, and refine probability estimates
    for prediction market opportunities.

    The Supervisor agent applies Tetlock's superforecasting principles to synthesize
    all inputs into a final probability, edge, confidence, and trade recommendation.

    This module runs in SHADOW MODE — it does not affect actual trading.
    It logs every evaluation to a JSONL file for future optimization.

Architecture:
    1. ResearcherAgent   — finds facts, data, evidence
    2. BaseRateAgent     — reference class forecasting, outside view
    3. NarrativeAgent    — sentiment, media narratives, crowd positioning
    4. QuantAgent        — statistical edge from quant module outputs
    5. RiskAgent         — portfolio risk, position sizing, worst-case
    6. SupervisorAgent   — synthesizes all 5, applies superforecasting, final call

Cost:
    Full evaluate()  = 6 API calls (~$0.02 with Haiku, ~$0.09 with Sonnet)
    quick_evaluate() = 2 API calls (~$0.007 with Haiku, ~$0.03 with Sonnet)

    Use quick_evaluate() for routine scanning. Reserve full evaluate() for
    opportunities where quant modules detect edge > 3%.

Usage:
    from multi_agent_orchestrator import MultiAgentOrchestrator

    orch = MultiAgentOrchestrator()
    result = orch.evaluate_opportunity(
        market_title="Will BTC be above $100K on 2026-04-30?",
        market_price=62.0,
        quant_data={"edge": 0.08, "vpin": 0.3, "ensemble_consensus": 0.70},
        portfolio={"positions": [...], "total_value": 5000}
    )

Env Vars:
    ANTHROPIC_API_KEY   — required for API calls
    AGENT_ENABLED       — "true" to enable (default "false", prevents accidental spend)
    AGENT_MODEL         — model override (default "claude-haiku-4-20250414")
    AGENT_RATE_LIMIT    — max API calls per hour (default 20)
    SHADOW_LOG_DIR      — log directory (default "/tmp/agent_shadow_logs")
"""

import json
import logging
import os
import re
import time
import threading
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

log = logging.getLogger("multi_agent")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
AGENT_ENABLED = os.getenv("AGENT_ENABLED", "false").lower() == "true"
AGENT_MODEL = os.getenv("AGENT_MODEL", "claude-haiku-4-20250414")
AGENT_RATE_LIMIT = int(os.getenv("AGENT_RATE_LIMIT", "20"))  # calls/hour
SHADOW_LOG_DIR = os.getenv("SHADOW_LOG_DIR", "/tmp/agent_shadow_logs")
AGENT_LOG_FILE = os.path.join(SHADOW_LOG_DIR, "multi_agent_decisions.jsonl")

# ---------------------------------------------------------------------------
# Virtual Portfolio Variants
# ---------------------------------------------------------------------------

AGENT_VIRTUAL_PORTFOLIOS = [
    {
        "name": "agent-aggressive",
        "debate_rounds": 1,
        "confidence_threshold": 0.6,
        "use_quant": False,
        "description": "Fast, low-threshold — catches more opportunities but noisier",
    },
    {
        "name": "agent-conservative",
        "debate_rounds": 3,
        "confidence_threshold": 0.8,
        "use_quant": False,
        "description": "High-confidence only — fewer trades, higher expected accuracy",
    },
    {
        "name": "full-agent",
        "debate_rounds": 2,
        "confidence_threshold": 0.7,
        "use_quant": True,
        "description": "Balanced — quant + AI agents, 2 debate rounds",
    },
    {
        "name": "quant-only",
        "debate_rounds": 0,
        "confidence_threshold": 0.7,
        "use_quant": True,
        "description": "No AI agents — pure quant module consensus (baseline)",
    },
]

# ---------------------------------------------------------------------------
# Agent System Prompts
# ---------------------------------------------------------------------------

RESEARCHER_PROMPT = """You are a research analyst specializing in prediction markets. \
Given a prediction market question, find the most relevant facts, data points, recent \
news, and evidence that bear on the outcome.

Rules:
- Be SPECIFIC: cite dates, numbers, percentages, sources by name.
- Distinguish between hard data (official statistics, verified reports) and soft signals \
(rumors, speculation, unnamed sources).
- List facts that support YES and facts that support NO separately.
- Flag any key unknowns or information gaps.
- Do NOT give a probability estimate — that is another agent's job.

Output format:
FACTS SUPPORTING YES:
1. ...
2. ...

FACTS SUPPORTING NO:
1. ...
2. ...

KEY UNKNOWNS:
1. ...
"""

BASERATE_PROMPT = """You are a superforecaster trained in Philip Tetlock's methodology. \
Given a prediction market question and research findings, estimate the base rate \
probability using reference class forecasting and the outside view.

Your process (follow EXACTLY):
1. REFERENCE CLASS: What is the broadest relevant reference class? What is the base \
rate for that class? (e.g., "Of all bills introduced in Congress, ~5% become law.")
2. NARROW THE CLASS: Are there sub-classes with different base rates? Adjust.
3. INSIDE VIEW ADJUSTMENT: Given the specific research findings, adjust up or down. \
Each adjustment must be explicitly justified and small (typically 1-10 percentage points).
4. FINAL ESTIMATE: State your probability to the nearest 1%.

Rules:
- ALWAYS start from the outside view / base rate. Never anchor on the current market price.
- Show your math. If base rate is 15% and you adjust +8%, state "15% + 8% = 23%".
- Express uncertainty honestly. If evidence is thin, say so.
- Apply Tetlock's granularity: use specific numbers, not "likely" or "unlikely".

Output format:
REFERENCE CLASS: [description] — BASE RATE: [X%]
ADJUSTMENTS: [+/-X% for reason]
FINAL PROBABILITY: [X%]
CONFIDENCE IN ESTIMATE: [low/medium/high]
REASONING: [2-3 sentences]
"""

NARRATIVE_PROMPT = """You are a sentiment and narrative analyst for prediction markets. \
Given a prediction market question and its current price, analyze how public perception, \
media narratives, and crowd psychology might affect the outcome or the market price.

Your analysis must cover:
1. DOMINANT NARRATIVE: What is the prevailing media/public narrative about this event?
2. CROWD POSITIONING: Is the crowd over-confident or under-confident? Is there herding?
3. CONTRARIAN SIGNALS: What would have to be true for the opposite outcome? Is anyone \
making that case?
4. NARRATIVE CATALYSTS: What upcoming events could shift the narrative (scheduled \
announcements, deadlines, speeches)?
5. SENTIMENT SKEW: Is sentiment systematically biased in one direction? (e.g., \
recency bias, availability bias, political tribalism)

Rules:
- Separate what WILL happen from what people THINK will happen.
- Identify which direction narrative bias pushes the market price (inflated or deflated).
- Be specific about timeframes and triggers.

Output format:
DOMINANT NARRATIVE: [description]
CROWD POSITION: [over/under-confident, direction]
CONTRARIAN CASE: [what the minority argues]
UPCOMING CATALYSTS: [list with dates if known]
NET NARRATIVE BIAS: [pushes price UP/DOWN by roughly X%]
"""

QUANT_PROMPT = """You are a quantitative analyst for prediction markets. Given market \
data and outputs from quant modules, calculate the statistical edge and assess trade \
quality from a purely numerical perspective.

Quant metrics to evaluate (when available):
- TIME-DECAY ADJUSTED EDGE: How does the edge change as expiration approaches?
- VPIN TOXICITY: Is there informed trading? (VPIN > 0.7 = toxic, avoid)
- MARKET IMPACT: Will our order move the price? What is the expected slippage?
- ENSEMBLE CONSENSUS: Do multiple models agree on direction?
- BAYESIAN POSTERIOR: What is the updated probability after incorporating new data?
- FEATURE SIGNALS: Which features are firing? Are they consistent?
- CORRELATION: Is this trade correlated with existing positions?

Rules:
- If quant data is sparse or missing, say so explicitly — do not fabricate numbers.
- Calculate expected value: EV = (prob * payout) - ((1-prob) * cost)
- Flag any red flags: high VPIN, low volume, wide spread, stale data.
- Be precise with numbers. Round to 2 decimal places.

Output format:
QUANT PROBABILITY: [X%]
STATISTICAL EDGE: [X%] (your prob minus market price)
EXPECTED VALUE PER $1: [$X.XX]
RED FLAGS: [list or "none"]
TRADE QUALITY: [A/B/C/D/F]
REASONING: [2-3 sentences]
"""

RISK_PROMPT = """You are a risk manager for a prediction market portfolio. Your PRIMARY \
job is preventing catastrophic losses. Given a proposed trade and current portfolio \
state, evaluate all risks.

Your analysis must cover:
1. POSITION SIZING: What is the optimal bet size? Use fractional Kelly criterion \
(0.75x Kelly for live, 1.0x for paper).
2. PORTFOLIO CONCENTRATION: After this trade, what % of portfolio is in one market, \
one category, or one expiration window?
3. CORRELATION RISK: Is this trade correlated with existing positions? Does it \
increase or decrease portfolio diversification?
4. WORST CASE: If this trade goes to zero, what is the portfolio impact?
5. LIQUIDITY RISK: Can we exit this position if needed? What is the expected exit cost?
6. TIMING RISK: Is there event risk (binary outcome) that could gap against us?

Rules:
- When in doubt, recommend SMALLER position sizes.
- Max single position: 10% of portfolio (hard cap).
- Max correlated cluster: 25% of portfolio.
- If VPIN > 0.7 or spread > 5%, recommend SKIP regardless of edge.
- Always calculate the position size in dollars, not just percentages.

Output format:
KELLY FRACTION: [X%] of bankroll
RECOMMENDED POSITION: $[X] ([X]% of portfolio)
CONCENTRATION AFTER TRADE: [X]% in this category
WORST CASE LOSS: $[X] ([X]% of portfolio)
RISK RATING: [LOW/MEDIUM/HIGH/EXTREME]
RECOMMENDATION: [PROCEED/REDUCE SIZE/SKIP]
REASONING: [2-3 sentences]
"""

SUPERVISOR_PROMPT = """You are the Chief Investment Officer supervising 5 specialist \
analysts for a prediction market fund. You have received their independent analyses \
and must now synthesize a FINAL decision.

Your process (follow EXACTLY):
1. READ all 5 analyst reports carefully.
2. IDENTIFY DISAGREEMENTS: Where do analysts disagree? Who has the stronger argument?
3. WEIGHT BY TRACK RECORD: If agent_weights are provided, weight more accurate agents \
higher. Otherwise, weight equally.
4. PRE-MORTEM: Assume this trade LOSES money. What went wrong? Is that scenario plausible?
5. APPLY TETLOCK'S COMMANDMENTS:
   - Triage: Is this question even forecastable?
   - Break impossible questions into sub-questions.
   - Balance inside and outside views.
   - Update incrementally (don't over-react to new info).
   - Look for clashing causal forces.
   - Distinguish signal from noise.
   - Be humble about what you don't know.
6. SYNTHESIZE: Combine all inputs into a final probability. Show how you weighted each.
7. DECIDE: Buy YES, Buy NO, or SKIP. Include position size.

Output format:
ANALYST SUMMARY:
- Researcher: [key finding]
- Base Rate: [X%]
- Narrative: [bias direction]
- Quant: [edge X%]
- Risk: [rating]

DISAGREEMENTS: [description]
PRE-MORTEM: [what could go wrong]
WEIGHTING: [how you weighted each input]

FINAL PROBABILITY: [X%]
EDGE VS MARKET: [X%]
CONFIDENCE: [0.0-1.0]
RECOMMENDATION: [BUY_YES / BUY_NO / SKIP]
POSITION SIZE: [$X]
REASONING: [3-5 sentences justifying the final call]
"""

# ---------------------------------------------------------------------------
# Rate Limiter
# ---------------------------------------------------------------------------


class RateLimiter:
    """Simple sliding-window rate limiter for API calls."""

    def __init__(self, max_calls_per_hour: int):
        self.max_calls = max_calls_per_hour
        self.calls: list[float] = []
        self.lock = threading.Lock()

    def acquire(self) -> bool:
        """Return True if a call is allowed, False if rate-limited."""
        now = time.time()
        with self.lock:
            # Prune calls older than 1 hour
            self.calls = [t for t in self.calls if now - t < 3600]
            if len(self.calls) >= self.max_calls:
                return False
            self.calls.append(now)
            return True

    def remaining(self) -> int:
        """How many calls remain in the current window."""
        now = time.time()
        with self.lock:
            self.calls = [t for t in self.calls if now - t < 3600]
            return max(0, self.max_calls - len(self.calls))


# ---------------------------------------------------------------------------
# Shadow Logger
# ---------------------------------------------------------------------------


def shadow_log(entry: dict):
    """Append a JSON line to the agent shadow log file."""
    Path(SHADOW_LOG_DIR).mkdir(parents=True, exist_ok=True)
    entry["_module"] = "multi_agent_orchestrator"
    entry["_ts"] = time.time()
    try:
        with open(AGENT_LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        log.warning("Shadow log write failed: %s", e)


# ---------------------------------------------------------------------------
# Response Parser
# ---------------------------------------------------------------------------


def parse_supervisor_output(text: str) -> dict:
    """Extract structured fields from Supervisor agent's free-text response.

    Returns a dict with keys: final_probability, edge, confidence,
    recommendation, position_size. Missing fields are None.
    """
    result: dict = {
        "final_probability": None,
        "edge": None,
        "confidence": None,
        "recommendation": None,
        "position_size": None,
    }

    # FINAL PROBABILITY: 62%
    m = re.search(r"FINAL PROBABILITY:\s*([\d.]+)%", text)
    if m:
        result["final_probability"] = float(m.group(1)) / 100.0

    # EDGE VS MARKET: 5%  or  EDGE: 5%
    m = re.search(r"EDGE(?:\s+VS\s+MARKET)?:\s*([+-]?[\d.]+)%", text)
    if m:
        result["edge"] = float(m.group(1)) / 100.0

    # CONFIDENCE: 0.75
    m = re.search(r"CONFIDENCE:\s*([\d.]+)", text)
    if m:
        result["confidence"] = float(m.group(1))

    # RECOMMENDATION: BUY_YES / BUY_NO / SKIP
    m = re.search(r"RECOMMENDATION:\s*(BUY_YES|BUY_NO|SKIP)", text, re.IGNORECASE)
    if m:
        result["recommendation"] = m.group(1).upper()

    # POSITION SIZE: $150
    m = re.search(r"POSITION SIZE:\s*\$?([\d,.]+)", text)
    if m:
        result["position_size"] = float(m.group(1).replace(",", ""))

    return result


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------


class MultiAgentOrchestrator:
    """6-agent Claude-powered orchestrator for prediction market evaluation.

    Shadow-mode only: logs all decisions, never executes trades.
    """

    def __init__(self, agent_weights: Optional[dict] = None):
        """Initialize orchestrator.

        Args:
            agent_weights: Optional dict mapping agent names to float weights
                (from MetaLearner). Higher weight = more influence in
                supervisor synthesis. Default: equal weights.
        """
        self.client = None
        self.rate_limiter = RateLimiter(AGENT_RATE_LIMIT)
        self.agent_weights = agent_weights or {
            "researcher": 1.0,
            "base_rate": 1.0,
            "narrative": 1.0,
            "quant": 1.0,
            "risk": 1.0,
        }
        self.total_api_calls = 0
        self.total_cost_estimate = 0.0

        if AGENT_ENABLED and ANTHROPIC_API_KEY:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
                log.info("Multi-agent orchestrator initialized (model=%s)", AGENT_MODEL)
            except ImportError:
                log.warning("anthropic package not installed — agents disabled")
            except Exception as e:
                log.warning("Failed to init Anthropic client: %s", e)
        elif not AGENT_ENABLED:
            log.info("Multi-agent orchestrator disabled (AGENT_ENABLED != 'true')")
        else:
            log.info("Multi-agent orchestrator disabled (no ANTHROPIC_API_KEY)")

    def _call_agent(
        self, agent_name: str, system_prompt: str, user_message: str, max_tokens: int = 500
    ) -> str:
        """Call Claude API with a specific agent persona.

        Args:
            agent_name: Name for logging (e.g., "researcher", "supervisor").
            system_prompt: The agent's system prompt defining its role.
            user_message: The user-turn content with market data.
            max_tokens: Max response tokens.

        Returns:
            Agent response text, or error string if call fails.
        """
        if not self.client:
            return "[AGENT DISABLED] No API client available."

        if not self.rate_limiter.acquire():
            remaining = self.rate_limiter.remaining()
            log.warning(
                "Rate limit hit for agent '%s' (remaining=%d)", agent_name, remaining
            )
            return f"[RATE LIMITED] {AGENT_RATE_LIMIT} calls/hour exceeded."

        start = time.time()
        try:
            response = self.client.messages.create(
                model=AGENT_MODEL,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            text = response.content[0].text
            elapsed = time.time() - start
            self.total_api_calls += 1

            # Estimate cost (rough, based on typical token counts)
            input_tokens = response.usage.input_tokens if hasattr(response, "usage") else 0
            output_tokens = response.usage.output_tokens if hasattr(response, "usage") else 0
            if "haiku" in AGENT_MODEL.lower():
                cost = (input_tokens * 0.25 + output_tokens * 1.25) / 1_000_000
            elif "sonnet" in AGENT_MODEL.lower():
                cost = (input_tokens * 3.0 + output_tokens * 15.0) / 1_000_000
            else:
                cost = (input_tokens * 3.0 + output_tokens * 15.0) / 1_000_000
            self.total_cost_estimate += cost

            shadow_log({
                "event": "agent_call",
                "agent": agent_name,
                "model": AGENT_MODEL,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_estimate": round(cost, 6),
                "elapsed_s": round(elapsed, 3),
            })

            log.debug(
                "Agent '%s' responded in %.2fs (%d in / %d out tokens, ~$%.4f)",
                agent_name, elapsed, input_tokens, output_tokens, cost,
            )
            return text

        except Exception as e:
            log.error("Agent '%s' call failed: %s", agent_name, e)
            shadow_log({"event": "agent_error", "agent": agent_name, "error": str(e)})
            return f"[ERROR] {agent_name}: {e}"

    # -------------------------------------------------------------------
    # Full 6-Agent Evaluation
    # -------------------------------------------------------------------

    def evaluate_opportunity(
        self,
        market_title: str,
        market_price: float,
        quant_data: Optional[dict] = None,
        portfolio: Optional[dict] = None,
    ) -> dict:
        """Run full 6-agent evaluation on a market opportunity.

        This makes 6 sequential API calls (agents build on each other's output).
        Total cost: ~$0.02 (Haiku) to ~$0.09 (Sonnet).

        Args:
            market_title: The prediction market question/title.
            market_price: Current market price in cents (0-100).
            quant_data: Dict of quant module outputs (edge, vpin, ensemble, etc.).
            portfolio: Dict describing current portfolio positions.

        Returns:
            Dict with all agent outputs, parsed supervisor decision, and metadata.
        """
        eval_start = time.time()

        if not self.client:
            result = {
                "ts": time.time(),
                "market": market_title,
                "market_price": market_price,
                "status": "disabled",
                "reason": "AGENT_ENABLED is false or no API key",
            }
            shadow_log(result)
            return result

        # Include agent weights in supervisor context if non-default
        weight_info = ""
        if self.agent_weights:
            weight_info = f"\nAgent reliability weights: {json.dumps(self.agent_weights)}"

        # --- Agent 1: Researcher ---
        researcher_output = self._call_agent(
            "researcher",
            RESEARCHER_PROMPT,
            f"Prediction market question: {market_title}",
        )

        # --- Agent 2: Base Rate (uses researcher findings) ---
        baserate_input = (
            f"Prediction market question: {market_title}\n\n"
            f"Research findings:\n{researcher_output[:800]}"
        )
        baserate_output = self._call_agent("base_rate", BASERATE_PROMPT, baserate_input)

        # --- Agent 3: Narrative (uses market price) ---
        narrative_input = (
            f"Prediction market question: {market_title}\n"
            f"Current market price: {market_price}c (implies {market_price}% probability)"
        )
        narrative_output = self._call_agent("narrative", NARRATIVE_PROMPT, narrative_input)

        # --- Agent 4: Quant (uses quant module outputs) ---
        quant_input = (
            f"Prediction market question: {market_title}\n"
            f"Current market price: {market_price}c\n"
            f"Quant module outputs:\n{json.dumps(quant_data or {}, indent=2)}"
        )
        quant_output = self._call_agent("quant", QUANT_PROMPT, quant_input)

        # --- Agent 5: Risk (uses portfolio data) ---
        risk_input = (
            f"Proposed trade: Buy '{market_title}' at {market_price}c\n"
            f"Current portfolio state:\n{json.dumps(portfolio or {}, indent=2)}"
        )
        risk_output = self._call_agent("risk", RISK_PROMPT, risk_input)

        # --- Agent 6: Supervisor (synthesizes all) ---
        supervisor_input = (
            f"Market: {market_title}\n"
            f"Current price: {market_price}c\n\n"
            f"=== RESEARCHER REPORT ===\n{researcher_output[:500]}\n\n"
            f"=== BASE RATE ANALYSIS ===\n{baserate_output[:500]}\n\n"
            f"=== NARRATIVE ANALYSIS ===\n{narrative_output[:500]}\n\n"
            f"=== QUANT ANALYSIS ===\n{quant_output[:500]}\n\n"
            f"=== RISK ASSESSMENT ===\n{risk_output[:500]}"
            f"{weight_info}"
        )
        supervisor_output = self._call_agent(
            "supervisor", SUPERVISOR_PROMPT, supervisor_input, max_tokens=800
        )

        # Parse structured fields from supervisor output
        parsed = parse_supervisor_output(supervisor_output)

        elapsed = time.time() - eval_start
        result = {
            "ts": time.time(),
            "market": market_title,
            "market_price": market_price,
            "status": "complete",
            "mode": "full",
            "agents": {
                "researcher": researcher_output[:1000],
                "base_rate": baserate_output[:1000],
                "narrative": narrative_output[:1000],
                "quant": quant_output[:1000],
                "risk": risk_output[:1000],
                "supervisor": supervisor_output[:1500],
            },
            "decision": parsed,
            "agent_weights_used": self.agent_weights,
            "elapsed_s": round(elapsed, 2),
            "api_calls": 6,
            "model": AGENT_MODEL,
        }

        shadow_log(result)
        log.info(
            "Full evaluation: '%s' => prob=%.2f, edge=%.2f, rec=%s (%.1fs)",
            market_title,
            parsed.get("final_probability") or 0,
            parsed.get("edge") or 0,
            parsed.get("recommendation") or "PARSE_FAIL",
            elapsed,
        )

        return result

    # -------------------------------------------------------------------
    # Quick (2-Agent) Evaluation
    # -------------------------------------------------------------------

    def quick_evaluate(self, market_title: str, market_price: float) -> dict:
        """Lightweight evaluation using BaseRate + Supervisor only.

        Saves API calls for high-volume scanning. Use full evaluate_opportunity()
        only when quant modules detect edge > 3%.

        Args:
            market_title: The prediction market question/title.
            market_price: Current market price in cents (0-100).

        Returns:
            Dict with base_rate and supervisor outputs + parsed decision.
        """
        eval_start = time.time()

        if not self.client:
            result = {
                "ts": time.time(),
                "market": market_title,
                "market_price": market_price,
                "status": "disabled",
                "quick_mode": True,
            }
            shadow_log(result)
            return result

        baserate_output = self._call_agent(
            "base_rate",
            BASERATE_PROMPT,
            f"Prediction market question: {market_title}\nCurrent price: {market_price}c",
            max_tokens=300,
        )

        supervisor_input = (
            f"Market: {market_title}\n"
            f"Current price: {market_price}c\n\n"
            f"=== BASE RATE ANALYSIS ===\n{baserate_output}\n\n"
            f"NOTE: This is a quick scan. Only base-rate analysis is available. "
            f"Be appropriately uncertain without researcher, narrative, quant, or risk inputs."
        )
        supervisor_output = self._call_agent(
            "supervisor", SUPERVISOR_PROMPT, supervisor_input, max_tokens=400
        )

        parsed = parse_supervisor_output(supervisor_output)
        elapsed = time.time() - eval_start

        result = {
            "ts": time.time(),
            "market": market_title,
            "market_price": market_price,
            "status": "complete",
            "mode": "quick",
            "agents": {
                "base_rate": baserate_output[:500],
                "supervisor": supervisor_output[:600],
            },
            "decision": parsed,
            "elapsed_s": round(elapsed, 2),
            "api_calls": 2,
            "model": AGENT_MODEL,
        }

        shadow_log(result)
        log.info(
            "Quick evaluation: '%s' => prob=%.2f, rec=%s (%.1fs)",
            market_title,
            parsed.get("final_probability") or 0,
            parsed.get("recommendation") or "PARSE_FAIL",
            elapsed,
        )

        return result

    # -------------------------------------------------------------------
    # Debate Mode (Multiple Rounds)
    # -------------------------------------------------------------------

    def evaluate_with_debate(
        self,
        market_title: str,
        market_price: float,
        debate_rounds: int = 2,
        quant_data: Optional[dict] = None,
        portfolio: Optional[dict] = None,
    ) -> dict:
        """Run evaluation with multiple debate rounds between agents.

        After the initial evaluation, the Supervisor identifies disagreements
        and asks agents to respond to each other's arguments. Each debate round
        adds 2 API calls (BaseRate re-evaluation + Supervisor re-synthesis).

        Args:
            market_title: The prediction market question/title.
            market_price: Current market price in cents (0-100).
            debate_rounds: Number of additional debate rounds after initial eval.
            quant_data: Dict of quant module outputs.
            portfolio: Dict describing current portfolio.

        Returns:
            Dict with all rounds of debate + final converged decision.
        """
        # Start with a full evaluation
        initial = self.evaluate_opportunity(market_title, market_price, quant_data, portfolio)
        if initial.get("status") != "complete":
            return initial

        rounds = [initial]

        for r in range(debate_rounds):
            prev_decision = rounds[-1].get("decision", {})
            prev_supervisor = rounds[-1].get("agents", {}).get("supervisor", "")

            # Have BaseRate agent reconsider given supervisor feedback
            reconsider_input = (
                f"Market: {market_title} at {market_price}c\n\n"
                f"Your previous base-rate estimate was incorporated into a supervisor "
                f"analysis. Here is the supervisor's synthesis:\n\n"
                f"{prev_supervisor[:600]}\n\n"
                f"Reconsider your estimate. Has the supervisor raised valid points that "
                f"should shift your probability? Or do you stand by your original estimate? "
                f"Provide an UPDATED probability."
            )
            updated_baserate = self._call_agent(
                f"base_rate_round_{r+1}", BASERATE_PROMPT, reconsider_input, max_tokens=300
            )

            # Supervisor re-synthesizes
            resynth_input = (
                f"Market: {market_title} at {market_price}c\n\n"
                f"DEBATE ROUND {r+1}: The base-rate forecaster has reconsidered:\n"
                f"{updated_baserate[:400]}\n\n"
                f"Your previous decision:\n{prev_supervisor[:400]}\n\n"
                f"Has anything changed? Update your FINAL PROBABILITY and RECOMMENDATION "
                f"if warranted. If your view is unchanged, restate it with higher confidence."
            )
            updated_supervisor = self._call_agent(
                f"supervisor_round_{r+1}", SUPERVISOR_PROMPT, resynth_input, max_tokens=500
            )

            parsed = parse_supervisor_output(updated_supervisor)
            round_result = {
                "round": r + 1,
                "base_rate_update": updated_baserate[:500],
                "supervisor_update": updated_supervisor[:600],
                "decision": parsed,
            }
            rounds.append(round_result)

        # Final result uses the last round's decision
        final_decision = rounds[-1].get("decision", rounds[0].get("decision", {}))

        result = {
            "ts": time.time(),
            "market": market_title,
            "market_price": market_price,
            "status": "complete",
            "mode": "debate",
            "debate_rounds": debate_rounds,
            "rounds": rounds,
            "final_decision": final_decision,
            "total_api_calls": 6 + (2 * debate_rounds),
            "model": AGENT_MODEL,
        }

        shadow_log(result)
        return result

    # -------------------------------------------------------------------
    # Virtual Portfolio Evaluation
    # -------------------------------------------------------------------

    def evaluate_all_portfolios(
        self,
        market_title: str,
        market_price: float,
        quant_data: Optional[dict] = None,
        portfolio: Optional[dict] = None,
    ) -> list[dict]:
        """Run evaluation for each virtual portfolio variant.

        Each variant has different debate rounds and confidence thresholds,
        letting us compare strategies after resolutions.

        Returns:
            List of results, one per virtual portfolio variant.
        """
        results = []

        for vp in AGENT_VIRTUAL_PORTFOLIOS:
            if vp["debate_rounds"] == 0:
                # quant-only: no agent calls, just pass-through quant data
                result = {
                    "ts": time.time(),
                    "market": market_title,
                    "market_price": market_price,
                    "portfolio_variant": vp["name"],
                    "mode": "quant_only",
                    "decision": {
                        "final_probability": (quant_data or {}).get("ensemble_consensus"),
                        "edge": (quant_data or {}).get("edge"),
                        "confidence": (quant_data or {}).get("agreement_pct"),
                        "recommendation": "SKIP" if not quant_data else None,
                    },
                }
                shadow_log(result)
                results.append(result)
                continue

            eval_result = self.evaluate_with_debate(
                market_title=market_title,
                market_price=market_price,
                debate_rounds=vp["debate_rounds"],
                quant_data=quant_data if vp.get("use_quant") else None,
                portfolio=portfolio,
            )

            # Apply confidence threshold
            decision = eval_result.get("final_decision", {})
            conf = decision.get("confidence") or 0
            if conf < vp["confidence_threshold"]:
                decision["recommendation"] = "SKIP"
                decision["skip_reason"] = (
                    f"Confidence {conf:.2f} < threshold {vp['confidence_threshold']}"
                )

            eval_result["portfolio_variant"] = vp["name"]
            eval_result["confidence_threshold"] = vp["confidence_threshold"]
            results.append(eval_result)

        return results

    # -------------------------------------------------------------------
    # Status
    # -------------------------------------------------------------------

    def status(self) -> dict:
        """Return orchestrator status for health checks."""
        return {
            "enabled": AGENT_ENABLED,
            "api_connected": self.client is not None,
            "model": AGENT_MODEL,
            "rate_limit": AGENT_RATE_LIMIT,
            "rate_limit_remaining": self.rate_limiter.remaining(),
            "total_api_calls": self.total_api_calls,
            "total_cost_estimate": round(self.total_cost_estimate, 4),
            "agent_weights": self.agent_weights,
            "virtual_portfolios": [vp["name"] for vp in AGENT_VIRTUAL_PORTFOLIOS],
        }


# ---------------------------------------------------------------------------
# Convenience: module-level singleton
# ---------------------------------------------------------------------------

_orchestrator: Optional[MultiAgentOrchestrator] = None


def get_orchestrator(agent_weights: Optional[dict] = None) -> MultiAgentOrchestrator:
    """Get or create the module-level orchestrator singleton."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MultiAgentOrchestrator(agent_weights=agent_weights)
    return _orchestrator
