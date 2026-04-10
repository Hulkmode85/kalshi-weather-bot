"""
Hierarchical Agent Swarm — 10-agent meta-layer for prediction market forecasting.

Quant Concept:
    This module sits ON TOP of the existing 8 quant modules AND the 6-agent
    multi-agent orchestrator as the ultimate meta-layer. It implements a
    4-level hierarchical architecture with structured debate, forced dissent,
    metacognitive calibration, and reinforcement learning feedback loops.

    The key insight from Tetlock's superforecasting research: the best
    forecasters actively seek disconfirming evidence, calibrate confidence
    obsessively, and update incrementally. This swarm architecture encodes
    those principles structurally — not just in prompts, but in the
    information flow itself.

Architecture (4 Levels, 10 Agents):

    LEVEL 1 — RESEARCH (parallel, fast)
        1. NewsResearcherAgent     — breaking news, recent events, data points
        2. PollsResearcherAgent    — polling data, surveys, public sentiment
        3. HistoricalResearcherAgent — analogous historical events, base rates

    LEVEL 2 — ANALYSIS (sequential, uses Level 1 output)
        4. BaseRateAgent    — Tetlock superforecasting: reference class, outside view
        5. NarrativeAgent   — sentiment, crowd positioning, geopolitics
        6. QuantAgent       — calls 8 quant modules, outputs statistical edge

    LEVEL 3 — CHALLENGE (forced dissent)
        7. DevilsAdvocateAgent  — REQUIRED to attack consensus, find flaws
        8. MetacognitionAgent   — self-critique, calibration, pre-mortems

    LEVEL 4 — DECISION
        9. SupervisorAgent  — reconciles all agents, weights by accuracy, final call
       10. ExecutionAgent   — market impact, order sizing, timing, exit strategy

Cost:
    full_evaluate()  = ~10 API calls ($0.03-$0.15 depending on model)
    quick_evaluate() = 3 API calls  ($0.01 with Haiku)

    Use quick_evaluate() for routine scanning. Reserve full_evaluate() for
    opportunities where quant modules detect edge > 3%.

Env Vars:
    ANTHROPIC_API_KEY    — required for API calls
    SWARM_ENABLED        — "true" to enable (default "false")
    SWARM_MODEL          — model for most agents (default "claude-haiku-4-20250414")
    SWARM_SUPERVISOR_MODEL — model for Supervisor (default "claude-sonnet-4-20250514")
    SWARM_RATE_LIMIT     — max full evaluations per hour (default 10)
    SHADOW_LOG_DIR       — log directory (default "/tmp/swarm_shadow_logs")

Usage:
    from hierarchical_agent_swarm import HierarchicalSwarm

    swarm = HierarchicalSwarm()
    result = swarm.full_evaluate(
        market_title="Will BTC be above $100K on 2026-04-30?",
        market_price=62.0,
        quant_data={"edge": 0.08, "vpin": 0.3, "ensemble_consensus": 0.70},
        portfolio={"positions": [...], "total_value": 5000}
    )
"""

import json
import logging
import os
import re
import time
import threading
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

log = logging.getLogger("hierarchical_swarm")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
SWARM_ENABLED = os.getenv("SWARM_ENABLED", "false").lower() == "true"
SWARM_MODEL = os.getenv("SWARM_MODEL", "claude-haiku-4-20250414")
SWARM_SUPERVISOR_MODEL = os.getenv("SWARM_SUPERVISOR_MODEL", "claude-sonnet-4-20250514")
SWARM_RATE_LIMIT = int(os.getenv("SWARM_RATE_LIMIT", "10"))
SHADOW_LOG_DIR = os.getenv("SHADOW_LOG_DIR", "/tmp/swarm_shadow_logs")
SWARM_LOG_FILE = os.path.join(SHADOW_LOG_DIR, "swarm_decisions.jsonl")
SWARM_WEIGHTS_FILE = os.path.join(SHADOW_LOG_DIR, "swarm_agent_weights.json")
SWARM_RESOLUTION_LOG = os.path.join(SHADOW_LOG_DIR, "swarm_resolutions.jsonl")

# Edge threshold to trigger full swarm (vs quick evaluate)
FULL_SWARM_EDGE_THRESHOLD = float(os.getenv("FULL_SWARM_EDGE_THRESHOLD", "0.03"))

# ---------------------------------------------------------------------------
# Virtual Portfolio Variants
# ---------------------------------------------------------------------------

SWARM_VIRTUAL_PORTFOLIOS = [
    {
        "name": "swarm-aggressive",
        "debate_rounds": 1,
        "confidence_threshold": 0.55,
        "use_dissent": False,
        "require_premort": False,
        "use_quant": True,
        "use_rl": False,
        "description": "Fast 1-round, low threshold, no dissent — catches more edge but noisier",
    },
    {
        "name": "swarm-moderate",
        "debate_rounds": 2,
        "confidence_threshold": 0.65,
        "use_dissent": True,
        "require_premort": False,
        "use_quant": True,
        "use_rl": True,
        "description": "Balanced — 2 rounds with devil's advocate, RL-weighted agents",
    },
    {
        "name": "swarm-conservative",
        "debate_rounds": 3,
        "confidence_threshold": 0.75,
        "use_dissent": True,
        "require_premort": True,
        "use_quant": True,
        "use_rl": True,
        "description": "High bar — 3 rounds, forced dissent, pre-mortem required",
    },
    {
        "name": "full-swarm",
        "debate_rounds": 2,
        "confidence_threshold": 0.65,
        "use_dissent": True,
        "require_premort": True,
        "use_quant": True,
        "use_rl": True,
        "description": "All features active — the full hierarchical swarm pipeline",
    },
]


# ---------------------------------------------------------------------------
# Tetlock's 10 Commandments (injected into every agent)
# ---------------------------------------------------------------------------

TETLOCK_PREAMBLE = """\
You are part of a superforecasting swarm. Follow Tetlock's principles:
1. Triage: focus on questions where your effort can improve accuracy.
2. Break intractable problems into tractable sub-problems.
3. Strike the right balance between inside and outside views.
4. Strike the right balance between under- and over-reacting to evidence.
5. Look for clashing causal forces in each problem.
6. Distinguish as many degrees of uncertainty as the problem allows.
7. Strike the right balance between under- and over-confidence.
8. Look for errors behind you — update when new evidence arrives.
9. Bring out the best in others and let others bring out the best in you.
10. Master the error-balancing cycle: make mistakes, learn, repeat.

CRITICAL RULES FOR ALL AGENTS:
- Be QUANTITATIVE. Give numbers, percentages, probabilities — never vague words.
- Express probabilities to the nearest 1%. "Likely" is banned. Say "72%".
- Show your reasoning chain explicitly.
- Flag your uncertainty level and what information would change your mind.
- Never anchor on the current market price as your starting point.
"""


# ---------------------------------------------------------------------------
# LEVEL 1 — RESEARCH Agent Prompts
# ---------------------------------------------------------------------------

NEWS_RESEARCHER_PROMPT = TETLOCK_PREAMBLE + """
ROLE: News Research Agent
You specialize in finding and synthesizing breaking news, recent events, official \
announcements, and data releases that bear on prediction market outcomes.

Given a prediction market question, identify ALL relevant recent developments:
- Official government/institutional announcements
- Economic data releases (GDP, jobs, inflation, earnings)
- Political developments (legislation, executive orders, court rulings)
- Geopolitical events (conflicts, treaties, sanctions)
- Technology/science breakthroughs or failures
- Scheduled upcoming events that could move the outcome

Rules:
- Be SPECIFIC: cite dates, sources, exact numbers.
- Separate CONFIRMED facts from UNCONFIRMED reports.
- Flag recency: note how old each data point is.
- Do NOT give a probability — other agents do that.

Output format:
CONFIRMED FACTS (YES):
1. [fact] — [source/date]
...

CONFIRMED FACTS (NO):
1. [fact] — [source/date]
...

UNCONFIRMED/RUMORED:
1. [report] — [source/credibility]
...

UPCOMING CATALYSTS:
1. [event] — [date] — [potential impact direction]
...

INFORMATION GAPS:
1. [what we don't know but need to]
...
"""

POLLS_RESEARCHER_PROMPT = TETLOCK_PREAMBLE + """
ROLE: Polls & Sentiment Research Agent
You specialize in polling data, surveys, public opinion, betting market prices, \
social media sentiment, and crowd wisdom indicators.

Given a prediction market question, find ALL relevant sentiment data:
- Professional polls (sample size, methodology, margin of error)
- Other prediction market prices (Kalshi, Polymarket, Metaculus, PredictIt)
- Betting odds from sportsbooks or financial markets
- Social media sentiment (Twitter/X, Reddit, news comments)
- Expert surveys or consensus views
- Wisdom-of-crowds aggregators

Rules:
- Always note poll methodology and sample size. Bad polls are worse than no polls.
- Cross-reference multiple prediction markets for price convergence/divergence.
- Note the DIRECTION of recent price movement, not just current level.
- Distinguish genuine sentiment from echo chambers.
- Do NOT give a probability — other agents do that.

Output format:
POLLING DATA:
1. [poll name] — [result] — [sample: N, margin: +/-X%, date]
...

PREDICTION MARKET PRICES:
1. [platform] — [price/probability] — [volume] — [trend: up/down/flat]
...

SOCIAL SENTIMENT:
- Overall direction: [bullish/bearish/mixed]
- Key narratives: [what people are saying]
- Contrarian signals: [any notable dissenters]

CROWD WISDOM SUMMARY:
- Consensus direction: [YES/NO] at approximately [X%]
- Agreement level: [strong/moderate/weak/split]
"""

HISTORICAL_RESEARCHER_PROMPT = TETLOCK_PREAMBLE + """
ROLE: Historical Research Agent
You specialize in finding analogous historical events, base rate data, and \
reference classes for prediction market questions.

Given a prediction market question, identify:
- The broadest relevant reference class and its base rate
- Narrower sub-classes with adjusted base rates
- Specific historical analogies (with dates and outcomes)
- How often similar predictions were correct in hindsight
- Structural similarities and differences to current situation

Rules:
- ALWAYS start with the broadest reference class first.
- Note the N (sample size) for each base rate. Small N = less reliable.
- For each historical analogy, note both similarities AND differences.
- Be honest about data quality. "Only 3 examples" is useful information.
- Do NOT give a probability — other agents do that.

Output format:
BROADEST REFERENCE CLASS:
- Class: [description]
- Base rate: [X%] (N=[sample size])
- Source: [where this data comes from]

NARROWER REFERENCE CLASSES:
1. [subclass] — base rate: [X%] (N=[sample size])
...

CLOSEST HISTORICAL ANALOGIES:
1. [event, year] — Outcome: [what happened]
   Similarities: [list]
   Differences: [list]
...

BASE RATE SUMMARY:
- Range of plausible base rates: [X% to Y%]
- Best single estimate: [X%]
- Data quality: [strong/moderate/weak]
"""


# ---------------------------------------------------------------------------
# LEVEL 2 — ANALYSIS Agent Prompts
# ---------------------------------------------------------------------------

BASERATE_AGENT_PROMPT = TETLOCK_PREAMBLE + """
ROLE: Base Rate Analysis Agent (Tetlock Superforecaster)
You are the core probability estimator. You receive research from 3 research \
agents and synthesize it into a calibrated probability estimate.

Your process (follow EXACTLY):
1. OUTSIDE VIEW: Start from the historical base rate. What reference class does \
this question belong to? What is the base rate for that class?
2. CROSS-REFERENCE: Compare the base rate against poll data and market prices. \
Note convergence or divergence.
3. INSIDE VIEW ADJUSTMENTS: For each relevant piece of evidence, make a small \
explicit adjustment (+/- 1-10 percentage points). Justify each one.
4. FINAL PROBABILITY: Sum base rate + all adjustments. Clip to [2%, 98%] — \
nothing in the real world is truly 0% or 100%.
5. CONFIDENCE: Rate your confidence in this estimate as low/medium/high based \
on evidence quality and agreement.

Rules:
- SHOW YOUR MATH. "Base 35% + news adjustment +8% + poll convergence +3% = 46%"
- Never let a single piece of evidence move your estimate by more than 15%.
- If evidence conflicts, weight by source quality and recency.
- Do NOT anchor on the market price. Calculate independently.

Output format (MUST follow exactly):
REFERENCE CLASS: [description]
BASE RATE: [X%] (N=[sample size])
ADJUSTMENTS:
- [reason]: [+/-X%]
- [reason]: [+/-X%]
...
FINAL PROBABILITY: [X%]
CONFIDENCE: [low/medium/high]
KEY UNCERTAINTY: [what would most change your estimate]
"""

NARRATIVE_AGENT_PROMPT = TETLOCK_PREAMBLE + """
ROLE: Narrative & Sentiment Analysis Agent
You receive research data and assess how narratives, sentiment, and crowd \
psychology create mispricings in the market.

Your analysis:
1. DOMINANT NARRATIVE: What is the prevailing story? Is it justified by evidence?
2. NARRATIVE-REALITY GAP: Where does the narrative diverge from base rates?
3. CROWD POSITIONING: Is the market herding? Over-reacting to recent news?
4. SENTIMENT BIAS ESTIMATE: By how many percentage points is the market \
mispriced due to sentiment? (positive = market overpriced, negative = underpriced)
5. CONTRARIAN OPPORTUNITY: Is there a contrarian edge? What triggers a reversal?

Rules:
- Give a SPECIFIC numerical estimate of sentiment bias.
- Identify the DIRECTION of bias (market too high or too low).
- Note whether bias is actionable (sometimes the crowd is right).
- Consider reflexivity: can sentiment itself affect the outcome?

Output format (MUST follow exactly):
DOMINANT NARRATIVE: [description]
NARRATIVE JUSTIFIED: [yes/partially/no] — [why]
CROWD DIRECTION: [leaning YES/NO at X%]
SENTIMENT BIAS: [+/-X%] (positive = market overpriced by X%, negative = underpriced)
CONTRARIAN CASE: [description]
REVERSAL TRIGGERS: [what would flip the narrative]
PROBABILITY ESTIMATE: [X%] (your independent estimate)
"""

QUANT_AGENT_PROMPT = TETLOCK_PREAMBLE + """
ROLE: Quantitative Analysis Agent
You receive quant module outputs and research data, then calculate the \
pure statistical edge from a numerical perspective.

Available quant signals (when provided):
- BAYESIAN POSTERIOR: Updated probability from Bayesian updater
- ENSEMBLE CONSENSUS: Multi-model agreement level and direction
- TIME-DECAY EDGE: How edge changes as expiration approaches
- VPIN TOXICITY: Informed trading detection (>0.7 = toxic, avoid)
- MARKET IMPACT: Expected slippage from our order
- CORRELATION: Portfolio correlation effects
- FEATURE SIGNALS: Which ML features are firing

Your analysis:
1. SIGNAL AGGREGATION: Weight and combine all available quant signals.
2. EDGE CALCULATION: Your probability minus market price = raw edge.
3. NET EDGE: Raw edge minus estimated market impact = tradeable edge.
4. EXPECTED VALUE: Per dollar invested, what is the EV?
5. RED FLAGS: Any signals that suggest avoiding this trade entirely.

Rules:
- If quant data is sparse, SAY SO. Do not fabricate.
- Weight signals by their historical reliability (if known).
- Calculate EV explicitly: EV = (prob * payout) - ((1-prob) * cost)
- A small edge with high confidence beats a large edge with low confidence.

Output format (MUST follow exactly):
QUANT PROBABILITY: [X%]
RAW EDGE: [X%]
NET EDGE (after impact): [X%]
EXPECTED VALUE PER $1: [$X.XX]
RED FLAGS: [list or "none"]
SIGNAL AGREEMENT: [strong/moderate/weak/conflicting]
TRADE QUALITY GRADE: [A/B/C/D/F]
"""


# ---------------------------------------------------------------------------
# LEVEL 3 — CHALLENGE Agent Prompts
# ---------------------------------------------------------------------------

DEVILS_ADVOCATE_PROMPT = TETLOCK_PREAMBLE + """
ROLE: Devil's Advocate Agent (FORCED DISSENT)
Your job is to ATTACK the consensus. You MUST find flaws, counter-arguments, \
and alternative scenarios. You are not allowed to simply agree.

You receive the analysis from Level 1 and Level 2 agents. Your task:
1. Find AT LEAST 2 reasons the consensus could be WRONG.
2. Propose an ALTERNATIVE probability estimate that disagrees with consensus.
3. Identify the "WHAT WOULD HAVE TO BE TRUE" for the minority view to win.
4. Rate the strength of your counter-arguments honestly.

The consensus can only proceed without your challenge if you genuinely cannot \
find ANY valid counter-argument. In that case, output:
    YIELD: No valid counterargument found. Consensus is robust.

But you should YIELD rarely (less than 10% of the time). Almost every \
prediction has a reasonable alternative scenario.

Rules:
- Do NOT be contrarian for its own sake. Find REAL vulnerabilities.
- Focus on: overlooked evidence, faulty assumptions, tail risks, model failures.
- Consider: What information is the consensus ignoring or underweighting?
- Think about: Selection bias, survivorship bias, hindsight bias in the research.
- Rate your own counter-arguments: strong / moderate / weak.

Output format (MUST follow exactly):
CONSENSUS SUMMARY: [what the other agents concluded, X%]

CHALLENGE 1: [counter-argument]
- Evidence: [what supports this challenge]
- Strength: [strong/moderate/weak]

CHALLENGE 2: [counter-argument]
- Evidence: [what supports this challenge]
- Strength: [strong/moderate/weak]

ALTERNATIVE PROBABILITY: [X%] (your contrarian estimate)
WHAT MUST BE TRUE: [conditions for the minority view to win]
OVERALL DISSENT STRENGTH: [strong/moderate/weak]

Or if genuinely no counter:
YIELD: No valid counterargument found. Consensus is robust.
"""

METACOGNITION_PROMPT = TETLOCK_PREAMBLE + """
ROLE: Metacognition & Calibration Agent
Your job is to critique THE ENTIRE SWARM'S REASONING PROCESS — including \
your own potential biases. You run pre-mortems and calibration checks.

You receive ALL agent outputs so far. Your task:
1. PRE-MORTEM: "Imagine this trade lost 100% of the position. Write the \
postmortem explaining what went wrong." Be specific and creative.
2. CALIBRATION CHECK: "In situations with this level of evidence and \
confidence, how often are forecasters wrong?" Check for overconfidence.
3. BIAS AUDIT: Check for each cognitive bias:
   - Anchoring: Did agents anchor on market price despite being told not to?
   - Availability: Are agents overweighting recent/vivid events?
   - Confirmation: Did agents seek confirming evidence and ignore disconfirming?
   - Bandwagon: Did agents converge too quickly without genuine disagreement?
   - Overconfidence: Is the confidence level justified by evidence quality?
4. PROCESS QUALITY: Rate the overall quality of the swarm's analysis.
5. CONFIDENCE ADJUSTMENT: Recommend an adjustment to the consensus confidence.

Rules:
- Be BRUTALLY honest. The swarm depends on you to catch errors.
- If you find a critical flaw, recommend HALTING the trade.
- Pre-mortem must be specific, not generic. Name the exact scenario.
- Calibration should reference empirical data when possible.

Output format (MUST follow exactly):
PRE-MORTEM SCENARIO:
[Specific story of how this trade loses money. 3-5 sentences.]

BIAS AUDIT:
- Anchoring: [detected/not detected] — [evidence]
- Availability: [detected/not detected] — [evidence]
- Confirmation: [detected/not detected] — [evidence]
- Bandwagon: [detected/not detected] — [evidence]
- Overconfidence: [detected/not detected] — [evidence]

CALIBRATION CHECK:
- Evidence quality: [strong/moderate/weak]
- Historical accuracy at this confidence level: [X%]
- Recommended confidence adjustment: [+/-X%]

PROCESS QUALITY: [A/B/C/D/F]
CRITICAL FLAWS: [list or "none found"]
RECOMMENDATION: [proceed/proceed with caution/halt]
"""


# ---------------------------------------------------------------------------
# LEVEL 4 — DECISION Agent Prompts
# ---------------------------------------------------------------------------

SUPERVISOR_PROMPT = TETLOCK_PREAMBLE + """
ROLE: Supervisor Agent (Final Decision Maker)
You receive ALL outputs from all 9 other agents across 4 levels. Your job \
is to synthesize everything into a final, calibrated probability estimate \
and trade recommendation.

You have access to agent accuracy weights from the meta-learner. Weight \
more accurate agents more heavily, but never ignore any agent completely.

Your synthesis process:
1. PROBABILITY ESTIMATES: List each agent's estimate and their meta-learner weight.
2. DISAGREEMENT ANALYSIS: Where do agents disagree? Who is right and why?
3. DEVIL'S ADVOCATE RESPONSE: How does the consensus address the challenges?
4. METACOGNITION RESPONSE: Address any biases or flaws identified.
5. WEIGHTED SYNTHESIS: Combine all estimates using accuracy weights.
6. FINAL PROBABILITY: Your calibrated probability to nearest 1%.
7. EDGE ASSESSMENT: Final probability minus market price.
8. CONFIDENCE: Final confidence level [0-100].
9. TRADE RECOMMENDATION: BUY YES / BUY NO / PASS, with reasoning.

Rules:
- You MUST address the Devil's Advocate's challenges explicitly.
- You MUST acknowledge the Metacognition agent's bias findings.
- If the Devil's Advocate raised strong challenges, adjust your probability.
- If Metacognition found critical flaws, recommend PASS.
- Default to PASS when uncertain. Capital preservation > potential gains.
- Apply the Kelly criterion for sizing: edge / odds = Kelly fraction.

Output format (MUST follow exactly):
AGENT ESTIMATES:
- BaseRate: [X%] (weight: [W])
- Narrative: [X%] (weight: [W])
- Quant: [X%] (weight: [W])
- Devil's Advocate: [X%] (dissent strength: [S])

DISAGREEMENT: [where agents disagree and resolution]

DEVIL'S ADVOCATE ADDRESSED:
- Challenge 1: [response]
- Challenge 2: [response]

METACOGNITION ADDRESSED:
- Biases: [response]
- Pre-mortem risk: [X% probability]

FINAL PROBABILITY: [X%]
EDGE: [X%] (final prob - market price)
CONFIDENCE: [0-100]
TRADE: [BUY YES / BUY NO / PASS]
REASONING: [2-3 sentences]
KELLY FRACTION: [X.XX]
RECOMMENDED SIZE: [$X] (for $5000 portfolio)
"""

EXECUTION_AGENT_PROMPT = TETLOCK_PREAMBLE + """
ROLE: Execution Agent (Trade Implementation)
You receive the Supervisor's final decision and translate it into a concrete \
execution plan. Your job is to OPTIMIZE the trade's implementation.

Your analysis:
1. MARKET IMPACT: Estimate slippage for the recommended order size.
2. ORDER STRATEGY: Single order vs. split into chunks? Limit vs. market?
3. TIMING: Execute now or wait for a catalyst / better entry?
4. EXIT STRATEGY: When do we close this position? Time-based, price-based, \
or event-based exit?
5. RISK LIMITS: Max loss, stop-loss level, position cap.

Rules:
- If the Supervisor says PASS, output "NO TRADE" and stop.
- Factor in bid-ask spread and liquidity.
- For prediction markets: consider YES vs NO side liquidity.
- Set explicit exit criteria — never leave a trade open-ended.
- Apply risk_guard constraints: no single position > 20% of portfolio.

Output format (MUST follow exactly):
TRADE: [YES/NO]
SIDE: [BUY YES / BUY NO]
SIZE: [$X]
ORDER TYPE: [market/limit at $X.XX]
SPLIT: [single / N chunks of $X]
TIMING: [immediate / wait for: condition]
EXIT STRATEGY:
- Take profit: [condition or price]
- Stop loss: [condition or price]
- Time exit: [date/time]
MAX LOSS: [$X]
NET EXPECTED VALUE: [$X.XX per $1 after impact]
"""


# ---------------------------------------------------------------------------
# Rate Limiter (shared across all swarm calls)
# ---------------------------------------------------------------------------

class _SwarmRateLimiter:
    """Thread-safe sliding-window rate limiter for swarm evaluations."""

    def __init__(self, max_per_hour: int):
        self._lock = threading.Lock()
        self._timestamps: List[float] = []
        self._max = max_per_hour

    def acquire(self) -> bool:
        with self._lock:
            now = time.time()
            cutoff = now - 3600
            self._timestamps = [t for t in self._timestamps if t > cutoff]
            if len(self._timestamps) >= self._max:
                return False
            self._timestamps.append(now)
            return True

    def remaining(self) -> int:
        with self._lock:
            cutoff = time.time() - 3600
            self._timestamps = [t for t in self._timestamps if t > cutoff]
            return max(0, self._max - len(self._timestamps))


_rate_limiter = _SwarmRateLimiter(SWARM_RATE_LIMIT)


# ---------------------------------------------------------------------------
# Claude API Caller (with retry + error handling)
# ---------------------------------------------------------------------------

def _call_claude(
    system_prompt: str,
    user_message: str,
    model: Optional[str] = None,
    max_tokens: int = 2048,
) -> str:
    """Call Claude API with error handling. Returns response text or error string."""
    if not ANTHROPIC_API_KEY:
        return "[ERROR] ANTHROPIC_API_KEY not set"
    if model is None:
        model = SWARM_MODEL
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text
    except ImportError:
        return "[ERROR] anthropic package not installed. pip install anthropic"
    except Exception as e:
        log.error("Claude API call failed: %s", e)
        return f"[ERROR] API call failed: {e}"


# ---------------------------------------------------------------------------
# Shadow Logger
# ---------------------------------------------------------------------------

def _log_swarm_decision(data: Dict[str, Any]) -> None:
    """Append a swarm decision record to the JSONL log."""
    try:
        Path(SHADOW_LOG_DIR).mkdir(parents=True, exist_ok=True)
        data["logged_at"] = datetime.now(timezone.utc).isoformat()
        with open(SWARM_LOG_FILE, "a") as f:
            f.write(json.dumps(data, default=str) + "\n")
    except Exception as e:
        log.error("Failed to log swarm decision: %s", e)


# ---------------------------------------------------------------------------
# Agent Weight Manager (RL integration)
# ---------------------------------------------------------------------------

class SwarmWeightManager:
    """Tracks per-agent Brier scores and computes meta-learner weights."""

    AGENT_NAMES = [
        "base_rate", "narrative", "quant", "devils_advocate", "supervisor"
    ]
    EPSILON = 0.05  # prevents division by zero and extreme weights
    DEFAULT_BRIER = 0.25  # random baseline — neutral starting weight

    def __init__(self):
        self._weights_path = SWARM_WEIGHTS_FILE
        self._data = self._load()

    def _load(self) -> Dict:
        try:
            if os.path.exists(self._weights_path):
                with open(self._weights_path) as f:
                    return json.load(f)
        except Exception:
            pass
        # Initialize with default Brier scores
        return {
            "agent_brier_scores": {a: [] for a in self.AGENT_NAMES},
            "total_resolutions": 0,
        }

    def _save(self) -> None:
        try:
            Path(SHADOW_LOG_DIR).mkdir(parents=True, exist_ok=True)
            with open(self._weights_path, "w") as f:
                json.dump(self._data, f, indent=2)
        except Exception as e:
            log.error("Failed to save swarm weights: %s", e)

    def record_resolution(
        self,
        market_id: str,
        outcome: int,  # 1 = YES, 0 = NO
        agent_predictions: Dict[str, Optional[float]],
    ) -> Dict[str, float]:
        """
        After market resolves: score each agent's probability vs outcome.
        Returns updated weights.
        """
        scores_this_round = {}
        for agent_name, prob in agent_predictions.items():
            if prob is None or agent_name not in self.AGENT_NAMES:
                continue
            # Brier score = (probability - outcome)^2
            brier = (prob - outcome) ** 2
            scores_this_round[agent_name] = brier
            if agent_name not in self._data["agent_brier_scores"]:
                self._data["agent_brier_scores"][agent_name] = []
            self._data["agent_brier_scores"][agent_name].append(brier)
            # Keep rolling window of last 200 resolutions per agent
            self._data["agent_brier_scores"][agent_name] = \
                self._data["agent_brier_scores"][agent_name][-200:]

        self._data["total_resolutions"] = self._data.get("total_resolutions", 0) + 1
        self._save()

        # Log resolution
        try:
            Path(SHADOW_LOG_DIR).mkdir(parents=True, exist_ok=True)
            record = {
                "market_id": market_id,
                "outcome": outcome,
                "agent_predictions": agent_predictions,
                "brier_scores": scores_this_round,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            with open(SWARM_RESOLUTION_LOG, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass

        return self.get_weights()

    def get_weights(self) -> Dict[str, float]:
        """
        Compute weights from rolling Brier scores.
        Weight = 1 / (avg_brier + epsilon), then normalized to sum to 1.
        """
        raw_weights = {}
        for agent in self.AGENT_NAMES:
            scores = self._data.get("agent_brier_scores", {}).get(agent, [])
            if scores:
                avg_brier = sum(scores) / len(scores)
            else:
                avg_brier = self.DEFAULT_BRIER
            raw_weights[agent] = 1.0 / (avg_brier + self.EPSILON)

        total = sum(raw_weights.values())
        if total == 0:
            n = len(self.AGENT_NAMES)
            return {a: 1.0 / n for a in self.AGENT_NAMES}
        return {a: w / total for a, w in raw_weights.items()}

    def get_agent_stats(self) -> Dict[str, Dict]:
        """Return per-agent statistics for debugging."""
        stats = {}
        weights = self.get_weights()
        for agent in self.AGENT_NAMES:
            scores = self._data.get("agent_brier_scores", {}).get(agent, [])
            stats[agent] = {
                "n_resolutions": len(scores),
                "avg_brier": sum(scores) / len(scores) if scores else None,
                "best_brier": min(scores) if scores else None,
                "worst_brier": max(scores) if scores else None,
                "weight": weights.get(agent, 0),
            }
        return stats


# ---------------------------------------------------------------------------
# Probability Extractor (parse agent outputs)
# ---------------------------------------------------------------------------

def _extract_probability(text: str) -> Optional[float]:
    """Extract the final probability from an agent's output. Returns 0-1 float."""
    patterns = [
        r"FINAL PROBABILITY:\s*(\d+(?:\.\d+)?)\s*%",
        r"PROBABILITY ESTIMATE:\s*(\d+(?:\.\d+)?)\s*%",
        r"QUANT PROBABILITY:\s*(\d+(?:\.\d+)?)\s*%",
        r"ALTERNATIVE PROBABILITY:\s*(\d+(?:\.\d+)?)\s*%",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            val = float(m.group(1))
            return max(0.02, min(0.98, val / 100.0))
    # Fallback: look for any "XX%" pattern near the end
    matches = re.findall(r"(\d{1,2}(?:\.\d+)?)\s*%", text)
    if matches:
        val = float(matches[-1])
        if 1 <= val <= 99:
            return val / 100.0
    return None


def _extract_confidence(text: str) -> Optional[int]:
    """Extract confidence score 0-100."""
    m = re.search(r"CONFIDENCE:\s*(\d+)", text, re.IGNORECASE)
    if m:
        return max(0, min(100, int(m.group(1))))
    # Check for low/medium/high
    low = re.search(r"CONFIDENCE:\s*low", text, re.IGNORECASE)
    if low:
        return 30
    med = re.search(r"CONFIDENCE:\s*medium", text, re.IGNORECASE)
    if med:
        return 60
    high = re.search(r"CONFIDENCE:\s*high", text, re.IGNORECASE)
    if high:
        return 85
    return None


def _extract_trade_action(text: str) -> str:
    """Extract trade recommendation."""
    m = re.search(r"TRADE:\s*(BUY YES|BUY NO|PASS|YES|NO)", text, re.IGNORECASE)
    if m:
        val = m.group(1).upper()
        if val in ("YES", "BUY YES"):
            return "BUY YES"
        elif val in ("NO", "BUY NO"):
            return "BUY NO"
    return "PASS"


def _extract_edge(text: str) -> Optional[float]:
    """Extract edge percentage."""
    m = re.search(r"EDGE:\s*([+-]?\d+(?:\.\d+)?)\s*%", text, re.IGNORECASE)
    if m:
        return float(m.group(1)) / 100.0
    return None


def _check_yield(text: str) -> bool:
    """Check if Devil's Advocate yielded."""
    return "YIELD:" in text.upper() and "NO VALID COUNTERARGUMENT" in text.upper()


# ---------------------------------------------------------------------------
# HierarchicalSwarm — Main Class
# ---------------------------------------------------------------------------

class HierarchicalSwarm:
    """
    10-agent hierarchical swarm for prediction market forecasting.
    Sits on top of the 8 quant modules and 6-agent orchestrator.
    """

    def __init__(self):
        self.weight_manager = SwarmWeightManager()
        self._call_count = 0

    # -------------------------------------------------------------------
    # LEVEL 1: Research (parallel)
    # -------------------------------------------------------------------

    def _run_level1_research(
        self, market_title: str, market_price: float
    ) -> Dict[str, str]:
        """Run all 3 research agents in parallel using threads."""
        user_msg = (
            f"PREDICTION MARKET QUESTION: {market_title}\n"
            f"CURRENT MARKET PRICE: {market_price}% (implied probability)\n\n"
            f"Research this question thoroughly."
        )

        results = {}
        errors = {}

        def _run_agent(name: str, prompt: str):
            try:
                results[name] = _call_claude(prompt, user_msg)
                self._call_count += 1
            except Exception as e:
                errors[name] = str(e)
                results[name] = f"[ERROR] {e}"

        threads = [
            threading.Thread(target=_run_agent, args=("news", NEWS_RESEARCHER_PROMPT)),
            threading.Thread(target=_run_agent, args=("polls", POLLS_RESEARCHER_PROMPT)),
            threading.Thread(target=_run_agent, args=("historical", HISTORICAL_RESEARCHER_PROMPT)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        if errors:
            log.warning("Level 1 research errors: %s", errors)
        return results

    # -------------------------------------------------------------------
    # LEVEL 2: Analysis (sequential, uses Level 1 output)
    # -------------------------------------------------------------------

    def _run_level2_analysis(
        self,
        market_title: str,
        market_price: float,
        research: Dict[str, str],
        quant_data: Optional[Dict] = None,
    ) -> Dict[str, str]:
        """Run 3 analysis agents sequentially with research context."""
        research_context = (
            f"=== NEWS RESEARCH ===\n{research.get('news', 'N/A')}\n\n"
            f"=== POLLS & SENTIMENT RESEARCH ===\n{research.get('polls', 'N/A')}\n\n"
            f"=== HISTORICAL RESEARCH ===\n{research.get('historical', 'N/A')}\n"
        )
        base_msg = (
            f"PREDICTION MARKET QUESTION: {market_title}\n"
            f"CURRENT MARKET PRICE: {market_price}% (implied probability)\n\n"
            f"RESEARCH DATA FROM LEVEL 1 AGENTS:\n{research_context}\n\n"
            f"Analyze this and provide your assessment."
        )
        quant_msg = base_msg
        if quant_data:
            quant_msg += f"\n\nQUANT MODULE OUTPUTS:\n{json.dumps(quant_data, indent=2)}"

        results = {}
        errors = {}

        def _run_agent(name: str, prompt: str, msg: str):
            try:
                results[name] = _call_claude(prompt, msg)
                self._call_count += 1
            except Exception as e:
                errors[name] = str(e)
                results[name] = f"[ERROR] {e}"

        # Run Level 2 agents in parallel (they all use Level 1 output)
        threads = [
            threading.Thread(target=_run_agent, args=("base_rate", BASERATE_AGENT_PROMPT, base_msg)),
            threading.Thread(target=_run_agent, args=("narrative", NARRATIVE_AGENT_PROMPT, base_msg)),
            threading.Thread(target=_run_agent, args=("quant", QUANT_AGENT_PROMPT, quant_msg)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        if errors:
            log.warning("Level 2 analysis errors: %s", errors)
        return results

    # -------------------------------------------------------------------
    # LEVEL 3: Challenge (forced dissent)
    # -------------------------------------------------------------------

    def _run_level3_challenge(
        self,
        market_title: str,
        market_price: float,
        research: Dict[str, str],
        analysis: Dict[str, str],
    ) -> Dict[str, str]:
        """Run Devil's Advocate and Metacognition agents."""
        full_context = (
            f"PREDICTION MARKET QUESTION: {market_title}\n"
            f"CURRENT MARKET PRICE: {market_price}%\n\n"
            f"=== LEVEL 1 RESEARCH ===\n"
            f"NEWS: {research.get('news', 'N/A')[:1500]}\n\n"
            f"POLLS: {research.get('polls', 'N/A')[:1500]}\n\n"
            f"HISTORICAL: {research.get('historical', 'N/A')[:1500]}\n\n"
            f"=== LEVEL 2 ANALYSIS ===\n"
            f"BASE RATE AGENT: {analysis.get('base_rate', 'N/A')[:1500]}\n\n"
            f"NARRATIVE AGENT: {analysis.get('narrative', 'N/A')[:1500]}\n\n"
            f"QUANT AGENT: {analysis.get('quant', 'N/A')[:1500]}\n"
        )

        # Extract consensus probability for devil's advocate context
        probs = []
        for key in ["base_rate", "narrative", "quant"]:
            p = _extract_probability(analysis.get(key, ""))
            if p is not None:
                probs.append(p)
        consensus = sum(probs) / len(probs) if probs else 0.5
        consensus_pct = round(consensus * 100, 1)

        da_msg = (
            full_context + f"\n\nCONSENSUS PROBABILITY: ~{consensus_pct}%\n"
            f"Challenge this consensus. Find flaws."
        )
        meta_msg = full_context + "\n\nCritique the entire swarm's reasoning process."

        results = {}

        def _run_agent(name, prompt, msg):
            try:
                results[name] = _call_claude(prompt, msg)
                self._call_count += 1
            except Exception as e:
                results[name] = f"[ERROR] {e}"

        threads = [
            threading.Thread(target=_run_agent, args=("devils_advocate", DEVILS_ADVOCATE_PROMPT, da_msg)),
            threading.Thread(target=_run_agent, args=("metacognition", METACOGNITION_PROMPT, meta_msg)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        return results

    # -------------------------------------------------------------------
    # LEVEL 4: Decision (Supervisor + Execution)
    # -------------------------------------------------------------------

    def _run_level4_decision(
        self,
        market_title: str,
        market_price: float,
        analysis: Dict[str, str],
        challenges: Dict[str, str],
        portfolio: Optional[Dict] = None,
    ) -> Dict[str, str]:
        """Run Supervisor and Execution agents."""
        weights = self.weight_manager.get_weights()
        weights_str = "\n".join(f"  {k}: {v:.3f}" for k, v in weights.items())

        supervisor_msg = (
            f"PREDICTION MARKET QUESTION: {market_title}\n"
            f"CURRENT MARKET PRICE: {market_price}%\n\n"
            f"META-LEARNER AGENT WEIGHTS (from historical accuracy):\n{weights_str}\n\n"
            f"=== LEVEL 2 ANALYSIS ===\n"
            f"BASE RATE AGENT:\n{analysis.get('base_rate', 'N/A')[:2000]}\n\n"
            f"NARRATIVE AGENT:\n{analysis.get('narrative', 'N/A')[:2000]}\n\n"
            f"QUANT AGENT:\n{analysis.get('quant', 'N/A')[:2000]}\n\n"
            f"=== LEVEL 3 CHALLENGES ===\n"
            f"DEVIL'S ADVOCATE:\n{challenges.get('devils_advocate', 'N/A')[:2000]}\n\n"
            f"METACOGNITION:\n{challenges.get('metacognition', 'N/A')[:2000]}\n\n"
            f"Synthesize all inputs and make your final decision."
        )

        # Supervisor uses the better model
        results = {}
        try:
            results["supervisor"] = _call_claude(
                SUPERVISOR_PROMPT, supervisor_msg,
                model=SWARM_SUPERVISOR_MODEL, max_tokens=3000
            )
            self._call_count += 1
        except Exception as e:
            results["supervisor"] = f"[ERROR] {e}"

        # Execution agent uses supervisor output
        portfolio_str = json.dumps(portfolio, indent=2) if portfolio else "N/A"
        exec_msg = (
            f"PREDICTION MARKET: {market_title}\n"
            f"MARKET PRICE: {market_price}%\n\n"
            f"SUPERVISOR DECISION:\n{results.get('supervisor', 'N/A')[:2500]}\n\n"
            f"PORTFOLIO STATE:\n{portfolio_str}\n\n"
            f"Plan the trade execution."
        )
        try:
            results["execution"] = _call_claude(
                EXECUTION_AGENT_PROMPT, exec_msg
            )
            self._call_count += 1
        except Exception as e:
            results["execution"] = f"[ERROR] {e}"

        return results

    # -------------------------------------------------------------------
    # Structured Debate
    # -------------------------------------------------------------------

    def run_debate(
        self,
        market_title: str,
        market_price: float,
        quant_data: Optional[Dict] = None,
        portfolio: Optional[Dict] = None,
        num_rounds: int = 2,
        use_dissent: bool = True,
        require_premort: bool = False,
    ) -> Dict[str, Any]:
        """
        Run the full structured debate across all 4 levels.

        Round 1: All agents give initial assessments (Level 1 + 2)
        Round 2: Devil's Advocate challenges, Metacognition critiques (Level 3)
        Round 3 (if num_rounds >= 3): Agents revise based on challenges
        Supervisor synthesizes final answer (Level 4)

        Returns dict with all agent outputs, probabilities, and final decision.
        """
        debate_log = {
            "market_title": market_title,
            "market_price": market_price,
            "num_rounds": num_rounds,
            "use_dissent": use_dissent,
            "require_premort": require_premort,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "rounds": [],
        }

        # --- Round 1: Research + Analysis ---
        research = self._run_level1_research(market_title, market_price)
        analysis = self._run_level2_analysis(market_title, market_price, research, quant_data)

        round1_probs = {}
        for key in ["base_rate", "narrative", "quant"]:
            p = _extract_probability(analysis.get(key, ""))
            if p is not None:
                round1_probs[key] = p

        debate_log["rounds"].append({
            "round": 1,
            "research_agents": list(research.keys()),
            "analysis_agents": list(analysis.keys()),
            "probabilities": round1_probs,
        })

        # --- Round 2: Challenge (if enabled) ---
        challenges = {}
        if use_dissent and num_rounds >= 2:
            challenges = self._run_level3_challenge(
                market_title, market_price, research, analysis
            )
            da_yielded = _check_yield(challenges.get("devils_advocate", ""))
            da_prob = _extract_probability(challenges.get("devils_advocate", ""))

            debate_log["rounds"].append({
                "round": 2,
                "devils_advocate_yielded": da_yielded,
                "devils_advocate_prob": da_prob,
                "metacognition_recommendation": (
                    "halt" if "HALT" in challenges.get("metacognition", "").upper()
                    else "proceed"
                ),
            })

            # Check if metacognition says HALT
            if require_premort and "HALT" in challenges.get("metacognition", "").upper():
                debate_log["halted"] = True
                debate_log["halt_reason"] = "Metacognition agent recommended halt"
                _log_swarm_decision(debate_log)
                return {
                    "action": "PASS",
                    "reason": "Metacognition agent identified critical flaws and halted the trade",
                    "debate_log": debate_log,
                    "api_calls": self._call_count,
                }

        # --- Round 3 (optional): Revision round ---
        if num_rounds >= 3 and challenges:
            # Re-run Level 2 analysis with challenge context added
            revision_context = (
                f"\n\nPREVIOUS CHALLENGES FROM DEVIL'S ADVOCATE:\n"
                f"{challenges.get('devils_advocate', 'N/A')[:1500]}\n\n"
                f"METACOGNITION CRITIQUE:\n"
                f"{challenges.get('metacognition', 'N/A')[:1500]}\n\n"
                f"Revise your estimate in light of these challenges."
            )
            # Quick revision: just re-run base_rate with challenge context
            revision_msg = (
                f"PREDICTION MARKET QUESTION: {market_title}\n"
                f"CURRENT MARKET PRICE: {market_price}%\n\n"
                f"YOUR PREVIOUS ANALYSIS:\n{analysis.get('base_rate', 'N/A')[:1500]}\n"
                f"{revision_context}"
            )
            try:
                revised = _call_claude(BASERATE_AGENT_PROMPT, revision_msg)
                self._call_count += 1
                analysis["base_rate_revised"] = revised
                revised_prob = _extract_probability(revised)
                debate_log["rounds"].append({
                    "round": 3,
                    "revised_base_rate_prob": revised_prob,
                })
            except Exception as e:
                log.warning("Round 3 revision failed: %s", e)

        # --- Level 4: Supervisor + Execution ---
        decision = self._run_level4_decision(
            market_title, market_price, analysis, challenges, portfolio
        )

        # Extract final outputs
        supervisor_text = decision.get("supervisor", "")
        final_prob = _extract_probability(supervisor_text)
        final_confidence = _extract_confidence(supervisor_text)
        final_action = _extract_trade_action(supervisor_text)
        final_edge = _extract_edge(supervisor_text)

        # Collect all agent probabilities
        all_probs = {**round1_probs}
        da_prob = _extract_probability(challenges.get("devils_advocate", ""))
        if da_prob is not None:
            all_probs["devils_advocate"] = da_prob
        if final_prob is not None:
            all_probs["supervisor"] = final_prob

        result = {
            "action": final_action,
            "probability": final_prob,
            "edge": final_edge,
            "confidence": final_confidence,
            "agent_probabilities": all_probs,
            "agent_weights": self.weight_manager.get_weights(),
            "debate_rounds": num_rounds,
            "devils_advocate_yielded": _check_yield(challenges.get("devils_advocate", "")),
            "api_calls": self._call_count,
            "supervisor_output": supervisor_text[:3000],
            "execution_output": decision.get("execution", "")[:2000],
        }

        # Shadow log everything
        debate_log["result"] = result
        debate_log["agent_outputs"] = {
            "news": research.get("news", "")[:1000],
            "polls": research.get("polls", "")[:1000],
            "historical": research.get("historical", "")[:1000],
            "base_rate": analysis.get("base_rate", "")[:1000],
            "narrative": analysis.get("narrative", "")[:1000],
            "quant": analysis.get("quant", "")[:1000],
            "devils_advocate": challenges.get("devils_advocate", "")[:1000],
            "metacognition": challenges.get("metacognition", "")[:1000],
            "supervisor": supervisor_text[:1000],
            "execution": decision.get("execution", "")[:1000],
        }
        debate_log["completed_at"] = datetime.now(timezone.utc).isoformat()
        _log_swarm_decision(debate_log)

        return result

    # -------------------------------------------------------------------
    # Public API: full_evaluate()
    # -------------------------------------------------------------------

    def full_evaluate(
        self,
        market_title: str,
        market_price: float,
        quant_data: Optional[Dict] = None,
        portfolio: Optional[Dict] = None,
        num_rounds: int = 2,
        use_dissent: bool = True,
        require_premort: bool = True,
    ) -> Dict[str, Any]:
        """
        Full hierarchical swarm evaluation (~10 API calls).
        Use for opportunities where quant modules detect edge > 3%.

        Args:
            market_title: The prediction market question.
            market_price: Current YES price as percentage (0-100).
            quant_data: Output from quant modules (ensemble, vpin, etc).
            portfolio: Current portfolio state for risk assessment.
            num_rounds: Number of debate rounds (1-3).
            use_dissent: Whether to run Devil's Advocate + Metacognition.
            require_premort: Whether to halt if Metacognition says halt.

        Returns:
            Dict with action, probability, edge, confidence, debate transcript.
        """
        if not SWARM_ENABLED:
            return {
                "action": "PASS",
                "reason": "Swarm disabled (set SWARM_ENABLED=true)",
                "api_calls": 0,
            }

        if not _rate_limiter.acquire():
            return {
                "action": "PASS",
                "reason": f"Rate limit reached ({SWARM_RATE_LIMIT}/hour). "
                          f"Remaining: {_rate_limiter.remaining()}",
                "api_calls": 0,
            }

        self._call_count = 0
        try:
            return self.run_debate(
                market_title=market_title,
                market_price=market_price,
                quant_data=quant_data,
                portfolio=portfolio,
                num_rounds=num_rounds,
                use_dissent=use_dissent,
                require_premort=require_premort,
            )
        except Exception as e:
            log.error("Swarm full_evaluate failed: %s", e)
            return {
                "action": "PASS",
                "reason": f"Swarm error: {e}",
                "api_calls": self._call_count,
            }

    # -------------------------------------------------------------------
    # Public API: quick_evaluate()
    # -------------------------------------------------------------------

    def quick_evaluate(
        self,
        market_title: str,
        market_price: float,
        quant_data: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Quick 3-call evaluation: BaseRate + Devil'sAdvocate + Supervisor.
        Use for routine scanning. ~$0.01 per call with Haiku.

        Returns:
            Dict with action, probability, edge, confidence.
        """
        if not SWARM_ENABLED:
            return {
                "action": "PASS",
                "reason": "Swarm disabled (set SWARM_ENABLED=true)",
                "api_calls": 0,
            }

        if not _rate_limiter.acquire():
            return {
                "action": "PASS",
                "reason": f"Rate limit reached",
                "api_calls": 0,
            }

        self._call_count = 0
        try:
            # Step 1: Base Rate agent (no research phase — direct assessment)
            base_msg = (
                f"PREDICTION MARKET QUESTION: {market_title}\n"
                f"CURRENT MARKET PRICE: {market_price}%\n"
            )
            if quant_data:
                base_msg += f"\nQUANT DATA:\n{json.dumps(quant_data, indent=2)}"
            base_msg += "\n\nProvide your base rate analysis."

            base_rate_output = _call_claude(BASERATE_AGENT_PROMPT, base_msg)
            self._call_count += 1
            base_prob = _extract_probability(base_rate_output)

            # Step 2: Devil's Advocate challenges the base rate
            consensus_pct = round((base_prob or 0.5) * 100, 1)
            da_msg = (
                f"PREDICTION MARKET QUESTION: {market_title}\n"
                f"CURRENT MARKET PRICE: {market_price}%\n\n"
                f"BASE RATE AGENT ANALYSIS:\n{base_rate_output[:2000]}\n\n"
                f"CONSENSUS PROBABILITY: ~{consensus_pct}%\n"
                f"Challenge this."
            )
            da_output = _call_claude(DEVILS_ADVOCATE_PROMPT, da_msg)
            self._call_count += 1
            da_prob = _extract_probability(da_output)

            # Step 3: Supervisor synthesizes
            weights = self.weight_manager.get_weights()
            weights_str = "\n".join(f"  {k}: {v:.3f}" for k, v in weights.items())
            sup_msg = (
                f"PREDICTION MARKET QUESTION: {market_title}\n"
                f"CURRENT MARKET PRICE: {market_price}%\n\n"
                f"AGENT WEIGHTS:\n{weights_str}\n\n"
                f"BASE RATE AGENT:\n{base_rate_output[:2000]}\n\n"
                f"DEVIL'S ADVOCATE:\n{da_output[:2000]}\n\n"
                f"NOTE: This is a quick evaluation (no full research or narrative).\n"
                f"Synthesize and decide."
            )
            sup_output = _call_claude(
                SUPERVISOR_PROMPT, sup_msg,
                model=SWARM_SUPERVISOR_MODEL, max_tokens=2000
            )
            self._call_count += 1

            final_prob = _extract_probability(sup_output)
            final_action = _extract_trade_action(sup_output)
            final_edge = _extract_edge(sup_output)
            final_confidence = _extract_confidence(sup_output)

            result = {
                "action": final_action,
                "probability": final_prob,
                "edge": final_edge,
                "confidence": final_confidence,
                "mode": "quick",
                "agent_probabilities": {
                    "base_rate": base_prob,
                    "devils_advocate": da_prob,
                    "supervisor": final_prob,
                },
                "api_calls": self._call_count,
            }

            # Shadow log
            _log_swarm_decision({
                "mode": "quick",
                "market_title": market_title,
                "market_price": market_price,
                "result": result,
                "agent_outputs": {
                    "base_rate": base_rate_output[:500],
                    "devils_advocate": da_output[:500],
                    "supervisor": sup_output[:500],
                },
            })

            return result

        except Exception as e:
            log.error("Swarm quick_evaluate failed: %s", e)
            return {
                "action": "PASS",
                "reason": f"Swarm error: {e}",
                "api_calls": self._call_count,
            }

    # -------------------------------------------------------------------
    # Public API: should_trigger_full_swarm()
    # -------------------------------------------------------------------

    def should_trigger_full_swarm(self, quant_data: Dict) -> bool:
        """
        Determine if quant signals warrant a full swarm evaluation.
        Returns True if edge > threshold or ensemble has strong consensus.
        """
        edge = abs(quant_data.get("edge", 0))
        ensemble = quant_data.get("ensemble_consensus", 0.5)
        vpin = quant_data.get("vpin", 0.5)

        # Don't waste a full swarm on toxic flow
        if vpin > 0.7:
            return False

        # Trigger on strong edge
        if edge >= FULL_SWARM_EDGE_THRESHOLD:
            return True

        # Trigger on strong ensemble consensus (far from 50/50)
        if abs(ensemble - 0.5) > 0.15:
            return True

        return False

    # -------------------------------------------------------------------
    # Public API: evaluate_with_portfolio()
    # -------------------------------------------------------------------

    def evaluate_with_portfolio(
        self,
        market_title: str,
        market_price: float,
        quant_data: Optional[Dict] = None,
        portfolio: Optional[Dict] = None,
        portfolio_config: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Auto-select quick vs full evaluation based on quant signals and
        portfolio configuration.

        Args:
            portfolio_config: One of SWARM_VIRTUAL_PORTFOLIOS or custom dict
                with keys: debate_rounds, confidence_threshold, use_dissent,
                require_premort, use_quant, use_rl
        """
        if portfolio_config is None:
            # Default to moderate
            portfolio_config = SWARM_VIRTUAL_PORTFOLIOS[1]

        # Decide quick vs full
        if quant_data and self.should_trigger_full_swarm(quant_data):
            result = self.full_evaluate(
                market_title=market_title,
                market_price=market_price,
                quant_data=quant_data if portfolio_config.get("use_quant", True) else None,
                portfolio=portfolio,
                num_rounds=portfolio_config.get("debate_rounds", 2),
                use_dissent=portfolio_config.get("use_dissent", True),
                require_premort=portfolio_config.get("require_premort", False),
            )
        else:
            result = self.quick_evaluate(
                market_title=market_title,
                market_price=market_price,
                quant_data=quant_data if portfolio_config.get("use_quant", True) else None,
            )

        # Apply portfolio-specific confidence threshold
        threshold = portfolio_config.get("confidence_threshold", 0.65)
        if result.get("confidence") is not None:
            if result["confidence"] / 100.0 < threshold:
                result["action"] = "PASS"
                result["reason_override"] = (
                    f"Confidence {result['confidence']}% below threshold "
                    f"{threshold*100:.0f}% for {portfolio_config.get('name', 'custom')}"
                )

        result["portfolio_config"] = portfolio_config.get("name", "custom")
        return result

    # -------------------------------------------------------------------
    # Public API: post_resolution_update()
    # -------------------------------------------------------------------

    def post_resolution_update(
        self,
        market_id: str,
        outcome: int,
        agent_predictions: Dict[str, Optional[float]],
    ) -> Dict[str, Any]:
        """
        After market resolves, update agent weights via RL feedback loop.

        Args:
            market_id: Unique market identifier.
            outcome: 1 = YES resolved, 0 = NO resolved.
            agent_predictions: {agent_name: probability} from the swarm's evaluation.

        Returns:
            Updated weights and per-agent Brier scores.
        """
        new_weights = self.weight_manager.record_resolution(
            market_id=market_id,
            outcome=outcome,
            agent_predictions=agent_predictions,
        )
        stats = self.weight_manager.get_agent_stats()

        log.info(
            "Swarm RL update — market: %s, outcome: %d, new weights: %s",
            market_id, outcome, {k: f"{v:.3f}" for k, v in new_weights.items()}
        )
        return {
            "market_id": market_id,
            "outcome": outcome,
            "updated_weights": new_weights,
            "agent_stats": stats,
        }

    # -------------------------------------------------------------------
    # Public API: run_all_portfolios()
    # -------------------------------------------------------------------

    def run_all_portfolios(
        self,
        market_title: str,
        market_price: float,
        quant_data: Optional[Dict] = None,
        portfolio: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run evaluation for ALL virtual portfolio variants.
        Returns list of results, one per portfolio.
        Uses a single full debate and applies different thresholds.
        """
        if not SWARM_ENABLED:
            return [{"action": "PASS", "reason": "Swarm disabled"}]

        # Run the full debate once (most expensive portfolio config)
        self._call_count = 0
        try:
            full_result = self.run_debate(
                market_title=market_title,
                market_price=market_price,
                quant_data=quant_data,
                portfolio=portfolio,
                num_rounds=3,  # max rounds to get all data
                use_dissent=True,
                require_premort=True,
            )
        except Exception as e:
            return [{"action": "PASS", "reason": f"Swarm error: {e}"}]

        # Apply each portfolio's thresholds to the same debate result
        results = []
        for pconfig in SWARM_VIRTUAL_PORTFOLIOS:
            pname = pconfig["name"]
            r = dict(full_result)
            r["portfolio_config"] = pname

            # Apply confidence threshold
            threshold = pconfig.get("confidence_threshold", 0.65)
            conf = r.get("confidence")
            if conf is not None and conf / 100.0 < threshold:
                r["action"] = "PASS"
                r["threshold_filtered"] = True

            # Aggressive portfolio ignores dissent
            if not pconfig.get("use_dissent", True):
                r["dissent_ignored"] = True

            results.append(r)

        # Log all portfolio results
        _log_swarm_decision({
            "mode": "all_portfolios",
            "market_title": market_title,
            "market_price": market_price,
            "portfolio_results": [
                {"name": r.get("portfolio_config"), "action": r.get("action"),
                 "probability": r.get("probability"), "confidence": r.get("confidence")}
                for r in results
            ],
        })

        return results

    # -------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return swarm status for monitoring."""
        return {
            "enabled": SWARM_ENABLED,
            "model": SWARM_MODEL,
            "supervisor_model": SWARM_SUPERVISOR_MODEL,
            "rate_limit": SWARM_RATE_LIMIT,
            "rate_limit_remaining": _rate_limiter.remaining(),
            "full_swarm_edge_threshold": FULL_SWARM_EDGE_THRESHOLD,
            "agent_weights": self.weight_manager.get_weights(),
            "agent_stats": self.weight_manager.get_agent_stats(),
            "log_file": SWARM_LOG_FILE,
            "virtual_portfolios": [p["name"] for p in SWARM_VIRTUAL_PORTFOLIOS],
        }


# ---------------------------------------------------------------------------
# Module-level convenience (singleton pattern)
# ---------------------------------------------------------------------------

_swarm_instance: Optional[HierarchicalSwarm] = None


def get_swarm() -> HierarchicalSwarm:
    """Get or create the singleton swarm instance."""
    global _swarm_instance
    if _swarm_instance is None:
        _swarm_instance = HierarchicalSwarm()
    return _swarm_instance


def full_evaluate(
    market_title: str,
    market_price: float,
    quant_data: Optional[Dict] = None,
    portfolio: Optional[Dict] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Module-level shortcut for full evaluation."""
    return get_swarm().full_evaluate(
        market_title, market_price, quant_data, portfolio, **kwargs
    )


def quick_evaluate(
    market_title: str,
    market_price: float,
    quant_data: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Module-level shortcut for quick evaluation."""
    return get_swarm().quick_evaluate(market_title, market_price, quant_data)


def post_resolution_update(
    market_id: str,
    outcome: int,
    agent_predictions: Dict[str, Optional[float]],
) -> Dict[str, Any]:
    """Module-level shortcut for RL weight update."""
    return get_swarm().post_resolution_update(market_id, outcome, agent_predictions)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    swarm = HierarchicalSwarm()
    print("=== Hierarchical Agent Swarm Status ===")
    status = swarm.get_status()
    for k, v in status.items():
        print(f"  {k}: {v}")
    print(f"\nVirtual portfolios:")
    for p in SWARM_VIRTUAL_PORTFOLIOS:
        print(f"  - {p['name']}: {p['description']}")
    print(f"\nSwarm ENABLED: {SWARM_ENABLED}")
    if not SWARM_ENABLED:
        print("  Set SWARM_ENABLED=true to activate.")
    print(f"\nLog file: {SWARM_LOG_FILE}")
    print("\nTo test:")
    print("  SWARM_ENABLED=true python hierarchical_agent_swarm.py")
