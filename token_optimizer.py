"""
token_optimizer.py — API Token Cost Optimizer

Quant Concept:
    Most AI API calls don't need the most expensive model. This module routes
    each call to the cheapest model capable of handling the task, compresses
    prompts to reduce token count, enables prompt caching for repeated system
    prompts, and batches non-real-time work for throughput efficiency.

    Model tiers (Anthropic pricing):
        HAIKU   ($0.25/1M in, $1.25/1M out) — 80% of calls: feature extraction,
                simple classification, data summarization, shadow signal evaluation
        SONNET  ($3.00/1M in, $15.00/1M out) — complex reasoning: multi-step
                analysis, agent orchestration
        OPUS   ($15.00/1M in, $75.00/1M out) — highest-stakes ONLY: live trade
                decisions, strategy changes

    Savings: 60-80% reduction vs naive always-use-Opus approach.
    Prompt caching: up to 90% savings on repeated system prompts and tool defs.

Usage:
    from token_optimizer import TokenOptimizer

    opt = TokenOptimizer()
    result = opt.route_call(
        task_type="SIMPLE",
        prompt="Classify this market signal as bullish/bearish/neutral",
        data_size=500
    )
    print(result)
    # {'recommended_model': 'claude-3-5-haiku-20241022', 'optimized_prompt': '...',
    #  'estimated_cost': 0.00035, 'cache_key': 'sys_abc123'}

    # Batch shadow evaluations
    results = opt.batch_evaluate(prompts_list, batch_size=50)

    # Savings report
    report = opt.get_savings_report()
    print(report)
    # {'total_cost': 12.45, 'naive_cost': 58.20, 'savings_pct': 78.6, ...}
"""

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TaskType(str, Enum):
    SIMPLE = "SIMPLE"
    MODERATE = "MODERATE"
    COMPLEX = "COMPLEX"


class ModelTier(str, Enum):
    HAIKU = "HAIKU"
    SONNET = "SONNET"
    OPUS = "OPUS"


# Model IDs
MODEL_IDS: Dict[ModelTier, str] = {
    ModelTier.HAIKU: "claude-3-5-haiku-20241022",
    ModelTier.SONNET: "claude-sonnet-4-20250514",
    ModelTier.OPUS: "claude-opus-4-20250514",
}

# Pricing per 1M tokens
MODEL_PRICING: Dict[ModelTier, Dict[str, float]] = {
    ModelTier.HAIKU: {"input": 0.25, "output": 1.25},
    ModelTier.SONNET: {"input": 3.00, "output": 15.00},
    ModelTier.OPUS: {"input": 15.00, "output": 75.00},
}

# Cached token pricing (prompt caching discount)
CACHE_WRITE_MULTIPLIER = 1.25   # 25% more to write cache
CACHE_READ_MULTIPLIER = 0.10    # 90% discount on cached reads

# Task type -> default model tier routing
TASK_ROUTING: Dict[TaskType, ModelTier] = {
    TaskType.SIMPLE: ModelTier.HAIKU,
    TaskType.MODERATE: ModelTier.SONNET,
    TaskType.COMPLEX: ModelTier.OPUS,
}

# Simple task keywords — route to Haiku even if not explicitly SIMPLE
SIMPLE_KEYWORDS = {
    "classify", "classification", "extract", "summarize", "summary",
    "tag", "label", "parse", "format", "convert", "count", "filter",
    "shadow", "signal_eval", "feature_extract", "data_summary",
}

# Complex task keywords — force Opus
COMPLEX_KEYWORDS = {
    "live_trade", "strategy_change", "portfolio_rebalance",
    "risk_override", "capital_allocation", "go_live",
}

# Default log directory
LOG_DIR = Path(os.getenv("TOKEN_OPT_LOG_DIR", "logs"))

# Batch settings
DEFAULT_BATCH_SIZE = 50
MAX_PROMPT_LENGTH_FOR_COMPRESSION = 4000  # chars


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RoutingResult:
    """Result of a model routing decision."""
    recommended_model: str
    model_tier: str
    optimized_prompt: str
    estimated_cost: float
    naive_cost: float
    savings_vs_naive: float
    cache_key: Optional[str]
    cache_hit: bool
    prompt_compressed: bool
    original_tokens_est: int
    optimized_tokens_est: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommended_model": self.recommended_model,
            "model_tier": self.model_tier,
            "optimized_prompt": self.optimized_prompt[:200] + "..." if len(self.optimized_prompt) > 200 else self.optimized_prompt,
            "estimated_cost": round(self.estimated_cost, 6),
            "naive_cost": round(self.naive_cost, 6),
            "savings_vs_naive": round(self.savings_vs_naive, 4),
            "cache_key": self.cache_key,
            "cache_hit": self.cache_hit,
            "prompt_compressed": self.prompt_compressed,
            "original_tokens_est": self.original_tokens_est,
            "optimized_tokens_est": self.optimized_tokens_est,
        }


@dataclass
class CostTracker:
    """Tracks cumulative costs and token usage."""
    total_calls: int = 0
    total_cost: float = 0.0
    naive_cost: float = 0.0
    tokens_by_model: Dict[str, int] = field(default_factory=lambda: {
        "HAIKU": 0, "SONNET": 0, "OPUS": 0
    })
    calls_by_model: Dict[str, int] = field(default_factory=lambda: {
        "HAIKU": 0, "SONNET": 0, "OPUS": 0
    })
    cache_hits: int = 0
    cache_misses: int = 0
    prompts_compressed: int = 0
    tokens_saved_by_compression: int = 0


# ---------------------------------------------------------------------------
# TokenOptimizer
# ---------------------------------------------------------------------------

class TokenOptimizer:
    """
    Routes AI API calls to the cheapest model capable of handling the task.
    Compresses prompts, manages prompt caching, and batches non-real-time work.
    Logs all routing decisions to shadow JSONL.
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        cache_enabled: bool = True,
        compression_enabled: bool = True,
        shadow_mode: bool = True,
    ):
        self.log_dir = Path(log_dir) if log_dir else LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "token_optimizer_shadow.jsonl"

        self.cache_enabled = cache_enabled
        self.compression_enabled = compression_enabled
        self.shadow_mode = shadow_mode

        # Prompt cache: hash -> (prompt, model_tier, timestamp)
        self._prompt_cache: Dict[str, Dict[str, Any]] = {}

        # Cost tracking
        self.tracker = CostTracker()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def route_call(
        self,
        task_type: str,
        prompt: str,
        data_size: int = 0,
        system_prompt: Optional[str] = None,
        force_model: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Route an AI call to the optimal model tier.

        Args:
            task_type: SIMPLE, MODERATE, or COMPLEX
            prompt: The user/task prompt
            data_size: Approximate size of accompanying data in tokens
            system_prompt: System prompt (cacheable)
            force_model: Override routing with specific tier (HAIKU/SONNET/OPUS)
            context: Additional context for logging

        Returns:
            Dict with recommended_model, optimized_prompt, estimated_cost, cache_key
        """
        ts = time.time()

        # Determine model tier
        if force_model:
            tier = ModelTier(force_model.upper())
        else:
            tier = self._determine_tier(task_type, prompt, data_size)

        # Compress prompt
        optimized_prompt = prompt
        compressed = False
        original_tokens = self._estimate_tokens(prompt + (system_prompt or "")) + data_size
        optimized_tokens = original_tokens

        if self.compression_enabled and len(prompt) > MAX_PROMPT_LENGTH_FOR_COMPRESSION:
            optimized_prompt = self.compress_prompt(prompt)
            compressed = True
            optimized_tokens = self._estimate_tokens(optimized_prompt + (system_prompt or "")) + data_size
            self.tracker.prompts_compressed += 1
            self.tracker.tokens_saved_by_compression += (original_tokens - optimized_tokens)

        # Cache check for system prompts
        cache_key = None
        cache_hit = False
        if self.cache_enabled and system_prompt:
            cache_key = self._cache_key(system_prompt)
            if cache_key in self._prompt_cache:
                cache_hit = True
                self.tracker.cache_hits += 1
            else:
                self._prompt_cache[cache_key] = {
                    "prompt": system_prompt,
                    "tier": tier.value,
                    "ts": ts,
                }
                self.tracker.cache_misses += 1

        # Cost estimation
        estimated_cost = self.estimate_cost(tier, optimized_tokens, cache_hit=cache_hit)
        naive_cost = self.estimate_cost(ModelTier.OPUS, original_tokens, cache_hit=False)
        savings = 1.0 - (estimated_cost / naive_cost) if naive_cost > 0 else 0.0

        # Update tracker
        self.tracker.total_calls += 1
        self.tracker.total_cost += estimated_cost
        self.tracker.naive_cost += naive_cost
        self.tracker.tokens_by_model[tier.value] += optimized_tokens
        self.tracker.calls_by_model[tier.value] += 1

        result = RoutingResult(
            recommended_model=MODEL_IDS[tier],
            model_tier=tier.value,
            optimized_prompt=optimized_prompt,
            estimated_cost=estimated_cost,
            naive_cost=naive_cost,
            savings_vs_naive=savings,
            cache_key=cache_key,
            cache_hit=cache_hit,
            prompt_compressed=compressed,
            original_tokens_est=original_tokens,
            optimized_tokens_est=optimized_tokens,
        )

        # Shadow log
        self._log_routing(result, task_type, data_size, context, ts)

        return result.to_dict()

    def compress_prompt(self, prompt: str) -> str:
        """
        Compress a prompt to reduce token count.

        Strategies:
        - Strip excessive whitespace and blank lines
        - Abbreviate common tool definition patterns
        - Remove redundant instructions
        - Shorten verbose data representations
        """
        compressed = prompt

        # 1. Collapse multiple whitespace/newlines
        compressed = re.sub(r'\n{3,}', '\n\n', compressed)
        compressed = re.sub(r'[ \t]{2,}', ' ', compressed)
        compressed = re.sub(r'^\s+$', '', compressed, flags=re.MULTILINE)

        # 2. Abbreviate common verbose patterns
        abbreviations = [
            (r'Please note that ', ''),
            (r'It is important to ', ''),
            (r'Make sure to ', ''),
            (r'You should ', ''),
            (r'In order to ', 'To '),
            (r'at this point in time', 'now'),
            (r'due to the fact that', 'because'),
            (r'in the event that', 'if'),
            (r'for the purpose of', 'to'),
            (r'on a daily basis', 'daily'),
            (r'a large number of', 'many'),
        ]
        for pattern, replacement in abbreviations:
            compressed = re.sub(pattern, replacement, compressed, flags=re.IGNORECASE)

        # 3. Shorten JSON-like data blocks (remove extra spacing in JSON)
        def compact_json_block(match: re.Match) -> str:
            try:
                data = json.loads(match.group(0))
                return json.dumps(data, separators=(',', ':'))
            except (json.JSONDecodeError, ValueError):
                return match.group(0)

        compressed = re.sub(
            r'\{[^{}]{50,}\}',
            compact_json_block,
            compressed,
        )

        # 4. Strip trailing whitespace per line
        compressed = '\n'.join(line.rstrip() for line in compressed.split('\n'))

        return compressed.strip()

    def estimate_cost(
        self,
        tier: ModelTier,
        total_tokens: int,
        output_ratio: float = 0.3,
        cache_hit: bool = False,
    ) -> float:
        """
        Estimate the cost of an API call.

        Args:
            tier: Model tier
            total_tokens: Total estimated tokens (input + output)
            output_ratio: Fraction of tokens that are output (default 30%)
            cache_hit: Whether system prompt is cached

        Returns:
            Estimated cost in USD
        """
        if isinstance(tier, str):
            tier = ModelTier(tier.upper())

        pricing = MODEL_PRICING[tier]
        input_tokens = int(total_tokens * (1 - output_ratio))
        output_tokens = int(total_tokens * output_ratio)

        if cache_hit:
            # 90% of input tokens are cached (system prompt)
            cached_tokens = int(input_tokens * 0.7)
            uncached_tokens = input_tokens - cached_tokens
            input_cost = (
                (uncached_tokens / 1_000_000) * pricing["input"]
                + (cached_tokens / 1_000_000) * pricing["input"] * CACHE_READ_MULTIPLIER
            )
        else:
            input_cost = (input_tokens / 1_000_000) * pricing["input"]

        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost

    def batch_evaluate(
        self,
        prompts: List[str],
        task_type: str = "SIMPLE",
        batch_size: int = DEFAULT_BATCH_SIZE,
        system_prompt: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Batch non-real-time evaluations (e.g., shadow signal evaluations).
        Groups prompts into batches for throughput efficiency.

        Args:
            prompts: List of prompts to evaluate
            task_type: Task type for all prompts in batch
            batch_size: Number of prompts per batch
            system_prompt: Shared system prompt (cached across batch)

        Returns:
            List of routing results for each prompt
        """
        results = []
        total_batches = (len(prompts) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(prompts))
            batch = prompts[start:end]

            batch_results = []
            for prompt in batch:
                result = self.route_call(
                    task_type=task_type,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    context={"batch_idx": batch_idx, "batch_size": len(batch)},
                )
                batch_results.append(result)

            results.extend(batch_results)

        return results

    def get_savings_report(self) -> Dict[str, Any]:
        """
        Get a comprehensive report on token usage and cost savings.

        Returns:
            Dict with total_cost, naive_cost, savings_pct, per-model breakdown
        """
        t = self.tracker
        savings_pct = (
            (1.0 - t.total_cost / t.naive_cost) * 100
            if t.naive_cost > 0 else 0.0
        )

        return {
            "total_calls": t.total_calls,
            "total_cost": round(t.total_cost, 4),
            "naive_cost_opus_only": round(t.naive_cost, 4),
            "savings_pct": round(savings_pct, 2),
            "savings_usd": round(t.naive_cost - t.total_cost, 4),
            "calls_by_model": dict(t.calls_by_model),
            "tokens_by_model": dict(t.tokens_by_model),
            "cache_hits": t.cache_hits,
            "cache_misses": t.cache_misses,
            "cache_hit_rate": round(
                t.cache_hits / max(1, t.cache_hits + t.cache_misses) * 100, 1
            ),
            "prompts_compressed": t.prompts_compressed,
            "tokens_saved_by_compression": t.tokens_saved_by_compression,
            "cost_per_call_avg": round(
                t.total_cost / max(1, t.total_calls), 6
            ),
            "haiku_pct": round(
                t.calls_by_model.get("HAIKU", 0) / max(1, t.total_calls) * 100, 1
            ),
        }

    def get_cache_control_headers(self, system_prompt: str) -> Dict[str, Any]:
        """
        Generate Anthropic cache_control headers for a system prompt.

        Returns:
            Dict suitable for inclusion in API request system message.
        """
        return {
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _determine_tier(
        self,
        task_type: str,
        prompt: str,
        data_size: int,
    ) -> ModelTier:
        """Determine the optimal model tier based on task characteristics."""
        prompt_lower = prompt.lower()

        # Check for complex keywords first (safety-critical)
        for kw in COMPLEX_KEYWORDS:
            if kw in prompt_lower:
                return ModelTier.OPUS

        # Check for simple keywords
        for kw in SIMPLE_KEYWORDS:
            if kw in prompt_lower:
                return ModelTier.HAIKU

        # Use explicit task type
        try:
            tt = TaskType(task_type.upper())
            return TASK_ROUTING[tt]
        except (ValueError, KeyError):
            pass

        # Heuristic: large data + simple task stays Haiku
        if data_size > 10000:
            # Large data usually means summarization -> Haiku
            return ModelTier.HAIKU

        # Default to Sonnet for unknown tasks
        return ModelTier.SONNET

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: ~4 chars per token for English."""
        return max(1, len(text) // 4)

    @staticmethod
    def _cache_key(text: str) -> str:
        """Generate a cache key from text content."""
        return f"sys_{hashlib.sha256(text.encode()).hexdigest()[:12]}"

    def _log_routing(
        self,
        result: RoutingResult,
        task_type: str,
        data_size: int,
        context: Optional[Dict[str, Any]],
        ts: float,
    ) -> None:
        """Log routing decision to shadow JSONL."""
        if not self.shadow_mode:
            return

        entry = {
            "ts": ts,
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)),
            "event": "routing_decision",
            "task_type": task_type,
            "model_tier": result.model_tier,
            "model_id": result.recommended_model,
            "estimated_cost": round(result.estimated_cost, 6),
            "naive_cost": round(result.naive_cost, 6),
            "savings_vs_naive": round(result.savings_vs_naive, 4),
            "cache_hit": result.cache_hit,
            "cache_key": result.cache_key,
            "prompt_compressed": result.prompt_compressed,
            "original_tokens": result.original_tokens_est,
            "optimized_tokens": result.optimized_tokens_est,
            "data_size": data_size,
            "context": context,
        }

        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            pass  # Never crash on log failure


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

_default_optimizer: Optional[TokenOptimizer] = None


def get_optimizer() -> TokenOptimizer:
    """Get or create the default TokenOptimizer singleton."""
    global _default_optimizer
    if _default_optimizer is None:
        _default_optimizer = TokenOptimizer()
    return _default_optimizer


def route_call(task_type: str, prompt: str, **kwargs) -> Dict[str, Any]:
    """Convenience wrapper: route a call using the default optimizer."""
    return get_optimizer().route_call(task_type=task_type, prompt=prompt, **kwargs)


def estimate_cost(tier: str, total_tokens: int, **kwargs) -> float:
    """Convenience wrapper: estimate cost using the default optimizer."""
    return get_optimizer().estimate_cost(ModelTier(tier.upper()), total_tokens, **kwargs)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    opt = TokenOptimizer(log_dir="logs")

    # Test routing
    print("=== Token Optimizer Self-Test ===\n")

    # Simple task -> Haiku
    r1 = opt.route_call("SIMPLE", "Classify this signal as bullish or bearish", data_size=100)
    print(f"SIMPLE task -> {r1['model_tier']} (expected HAIKU)")
    assert r1["model_tier"] == "HAIKU"

    # Moderate task -> Sonnet
    r2 = opt.route_call("MODERATE", "Analyze market regime and multi-step correlations", data_size=2000)
    print(f"MODERATE task -> {r2['model_tier']} (expected SONNET)")
    assert r2["model_tier"] == "SONNET"

    # Complex task -> Opus
    r3 = opt.route_call("COMPLEX", "Make live_trade decision for BTC position", data_size=5000)
    print(f"COMPLEX task -> {r3['model_tier']} (expected OPUS)")
    assert r3["model_tier"] == "OPUS"

    # Keyword override: shadow signal -> Haiku even if marked MODERATE
    r4 = opt.route_call("MODERATE", "Run shadow signal_eval on this data", data_size=500)
    print(f"Shadow signal eval -> {r4['model_tier']} (expected HAIKU)")
    assert r4["model_tier"] == "HAIKU"

    # Batch test
    batch_prompts = [f"Classify signal {i}" for i in range(120)]
    batch_results = opt.batch_evaluate(batch_prompts, task_type="SIMPLE", batch_size=50)
    print(f"\nBatch: {len(batch_results)} results from {len(batch_prompts)} prompts")
    assert len(batch_results) == 120

    # Compression test
    long_prompt = "Please note that " * 300 + "classify this signal"
    compressed = opt.compress_prompt(long_prompt)
    print(f"\nCompression: {len(long_prompt)} -> {len(compressed)} chars "
          f"({100 - len(compressed)/len(long_prompt)*100:.0f}% reduction)")

    # Savings report
    report = opt.get_savings_report()
    print(f"\n=== Savings Report ===")
    print(f"Total calls: {report['total_calls']}")
    print(f"Total cost: ${report['total_cost']}")
    print(f"Naive cost (all Opus): ${report['naive_cost_opus_only']}")
    print(f"Savings: {report['savings_pct']}% (${report['savings_usd']})")
    print(f"Haiku usage: {report['haiku_pct']}%")
    print(f"Cache hit rate: {report['cache_hit_rate']}%")

    print("\n[PASS] All tests passed")
