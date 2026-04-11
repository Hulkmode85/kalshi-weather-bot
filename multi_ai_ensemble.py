"""
multi_ai_ensemble.py — Multi-Provider AI Ensemble

Quant Concept:
    A single model has blind spots. Using multiple AI providers (Claude, GPT-4,
    Gemini) and aggregating their predictions via ensemble methods produces
    higher accuracy on critical decisions — the same principle behind ensemble
    ML models that consistently outperform single models.

    Ensemble methods:
        MAJORITY_VOTE:          Each model predicts direction, majority wins.
        WEIGHTED_AVERAGE:       Weight by historical accuracy per model.
        BAYESIAN_AGGREGATION:   Use each model's confidence as prior, combine
                                via Bayes' theorem for posterior probability.

    Key signals:
        Agreement boost:   When ALL models agree, confidence += 15%
        Disagreement cut:  When models strongly disagree, REDUCE position size
                          (uncertainty signal)

    Fallback: If only Claude is available, uses multi-persona approach
    (analyst, skeptic, optimist) to simulate ensemble diversity.

Usage:
    from multi_ai_ensemble import MultiAIEnsemble

    ensemble = MultiAIEnsemble()
    result = ensemble.predict(
        market_data={"price": 67500, "volume": 1.2e9, "trend": "up"},
        question="Will BTC close above $68,000 today?"
    )
    print(result)
    # {'ensemble_prediction': 'YES', 'confidence': 0.72, 'agreement_level': 'FULL',
    #  'per_model_predictions': {...}, 'recommended_sizing_adjustment': 1.15}
"""

import json
import math
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class EnsembleMethod(str, Enum):
    MAJORITY_VOTE = "MAJORITY_VOTE"
    WEIGHTED_AVERAGE = "WEIGHTED_AVERAGE"
    BAYESIAN_AGGREGATION = "BAYESIAN_AGGREGATION"


class AgreementLevel(str, Enum):
    FULL = "FULL"           # All models agree
    PARTIAL = "PARTIAL"     # Majority agrees
    SPLIT = "SPLIT"         # No clear majority (or 50/50)


class Provider(str, Enum):
    CLAUDE = "CLAUDE"
    GPT4 = "GPT4"
    GEMINI = "GEMINI"
    # Multi-persona fallback personas
    ANALYST = "CLAUDE_ANALYST"
    SKEPTIC = "CLAUDE_SKEPTIC"
    OPTIMIST = "CLAUDE_OPTIMIST"


# Agreement boost / disagreement penalty
AGREEMENT_CONFIDENCE_BOOST = 0.15
DISAGREEMENT_SIZING_CUT = 0.60     # Reduce to 60% of normal size
PARTIAL_AGREEMENT_SIZING = 0.85    # Reduce to 85% of normal size

# Default provider weights (updated by historical accuracy)
DEFAULT_WEIGHTS: Dict[str, float] = {
    Provider.CLAUDE: 0.40,
    Provider.GPT4: 0.35,
    Provider.GEMINI: 0.25,
}

# Persona weights for fallback mode
PERSONA_WEIGHTS: Dict[str, float] = {
    Provider.ANALYST: 0.45,
    Provider.SKEPTIC: 0.30,
    Provider.OPTIMIST: 0.25,
}

# Persona system prompts for multi-persona fallback
PERSONA_PROMPTS: Dict[str, str] = {
    Provider.ANALYST: (
        "You are a quantitative market analyst. Analyze the data objectively "
        "using statistical evidence. Focus on base rates, historical patterns, "
        "and measurable signals. Provide your prediction and confidence 0-1."
    ),
    Provider.SKEPTIC: (
        "You are a risk-focused market skeptic. Challenge the obvious narrative. "
        "Look for reasons the consensus could be wrong. Consider tail risks, "
        "adverse selection, and overconfidence. Provide your prediction and confidence 0-1."
    ),
    Provider.OPTIMIST: (
        "You are a momentum-focused market analyst. Look for positive signals, "
        "breakout patterns, and catalysts. Weight recent trends heavily. "
        "Provide your prediction and confidence 0-1."
    ),
}

LOG_DIR = Path(os.getenv("ENSEMBLE_LOG_DIR", "logs"))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ModelPrediction:
    """A single model's prediction."""
    provider: str
    prediction: str           # YES/NO or direction
    confidence: float         # 0.0 to 1.0
    reasoning: str = ""
    latency_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "prediction": self.prediction,
            "confidence": round(self.confidence, 4),
            "reasoning": self.reasoning[:300] if self.reasoning else "",
            "latency_ms": round(self.latency_ms, 1),
            "error": self.error,
        }


@dataclass
class EnsembleResult:
    """Aggregated ensemble prediction."""
    ensemble_prediction: str
    confidence: float
    agreement_level: str
    per_model_predictions: Dict[str, Dict[str, Any]]
    recommended_sizing_adjustment: float
    method_used: str
    num_models: int
    raw_yes_probability: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ensemble_prediction": self.ensemble_prediction,
            "confidence": round(self.confidence, 4),
            "agreement_level": self.agreement_level,
            "per_model_predictions": self.per_model_predictions,
            "recommended_sizing_adjustment": round(self.recommended_sizing_adjustment, 3),
            "method_used": self.method_used,
            "num_models": self.num_models,
            "raw_yes_probability": round(self.raw_yes_probability, 4),
        }


@dataclass
class PerformanceTracker:
    """Track per-model and ensemble accuracy over time."""
    total_predictions: int = 0
    correct_by_provider: Dict[str, int] = field(default_factory=dict)
    total_by_provider: Dict[str, int] = field(default_factory=dict)
    ensemble_correct: int = 0
    agreement_outcomes: Dict[str, Dict[str, int]] = field(default_factory=lambda: {
        "FULL": {"correct": 0, "total": 0},
        "PARTIAL": {"correct": 0, "total": 0},
        "SPLIT": {"correct": 0, "total": 0},
    })


# ---------------------------------------------------------------------------
# MultiAIEnsemble
# ---------------------------------------------------------------------------

class MultiAIEnsemble:
    """
    Multi-provider AI ensemble for higher-accuracy predictions.
    Aggregates predictions from multiple AI models using configurable
    ensemble methods. Falls back to multi-persona approach when only
    Claude is available.
    """

    def __init__(
        self,
        method: str = "WEIGHTED_AVERAGE",
        log_dir: Optional[str] = None,
        shadow_mode: bool = True,
        claude_caller: Optional[Callable] = None,
        openai_caller: Optional[Callable] = None,
        gemini_caller: Optional[Callable] = None,
    ):
        """
        Args:
            method: Ensemble method (MAJORITY_VOTE, WEIGHTED_AVERAGE, BAYESIAN_AGGREGATION)
            log_dir: Directory for shadow JSONL logs
            shadow_mode: Whether to log all decisions
            claude_caller: Optional callable(system_prompt, user_prompt) -> str
            openai_caller: Optional callable(system_prompt, user_prompt) -> str
            gemini_caller: Optional callable(system_prompt, user_prompt) -> str
        """
        self.method = EnsembleMethod(method)
        self.log_dir = Path(log_dir) if log_dir else LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "multi_ai_ensemble_shadow.jsonl"
        self.shadow_mode = shadow_mode

        # Provider callers (injectable for testing / real API integration)
        self.callers: Dict[str, Optional[Callable]] = {
            Provider.CLAUDE: claude_caller,
            Provider.GPT4: openai_caller,
            Provider.GEMINI: gemini_caller,
        }

        # Dynamic weights (updated by record_outcome)
        self.weights: Dict[str, float] = dict(DEFAULT_WEIGHTS)

        # Performance tracking
        self.tracker = PerformanceTracker()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        market_data: Dict[str, Any],
        question: str,
        method: Optional[str] = None,
        providers: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Get an ensemble prediction from multiple AI models.

        Args:
            market_data: Dict of market data (price, volume, signals, etc.)
            question: The prediction question (e.g., "Will BTC close above $68K?")
            method: Override ensemble method for this call
            providers: Specific providers to use (default: all available)

        Returns:
            Dict with ensemble_prediction, confidence, agreement_level,
            per_model_predictions, recommended_sizing_adjustment
        """
        ts = time.time()
        active_method = EnsembleMethod(method) if method else self.method

        # Determine available providers
        available = self._get_available_providers(providers)

        # If only Claude (or no providers), use multi-persona fallback
        if len(available) <= 1 and Provider.CLAUDE in available:
            predictions = self._multi_persona_predict(market_data, question)
            weights = PERSONA_WEIGHTS
            fallback = True
        elif not available:
            # No providers at all: return neutral
            return self._neutral_result(question, ts)
        else:
            predictions = self._multi_provider_predict(available, market_data, question)
            weights = {p: self.weights.get(p, 0.25) for p in available}
            fallback = False

        # Filter out failed predictions
        valid = [p for p in predictions if p.error is None]
        if not valid:
            return self._neutral_result(question, ts)

        # Aggregate
        result = self._aggregate(valid, weights, active_method)

        # Log
        self._log_prediction(result, market_data, question, fallback, ts)

        return result.to_dict()

    def record_outcome(
        self,
        prediction_ts: float,
        actual_outcome: str,
        per_model_predictions: Dict[str, Dict[str, Any]],
        ensemble_prediction: str,
    ) -> Dict[str, Any]:
        """
        Record the actual outcome to update model weights and accuracy tracking.

        Args:
            prediction_ts: Timestamp of the original prediction
            actual_outcome: What actually happened (YES/NO)
            per_model_predictions: The per-model predictions from predict()
            ensemble_prediction: The ensemble prediction from predict()

        Returns:
            Updated accuracy stats
        """
        actual = actual_outcome.upper()
        self.tracker.total_predictions += 1

        # Check ensemble accuracy
        if ensemble_prediction.upper() == actual:
            self.tracker.ensemble_correct += 1

        # Check per-model accuracy and update weights
        for provider_key, pred_data in per_model_predictions.items():
            provider = provider_key
            if provider not in self.tracker.total_by_provider:
                self.tracker.total_by_provider[provider] = 0
                self.tracker.correct_by_provider[provider] = 0

            self.tracker.total_by_provider[provider] += 1
            if pred_data.get("prediction", "").upper() == actual:
                self.tracker.correct_by_provider[provider] += 1

        # Recompute weights based on accuracy
        self._update_weights()

        # Log outcome
        if self.shadow_mode:
            entry = {
                "ts": time.time(),
                "event": "outcome_recorded",
                "prediction_ts": prediction_ts,
                "actual_outcome": actual,
                "ensemble_prediction": ensemble_prediction,
                "ensemble_correct": ensemble_prediction.upper() == actual,
                "updated_weights": {k: round(v, 4) for k, v in self.weights.items()},
            }
            self._write_log(entry)

        return self.get_performance_report()

    def get_performance_report(self) -> Dict[str, Any]:
        """Get accuracy and performance statistics."""
        t = self.tracker
        ensemble_acc = (
            t.ensemble_correct / max(1, t.total_predictions)
        )

        per_model_acc = {}
        for provider in set(list(t.total_by_provider.keys())):
            total = t.total_by_provider.get(provider, 0)
            correct = t.correct_by_provider.get(provider, 0)
            per_model_acc[provider] = {
                "accuracy": round(correct / max(1, total), 4),
                "total": total,
                "correct": correct,
            }

        return {
            "total_predictions": t.total_predictions,
            "ensemble_accuracy": round(ensemble_acc, 4),
            "per_model_accuracy": per_model_acc,
            "current_weights": {k: round(v, 4) for k, v in self.weights.items()},
            "agreement_outcomes": t.agreement_outcomes,
        }

    # ------------------------------------------------------------------
    # Aggregation methods
    # ------------------------------------------------------------------

    def _aggregate(
        self,
        predictions: List[ModelPrediction],
        weights: Dict[str, float],
        method: EnsembleMethod,
    ) -> EnsembleResult:
        """Aggregate predictions using the specified method."""
        if method == EnsembleMethod.MAJORITY_VOTE:
            return self._majority_vote(predictions, weights)
        elif method == EnsembleMethod.WEIGHTED_AVERAGE:
            return self._weighted_average(predictions, weights)
        elif method == EnsembleMethod.BAYESIAN_AGGREGATION:
            return self._bayesian_aggregation(predictions, weights)
        else:
            return self._weighted_average(predictions, weights)

    def _majority_vote(
        self,
        predictions: List[ModelPrediction],
        weights: Dict[str, float],
    ) -> EnsembleResult:
        """Simple majority vote: each model gets one vote."""
        yes_votes = sum(1 for p in predictions if p.prediction.upper() == "YES")
        no_votes = len(predictions) - yes_votes
        total = len(predictions)

        prediction = "YES" if yes_votes > no_votes else "NO"
        raw_yes_prob = yes_votes / total

        # Agreement level
        agreement = self._get_agreement_level(predictions)

        # Confidence = proportion of majority + individual confidences
        majority_pct = max(yes_votes, no_votes) / total
        avg_confidence = sum(p.confidence for p in predictions) / total
        confidence = (majority_pct * 0.6) + (avg_confidence * 0.4)

        # Apply agreement adjustments
        confidence, sizing_adj = self._apply_agreement_adjustments(
            confidence, agreement
        )

        per_model = {p.provider: p.to_dict() for p in predictions}

        return EnsembleResult(
            ensemble_prediction=prediction,
            confidence=min(0.99, confidence),
            agreement_level=agreement.value,
            per_model_predictions=per_model,
            recommended_sizing_adjustment=sizing_adj,
            method_used=EnsembleMethod.MAJORITY_VOTE.value,
            num_models=total,
            raw_yes_probability=raw_yes_prob,
        )

    def _weighted_average(
        self,
        predictions: List[ModelPrediction],
        weights: Dict[str, float],
    ) -> EnsembleResult:
        """Weighted average: weight by historical accuracy per model."""
        total_weight = 0.0
        weighted_yes_prob = 0.0

        for p in predictions:
            w = weights.get(p.provider, 0.25)
            # Convert prediction to probability
            prob_yes = p.confidence if p.prediction.upper() == "YES" else (1 - p.confidence)
            weighted_yes_prob += w * prob_yes
            total_weight += w

        if total_weight > 0:
            weighted_yes_prob /= total_weight

        prediction = "YES" if weighted_yes_prob >= 0.5 else "NO"
        confidence = abs(weighted_yes_prob - 0.5) * 2  # Scale 0.5-1.0 -> 0-1

        agreement = self._get_agreement_level(predictions)
        confidence, sizing_adj = self._apply_agreement_adjustments(
            confidence, agreement
        )

        per_model = {p.provider: p.to_dict() for p in predictions}

        return EnsembleResult(
            ensemble_prediction=prediction,
            confidence=min(0.99, confidence),
            agreement_level=agreement.value,
            per_model_predictions=per_model,
            recommended_sizing_adjustment=sizing_adj,
            method_used=EnsembleMethod.WEIGHTED_AVERAGE.value,
            num_models=len(predictions),
            raw_yes_probability=round(weighted_yes_prob, 4),
        )

    def _bayesian_aggregation(
        self,
        predictions: List[ModelPrediction],
        weights: Dict[str, float],
    ) -> EnsembleResult:
        """
        Bayesian aggregation: treat each model's confidence as evidence,
        combine via Bayes' theorem.

        Prior: 0.5 (no bias)
        Each model updates the prior based on its confidence and historical
        accuracy (weight).
        """
        # Start with uniform prior
        log_odds = 0.0  # log(0.5/0.5) = 0

        for p in predictions:
            w = weights.get(p.provider, 0.25)
            # Model's implied probability of YES
            prob_yes = p.confidence if p.prediction.upper() == "YES" else (1 - p.confidence)
            # Clamp to avoid log(0)
            prob_yes = max(0.01, min(0.99, prob_yes))

            # Weight the evidence by model accuracy
            # Higher weight = more informative signal
            evidence_strength = w * 2  # Scale weight to evidence multiplier
            model_log_odds = math.log(prob_yes / (1 - prob_yes))
            log_odds += evidence_strength * model_log_odds

        # Convert back to probability
        posterior_yes = 1.0 / (1.0 + math.exp(-log_odds))
        prediction = "YES" if posterior_yes >= 0.5 else "NO"
        confidence = abs(posterior_yes - 0.5) * 2

        agreement = self._get_agreement_level(predictions)
        confidence, sizing_adj = self._apply_agreement_adjustments(
            confidence, agreement
        )

        per_model = {p.provider: p.to_dict() for p in predictions}

        return EnsembleResult(
            ensemble_prediction=prediction,
            confidence=min(0.99, confidence),
            agreement_level=agreement.value,
            per_model_predictions=per_model,
            recommended_sizing_adjustment=sizing_adj,
            method_used=EnsembleMethod.BAYESIAN_AGGREGATION.value,
            num_models=len(predictions),
            raw_yes_probability=round(posterior_yes, 4),
        )

    # ------------------------------------------------------------------
    # Provider calling
    # ------------------------------------------------------------------

    def _get_available_providers(
        self, requested: Optional[List[str]] = None
    ) -> List[str]:
        """Determine which providers are available (or simulated in shadow mode)."""
        available = []

        # Claude is always available (shadow mode simulates if no key)
        if self.callers.get(Provider.CLAUDE) or os.getenv("ANTHROPIC_API_KEY") or self.shadow_mode:
            available.append(Provider.CLAUDE)

        # GPT-4 if key set or shadow mode
        if self.callers.get(Provider.GPT4) or os.getenv("OPENAI_API_KEY"):
            available.append(Provider.GPT4)
        elif self.shadow_mode:
            available.append(Provider.GPT4)

        # Gemini if key set or shadow mode
        if self.callers.get(Provider.GEMINI) or os.getenv("GOOGLE_API_KEY"):
            available.append(Provider.GEMINI)
        elif self.shadow_mode:
            available.append(Provider.GEMINI)

        if requested:
            requested_set = {r.upper() for r in requested}
            available = [p for p in available if p.value in requested_set or p in requested_set]

        return available

    def _multi_provider_predict(
        self,
        providers: List[str],
        market_data: Dict[str, Any],
        question: str,
    ) -> List[ModelPrediction]:
        """Get predictions from multiple providers."""
        predictions = []
        system_prompt = (
            "You are a quantitative market analyst. Analyze the provided market data "
            "and answer the question with YES or NO, plus a confidence between 0 and 1. "
            "Respond in JSON format: {\"prediction\": \"YES/NO\", \"confidence\": 0.XX, "
            "\"reasoning\": \"brief explanation\"}"
        )
        user_prompt = (
            f"Market data: {json.dumps(market_data, default=str)}\n\n"
            f"Question: {question}"
        )

        for provider in providers:
            caller = self.callers.get(provider)
            if caller is None:
                # Simulate prediction for shadow/paper mode
                predictions.append(self._simulate_prediction(provider, market_data, question))
                continue

            t0 = time.time()
            try:
                response = caller(system_prompt, user_prompt)
                latency = (time.time() - t0) * 1000
                pred = self._parse_prediction_response(provider, response, latency)
                predictions.append(pred)
            except Exception as e:
                predictions.append(ModelPrediction(
                    provider=provider,
                    prediction="ABSTAIN",
                    confidence=0.0,
                    error=str(e),
                    latency_ms=(time.time() - t0) * 1000,
                ))

        return predictions

    def _multi_persona_predict(
        self,
        market_data: Dict[str, Any],
        question: str,
    ) -> List[ModelPrediction]:
        """
        Fallback: use multi-persona approach with Claude only.
        Three personas (analyst, skeptic, optimist) provide diverse viewpoints.
        """
        predictions = []
        user_prompt = (
            f"Market data: {json.dumps(market_data, default=str)}\n\n"
            f"Question: {question}\n\n"
            "Respond in JSON: {\"prediction\": \"YES/NO\", \"confidence\": 0.XX, "
            "\"reasoning\": \"brief explanation\"}"
        )

        claude_caller = self.callers.get(Provider.CLAUDE)

        for persona, system_prompt in PERSONA_PROMPTS.items():
            if claude_caller:
                t0 = time.time()
                try:
                    response = claude_caller(system_prompt, user_prompt)
                    latency = (time.time() - t0) * 1000
                    pred = self._parse_prediction_response(persona, response, latency)
                    predictions.append(pred)
                except Exception as e:
                    predictions.append(ModelPrediction(
                        provider=persona,
                        prediction="ABSTAIN",
                        confidence=0.0,
                        error=str(e),
                    ))
            else:
                # Shadow mode: simulate persona predictions
                predictions.append(self._simulate_prediction(persona, market_data, question))

        return predictions

    def _simulate_prediction(
        self,
        provider: str,
        market_data: Dict[str, Any],
        question: str,
    ) -> ModelPrediction:
        """
        Simulate a prediction in shadow/paper mode when no API caller is set.
        Uses deterministic hashing for reproducibility.
        """
        # Create a deterministic but varied prediction based on input
        seed_str = f"{provider}:{json.dumps(market_data, sort_keys=True, default=str)}:{question}"
        seed_hash = hash(seed_str) & 0xFFFFFFFF
        pseudo_random = (seed_hash % 1000) / 1000.0

        # Different providers have slight biases for diversity
        bias = {
            Provider.CLAUDE: 0.0,
            Provider.GPT4: 0.05,
            Provider.GEMINI: -0.05,
            Provider.ANALYST: 0.0,
            Provider.SKEPTIC: -0.10,
            Provider.OPTIMIST: 0.10,
        }.get(provider, 0.0)

        adjusted = pseudo_random + bias
        prediction = "YES" if adjusted >= 0.5 else "NO"
        confidence = 0.5 + abs(adjusted - 0.5) * 0.8  # Scale to 0.5-0.9

        return ModelPrediction(
            provider=provider if isinstance(provider, str) else provider.value,
            prediction=prediction,
            confidence=round(confidence, 3),
            reasoning=f"[SIMULATED] shadow mode prediction for {provider}",
            latency_ms=0.0,
        )

    @staticmethod
    def _parse_prediction_response(
        provider: str,
        response: str,
        latency_ms: float,
    ) -> ModelPrediction:
        """Parse a model's JSON response into a ModelPrediction."""
        try:
            # Try to extract JSON from response
            json_match = None
            # Look for JSON block
            for pattern_start, pattern_end in [("{", "}"), ("```json", "```")]:
                start_idx = response.find(pattern_start)
                end_idx = response.rfind(pattern_end)
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response[start_idx:end_idx + len(pattern_end)]
                    json_str = json_str.strip("`").strip()
                    if json_str.startswith("json"):
                        json_str = json_str[4:].strip()
                    try:
                        json_match = json.loads(json_str)
                        break
                    except json.JSONDecodeError:
                        continue

            if json_match:
                return ModelPrediction(
                    provider=provider if isinstance(provider, str) else provider.value,
                    prediction=json_match.get("prediction", "ABSTAIN").upper(),
                    confidence=float(json_match.get("confidence", 0.5)),
                    reasoning=json_match.get("reasoning", ""),
                    latency_ms=latency_ms,
                )
            else:
                # Fallback: look for YES/NO in text
                upper = response.upper()
                if "YES" in upper:
                    pred = "YES"
                elif "NO" in upper:
                    pred = "NO"
                else:
                    pred = "ABSTAIN"
                return ModelPrediction(
                    provider=provider if isinstance(provider, str) else provider.value,
                    prediction=pred,
                    confidence=0.5,
                    reasoning=response[:200],
                    latency_ms=latency_ms,
                )
        except Exception as e:
            return ModelPrediction(
                provider=provider if isinstance(provider, str) else provider.value,
                prediction="ABSTAIN",
                confidence=0.0,
                error=f"Parse error: {e}",
                latency_ms=latency_ms,
            )

    # ------------------------------------------------------------------
    # Agreement & adjustment
    # ------------------------------------------------------------------

    @staticmethod
    def _get_agreement_level(predictions: List[ModelPrediction]) -> AgreementLevel:
        """Determine the agreement level among predictions."""
        valid = [p for p in predictions if p.prediction.upper() in ("YES", "NO")]
        if not valid:
            return AgreementLevel.SPLIT

        yes_count = sum(1 for p in valid if p.prediction.upper() == "YES")
        no_count = len(valid) - yes_count

        if yes_count == len(valid) or no_count == len(valid):
            return AgreementLevel.FULL
        elif max(yes_count, no_count) > len(valid) / 2:
            return AgreementLevel.PARTIAL
        else:
            return AgreementLevel.SPLIT

    @staticmethod
    def _apply_agreement_adjustments(
        confidence: float,
        agreement: AgreementLevel,
    ) -> Tuple[float, float]:
        """
        Apply confidence boost / sizing adjustment based on agreement level.

        Returns:
            (adjusted_confidence, sizing_adjustment_multiplier)
        """
        if agreement == AgreementLevel.FULL:
            # All models agree: boost confidence, increase sizing
            return (
                min(0.99, confidence + AGREEMENT_CONFIDENCE_BOOST),
                1.0 + AGREEMENT_CONFIDENCE_BOOST,  # 1.15x
            )
        elif agreement == AgreementLevel.PARTIAL:
            return (confidence, PARTIAL_AGREEMENT_SIZING)
        else:
            # SPLIT: strong disagreement -> reduce sizing significantly
            return (
                max(0.01, confidence * 0.7),
                DISAGREEMENT_SIZING_CUT,
            )

    # ------------------------------------------------------------------
    # Weight updates
    # ------------------------------------------------------------------

    def _update_weights(self) -> None:
        """Update provider weights based on historical accuracy."""
        t = self.tracker
        accuracies = {}

        for provider in t.total_by_provider:
            total = t.total_by_provider[provider]
            correct = t.correct_by_provider.get(provider, 0)
            if total >= 5:  # Minimum sample size
                accuracies[provider] = correct / total
            else:
                # Not enough data: keep default
                accuracies[provider] = DEFAULT_WEIGHTS.get(
                    provider, PERSONA_WEIGHTS.get(provider, 0.25)
                )

        # Normalize to sum to 1
        total_acc = sum(accuracies.values())
        if total_acc > 0:
            for provider, acc in accuracies.items():
                self.weights[provider] = acc / total_acc

    # ------------------------------------------------------------------
    # Neutral / error result
    # ------------------------------------------------------------------

    def _neutral_result(self, question: str, ts: float) -> Dict[str, Any]:
        """Return a neutral result when no providers are available."""
        result = EnsembleResult(
            ensemble_prediction="ABSTAIN",
            confidence=0.0,
            agreement_level=AgreementLevel.SPLIT.value,
            per_model_predictions={},
            recommended_sizing_adjustment=0.0,
            method_used=self.method.value,
            num_models=0,
            raw_yes_probability=0.5,
        )
        self._log_prediction(result, {}, question, False, ts)
        return result.to_dict()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_prediction(
        self,
        result: EnsembleResult,
        market_data: Dict[str, Any],
        question: str,
        fallback_mode: bool,
        ts: float,
    ) -> None:
        """Log ensemble prediction to shadow JSONL."""
        if not self.shadow_mode:
            return

        entry = {
            "ts": ts,
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts)),
            "event": "ensemble_prediction",
            "question": question[:200],
            "method": result.method_used,
            "prediction": result.ensemble_prediction,
            "confidence": round(result.confidence, 4),
            "agreement_level": result.agreement_level,
            "sizing_adjustment": round(result.recommended_sizing_adjustment, 3),
            "num_models": result.num_models,
            "raw_yes_prob": result.raw_yes_probability,
            "fallback_mode": fallback_mode,
            "per_model": {
                k: {"pred": v.get("prediction"), "conf": v.get("confidence")}
                for k, v in result.per_model_predictions.items()
            },
            "market_data_keys": list(market_data.keys()),
        }
        self._write_log(entry)

    def _write_log(self, entry: Dict[str, Any]) -> None:
        """Write a log entry to the JSONL file."""
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            pass  # Never crash on log failure


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

_default_ensemble: Optional[MultiAIEnsemble] = None


def get_ensemble() -> MultiAIEnsemble:
    """Get or create the default MultiAIEnsemble singleton."""
    global _default_ensemble
    if _default_ensemble is None:
        _default_ensemble = MultiAIEnsemble()
    return _default_ensemble


def predict(market_data: Dict[str, Any], question: str, **kwargs) -> Dict[str, Any]:
    """Convenience wrapper: get ensemble prediction using default instance."""
    return get_ensemble().predict(market_data=market_data, question=question, **kwargs)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Multi-AI Ensemble Self-Test ===\n")

    ensemble = MultiAIEnsemble(method="WEIGHTED_AVERAGE", log_dir="logs")

    # Test 1: Shadow mode prediction (no API callers)
    market = {"price": 67500, "volume": 1.2e9, "trend": "up", "rsi": 62}
    result = ensemble.predict(
        market_data=market,
        question="Will BTC close above $68,000 today?",
    )
    print(f"Prediction: {result['ensemble_prediction']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Agreement: {result['agreement_level']}")
    print(f"Sizing adj: {result['recommended_sizing_adjustment']}")
    print(f"Models used: {result['num_models']}")
    print(f"Per-model: {json.dumps(result['per_model_predictions'], indent=2)}")

    # Test 2: Different ensemble methods
    for method in ["MAJORITY_VOTE", "WEIGHTED_AVERAGE", "BAYESIAN_AGGREGATION"]:
        r = ensemble.predict(
            market_data=market,
            question="Will ETH break $4000?",
            method=method,
        )
        print(f"\n{method}: {r['ensemble_prediction']} "
              f"(conf={r['confidence']:.3f}, agree={r['agreement_level']})")

    # Test 3: Record outcome and check weight updates
    ensemble.record_outcome(
        prediction_ts=time.time() - 3600,
        actual_outcome="YES",
        per_model_predictions=result["per_model_predictions"],
        ensemble_prediction=result["ensemble_prediction"],
    )

    # Test 4: Performance report
    report = ensemble.get_performance_report()
    print(f"\n=== Performance Report ===")
    print(f"Total predictions: {report['total_predictions']}")
    print(f"Ensemble accuracy: {report['ensemble_accuracy']}")
    print(f"Current weights: {report['current_weights']}")

    # Test 5: Agreement levels
    # Simulate full agreement
    preds_full = [
        ModelPrediction("A", "YES", 0.8),
        ModelPrediction("B", "YES", 0.75),
        ModelPrediction("C", "YES", 0.9),
    ]
    assert MultiAIEnsemble._get_agreement_level(preds_full) == AgreementLevel.FULL

    # Simulate split
    preds_split = [
        ModelPrediction("A", "YES", 0.8),
        ModelPrediction("B", "NO", 0.75),
    ]
    assert MultiAIEnsemble._get_agreement_level(preds_split) == AgreementLevel.SPLIT

    print("\n[PASS] All tests passed")
