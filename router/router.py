"""
Module 4 — LightGBM-Based Router
==================================
Routes each segment to the appropriate model tier based on learned
features and heuristic fallbacks.

Research-grade features:
  - 30-dim feature vector with documented feature names
  - Calibrated probability thresholds with uncertainty zone
  - Safety override layer
  - Detailed routing explanation (for interpretability)
  - Feature importance tracking

Architecture:
  Step 1 — Safety check      → block unsafe content
  Step 2 — Feature extraction → 30-dim vector
  Step 3 — LightGBM P(strong) prediction
  Step 4 — Threshold routing with uncertainty fallback

Routing targets:
  weak_model      — cheap/fast tier (Llama-3.1-8B)
  strong_model    — expensive/capable tier (Llama-3.3-70B)
  safe_block      — unsafe content blocked
  verify_required — routed strong + flagged for verification
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config.labels import (
    INTENT_LABELS,
    LABEL2ID,
    NUM_INTENTS,
    ROUTE_BLOCK,
    ROUTE_STRONG,
    ROUTE_VERIFY,
    ROUTE_WEAK,
    STRONG_INTENTS,
    WEAK_INTENTS,
)

try:
    import lightgbm as lgb
    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False


# ═══════════════════════════════════════════════════════════════
#  FEATURE NAMES (documented, fixed order — critical for repro)
# ═══════════════════════════════════════════════════════════════

# First 19 are scalar features, then 11 intent one-hot = 30 total
SCALAR_FEATURE_NAMES = [
    "word_count",
    "char_count",
    "avg_word_length",
    "complexity_score",
    "intent_confidence",
    "depth",
    "is_dependent",
    "has_math_op",
    "has_numbers",
    "has_code_fence",
    "has_question",
    "math_keyword_count",
    "code_keyword_count",
    "reasoning_keyword_count",
    "sentence_count",
    "non_english_ratio",
    "uppercase_ratio",
    "punctuation_density",
    "unsafe_flag",
    "difficulty_score",
]

FEATURE_NAMES = SCALAR_FEATURE_NAMES + [f"intent_{lab}" for lab in INTENT_LABELS]
NUM_FEATURES = len(FEATURE_NAMES)  # 19 + 11 = 30


# ═══════════════════════════════════════════════════════════════
#  KEYWORD BANKS
# ═══════════════════════════════════════════════════════════════

_MATH_KW = {
    "solve", "calculate", "compute", "evaluate", "simplify",
    "integrate", "differentiate", "equation", "formula", "sum",
    "product", "percentage", "ratio", "average", "probability",
    "algebra", "geometry", "calculus", "matrix", "polynomial",
    "derivative", "integral", "eigenvalue", "determinant", "limit",
    "convergence", "divergence", "fourier", "laplace", "prove",
}

_CODE_KW = {
    "code", "program", "function", "class", "method", "algorithm",
    "implement", "debug", "compile", "script", "python", "java",
    "javascript", "typescript", "rust", "html", "sql", "api",
    "recursion", "binary", "stack", "queue", "sort", "search",
    "merge", "tree", "hash", "trie", "graph", "linked list",
    "data structure", "thread", "async", "concurrent",
}

_REASONING_KW = {
    "reason", "reasoning", "logic", "logical", "argue", "analyse",
    "analyze", "evaluate", "infer", "deduce", "conclude",
    "therefore", "hence", "thus", "given that", "assuming",
    "evidence", "hypothesis", "compare", "contrast", "critique",
    "synthesize", "implications", "trade-off", "assess",
}


# ═══════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_features(segment: Dict[str, Any]) -> np.ndarray:
    """
    Convert an annotated segment into a fixed-size 30-dim feature vector.

    Parameters
    ----------
    segment : dict
        Must contain at minimum 'text' and 'intent'.
        Optionally: complexity_score, intent_confidence, depth,
        depends_on, unsafe_candidate.

    Returns
    -------
    np.ndarray of shape (30,)
    """
    text = segment.get("text", "")
    intent = segment.get("intent", segment.get("intent_label", "explanation"))
    lower = text.lower()
    words = text.split()

    # ── Scalar features (19) ────────────────────────────────
    word_count = len(words)
    char_count = len(text)
    avg_word_length = (char_count / word_count) if word_count > 0 else 0.0
    complexity_score = float(segment.get("complexity_score", 0.3))
    intent_confidence = float(segment.get("intent_confidence", 0.5))
    depth = float(segment.get("depth", 0))
    is_dependent = 1.0 if segment.get("depends_on") else 0.0

    has_math_op = 1.0 if re.search(r"[=+\-*/^∫∑∂]", text) else 0.0
    has_numbers = 1.0 if re.search(r"\d", text) else 0.0
    has_code_fence = 1.0 if "```" in text else 0.0
    has_question = 1.0 if text.strip().endswith("?") else 0.0

    math_kw_count = sum(1 for kw in _MATH_KW if kw in lower)
    code_kw_count = sum(1 for kw in _CODE_KW if kw in lower)
    reasoning_kw_count = sum(1 for kw in _REASONING_KW if kw in lower)

    sentence_count = max(len(re.split(r"[.!?]+", text.strip())), 1)

    alpha_chars = sum(1 for ch in text if ch.isalpha())
    non_ascii = sum(1 for ch in text if ord(ch) > 127 and not ch.isspace())
    non_english_ratio = (non_ascii / alpha_chars) if alpha_chars > 0 else 0.0

    upper_chars = sum(1 for ch in text if ch.isupper())
    uppercase_ratio = (upper_chars / char_count) if char_count > 0 else 0.0

    punct_chars = sum(1 for ch in text if ch in ".,;:!?()[]{}\"'-")
    punctuation_density = (punct_chars / char_count) if char_count > 0 else 0.0

    unsafe_flag = 1.0 if segment.get("unsafe_candidate", False) else 0.0
    difficulty_score = complexity_score + 0.2 * reasoning_kw_count
    # ── Intent one-hot (11) ─────────────────────────────────
    intent_one_hot = [0.0] * len(INTENT_LABELS)

    vec = [
        word_count, char_count, avg_word_length,
        complexity_score, intent_confidence,
        depth, is_dependent,
        has_math_op, has_numbers, has_code_fence, has_question,
        math_kw_count, code_kw_count, reasoning_kw_count,
        sentence_count, non_english_ratio, uppercase_ratio,
        punctuation_density, unsafe_flag,
        *intent_one_hot,
    ]
    vec.append(difficulty_score)

    assert len(vec) == NUM_FEATURES, f"Expected {NUM_FEATURES} features, got {len(vec)}"
    return np.array(vec, dtype=np.float64)


def extract_features_batch(segments: List[Dict[str, Any]]) -> np.ndarray:
    """Extract features for multiple segments. Returns shape (N, 30)."""
    return np.array([extract_features(seg) for seg in segments], dtype=np.float64)


# ═══════════════════════════════════════════════════════════════
#  ROUTER CLASS
# ═══════════════════════════════════════════════════════════════

class LearningBasedRouter:
    """
    LightGBM-powered segment router with heuristic fallback.

    Parameters
    ----------
    model_path : str
        Path to saved LightGBM model file.
    strong_threshold : float
        P(strong) above this → route to strong model.
    weak_threshold : float
        P(strong) below this → route to weak model.
    complexity_escalation : float
        If weak-intent but complexity above this → escalate to strong.
    """

    def __init__(
        self,
        model_path: str = "models/router_model.txt",
        strong_threshold: float = 0.65,
        weak_threshold: float = 0.35,
        complexity_escalation: float = 0.60,
    ) -> None:
        self.model_path = model_path
        self.strong_threshold = strong_threshold
        self.weak_threshold = weak_threshold
        self.complexity_escalation = complexity_escalation
        self.model: Optional[lgb.Booster] = None

        if os.path.isfile(model_path) and _HAS_LIGHTGBM:
            self.load_model(model_path)

    def load_model(self, model_path: str) -> None:
        if not _HAS_LIGHTGBM:
            raise ImportError("LightGBM not installed")
        self.model = lgb.Booster(model_file=model_path)

    def save_model(self, path: str) -> None:
        if self.model is not None:
            self.model.save_model(path)

    # ─────────────────────────────────────────────────────────
    #  SINGLE SEGMENT ROUTING
    # ─────────────────────────────────────────────────────────

    def route(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route a single segment. Returns enriched segment dict with:
          - route: str (weak_model|strong_model|safe_block|verify_required)
          - route_confidence: float
          - route_method: str (ml|heuristic|safety)
          - route_reason: str
        """
        intent = segment.get("intent", segment.get("intent_label", "explanation"))
        complexity = float(segment.get("complexity_score", 0.5))
        intent_conf = float(segment.get("intent_confidence", 0.5))

        # ── Step 1: Safety override ─────────────────────────
        if segment.get("unsafe_candidate", False):
            return self._make_result(
                segment, ROUTE_BLOCK, 1.0, "safety",
                "Unsafe content detected — blocked before execution"
            )

        # ── Step 2: ML prediction ───────────────────────────
        if self.model is not None:
            features = extract_features(segment)
            prob_strong = float(self.model.predict([features])[0])

            if prob_strong > self.strong_threshold:
                # High confidence strong
                route = ROUTE_STRONG
                if intent_conf < 0.5:
                    route = ROUTE_VERIFY  # strong but uncertain intent
                return self._make_result(
                    segment, route, prob_strong, "ml",
                    f"P(strong)={prob_strong:.3f} > {self.strong_threshold}"
                )

            elif prob_strong < self.weak_threshold:
                return self._make_result(
                    segment, ROUTE_WEAK, 1.0 - prob_strong, "ml",
                    f"P(strong)={prob_strong:.3f} < {self.weak_threshold}"
                )

            else:
                # Uncertain zone — fall through to heuristic
                pass

        # ── Step 3: Heuristic fallback ──────────────────────
        route, reason = self._heuristic_route(intent, complexity, intent_conf)
        conf = max(complexity, 1.0 - complexity)  # proxy confidence

        return self._make_result(segment, route, conf, "heuristic", reason)

    def _heuristic_route(
        self, intent: str, complexity: float, intent_conf: float
    ) -> Tuple[str, str]:
        """Rule-based routing for uncertain ML predictions."""

        if intent in STRONG_INTENTS:
            if complexity < 0.30 and intent_conf > 0.90:
                return ROUTE_WEAK, f"Strong intent '{intent}' but trivial complexity ({complexity:.2f})"
            return ROUTE_STRONG, f"Strong intent '{intent}' with complexity {complexity:.2f}"

        if intent in WEAK_INTENTS:
            if complexity > self.complexity_escalation:
                return ROUTE_STRONG, f"Weak intent '{intent}' but high complexity ({complexity:.2f}) → escalated"
            return ROUTE_WEAK, f"Weak intent '{intent}' with complexity {complexity:.2f}"

        # Unknown intent
        if complexity > 0.50:
            return ROUTE_STRONG, f"Unknown intent '{intent}', complexity {complexity:.2f} → strong"
        return ROUTE_WEAK, f"Unknown intent '{intent}', complexity {complexity:.2f} → weak"

    @staticmethod
    def _make_result(
        segment: Dict[str, Any],
        route: str,
        confidence: float,
        method: str,
        reason: str,
    ) -> Dict[str, Any]:
        """Build the routing result dict."""
        result = dict(segment)
        result["route"] = route
        result["route_confidence"] = round(confidence, 4)
        result["route_method"] = method
        result["route_reason"] = reason
        return result

    # ─────────────────────────────────────────────────────────
    #  BATCH ROUTING
    # ─────────────────────────────────────────────────────────

    def route_all(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Route a list of segments."""
        return [self.route(seg) for seg in segments]

    # ─────────────────────────────────────────────────────────
    #  PRINT
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def print_routes(routed_segments: List[Dict[str, Any]]) -> None:
        """Pretty-print routing decisions."""
        print("=" * 70)
        print("  ROUTING DECISIONS")
        print("=" * 70)

        for seg in routed_segments:
            sid = seg.get("segment_id", "?")
            text = seg.get("text", "")[:60]
            route = seg.get("route", "?")
            conf = seg.get("route_confidence", 0)
            method = seg.get("route_method", "?")
            reason = seg.get("route_reason", "")

            icon = {
                ROUTE_STRONG: "🔴",
                ROUTE_WEAK: "🟢",
                ROUTE_BLOCK: "🚫",
                ROUTE_VERIFY: "🟡",
            }.get(route, "❓")

            print(f"\n  {icon} [{sid}] \"{text}{'…' if len(seg.get('text','')) > 60 else ''}\"")
            print(f"       route  = {route} (conf: {conf:.3f}, method: {method})")
            print(f"       reason = {reason}")

        print("\n" + "=" * 70)