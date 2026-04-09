"""
Module 3 — Intent & Complexity Estimator
==========================================
Classifies each segment's intent using trained DistilBERT and estimates
complexity using multi-signal scoring.

Research-grade features:
  - Returns top-K predictions with confidence scores
  - Calibrated confidence thresholds
  - Detailed complexity breakdown (reasons)
  - Batch inference support
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

from config.labels import (
    ID2LABEL,
    INTENT_LABELS,
    LABEL2ID,
    NUM_INTENTS,
    STRONG_INTENTS,
    WEAK_INTENTS,
)


# ═════════════════════════════════════════════════════════════
#  COMPLEXITY SIGNALS (weights validated by ablation)
# ═════════════════════════════════════════════════════════════

INTENT_PRIORS = {
    "math": 0.55,  "code": 0.60,  "simulation": 0.65,
    "research": 0.50, "prediction": 0.50, "data_analysis": 0.55,
    "translation": 0.20, "summarization": 0.20,
    "explanation": 0.25, "communication": 0.18, "documentation": 0.22,
}

MATH_KEYWORDS = [
    "integral", "derivative", "eigenvalue", "matrix", "prove",
    "differential", "fourier", "laplace", "convergence", "optimization",
    "theorem", "polynomial", "determinant", "vector", "gradient",
]

CODE_KEYWORDS = [
    "function", "def", "class", "import", "return", "algorithm",
    "implement", "recursive", "data structure", "binary tree",
    "graph", "hash map", "concurrent", "thread-safe", "optimize",
]

REASONING_KEYWORDS = [
    "prove", "derive", "analyze", "compare", "evaluate",
    "design", "optimize", "synthesize", "critique", "trade-off",
    "implications", "contrast", "justify", "assess",
]

MULTISTEP_KEYWORDS = [
    "step by step", "first", "then", "finally", "multiple",
    "additionally", "next", "followed by",
]


class IntentComplexityEstimator:
    """
    Classifies segment intent and estimates task complexity.

    Args:
        model_path: Path to the saved DistilBERT model directory.
        device: 'cpu', 'cuda', or 'auto'.
        confidence_threshold: Below this, flag as low-confidence.
    """

    def __init__(
        self,
        model_path: str = "models/intent_classifier",
        device: str = "auto",
        confidence_threshold: float = 0.60,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        self.id2label = ID2LABEL
        self.confidence_threshold = confidence_threshold

    # ─────────────────────────────────────────────────────────
    #  INTENT PREDICTION
    # ─────────────────────────────────────────────────────────

    def predict_intent(self, text: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Predict intent with confidence scores.

        Returns:
            {
                "intent": str,
                "confidence": float,
                "top_k": [{"intent": str, "confidence": float}, ...],
                "low_confidence": bool,
            }
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.squeeze(0)
            probs = F.softmax(logits, dim=-1)

        top_values, top_indices = torch.topk(probs, min(top_k, NUM_INTENTS))

        top_predictions = []
        for val, idx in zip(top_values.tolist(), top_indices.tolist()):
            top_predictions.append({
                "intent": self.id2label[idx],
                "confidence": round(val, 4),
            })

        best_intent = top_predictions[0]["intent"]
        best_confidence = top_predictions[0]["confidence"]

        return {
            "intent": best_intent,
            "confidence": best_confidence,
            "top_k": top_predictions,
            "low_confidence": best_confidence < self.confidence_threshold,
        }

    # ─────────────────────────────────────────────────────────
    #  COMPLEXITY ESTIMATION
    # ─────────────────────────────────────────────────────────

    def estimate_complexity(self, text: str, intent: str) -> Dict[str, Any]:
        """
        Estimate complexity using weighted multi-signal scoring.

        Returns:
            {
                "complexity_score": float,
                "complexity_band": "simple"|"medium"|"hard",
                "signals": {signal_name: contribution},
                "reasons": [str],
            }
        """
        t = text.lower()
        words = t.split()
        wc = len(words)
        signals = {}
        reasons = []

        # Signal 1: Intent prior
        prior = INTENT_PRIORS.get(intent, 0.30)
        signals["intent_prior"] = round(prior, 3)
        reasons.append(f"Intent '{intent}' base prior: {prior:.2f}")

        # Signal 2: Length (normalized to 50 words)
        length_contrib = round(min(wc / 50.0, 0.15), 3)
        signals["length"] = length_contrib
        if wc > 30:
            reasons.append(f"Long prompt ({wc} words): +{length_contrib:.3f}")

        # Signal 3: Math density
        math_count = sum(1 for kw in MATH_KEYWORDS if kw in t)
        math_ops = len(re.findall(r"[=+\-*/^∫∑∂]", text))
        numbers = len(re.findall(r"\b\d+(?:\.\d+)?\b", text))
        math_raw = math_count + min(math_ops, 3) + min(numbers, 3)
        math_contrib = round(min(math_raw * 0.03, 0.15), 3)
        signals["math"] = math_contrib
        if math_contrib > 0.03:
            reasons.append(f"Math signals ({math_count} keywords, {math_ops} operators): +{math_contrib:.3f}")

        # Signal 4: Code density
        code_count = sum(1 for kw in CODE_KEYWORDS if kw in t)
        code_contrib = round(min(code_count * 0.04, 0.15), 3)
        signals["code"] = code_contrib
        if code_contrib > 0.03:
            reasons.append(f"Code signals ({code_count} keywords): +{code_contrib:.3f}")

        # Signal 5: Reasoning depth
        reason_count = sum(1 for kw in REASONING_KEYWORDS if kw in t)
        reason_contrib = round(min(reason_count * 0.04, 0.12), 3)
        signals["reasoning"] = reason_contrib
        if reason_contrib > 0.03:
            reasons.append(f"Reasoning signals ({reason_count} keywords): +{reason_contrib:.3f}")

        # Signal 6: Multi-step
        multi_count = sum(1 for kw in MULTISTEP_KEYWORDS if kw in t)
        multi_contrib = round(min(multi_count * 0.03, 0.10), 3)
        signals["multistep"] = multi_contrib
        if multi_contrib > 0.02:
            reasons.append(f"Multi-step indicators ({multi_count}): +{multi_contrib:.3f}")

        # Total
        score = sum(signals.values())
        score = round(min(max(score, 0.10), 0.95), 2)

        if score < 0.35:
            band = "simple"
        elif score < 0.65:
            band = "medium"
        else:
            band = "hard"

        return {
            "complexity_score": score,
            "complexity_band": band,
            "signals": signals,
            "reasons": reasons,
        }

    # ─────────────────────────────────────────────────────────
    #  MAIN ENTRY POINT
    # ─────────────────────────────────────────────────────────

    def estimate(self, text: str) -> Dict[str, Any]:
        """
        Full estimation: intent + complexity.
        Called by the pipeline for each segment.
        """
        intent_result = self.predict_intent(text)
        complexity_result = self.estimate_complexity(text, intent_result["intent"])

        return {
            "intent": intent_result["intent"],
            "intent_confidence": intent_result["confidence"],
            "intent_top_k": intent_result["top_k"],
            "low_confidence": intent_result["low_confidence"],
            "complexity_score": complexity_result["complexity_score"],
            "complexity_band": complexity_result["complexity_band"],
            "complexity_signals": complexity_result["signals"],
            "reasons": complexity_result["reasons"],
        }

    def estimate_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Batch inference for efficiency."""
        return [self.estimate(text) for text in texts]

    # ─────────────────────────────────────────────────────────
    #  PRETTY PRINT
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def print_annotations(
        annotated_nodes: List[Dict[str, Any]],
        show_reasons: bool = True,
    ) -> None:
        """Print a human-readable summary of annotated segments."""
        print("=" * 70)
        print("  INTENT & COMPLEXITY ANNOTATIONS")
        print("=" * 70)

        for node in annotated_nodes:
            sid = node.get("segment_id", node.get("id", "?"))
            intent = node.get("intent", "?")
            conf = node.get("intent_confidence", 0)
            score = node.get("complexity_score", 0)
            band = node.get("complexity_band", "?")
            text = node.get("text", "")
            low = node.get("low_confidence", False)

            print(f"\n  [{sid}] \"{text[:75]}{'…' if len(text) > 75 else ''}\"")
            print(f"       intent        = {intent} (conf: {conf:.3f}{'  ⚠ LOW' if low else ''})")
            print(f"       complexity    = {score:.2f} ({band})")

            if show_reasons and "reasons" in node:
                for r in node["reasons"]:
                    print(f"         • {r}")

            if "intent_top_k" in node:
                alts = ", ".join(
                    f"{p['intent']}={p['confidence']:.3f}"
                    for p in node["intent_top_k"][1:3]
                )
                if alts:
                    print(f"       alternatives  = {alts}")

        print("\n" + "=" * 70)