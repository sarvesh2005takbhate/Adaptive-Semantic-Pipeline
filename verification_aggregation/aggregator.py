"""
Module 7+8 — Response Verification & Final Aggregation
========================================================
Validates outputs, estimates confidence, re-queries if needed,
aggregates into final answer, and collects feedback.

Module 7 (Verification):
  - Answer completeness check
  - Dependency consistency
  - Intent-specific validation (math/code/summarization etc.)
  - Confidence scoring
  - Re-query: weak→strong escalation or self-reflection

Module 8 (Aggregation):
  - Single-segment passthrough (zero overhead)
  - Multi-segment synthesis via model call
  - Feedback collection for router improvement
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("verification")


# ═══════════════════════════════════════════════════════════════
#  ANSWER VALIDATOR (intent-specific checks)
# ═══════════════════════════════════════════════════════════════

class AnswerValidator:
    """Validates answer quality per intent type."""

    def validate(self, text: str, intent: str, prompt: str = "") -> Dict[str, Any]:
        """
        Returns:
            valid: bool, score: float (0-1), format_ok: bool, reasons: list
        """
        if not text or not text.strip():
            return {"valid": False, "score": 0.0, "format_ok": False,
                    "reasons": ["empty_output"]}

        validator = {
            "math": self._validate_math,
            "code": self._validate_code,
            "translation": self._validate_translation,
            "summarization": self._validate_summarization,
            "data_analysis": self._validate_data_analysis,
            "simulation": self._validate_simulation,
        }.get(intent, self._validate_general)

        return validator(text, prompt)

    def _validate_math(self, text: str, prompt: str) -> Dict[str, Any]:
        reasons = []
        score = 0.5

        # Should contain numbers or mathematical expressions
        has_numbers = bool(re.search(r"\d", text))
        has_math = bool(re.search(r"[=+\-*/²³√∫∑]", text))
        has_result = bool(re.search(r"(?:answer|result|solution|equals?|=)\s*[:=]?\s*\S", text, re.I))

        if has_numbers:
            score += 0.15
        else:
            reasons.append("no_numbers_in_math_output")
        if has_math or has_result:
            score += 0.15
        if len(text.split()) >= 5:
            score += 0.10

        # Check for step-by-step reasoning
        if re.search(r"(?:step|first|then|therefore|thus|hence)", text, re.I):
            score += 0.10
            reasons.append("has_reasoning_steps")

        return {"valid": score > 0.6, "score": round(score, 2),
                "format_ok": has_numbers, "reasons": reasons}

    def _validate_code(self, text: str, prompt: str) -> Dict[str, Any]:
        reasons = []
        score = 0.5

        has_code = bool(re.search(r"```|def |class |function |import |return |if |for |while ", text))
        has_keywords = bool(re.search(r"\b(?:def|class|function|var|let|const|import)\b", text))

        if has_code:
            score += 0.25
        elif has_keywords:
            score += 0.15
        else:
            reasons.append("no_code_detected")

        # Try to extract and compile Python code
        code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
        if code_blocks:
            for block in code_blocks:
                try:
                    compile(block.strip(), "<string>", "exec")
                    score += 0.15
                    reasons.append("code_compiles")
                except SyntaxError:
                    reasons.append("syntax_error")
                break

        if len(text.split()) >= 10:
            score += 0.10

        return {"valid": score > 0.5, "score": round(min(score, 1.0), 2),
                "format_ok": has_code, "reasons": reasons}

    def _validate_translation(self, text: str, prompt: str) -> Dict[str, Any]:
        score = 0.6
        reasons = []

        if len(text.split()) < 2:
            return {"valid": False, "score": 0.2, "format_ok": False,
                    "reasons": ["too_short"]}

        # Translation should differ from input significantly
        if prompt:
            overlap = len(set(text.lower().split()) & set(prompt.lower().split()))
            total = max(len(text.split()), 1)
            overlap_ratio = overlap / total
            if overlap_ratio > 0.8:
                score -= 0.2
                reasons.append("high_overlap_with_source")
            else:
                score += 0.2

        return {"valid": score > 0.5, "score": round(score, 2),
                "format_ok": True, "reasons": reasons}

    def _validate_summarization(self, text: str, prompt: str) -> Dict[str, Any]:
        score = 0.6
        reasons = []

        # Summary should be shorter than input
        if prompt and len(text) < len(prompt):
            score += 0.2
            reasons.append("shorter_than_input")
        elif prompt:
            score -= 0.1
            reasons.append("longer_than_input")

        if len(text.split()) >= 5:
            score += 0.1

        return {"valid": score > 0.5, "score": round(score, 2),
                "format_ok": True, "reasons": reasons}

    def _validate_data_analysis(self, text: str, prompt: str) -> Dict[str, Any]:
        score = 0.5
        has_numbers = bool(re.search(r"\d", text))
        has_analysis_words = bool(re.search(
            r"\b(?:correlation|mean|median|trend|outlier|significant|distribution|result)\b",
            text, re.I))

        if has_numbers:
            score += 0.2
        if has_analysis_words:
            score += 0.2
        if len(text.split()) >= 10:
            score += 0.1

        return {"valid": score > 0.5, "score": round(score, 2),
                "format_ok": has_numbers, "reasons": []}

    def _validate_simulation(self, text: str, prompt: str) -> Dict[str, Any]:
        score = 0.5
        has_numbers = bool(re.search(r"\d", text))
        has_sim_words = bool(re.search(
            r"\b(?:simulation|iteration|result|converge|probability|model|run)\b",
            text, re.I))

        if has_numbers:
            score += 0.2
        if has_sim_words:
            score += 0.2
        if len(text.split()) >= 10:
            score += 0.1

        return {"valid": score > 0.5, "score": round(score, 2),
                "format_ok": True, "reasons": []}

    def _validate_general(self, text: str, prompt: str) -> Dict[str, Any]:
        score = 0.5
        wc = len(text.split())

        if wc >= 5:
            score += 0.2
        if wc >= 20:
            score += 0.1
        if text.strip().endswith((".", "!", "?", "```")):
            score += 0.1

        # Check it's not just a refusal
        if re.search(r"\bi (?:cannot|can't|won't) (?:help|assist)\b", text, re.I):
            score -= 0.3

        return {"valid": score > 0.4, "score": round(min(score, 1.0), 2),
                "format_ok": True, "reasons": []}


# ═══════════════════════════════════════════════════════════════
#  VERIFICATION CONTROLLER (Module 7)
# ═══════════════════════════════════════════════════════════════

class VerificationController:
    """
    Verifies all segment outputs, estimates confidence,
    and flags segments needing re-query.

    Usage:
        verifier = VerificationController()
        result = verifier.verify(execution_result, routed_segments, prompt_text)
    """

    def __init__(self, confidence_threshold: float = 0.60):
        self.validator = AnswerValidator()
        self.confidence_threshold = confidence_threshold
        self._stats = {"total": 0, "passed": 0, "failed": 0,
                       "requeried": 0, "escalated": 0}

    def verify(
        self,
        execution_result: Dict[str, Any],
        routed_segments: List[Dict[str, Any]],
        prompt_text: str = "",
    ) -> Dict[str, Any]:
        """
        Verify all outputs from Module 5.

        Returns:
            verified_outputs: list of verified segment dicts
            verification_stats: summary statistics
        """
        start = time.time()
        outputs = execution_result.get("subtask_outputs", [])

        # Build indices
        routed_idx = {s.get("segment_id", s.get("id", -1)): s for s in routed_segments}

        verified = []

        for output in outputs:
            sid = output["segment_id"]
            text = output.get("output", "")
            route_tier = output.get("route_tier", output.get("model_used", "unknown"))
            routed = routed_idx.get(sid, {})
            intent = routed.get("intent", "explanation")
            seg_text = routed.get("text", "")
            depends_on = routed.get("depends_on", [])

            self._stats["total"] += 1

            # Skip blocked/failed
            if output.get("status") in ("blocked", "failed", "skipped"):
                verified.append({
                    "segment_id": sid,
                    "verified_text": text,
                    "confidence": 0.0,
                    "verification_passed": False,
                    "model_used": route_tier,
                    "flags": ["status_" + output.get("status", "unknown")],
                })
                self._stats["failed"] += 1
                continue

            # ── Validate ────────────────────────────────────
            validation = self.validator.validate(text, intent, seg_text)

            # ── Dependency check ────────────────────────────
            dep_ok = self._check_dependencies(sid, text, depends_on, outputs)

            # ── Confidence scoring ──────────────────────────
            confidence = self._estimate_confidence(
                validation, dep_ok, text, route_tier, intent)

            # ── Collect flags ───────────────────────────────
            flags = list(validation.get("reasons", []))
            if not dep_ok:
                flags.append("dependency_inconsistent")
            if confidence < self.confidence_threshold:
                flags.append("low_confidence")

            passed = confidence >= self.confidence_threshold
            if passed:
                self._stats["passed"] += 1
            else:
                self._stats["failed"] += 1

            verified.append({
                "segment_id": sid,
                "verified_text": text,
                "confidence": round(confidence, 3),
                "verification_passed": passed,
                "model_used": route_tier,
                "intent": intent,
                "validation_score": validation["score"],
                "dependency_consistent": dep_ok,
                "needs_requery": not passed and "weak" in str(route_tier).lower(),
                "flags": flags,
            })

        verification_time = time.time() - start

        return {
            "verified_outputs": verified,
            "verification_stats": {
                **self._stats,
                "verification_time": round(verification_time, 4),
                "pass_rate": round(self._stats["passed"] / max(self._stats["total"], 1), 4),
            },
        }

    def _check_dependencies(
        self, sid: int, text: str,
        depends_on: List[int],
        all_outputs: List[Dict],
    ) -> bool:
        """Check if dependent segment's output references upstream content."""
        if not depends_on:
            return True  # no dependencies = always consistent

        output_idx = {o["segment_id"]: o.get("output", "") for o in all_outputs}

        for dep_id in depends_on:
            dep_text = output_idx.get(dep_id, "")
            if not dep_text:
                return False  # upstream missing

            # Check for some semantic overlap (lightweight)
            dep_words = set(dep_text.lower().split())
            seg_words = set(text.lower().split())
            common = dep_words & seg_words - {"the", "a", "an", "is", "are", "was", "in", "to", "of", "and", "for"}

            if len(common) < 2 and len(dep_words) > 5:
                return False  # no meaningful overlap

        return True

    def _estimate_confidence(
        self,
        validation: Dict,
        dep_ok: bool,
        text: str,
        route_tier: str,
        intent: str,
    ) -> float:
        """Multi-signal confidence estimation."""
        score = 0.3  # base

        # Validation score (0-1) contributes up to 0.35
        score += validation.get("score", 0.5) * 0.35

        # Dependency consistency
        if dep_ok:
            score += 0.15

        # Format ok
        if validation.get("format_ok", False):
            score += 0.10

        # Strong model bonus (more trustworthy)
        if "strong" in str(route_tier).lower():
            score += 0.10

        # Length penalty
        wc = len(text.split())
        if wc < 3:
            score -= 0.20
        elif wc > 500:
            score -= 0.05  # suspiciously long

        return round(min(max(score, 0.0), 1.0), 3)

    def stats(self) -> Dict[str, Any]:
        return dict(self._stats)

    def reset(self):
        self._stats = {"total": 0, "passed": 0, "failed": 0,
                       "requeried": 0, "escalated": 0}


# ═══════════════════════════════════════════════════════════════
#  FINAL AGGREGATOR (Module 8)
# ═══════════════════════════════════════════════════════════════

class FinalAggregator:
    """
    Aggregates verified outputs into a coherent final answer.

    Single-segment: passthrough (zero overhead).
    Multi-segment: synthesis via model or concatenation.
    """

    def __init__(self, model=None):
        self.model = model  # should have .infer(text) or .generate(text)

    def aggregate(
        self,
        verified_outputs: List[Dict[str, Any]],
        prompt_text: str = "",
    ) -> Dict[str, Any]:
        """
        Aggregate verified outputs into final answer.

        Returns:
            final_answer: str
            aggregation_method: str
            latency: float
            segments_used: int
        """
        start = time.time()

        # Filter to passed outputs only
        usable = [v for v in verified_outputs if v.get("verification_passed", True)]
        if not usable:
            usable = verified_outputs  # fallback: use all

        # Sort by segment_id
        usable = sorted(usable, key=lambda x: x.get("segment_id", 0))

        # ── Single segment: passthrough ─────────────────────
        if len(usable) == 1:
            text = usable[0].get("verified_text", "")
            return {
                "final_answer": text,
                "aggregation_method": "passthrough",
                "latency": round(time.time() - start, 4),
                "segments_used": 1,
            }

        # ── Multi segment: synthesize ───────────────────────
        parts = []
        for v in usable:
            sid = v.get("segment_id", "?")
            intent = v.get("intent", "")
            text = v.get("verified_text", "")
            label = f"[Task {sid}" + (f" - {intent}" if intent else "") + "]"
            parts.append(f"{label}\n{text}")

        combined = "\n\n".join(parts)

        # For 2 segments: smart concatenation (no model call needed)
        # For 3+ segments: model synthesis if available
        if len(usable) <= 2 or self.model is None:
            return {
                "final_answer": combined,
                "aggregation_method": "concatenation",
                "latency": round(time.time() - start, 4),
                "segments_used": len(usable),
            }

        # 3+ segments: try model synthesis
        try:
            synthesis_prompt = (
                f"The user asked: {prompt_text}\n\n"
                f"Here are the completed subtask results:\n\n{combined}\n\n"
                f"Synthesize these into one coherent final answer."
            )
            if hasattr(self.model, "generate"):
                final = self.model.generate(synthesis_prompt)
            elif hasattr(self.model, "infer"):
                final = self.model.infer(synthesis_prompt).get("output", combined)
            else:
                final = combined

            return {
                "final_answer": final,
                "aggregation_method": "model_synthesis",
                "latency": round(time.time() - start, 4),
                "segments_used": len(usable),
            }
        except Exception as e:
            log.warning(f"Synthesis failed: {e}, falling back to concatenation")

        # Fallback: concatenation
        return {
            "final_answer": combined,
            "aggregation_method": "concatenation",
            "latency": round(time.time() - start, 4),
            "segments_used": len(usable),
        }


# ═══════════════════════════════════════════════════════════════
#  FEEDBACK COLLECTOR (for router improvement)
# ═══════════════════════════════════════════════════════════════

class FeedbackCollector:
    """
    Collects feedback from verification for router online learning.
    Logs segments where weak model had low confidence → candidates
    for re-routing to strong model.
    """

    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def collect(
        self,
        verified_outputs: List[Dict[str, Any]],
        routed_segments: List[Dict[str, Any]],
    ):
        """Collect feedback from one prompt's results."""
        routed_idx = {s.get("segment_id", -1): s for s in routed_segments}

        for v in verified_outputs:
            sid = v["segment_id"]
            routed = routed_idx.get(sid, {})

            self.records.append({
                "segment_id": sid,
                "intent": v.get("intent", routed.get("intent", "unknown")),
                "complexity": routed.get("complexity_score", 0.5),
                "route_decision": v.get("model_used", "unknown"),
                "confidence": v.get("confidence", 0.0),
                "passed": v.get("verification_passed", False),
                "needs_escalation": (
                    not v.get("verification_passed", True)
                    and "weak" in str(v.get("model_used", "")).lower()
                ),
            })

    def get_escalation_candidates(self) -> List[Dict]:
        """Get segments that should have been routed to strong model."""
        return [r for r in self.records if r.get("needs_escalation")]

    def stats(self) -> Dict[str, Any]:
        if not self.records:
            return {"total": 0}
        passed = sum(1 for r in self.records if r["passed"])
        escalation = len(self.get_escalation_candidates())
        return {
            "total": len(self.records),
            "passed": passed,
            "failed": len(self.records) - passed,
            "pass_rate": round(passed / len(self.records), 4),
            "escalation_candidates": escalation,
        }

    def reset(self):
        self.records.clear()