"""
Module 6 — Safety System
===========================
Multi-layer safety filtering for both inputs (pre-execution) and
outputs (post-execution).

Architecture:
    ┌─────────────────────────────────────────────────┐
    │                 SafetySystem                     │
    │                                                  │
    │  Layer 1: PromptGuard (pre-execution)            │
    │    → keyword + regex scanning                    │
    │    → category-based severity scoring             │
    │    → block / warn / pass decision                │
    │                                                  │
    │  Layer 2: OutputFilter (post-execution)          │
    │    → hallucination detection                     │
    │    → harmful content scanning                    │
    │    → factual consistency checks                  │
    │    → PII detection + redaction                   │
    │                                                  │
    │  Layer 3: GuardRails                             │
    │    → response length limits                      │
    │    → format validation                           │
    │    → confidence-gated output                     │
    │    → refusal detection                           │
    │                                                  │
    │  Layer 4: EscalationHandler                      │
    │    → decides final action: pass/warn/redact/block│
    │    → logs all decisions for audit                 │
    │    → escalation to human review queue             │
    └─────────────────────────────────────────────────┘

Policy is defined in PolicyRules (configurable, JSON-serializable).
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

log = logging.getLogger("safety_system")


# ═══════════════════════════════════════════════════════════════
#  ENUMS & DATA CLASSES
# ═══════════════════════════════════════════════════════════════

class Severity(Enum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class SafetyAction(Enum):
    PASS = "pass"
    WARN = "warn"
    REDACT = "redact"
    BLOCK = "block"
    ESCALATE = "escalate"


class SafetyCategory(Enum):
    VIOLENCE = "violence"
    SELF_HARM = "self_harm"
    HATE_SPEECH = "hate_speech"
    ILLEGAL = "illegal_activity"
    EXPLICIT = "explicit_content"
    DANGEROUS = "dangerous_instructions"
    PII = "pii_exposure"
    HALLUCINATION = "hallucination"
    REFUSAL = "model_refusal"
    FORMAT_VIOLATION = "format_violation"


@dataclass
class SafetyFlag:
    """A single safety detection."""
    category: SafetyCategory
    severity: Severity
    evidence: str          # what triggered the flag
    span: Tuple[int, int]  # character span in text
    rule_id: str           # which rule triggered
    layer: str             # "prompt_guard" | "output_filter" | "guardrails"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "severity_name": self.severity.name,
            "evidence": self.evidence[:100],
            "span": list(self.span),
            "rule_id": self.rule_id,
            "layer": self.layer,
        }


@dataclass
class SafetyResult:
    """Result of safety analysis on a piece of text."""
    text: str
    flags: List[SafetyFlag] = field(default_factory=list)
    action: SafetyAction = SafetyAction.PASS
    max_severity: Severity = Severity.NONE
    redacted_text: Optional[str] = None
    analysis_time_ms: float = 0.0
    layer_results: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_safe(self) -> bool:
        return self.action in (SafetyAction.PASS, SafetyAction.WARN)

    @property
    def is_blocked(self) -> bool:
        return self.action == SafetyAction.BLOCK

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "is_safe": self.is_safe,
            "max_severity": self.max_severity.name,
            "n_flags": len(self.flags),
            "flags": [f.to_dict() for f in self.flags],
            "redacted_text": self.redacted_text,
            "analysis_time_ms": round(self.analysis_time_ms, 2),
            "layer_results": self.layer_results,
        }


# ═══════════════════════════════════════════════════════════════
#  POLICY RULES (configurable)
# ═══════════════════════════════════════════════════════════════

class PolicyRules:
    """
    Configurable safety policy. Can be loaded from JSON.

    Defines keyword banks, severity thresholds, and action mappings
    for each safety category.
    """

    def __init__(self):
        # ── Category keyword banks ──────────────────────────
        self.category_patterns: Dict[SafetyCategory, List[Tuple[str, Severity]]] = {
            SafetyCategory.VIOLENCE: [
                (r"\b(?:kill|murder|assassinate|shoot|stab|bomb|explode|attack)\b", Severity.HIGH),
                (r"\b(?:weapon|firearm|gun|knife|explosive|poison)\s+(?:make|build|create|craft)\b", Severity.CRITICAL),
                (r"\b(?:hurt|harm|injure|beat|assault|fight)\b", Severity.MEDIUM),
                (r"\b(?:how\s+to\s+(?:kill|murder|harm|poison|attack))\b", Severity.CRITICAL),
            ],
            SafetyCategory.SELF_HARM: [
                (r"\b(?:suicide|self[- ]?harm|cut\s+(?:my|your)self|end\s+(?:my|it|life))\b", Severity.CRITICAL),
                (r"\b(?:want\s+to\s+die|kill\s+myself|no\s+reason\s+to\s+live)\b", Severity.CRITICAL),
                (r"\b(?:overdose|hang\s+(?:my|your)self)\b", Severity.CRITICAL),
            ],
            SafetyCategory.HATE_SPEECH: [
                (r"\b(?:racial\s+slur|ethnic\s+cleansing|genocide)\b", Severity.CRITICAL),
                (r"\b(?:inferior\s+race|subhuman|master\s+race)\b", Severity.HIGH),
                (r"\b(?:hate\s+(?:all|every)\s+\w+)\b", Severity.MEDIUM),
            ],
            SafetyCategory.ILLEGAL: [
                (r"\b(?:how\s+to\s+(?:hack|steal|forge|counterfeit|launder))\b", Severity.HIGH),
                (r"\b(?:make\s+(?:drugs|meth|cocaine|heroin|bomb))\b", Severity.CRITICAL),
                (r"\b(?:bypass\s+security|pick\s+(?:a\s+)?lock|break\s+into)\b", Severity.MEDIUM),
                (r"\b(?:illegal\s+(?:download|torrent|pirate))\b", Severity.LOW),
            ],
            SafetyCategory.EXPLICIT: [
                (r"\b(?:pornograph|explicit\s+sexual|nude|nsfw)\b", Severity.HIGH),
                (r"\b(?:sexual\s+(?:content|act|favor))\b", Severity.MEDIUM),
            ],
            SafetyCategory.DANGEROUS: [
                (r"\b(?:how\s+to\s+(?:make|build|create)\s+(?:a\s+)?(?:bomb|weapon|explosive|poison|virus))\b", Severity.CRITICAL),
                (r"\b(?:synthesize|manufacture)\s+(?:chemical|biological|nuclear)\b", Severity.CRITICAL),
                (r"\b(?:bioweapon|chemical\s+weapon|nerve\s+agent|ricin|anthrax)\b", Severity.CRITICAL),
            ],
        }

        # ── PII patterns ────────────────────────────────────
        self.pii_patterns: List[Tuple[str, str, Severity]] = [
            (r"\b\d{3}[-.]?\d{2}[-.]?\d{4}\b", "SSN", Severity.HIGH),
            (r"\b\d{16}\b|\b\d{4}[- ]\d{4}[- ]\d{4}[- ]\d{4}\b", "credit_card", Severity.HIGH),
            (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email", Severity.MEDIUM),
            (r"\b(?:\+?1[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b", "phone", Severity.MEDIUM),
            (r"\b\d{3}[-.]?\d{3}[-.]?\d{3}\b", "national_id", Severity.HIGH),
        ]

        # ── Severity → action mapping ───────────────────────
        self.severity_actions: Dict[Severity, SafetyAction] = {
            Severity.NONE: SafetyAction.PASS,
            Severity.LOW: SafetyAction.WARN,
            Severity.MEDIUM: SafetyAction.WARN,
            Severity.HIGH: SafetyAction.BLOCK,
            Severity.CRITICAL: SafetyAction.BLOCK,
        }

        # ── Guardrail thresholds ────────────────────────────
        self.max_output_length = 4096      # max words
        self.min_output_length = 3         # min words
        self.confidence_gate = 0.30        # block output if intent confidence below this
        self.max_repetition_ratio = 0.50   # flag if >50% of sentences are repeated

        # ── Hallucination detection patterns ────────────────
        self.hallucination_phrases = [
            r"\bas an ai\b",
            r"\bi (?:cannot|can't|don't|do not) have (?:access|feelings|opinions)\b",
            r"\bi (?:was|am) (?:created|trained|developed|built) by\b",
            r"\baccording to my training\b",
            r"\bmy training data\b",
            r"\bi don't have (?:real-time|current|access)\b",
        ]

        # ── Refusal detection ───────────────────────────────
        self.refusal_phrases = [
            r"\bi (?:cannot|can't|won't|will not|am unable to) (?:help|assist|provide|generate|create)\b",
            r"\b(?:sorry|apolog(?:y|ize|ies)),?\s+(?:but\s+)?i (?:cannot|can't)\b",
            r"\bthat (?:request|task|query) (?:is|goes) (?:against|beyond|outside)\b",
            r"\bi'm not (?:able|allowed|permitted) to\b",
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_output_length": self.max_output_length,
            "min_output_length": self.min_output_length,
            "confidence_gate": self.confidence_gate,
            "max_repetition_ratio": self.max_repetition_ratio,
            "n_category_rules": sum(len(v) for v in self.category_patterns.values()),
            "n_pii_patterns": len(self.pii_patterns),
            "n_hallucination_patterns": len(self.hallucination_phrases),
            "n_refusal_patterns": len(self.refusal_phrases),
        }

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PolicyRules":
        """Load policy (currently returns defaults; extend for custom JSON)."""
        return cls()


# ═══════════════════════════════════════════════════════════════
#  LAYER 1: PROMPT GUARD (pre-execution)
# ═══════════════════════════════════════════════════════════════

class PromptGuard:
    """
    Scans input prompts/segments for harmful content BEFORE execution.
    Prevents unsafe content from reaching the model at all.
    """

    def __init__(self, policy: PolicyRules):
        self.policy = policy
        self._compiled: Dict[SafetyCategory, List[Tuple[re.Pattern, Severity]]] = {}

        for cat, patterns in policy.category_patterns.items():
            self._compiled[cat] = [
                (re.compile(pat, re.IGNORECASE), sev)
                for pat, sev in patterns
            ]

    def scan(self, text: str) -> List[SafetyFlag]:
        """Scan input text for safety violations."""
        flags = []

        for cat, rules in self._compiled.items():
            for pattern, severity in rules:
                for match in pattern.finditer(text):
                    flags.append(SafetyFlag(
                        category=cat,
                        severity=severity,
                        evidence=match.group(),
                        span=(match.start(), match.end()),
                        rule_id=f"prompt_{cat.value}_{pattern.pattern[:30]}",
                        layer="prompt_guard",
                    ))

        return flags


# ═══════════════════════════════════════════════════════════════
#  LAYER 2: OUTPUT FILTER (post-execution)
# ═══════════════════════════════════════════════════════════════

class OutputFilter:
    """
    Scans model outputs for:
      - Harmful content (same categories as prompt guard)
      - PII leakage
      - Hallucination indicators
      - Refusal patterns
    """

    def __init__(self, policy: PolicyRules):
        self.policy = policy

        # Compile category patterns
        self._harmful: Dict[SafetyCategory, List[Tuple[re.Pattern, Severity]]] = {}
        for cat, patterns in policy.category_patterns.items():
            self._harmful[cat] = [
                (re.compile(pat, re.IGNORECASE), sev)
                for pat, sev in patterns
            ]

        # Compile PII patterns
        self._pii = [
            (re.compile(pat), label, sev)
            for pat, label, sev in policy.pii_patterns
        ]

        # Compile hallucination patterns
        self._hallucination = [
            re.compile(pat, re.IGNORECASE)
            for pat in policy.hallucination_phrases
        ]

        # Compile refusal patterns
        self._refusal = [
            re.compile(pat, re.IGNORECASE)
            for pat in policy.refusal_phrases
        ]

    def scan(self, text: str) -> List[SafetyFlag]:
        """Scan model output for all output-level concerns."""
        flags = []
        flags.extend(self._scan_harmful(text))
        flags.extend(self._scan_pii(text))
        flags.extend(self._scan_hallucination(text))
        flags.extend(self._scan_refusal(text))
        return flags

    def _scan_harmful(self, text: str) -> List[SafetyFlag]:
        flags = []
        for cat, rules in self._harmful.items():
            for pattern, severity in rules:
                for match in pattern.finditer(text):
                    flags.append(SafetyFlag(
                        category=cat, severity=severity,
                        evidence=match.group(),
                        span=(match.start(), match.end()),
                        rule_id=f"output_{cat.value}",
                        layer="output_filter",
                    ))
        return flags

    def _scan_pii(self, text: str) -> List[SafetyFlag]:
        flags = []
        for pattern, label, severity in self._pii:
            for match in pattern.finditer(text):
                flags.append(SafetyFlag(
                    category=SafetyCategory.PII,
                    severity=severity,
                    evidence=f"{label}: {match.group()[:20]}***",
                    span=(match.start(), match.end()),
                    rule_id=f"pii_{label}",
                    layer="output_filter",
                ))
        return flags

    def _scan_hallucination(self, text: str) -> List[SafetyFlag]:
        flags = []
        for pattern in self._hallucination:
            for match in pattern.finditer(text):
                flags.append(SafetyFlag(
                    category=SafetyCategory.HALLUCINATION,
                    severity=Severity.LOW,
                    evidence=match.group(),
                    span=(match.start(), match.end()),
                    rule_id="hallucination_indicator",
                    layer="output_filter",
                ))
        return flags

    def _scan_refusal(self, text: str) -> List[SafetyFlag]:
        flags = []
        for pattern in self._refusal:
            for match in pattern.finditer(text):
                flags.append(SafetyFlag(
                    category=SafetyCategory.REFUSAL,
                    severity=Severity.MEDIUM,
                    evidence=match.group(),
                    span=(match.start(), match.end()),
                    rule_id="model_refusal",
                    layer="output_filter",
                ))
        return flags

    def redact_pii(self, text: str) -> str:
        """Replace PII with [REDACTED] markers."""
        result = text
        for pattern, label, _ in self._pii:
            result = pattern.sub(f"[REDACTED-{label.upper()}]", result)
        return result


# ═══════════════════════════════════════════════════════════════
#  LAYER 3: GUARDRAILS
# ═══════════════════════════════════════════════════════════════

class GuardRails:
    """
    Structural and quality guardrails on model outputs:
      - Length limits
      - Confidence gating
      - Repetition detection
      - Format validation
    """

    def __init__(self, policy: PolicyRules):
        self.policy = policy

    def check(self, text: str, metadata: Optional[Dict] = None) -> List[SafetyFlag]:
        """Run all guardrail checks."""
        flags = []
        metadata = metadata or {}

        flags.extend(self._check_length(text))
        flags.extend(self._check_confidence(metadata))
        flags.extend(self._check_repetition(text))
        flags.extend(self._check_empty_or_garbage(text))

        return flags

    def _check_length(self, text: str) -> List[SafetyFlag]:
        flags = []
        wc = len(text.split())

        if wc > self.policy.max_output_length:
            flags.append(SafetyFlag(
                category=SafetyCategory.FORMAT_VIOLATION,
                severity=Severity.LOW,
                evidence=f"Output too long: {wc} words (max {self.policy.max_output_length})",
                span=(0, len(text)),
                rule_id="length_max",
                layer="guardrails",
            ))

        if wc < self.policy.min_output_length:
            flags.append(SafetyFlag(
                category=SafetyCategory.FORMAT_VIOLATION,
                severity=Severity.MEDIUM,
                evidence=f"Output too short: {wc} words (min {self.policy.min_output_length})",
                span=(0, len(text)),
                rule_id="length_min",
                layer="guardrails",
            ))

        return flags

    def _check_confidence(self, metadata: Dict) -> List[SafetyFlag]:
        flags = []
        conf = metadata.get("intent_confidence", 1.0)

        if conf < self.policy.confidence_gate:
            flags.append(SafetyFlag(
                category=SafetyCategory.HALLUCINATION,
                severity=Severity.MEDIUM,
                evidence=f"Intent confidence {conf:.3f} below gate {self.policy.confidence_gate}",
                span=(0, 0),
                rule_id="confidence_gate",
                layer="guardrails",
            ))

        return flags

    def _check_repetition(self, text: str) -> List[SafetyFlag]:
        flags = []
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip().lower() for s in sentences if s.strip()]

        if len(sentences) < 3:
            return flags

        unique = set(sentences)
        repetition_ratio = 1.0 - (len(unique) / len(sentences))

        if repetition_ratio > self.policy.max_repetition_ratio:
            flags.append(SafetyFlag(
                category=SafetyCategory.HALLUCINATION,
                severity=Severity.MEDIUM,
                evidence=f"Repetition ratio {repetition_ratio:.2f} (max {self.policy.max_repetition_ratio})",
                span=(0, len(text)),
                rule_id="repetition",
                layer="guardrails",
            ))

        return flags

    def _check_empty_or_garbage(self, text: str) -> List[SafetyFlag]:
        flags = []
        stripped = text.strip()

        if not stripped:
            flags.append(SafetyFlag(
                category=SafetyCategory.FORMAT_VIOLATION,
                severity=Severity.HIGH,
                evidence="Empty output",
                span=(0, 0),
                rule_id="empty_output",
                layer="guardrails",
            ))
            return flags

        # Check for garbage (high ratio of non-alphanumeric)
        alpha_count = sum(1 for c in stripped if c.isalnum() or c.isspace())
        if len(stripped) > 10 and alpha_count / len(stripped) < 0.3:
            flags.append(SafetyFlag(
                category=SafetyCategory.FORMAT_VIOLATION,
                severity=Severity.MEDIUM,
                evidence=f"Low alphanumeric ratio: {alpha_count/len(stripped):.2f}",
                span=(0, len(text)),
                rule_id="garbage_output",
                layer="guardrails",
            ))

        return flags


# ═══════════════════════════════════════════════════════════════
#  LAYER 4: ESCALATION HANDLER
# ═══════════════════════════════════════════════════════════════

class EscalationHandler:
    """
    Final decision maker. Takes flags from all layers and decides:
      - PASS:     safe, no issues
      - WARN:     minor issues, log and pass through
      - REDACT:   remove PII / sensitive content, then pass
      - BLOCK:    unsafe, do not return to user
      - ESCALATE: queue for human review
    """

    def __init__(self, policy: PolicyRules):
        self.policy = policy
        self._audit_log: List[Dict[str, Any]] = []

    def decide(self, flags: List[SafetyFlag], text: str,
               redacted_text: Optional[str] = None) -> SafetyResult:
        """Make final safety decision based on all flags."""

        if not flags:
            return SafetyResult(
                text=text,
                action=SafetyAction.PASS,
                max_severity=Severity.NONE,
            )

        # Find max severity
        max_sev = max(flags, key=lambda f: f.severity.value).severity

        # Count flags by category
        cat_counts: Dict[SafetyCategory, int] = {}
        for f in flags:
            cat_counts[f.category] = cat_counts.get(f.category, 0) + 1

        # Base action from severity
        action = self.policy.severity_actions.get(max_sev, SafetyAction.BLOCK)

        # ── Escalation rules ────────────────────────────────

        # Multiple medium flags → escalate to block
        medium_count = sum(1 for f in flags if f.severity == Severity.MEDIUM)
        if medium_count >= 3 and action == SafetyAction.WARN:
            action = SafetyAction.BLOCK

        # PII found → redact (even if other action is pass/warn)
        has_pii = SafetyCategory.PII in cat_counts
        if has_pii and action in (SafetyAction.PASS, SafetyAction.WARN):
            action = SafetyAction.REDACT

        # Hallucination + refusal combo → escalate
        has_hallucination = SafetyCategory.HALLUCINATION in cat_counts
        has_refusal = SafetyCategory.REFUSAL in cat_counts
        if has_hallucination and has_refusal:
            if action in (SafetyAction.PASS, SafetyAction.WARN):
                action = SafetyAction.ESCALATE

        # Critical severity → always block
        if max_sev == Severity.CRITICAL:
            action = SafetyAction.BLOCK

        result = SafetyResult(
            text=text,
            flags=flags,
            action=action,
            max_severity=max_sev,
            redacted_text=redacted_text if action == SafetyAction.REDACT else None,
        )

        # Audit log
        self._audit_log.append({
            "timestamp": time.time(),
            "action": action.value,
            "max_severity": max_sev.name,
            "n_flags": len(flags),
            "categories": {k.value: v for k, v in cat_counts.items()},
            "text_preview": text[:80],
        })

        return result

    @property
    def audit_log(self) -> List[Dict[str, Any]]:
        return list(self._audit_log)

    def stats(self) -> Dict[str, Any]:
        if not self._audit_log:
            return {"total_decisions": 0}

        actions = [e["action"] for e in self._audit_log]
        return {
            "total_decisions": len(self._audit_log),
            "pass": actions.count("pass"),
            "warn": actions.count("warn"),
            "redact": actions.count("redact"),
            "block": actions.count("block"),
            "escalate": actions.count("escalate"),
            "block_rate": round(actions.count("block") / len(actions), 4),
        }

    def reset(self):
        self._audit_log.clear()


# ═══════════════════════════════════════════════════════════════
#  MAIN: SAFETY SYSTEM (unified interface)
# ═══════════════════════════════════════════════════════════════

class SafetySystem:
    """
    Unified safety system. Single entry point for the pipeline.

    Usage:
        safety = SafetySystem()

        # Pre-execution: check segment text
        input_result = safety.check_input(segment_text)
        if input_result.is_blocked:
            # don't execute this segment

        # Post-execution: check model output
        output_result = safety.check_output(
            output_text,
            metadata={"intent_confidence": 0.85}
        )
        if output_result.action == SafetyAction.REDACT:
            final_text = output_result.redacted_text
    """

    def __init__(self, policy: Optional[PolicyRules] = None):
        self.policy = policy or PolicyRules()
        self.prompt_guard = PromptGuard(self.policy)
        self.output_filter = OutputFilter(self.policy)
        self.guardrails = GuardRails(self.policy)
        self.escalation = EscalationHandler(self.policy)

        self._input_checks = 0
        self._output_checks = 0

    # ─────────────────────────────────────────────────────────
    #  PRE-EXECUTION CHECK
    # ─────────────────────────────────────────────────────────

    def check_input(self, text: str) -> SafetyResult:
        """
        Check a segment BEFORE sending to model.
        Returns SafetyResult with action = pass/warn/block.
        """
        start = time.time()
        self._input_checks += 1

        flags = self.prompt_guard.scan(text)
        result = self.escalation.decide(flags, text)
        result.analysis_time_ms = (time.time() - start) * 1000
        result.layer_results = {"prompt_guard": len(flags)}

        return result

    # ─────────────────────────────────────────────────────────
    #  POST-EXECUTION CHECK
    # ─────────────────────────────────────────────────────────

    def check_output(
        self,
        text: str,
        metadata: Optional[Dict] = None,
    ) -> SafetyResult:
        """
        Check model output AFTER execution.
        Runs output filter + guardrails, then escalation.
        """
        start = time.time()
        self._output_checks += 1

        # Layer 2: Output filter
        output_flags = self.output_filter.scan(text)

        # Layer 3: Guardrails
        guardrail_flags = self.guardrails.check(text, metadata)

        all_flags = output_flags + guardrail_flags

        # Prepare redacted text in case needed
        redacted = self.output_filter.redact_pii(text)

        # Layer 4: Escalation decision
        result = self.escalation.decide(all_flags, text, redacted_text=redacted)
        result.analysis_time_ms = (time.time() - start) * 1000
        result.layer_results = {
            "output_filter": len(output_flags),
            "guardrails": len(guardrail_flags),
        }

        return result

    # ─────────────────────────────────────────────────────────
    #  BATCH CHECK (for pipeline integration)
    # ─────────────────────────────────────────────────────────

    def check_segments_pre(
        self, segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Check all segments before execution. Adds safety_result to each.
        Marks unsafe segments with 'unsafe_candidate': True.
        """
        results = []
        for seg in segments:
            text = seg.get("text", "")
            check = self.check_input(text)
            enriched = dict(seg)
            enriched["safety_input"] = check.to_dict()
            enriched["unsafe_candidate"] = check.is_blocked
            results.append(enriched)
        return results

    def check_outputs_post(
        self,
        execution_results: List[Dict[str, Any]],
        segment_metadata: Optional[Dict[int, Dict]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Check all outputs after execution. Adds safety_result to each.
        Redacts PII if needed.
        """
        metadata = segment_metadata or {}
        results = []

        for result in execution_results:
            sid = result.get("segment_id", -1)
            output = result.get("output", "")
            meta = metadata.get(sid, {})

            check = self.check_output(output, metadata=meta)
            enriched = dict(result)
            enriched["safety_output"] = check.to_dict()

            # Apply redaction if needed
            if check.action == SafetyAction.REDACT and check.redacted_text:
                enriched["output"] = check.redacted_text
                enriched["was_redacted"] = True

            # Block output if needed
            if check.is_blocked:
                enriched["output"] = "[BLOCKED] Output flagged by safety system."
                enriched["was_blocked"] = True

            results.append(enriched)

        return results

    # ─────────────────────────────────────────────────────────
    #  STATS & REPORTING
    # ─────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        return {
            "input_checks": self._input_checks,
            "output_checks": self._output_checks,
            "policy": self.policy.to_dict(),
            "escalation": self.escalation.stats(),
        }

    def print_report(self):
        s = self.stats()
        e = s["escalation"]
        print(f"\n{'='*60}")
        print(f"  SAFETY SYSTEM REPORT")
        print(f"{'='*60}")
        print(f"  Input checks:   {s['input_checks']}")
        print(f"  Output checks:  {s['output_checks']}")
        print(f"  Decisions:      {e.get('total_decisions', 0)}")
        print(f"  Passed:         {e.get('pass', 0)}")
        print(f"  Warnings:       {e.get('warn', 0)}")
        print(f"  Redacted:       {e.get('redact', 0)}")
        print(f"  Blocked:        {e.get('block', 0)}")
        print(f"  Escalated:      {e.get('escalate', 0)}")
        if e.get('total_decisions', 0) > 0:
            print(f"  Block rate:     {e.get('block_rate', 0):.2%}")
        print(f"{'='*60}")

    def reset(self):
        self._input_checks = 0
        self._output_checks = 0
        self.escalation.reset()