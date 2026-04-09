"""
Module 1+2 — Semantic Decomposer & Dependency Graph Builder
=============================================================
Splits a multi-task prompt into segments AND builds a DAG
execution plan in one pass.

Merged because decomposition and dependency detection are tightly
coupled — detecting dependencies during splitting avoids a second
NLP pass and prevents inconsistencies.

Output per prompt:
  {
    "segments": [
      {"segment_id": 1, "text": "...", "depends_on": []},
      {"segment_id": 2, "text": "...", "depends_on": [1]},
    ],
    "execution_plan": [
      {"step": 1, "mode": "parallel", "segment_ids": [1, 3]},
      {"step": 2, "mode": "sequential", "segment_ids": [2]},
    ],
    "dag": {"nodes": [...], "edges": [...]},
    "depth_levels": {0: [1, 3], 1: [2]},
    "stats": {"n_segments": 3, "n_parallel": 2, "n_sequential": 1,
              "max_depth": 1, "parallelism_ratio": 0.67}
  }

Fixes over original:
  • "Do two things: first... Second..." pattern handled (was over-splitting)
  • "First X, and then use that to Y" pattern handled
  • DAG uses Kahn's algorithm for topological sort
  • Cycle detection (safety)
  • Parallelism ratio metric
  • 1-based segment IDs throughout
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import spacy

log = logging.getLogger("decomposer")

# ═══════════════════════════════════════════════════════════════
#  PRE-COMPILED PATTERNS
# ═══════════════════════════════════════════════════════════════

_RE_WHITESPACE = re.compile(r"\s+")
_RE_MATH_SIGNAL = re.compile(r"[=+\-*/^∫∑∂]")
_RE_SEMICOLON = re.compile(r"\s*;\s*")
_RE_LEADING_CONJ = re.compile(
    r"^(and|then|also|but|next|after\s+that|finally|subsequently)\b",
    re.IGNORECASE,
)
_RE_STRIP_EDGES = re.compile(r"^[\s,;:\-]+|[\s,;:\-]+$")
_RE_GREETING = re.compile(
    r"^(hi|hello|hey|good\s+(?:morning|afternoon|evening|day))"
    r"(?:\s+there|\s+assistant|\s+friend)?[!.,\s]*$",
    re.IGNORECASE,
)
_RE_QUOTED = re.compile(r"(['\"])(?:(?!\1).)*\1")
_RE_DEPENDENT_OPENER = re.compile(
    r"^(based on|using|given|considering|according to|assuming|from)\b",
    re.IGNORECASE,
)
_RE_CONTINUATION = re.compile(
    r"^(justify|clarify|elaborate|support)\s+"
    r"(why|your\s+\w+|the\s+(answer|choice|reasoning|result|decision|reason))",
    re.IGNORECASE,
)

# ── Multi-task framing patterns (CRITICAL FIX for over-splitting) ──
# These patterns frame 2 tasks but contain structural words like
# "first", "second" that the conjunction splitter was treating as splits.
_RE_TWO_THINGS = re.compile(
    r"^(?:do\s+(?:two|2)\s+things|I\s+need\s+help\s+with\s+(?:two|2)\s+tasks?)"
    r"\s*[:.]\s*",
    re.IGNORECASE,
)
_RE_FIRST_SECOND = re.compile(
    r"^first[,.]?\s+(.*?)\s*[.]?\s*second(?:ly)?[,.]?\s+(.*)",
    re.IGNORECASE | re.DOTALL,
)
_RE_NUMBERED_TASKS = re.compile(
    r"^(?:1[.)]\s*)(.*?)(?:\s*2[.)]\s*)(.*)",
    re.IGNORECASE | re.DOTALL,
)

# "First X, and then use that to Y" — dependent pair
_RE_FIRST_THEN_USE = re.compile(
    r"^first\s+(.*?),?\s+and\s+then\s+use\s+that\s+to\s+(.*)",
    re.IGNORECASE | re.DOTALL,
)


_RE_CONNECTOR_SPLIT = re.compile(
    r"^(.*?)[.!?]\s*(?:while you'?re at it|in addition|additionally|separately|also)[,.]?\s+(.*)",
    re.IGNORECASE | re.DOTALL,
)
 
# "X. Once that is done, Y" / "X. Based on the result, Y" — dependent pair
_RE_DEPENDENT_CONNECTOR = re.compile(
    r"^(.*?)[.!?]\s*(?:once (?:that is|this is) done|based on (?:the |your )?(?:result|findings|output)"
    r"|using the (?:output|result|information))[,.]?\s+(.*)",
    re.IGNORECASE | re.DOTALL,
)
 
_MAX_SPLIT_DEPTH = 6
_MIN_SEGMENT_WORDS = 3
 


# ═══════════════════════════════════════════════════════════════
#  DECOMPOSER + DAG BUILDER
# ═══════════════════════════════════════════════════════════════

class SemanticDecomposer:
    """
    Rule-based semantic decomposer with integrated DAG builder.

    Usage:
        decomposer = SemanticDecomposer()
        result = decomposer.decompose("Write code to sort a list and also translate it to French")
        # result["segments"], result["execution_plan"], result["dag"], etc.
    """

    SPLIT_CONJUNCTIONS = {
        "and", "then", "also", "but",
        "next", "finally", "additionally", "subsequently",
        "and then", "after that", "followed by", "in addition",
    }

    DEPENDENCY_MARKERS = {
        "it", "that", "those", "these", "this",
        "result", "answer", "output", "above",
    }

    DEPENDENCY_PHRASES = [
        "use that", "use it", "use the result", "use the answer",
        "that result", "the result", "the answer", "the output",
        "based on that", "based on the above",
        "using the above", "using that", "using it",
        "then use", "from the above", "from that",
        "with that", "with the result",
        "based on your findings", "using those results",
        "using the output", "using the information",
        "based on the result", "once that is done",
        "after completing that", "using the information obtained",
    ]

    ACTION_PREFIXES = {
        "find", "name", "write", "choose", "list", "identify",
        "explain", "compare", "tell", "describe", "solve", "output",
        "summarize", "translate", "generate", "give", "compute",
        "calculate", "use", "return", "classify", "categorize",
        "rewrite", "debug", "create", "define", "determine",
        "evaluate", "provide", "suggest", "recommend",
        "state", "indicate", "mark", "read", "infer",
        "reason", "integrate", "implement", "analyze",
        "predict", "forecast", "estimate", "simulate",
        "model", "design", "draft", "compose", "document",
        "build", "develop", "test", "deploy", "optimize",
        "convert", "extract", "parse", "validate", "check",
        "run", "execute", "train", "cluster", "segment",
        "perform", "normalize", "detect", "survey",
    }

    _ANALYTICAL_STARTERS = {
        "explain", "analyze", "analyse", "describe", "discuss",
        "evaluate", "assess", "critique", "review", "justify",
        "elaborate", "summarize", "summarise", "interpret",
    }

    _RELATIONAL_NOUNS = {
        "algorithm", "algorithms", "approach", "approaches",
        "method", "methods", "implementation", "implementations",
        "step", "steps", "process", "processes",
        "output", "outputs", "result", "results",
        "answer", "answers", "solution", "solutions",
        "formula", "formulas", "equation", "equations",
        "choice", "choices", "decision", "decisions",
        "rationale", "reasoning", "findings",
        "translation", "summary", "classification",
    }

    _MULTIWORD_RELATIONAL = {
        "data structure", "data structures",
        "time complexity", "space complexity",
    }

    def __init__(self, model_name: str = "en_core_web_sm", debug: bool = False):
        self.nlp = self._load_nlp(model_name)
        self.debug = debug
        action_alt = "|".join(sorted(self.ACTION_PREFIXES))
        self._re_comma_action = re.compile(
            rf",\s*(then\s+)?(?=(?:{action_alt})\b)", re.IGNORECASE,
        )
        self._re_action_start = re.compile(
            rf"^(?:{action_alt})\b", re.IGNORECASE,
        )

    @staticmethod
    def _load_nlp(model_name: str):
        try:
            return spacy.load(model_name)
        except OSError:
            nlp = spacy.blank("en")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            return nlp

    # ─────────────────────────────────────────────────────────
    #  HELPERS
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(text: str) -> str:
        text = _RE_WHITESPACE.sub(" ", text)
        return _RE_STRIP_EDGES.sub("", text)

    @staticmethod
    def _word_count_str(text: str) -> int:
        return len(text.split())

    def _log(self, msg: str):
        if self.debug:
            print(f"[DECOMPOSER] {msg}")

    @staticmethod
    def _inside_quote(text: str, idx: int) -> bool:
        for m in _RE_QUOTED.finditer(text):
            if m.start() <= idx < m.end():
                return True
        return False

    # ─────────────────────────────────────────────────────────
    #  STRUCTURED MULTI-TASK DETECTION (CRITICAL FIX)
    # ─────────────────────────────────────────────────────────

    def _try_structured_split(self, text: str) -> Optional[List[Tuple[str, bool]]]:
        """
        Detect structured multi-task patterns BEFORE general splitting.
        Returns list of (segment_text, is_dependent) or None.

        This fixes the over-splitting bug where "Do two things: first X. Second Y"
        was being split into 4 segments instead of 2.
        """
        normalized = self._normalize(text)

        # Pattern: "First X, and then use that to Y" (dependent)
        m = _RE_FIRST_THEN_USE.match(normalized)
        if m:
            t1 = self._normalize(m.group(1))
            t2 = self._normalize(m.group(2))
            if t1 and t2:
                self._log(f"Structured split (first...then use): [{t1}] → [{t2}]")
                return [(t1, False), (t2, True)]

        # Pattern: "Do two things: first X. Second Y"
        stripped = _RE_TWO_THINGS.sub("", normalized)
        if stripped != normalized:
            # Removed the framing, now split on "Second"
            m = _RE_FIRST_SECOND.match(stripped)
            if m:
                t1 = self._normalize(m.group(1))
                t2 = self._normalize(m.group(2))
                if t1 and t2:
                    dep = self._has_dependency_signal(t2)
                    self._log(f"Structured split (two things): [{t1}] + [{t2}]")
                    return [(t1, False), (t2, dep)]

        # Pattern: "First X. Second Y" (without "do two things" prefix)
        m = _RE_FIRST_SECOND.match(normalized)
        if m:
            t1 = self._normalize(m.group(1))
            t2 = self._normalize(m.group(2))
            if t1 and t2 and self._word_count_str(t1) >= 3 and self._word_count_str(t2) >= 3:
                dep = self._has_dependency_signal(t2)
                self._log(f"Structured split (first/second): [{t1}] + [{t2}]")
                return [(t1, False), (t2, dep)]

        # Pattern: "1) X  2) Y"
        m = _RE_NUMBERED_TASKS.match(normalized)
        if m:
            t1 = self._normalize(m.group(1))
            t2 = self._normalize(m.group(2))
            if t1 and t2:
                dep = self._has_dependency_signal(t2)
                return [(t1, False), (t2, dep)]
            
        m = _RE_CONNECTOR_SPLIT.match(normalized)
        if m:
            t1 = self._normalize(m.group(1))
            t2 = self._normalize(m.group(2))
            if t1 and t2 and self._word_count_str(t1) >= 3 and self._word_count_str(t2) >= 3:
                self._log(f"Structured split (connector): [{t1}] + [{t2}]")
                return [(t1, False), (t2, False)]
 
        # Pattern: "X. Once that is done, Y" / "X. Based on the result, Y" (dependent)
        m = _RE_DEPENDENT_CONNECTOR.match(normalized)
        if m:
            t1 = self._normalize(m.group(1))
            t2 = self._normalize(m.group(2))
            if t1 and t2 and self._word_count_str(t1) >= 3 and self._word_count_str(t2) >= 3:
                self._log(f"Structured split (dependent connector): [{t1}] → [{t2}]")
                return [(t1, False), (t2, True)]

        return None

    # ─────────────────────────────────────────────────────────
    #  DEPENDENCY DETECTION
    # ─────────────────────────────────────────────────────────

    def _has_dependency_signal(self, text: str) -> bool:
        """Check if text has any dependency signal (lightweight check)."""
        lowered = text.lower()
        if any(p in lowered for p in self.DEPENDENCY_PHRASES):
            return True
        first_word = lowered.split()[0] if lowered.split() else ""
        if first_word in self.DEPENDENCY_MARKERS:
            return True
        return False

    def _is_dependent(self, fragment: str, doc=None, is_multi: bool = True) -> bool:
        normalized = self._normalize(fragment)
        if not normalized:
            return False

        # Strong: explicit phrases
        lowered = normalized.lower()
        if any(p in lowered for p in self.DEPENDENCY_PHRASES):
            return True

        if doc is None:
            doc = self.nlp(normalized)
        if not len(doc):
            return False

        first = doc[0].text.lower()
        if first in self.DEPENDENCY_MARKERS:
            return True

        # Bridging reference (only in multi-segment)
        if is_multi and self._has_bridging_reference(normalized):
            if any(p in normalized.lower() for p in self.DEPENDENCY_PHRASES):
                return True

        return any(
            t.dep_ in {"nsubj", "dobj", "pobj", "attr", "nsubjpass"}
            and t.lower_ in self.DEPENDENCY_MARKERS
            for t in doc
        )

    def _has_bridging_reference(self, text: str) -> bool:
        words = text.lower().split()
        if len(words) < 3:
            return False
        first = words[0].rstrip(".,;:!?")
        if first not in self._ANALYTICAL_STARTERS:
            return False
        for i in range(1, len(words)):
            w = words[i].rstrip(".,;:!?")
            if w in {"the", "each", "every"}:
                for j in range(i + 1, min(i + 5, len(words))):
                    candidate = words[j].rstrip(".,;:!?")
                    if candidate in self._RELATIONAL_NOUNS:
                        return True
                    if j + 1 < len(words):
                        pair = candidate + " " + words[j + 1].rstrip(".,;:!?")
                        if pair in self._MULTIWORD_RELATIONAL:
                            return True
        return False

    # ─────────────────────────────────────────────────────────
    #  TASK DETECTION
    # ─────────────────────────────────────────────────────────

    def _looks_like_task(self, fragment: str, doc=None) -> bool:
        normalized = self._normalize(fragment)
        if len(normalized) < 3:
            return False
        if doc is None:
            doc = self.nlp(normalized)
        if not len(doc):
            return False
        if " and " in normalized and any(w in normalized for w in ["explain", "summarize", "translate"]):
            return True

        first_token = doc[0].text.lower()
        token_count = sum(1 for t in doc if not t.is_space and not t.is_punct)
        root = next((t for t in doc if t.head == t), None)
        has_subject = any(t.dep_ in {"nsubj", "nsubjpass", "expl"} for t in doc)
        has_predicate = any(t.pos_ in {"VERB", "AUX"} for t in doc)
        has_math = bool(_RE_MATH_SIGNAL.search(normalized))
        has_question = normalized.endswith("?") or any(
            t.lower_ in {"who", "what", "when", "where", "why", "how"} for t in doc[:2]
        )
        has_imperative = (
            root is not None
            and root.pos_ in {"VERB", "AUX"}
            and (root.i == 0 or first_token in self.ACTION_PREFIXES)
            and token_count >= 2
        )
        has_statement = root is not None and root.pos_ in {"VERB", "AUX"} and has_subject

        if first_token in self.ACTION_PREFIXES and token_count < 2 and not has_math:
            return False

        return has_math or has_question or has_imperative or has_statement or (
            has_predicate and first_token in self.ACTION_PREFIXES and token_count >= 2
        )

    # ─────────────────────────────────────────────────────────
    #  CONJUNCTION SPLITTING
    # ─────────────────────────────────────────────────────────

    def _candidate_split_indices(self, sentence: str, doc) -> List[tuple]:
        indices: list[tuple] = []
        for token in doc:
            if token.dep_ == "cc" and token.text.lower() in self.SPLIT_CONJUNCTIONS:
                if self._inside_quote(sentence, token.idx):
                    continue
                if token.text.lower() == "and":
                    left_ctx = sentence[:token.idx].rstrip()
                    right_ctx = sentence[token.idx + len(token.text):].lstrip()
                    if (_RE_MATH_SIGNAL.search(left_ctx[-15:] if len(left_ctx) >= 15 else left_ctx)
                        and _RE_MATH_SIGNAL.search(right_ctx[:15] if len(right_ctx) >= 15 else right_ctx)):
                        continue
                indices.append((token.idx, token.text.lower()))

        for m in _RE_SEMICOLON.finditer(sentence):
            if not self._inside_quote(sentence, m.start()):
                indices.append((m.start(), ";"))

        for m in self._re_comma_action.finditer(sentence):
            if not self._inside_quote(sentence, m.start()):
                indices.append((m.start(), ","))

        action_alt = "|".join(sorted(self.ACTION_PREFIXES))
        for m in re.finditer(rf"\band\s+then\s+(?=({action_alt})\b)", sentence, flags=re.IGNORECASE):
            idx = m.start()
            if not self._inside_quote(sentence, idx) and idx not in {i for i, _ in indices}:
                indices.append((idx, "and"))

        return sorted(set(indices), key=lambda x: x[0])

    def split_conjunction(self, sentence: str, _depth: int = 0) -> List[str]:
        sentence = self._normalize(sentence)
        if not sentence:
            return []
        if _depth >= _MAX_SPLIT_DEPTH:
            return [sentence]

        doc = self.nlp(sentence)
        for split_idx, splitter in self._candidate_split_indices(sentence, doc):
            left = self._normalize(sentence[:split_idx])
            right = self._normalize(sentence[split_idx:])
            if right:
                prev = None
                while prev != right:
                    prev = right
                    right = _RE_LEADING_CONJ.sub("", right, count=1).strip(" ,;:-")

            if splitter == "," and _RE_DEPENDENT_OPENER.match(left):
                continue
            if splitter in {"and", "but"}:
                right_words = right.split()
                first_right = right_words[0].lower() if right_words else ""
                if not (self._re_action_start.match(right)
                        or first_right in {"who", "what", "when", "where", "why", "how"}
                        or first_right in {"also", "separately", "additionally"}):
                    continue
            if _RE_CONTINUATION.match(right):
                continue

            right_words = [w for w in right.split() if w.strip(".,;:!?")]
            if (len(right_words) <= 3 and right_words
                and right_words[0].lower() == "explain"
                and len(right_words) >= 2
                and right_words[1].lower().rstrip(".") in {"why"}):
                continue

            if splitter in {"then", "also", "next", "finally", "subsequently"}:
                if not self._re_action_start.match(right):
                    continue

            if self._looks_like_task(left) and self._looks_like_task(right):
                left_parts = self.split_conjunction(left, _depth + 1)
                right_parts = self.split_conjunction(right, _depth + 1)
                return left_parts + right_parts

        return [sentence]

    def _ensure_sentence_boundaries(self, text: str) -> str:
        if any(ch in text for ch in ".!?;"):
            return text
        patched = self._re_comma_action.sub(". ", text)
        if patched == text:
            patched = re.sub(
                r",\s*(?=" + "|".join(sorted(self.ACTION_PREFIXES)) + r"\b)",
                ". ", text, flags=re.IGNORECASE,
            )
        return patched

    # ─────────────────────────────────────────────────────────
    #  DAG BUILDER (Kahn's Algorithm)
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def build_dag(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build a DAG from segments and compute execution plan.

        Uses Kahn's algorithm for topological sort.
        Groups segments by depth level for parallel execution.
        """
        n = len(segments)
        if n == 0:
            return {
                "dag": {"nodes": [], "edges": []},
                "topological_order": [],
                "depth_levels": {},
                "execution_plan": [],
                "stats": {"n_segments": 0, "n_parallel": 0, "n_sequential": 0,
                          "max_depth": 0, "parallelism_ratio": 0.0},
            }

        # Build adjacency + in-degree
        nodes = [seg["segment_id"] for seg in segments]
        edges = []
        in_degree = {sid: 0 for sid in nodes}
        adj = defaultdict(list)

        for seg in segments:
            sid = seg["segment_id"]
            for dep in seg.get("depends_on", []):
                if dep in in_degree:  # valid dependency
                    edges.append((dep, sid))
                    adj[dep].append(sid)
                    in_degree[sid] += 1

        # Kahn's algorithm — topological sort
        queue = deque([n for n in nodes if in_degree[n] == 0])
        topo_order = []
        depth = {n: 0 for n in nodes}

        while queue:
            node = queue.popleft()
            topo_order.append(node)
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                depth[neighbor] = max(depth[neighbor], depth[node] + 1)
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Cycle detection
        if len(topo_order) != n:
            log.warning("Cycle detected in dependency graph! Falling back to sequential.")
            return {
                "dag": {"nodes": nodes, "edges": edges, "has_cycle": True},
                "topological_order": nodes,
                "depth_levels": {0: nodes},
                "execution_plan": [
                    {"step": i + 1, "mode": "sequential", "segment_ids": [sid]}
                    for i, sid in enumerate(nodes)
                ],
                "stats": {"n_segments": n, "n_parallel": 0, "n_sequential": n,
                          "max_depth": n - 1, "parallelism_ratio": 0.0},
            }

        # Group by depth level
        depth_levels: Dict[int, List[int]] = defaultdict(list)
        for sid in topo_order:
            depth_levels[depth[sid]].append(sid)

        # Build execution plan
        execution_plan = []
        for d in sorted(depth_levels.keys()):
            sids = depth_levels[d]
            mode = "parallel" if len(sids) > 1 else "sequential"
            execution_plan.append({
                "step": d + 1,
                "mode": mode,
                "segment_ids": sids,
            })

        max_depth = max(depth.values()) if depth else 0
        n_parallel = sum(len(sids) for sids in depth_levels.values() if len(sids) > 1)
        n_sequential = n - n_parallel
        parallelism_ratio = n_parallel / n if n > 0 else 0.0

        return {
            "dag": {"nodes": nodes, "edges": edges, "has_cycle": False},
            "topological_order": topo_order,
            "depth_levels": dict(depth_levels),
            "execution_plan": execution_plan,
            "stats": {
                "n_segments": n,
                "n_parallel": n_parallel,
                "n_sequential": n_sequential,
                "max_depth": max_depth,
                "parallelism_ratio": round(parallelism_ratio, 3),
            },
        }

    # ─────────────────────────────────────────────────────────
    #  CONFIDENCE SCORING
    # ─────────────────────────────────────────────────────────

    def _compute_confidence(self, text: str, doc) -> float:
        score = 0.5
        first = doc[0].text.lower() if len(doc) > 0 else ""
        if first in self.ACTION_PREFIXES:
            score += 0.25
        if any(t.pos_ == "VERB" for t in doc):
            score += 0.10
        wc = self._word_count_str(text)
        if 5 <= wc <= 40:
            score += 0.10
        elif wc < 3:
            score -= 0.20
        if text.rstrip().endswith("?"):
            score += 0.05
        return round(min(max(score, 0.1), 1.0), 2)

    # ─────────────────────────────────────────────────────────
    #  MAIN ENTRY POINT
    # ─────────────────────────────────────────────────────────

    def decompose(self, text: str) -> Dict[str, Any]:
        """
        Decompose a prompt into segments + build execution DAG.

        Returns dict with: segments, execution_plan, dag, depth_levels, stats
        """
        text = text.strip()
        if not text:
            return self._empty_result()

        # Code blocks → single segment
        if "```" in text:
            segments = [{"segment_id": 1, "text": self._normalize(text),
                        "depends_on": [], "confidence": 1.0}]
            dag_result = self.build_dag(segments)
            return {"segments": segments, **dag_result}

        # ── Try structured multi-task patterns FIRST ────────
        structured = self._try_structured_split(text)
        if structured:
            segments = []
            for i, (seg_text, is_dep) in enumerate(structured):
                seg_text = self._normalize(seg_text)
                if not seg_text or self._word_count_str(seg_text) < _MIN_SEGMENT_WORDS:
                    continue
                doc = self.nlp(seg_text)
                depends_on = [segments[-1]["segment_id"]] if is_dep and segments else []
                segments.append({
                    "segment_id": len(segments) + 1,
                    "text": seg_text,
                    "depends_on": depends_on,
                    "confidence": self._compute_confidence(seg_text, doc),
                })

            if segments:
                dag_result = self.build_dag(segments)
                return {"segments": segments, **dag_result}

        # ── General splitting ───────────────────────────────
        # Strip greeting
        for m in _RE_GREETING.finditer(text):
            if m.start() < 5:
                text = text[m.end():].lstrip(" ,;")
                break

        text = self._ensure_sentence_boundaries(text)
        doc = self.nlp(text)
        raw_segments: list[str] = []

        for sent in doc.sents:
            sentence = self._normalize(sent.text)
            if len(sentence) < 3:
                continue
            sub_sents = re.split(
                r"\.\s+(?=" + "|".join(sorted(self.ACTION_PREFIXES)) + r"\b)",
                sentence, flags=re.IGNORECASE,
            )
            for sub in sub_sents:
                sub = self._normalize(sub)
                if len(sub) < 3:
                    continue
                raw_segments.extend(self.split_conjunction(sub))

        # ── Merge verification caveats ──────────────────────
        merged: list[str] = []
        skip = False
        for i, seg in enumerate(raw_segments):
            if skip:
                skip = False
                continue
            if (seg.rstrip().endswith("?")
                and i + 1 < len(raw_segments)
                and re.match(r"^(mark|state|indicate|note|flag)\b", raw_segments[i + 1], re.I)):
                merged.append(seg + " " + raw_segments[i + 1])
                skip = True
            else:
                merged.append(seg)

        CONNECTOR_ONLY = {
            "while you're at it", "while youre at it",
            "on top of that", "in addition to that",
            "apart from that", "besides that",
            "once that is done", "after completing that",
            "additionally", "also", "furthermore",
        }

        # ── Build segments with dependencies ────────────────
        is_multi = len(merged) > 1
        segments = []
        for index, seg_text in enumerate(merged):
            cleaned = self._normalize(seg_text)
            if not cleaned or self._word_count_str(cleaned) < _MIN_SEGMENT_WORDS:
                continue
 
            # Skip if it's just a connector phrase
            if cleaned.lower().rstrip(".,;:!? ") in CONNECTOR_ONLY:
                self._log(f"Dropped connector phrase: {cleaned!r}")
                continue
 
            seg_doc = self.nlp(cleaned)
            depends_on: List[int] = []
            if index > 0 and is_multi:
                if self._is_dependent(cleaned, seg_doc, is_multi=True):
                    depends_on = [segments[-1]["segment_id"]] if segments else []
 
            segments.append({
                "segment_id": len(segments) + 1,
                "text": cleaned,
                "depends_on": depends_on,
                "confidence": self._compute_confidence(cleaned, seg_doc),
            })
 
        if not segments:
            # Fallback: entire prompt as single segment
            segments = [{"segment_id": 1, "text": self._normalize(text),
                        "depends_on": [], "confidence": 0.5}]
 
        dag_result = self.build_dag(segments)
        return {"segments": segments, **dag_result}
    


    def _empty_result(self) -> Dict[str, Any]:
         return {
            "segments": [],
            "dag": {"nodes": [], "edges": []},
            "topological_order": [],
            "depth_levels": {},
            "execution_plan": [],
            "stats": {"n_segments": 0, "n_parallel": 0, "n_sequential": 0,
                      "max_depth": 0, "parallelism_ratio": 0.0},
        }

    # ─────────────────────────────────────────────────────────
    #  BATCH + PRINT
    # ─────────────────────────────────────────────────────────

    def decompose_batch(self, prompts: List[str]) -> List[Dict[str, Any]]:
        return [self.decompose(p) for p in prompts]

    @staticmethod
    def print_result(result: Dict[str, Any]) -> None:
        """Pretty-print decomposition result."""
        print("=" * 60)
        print("  DECOMPOSITION + EXECUTION PLAN")
        print("=" * 60)

        for seg in result["segments"]:
            dep_str = f" → depends on {seg['depends_on']}" if seg["depends_on"] else " (independent)"
            print(f"\n  [{seg['segment_id']}] \"{seg['text'][:70]}{'…' if len(seg['text']) > 70 else ''}\"")
            print(f"       {dep_str}  (conf: {seg['confidence']:.2f})")

        print(f"\n  Execution Plan:")
        for step in result["execution_plan"]:
            print(f"    Step {step['step']}: {step['mode']} → segments {step['segment_ids']}")

        s = result["stats"]
        print(f"\n  Stats: {s['n_segments']} segments, "
              f"{s['n_parallel']} parallel, {s['n_sequential']} sequential, "
              f"parallelism={s['parallelism_ratio']:.2f}")
        print("=" * 60)

    # ─────────────────────────────────────────────────────────
    #  EVALUATION
    # ─────────────────────────────────────────────────────────

    @classmethod
    def evaluate_dataset(
        cls,
        decomposer: "SemanticDecomposer",
        dataset_path: str,
        output_path: Optional[str] = None,
        max_samples: int = 0,
    ) -> Dict[str, Any]:
        """Evaluate decomposer + DAG on dataset with comprehensive metrics."""
        data = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))

        if max_samples > 0:
            data = data[:max_samples]

        count_correct = 0
        over_split = 0
        under_split = 0
        total = len(data)

        dep_tp, dep_fp, dep_fn = 0, 0, 0
        single_correct, single_total = 0, 0
        multi_correct, multi_total = 0, 0

        parallelism_ratios = []
        latencies = []
        errors = []

        for item in data:
            prompt = item["prompt"]
            gold_segs = item["segments"]
            gold_count = len(gold_segs)
            is_multi = gold_count > 1

            start = time.time()
            result = decomposer.decompose(prompt)
            latency = time.time() - start
            latencies.append(latency)

            pred_segs = result["segments"]
            pred_count = len(pred_segs)
            parallelism_ratios.append(result["stats"]["parallelism_ratio"])

            if pred_count == gold_count:
                count_correct += 1
                if is_multi:
                    multi_correct += 1
                else:
                    single_correct += 1
            elif pred_count > gold_count:
                over_split += 1
            else:
                under_split += 1

            if is_multi:
                multi_total += 1
            else:
                single_total += 1

            # Dependency eval
            gold_dep_ids = {s["segment_id"] for s in gold_segs if s.get("depends_on")}
            pred_dep_ids = {s["segment_id"] for s in pred_segs if s.get("depends_on")}
            dep_tp += len(gold_dep_ids & pred_dep_ids)
            dep_fp += len(pred_dep_ids - gold_dep_ids)
            dep_fn += len(gold_dep_ids - pred_dep_ids)

            if pred_count != gold_count:
                errors.append({
                    "prompt": prompt[:120],
                    "gold_count": gold_count,
                    "pred_count": pred_count,
                    "error": pred_count - gold_count,
                })

        lat_arr = np.array(latencies)
        par_arr = np.array(parallelism_ratios)

        dep_prec = dep_tp / (dep_tp + dep_fp) if (dep_tp + dep_fp) > 0 else 0
        dep_rec = dep_tp / (dep_tp + dep_fn) if (dep_tp + dep_fn) > 0 else 0
        dep_f1 = 2 * dep_prec * dep_rec / (dep_prec + dep_rec) if (dep_prec + dep_rec) > 0 else 0

        results = {
            "total_prompts": total,
            "segmentation": {
                "count_accuracy": round(count_correct / total, 4) if total else 0,
                "over_split_rate": round(over_split / total, 4) if total else 0,
                "under_split_rate": round(under_split / total, 4) if total else 0,
                "exact_match": count_correct,
                "over_split": over_split,
                "under_split": under_split,
            },
            "single_task": {
                "total": single_total, "correct": single_correct,
                "accuracy": round(single_correct / single_total, 4) if single_total else 0,
            },
            "multi_task": {
                "total": multi_total, "correct": multi_correct,
                "accuracy": round(multi_correct / multi_total, 4) if multi_total else 0,
            },
            "dependency_detection": {
                "precision": round(dep_prec, 4), "recall": round(dep_rec, 4), "f1": round(dep_f1, 4),
                "true_positives": dep_tp, "false_positives": dep_fp, "false_negatives": dep_fn,
            },
            "parallelism": {
                "mean_ratio": round(float(par_arr.mean()), 4),
                "median_ratio": round(float(np.median(par_arr)), 4),
            },
            "latency": {
                "mean_ms": round(float(lat_arr.mean() * 1000), 2),
                "p95_ms": round(float(np.percentile(lat_arr, 95) * 1000), 2),
            },
            "top_errors": sorted(errors, key=lambda x: abs(x["error"]), reverse=True)[:20],
        }

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

        seg = results["segmentation"]
        dep = results["dependency_detection"]
        print(f"\n{'='*60}")
        print(f"  DECOMPOSER + DAG EVALUATION")
        print(f"{'='*60}")
        print(f"  Count accuracy:       {seg['count_accuracy']:.4f}")
        print(f"  Single-task:          {results['single_task']['accuracy']:.4f}")
        print(f"  Multi-task:           {results['multi_task']['accuracy']:.4f}")
        print(f"  Over/under split:     {seg['over_split']}/{seg['under_split']}")
        print(f"  Dependency P/R/F1:    {dep['precision']:.3f}/{dep['recall']:.3f}/{dep['f1']:.3f}")
        print(f"  Mean parallelism:     {results['parallelism']['mean_ratio']:.3f}")
        print(f"  Mean latency:         {results['latency']['mean_ms']:.1f}ms")
        print(f"{'='*60}")

        return results


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Decompose + DAG evaluation")
    parser.add_argument("--data", type=str, default="data/dataset.jsonl")
    parser.add_argument("--output", type=str, default="evaluation/decomposer_dag_results.json")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--demo", type=str, default="", help="Demo with a single prompt")
    args = parser.parse_args()

    decomposer = SemanticDecomposer(debug=args.debug)

    if args.demo:
        result = decomposer.decompose(args.demo)
        SemanticDecomposer.print_result(result)
    else:
        SemanticDecomposer.evaluate_dataset(
            decomposer, dataset_path=args.data, output_path=args.output,
            max_samples=args.max_samples,
        )


if __name__ == "__main__":
    main()