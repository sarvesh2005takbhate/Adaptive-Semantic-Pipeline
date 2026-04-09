"""
Module 5 — Execution Engine
==============================
Orchestrates parallel/sequential execution of routed segments.

Consolidates: scheduler, context_manager, result_cache, failure_handler.

Architecture:
    ExecutionEngine.execute(routed_segments, execution_plan)
      → Scheduler reads plan steps
      → Parallel steps use ThreadPoolExecutor
      → Sequential steps pass upstream results via ExecutionContext
      → FailureHandler manages retries/escalation
      → ResultCache avoids redundant calls
      → ModelDispatcher routes to weak/strong backends
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from enum import Enum
from typing import Any, Dict, List, Optional

log = logging.getLogger("execution_engine")

GLOBAL_CACHE = {}  # For demonstration; in production, use a proper cache class
# ═══════════════════════════════════════════════════════════════
#  SEGMENT STATUS
# ═══════════════════════════════════════════════════════════════

class SegmentStatus(Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


# ═══════════════════════════════════════════════════════════════
#  EXECUTION CONTEXT (thread-safe shared state)
# ═══════════════════════════════════════════════════════════════

class ExecutionContext:
    """Thread-safe store for segment results, statuses, and timings."""

    def __init__(self, prompt_id: int, prompt_text: str = ""):
        self._prompt_id = prompt_id
        self._prompt_text = prompt_text
        self._lock = threading.Lock()
        self._status: Dict[int, SegmentStatus] = {}
        self._results: Dict[int, Dict[str, Any]] = {}
        self._errors: Dict[int, Dict[str, Any]] = {}
        self._timing: Dict[int, Dict[str, float]] = {}

    def register_segment(self, segment_id: int):
        with self._lock:
            self._status[segment_id] = SegmentStatus.PENDING

    def set_status(self, sid: int, status: SegmentStatus):
        with self._lock:
            self._status[sid] = status
            if status == SegmentStatus.RUNNING:
                self._timing[sid] = {"start": time.time()}

    def store_result(self, sid: int, result: Dict[str, Any]):
        with self._lock:
            self._results[sid] = result
            self._status[sid] = SegmentStatus.COMPLETED
            if sid in self._timing:
                self._timing[sid]["end"] = time.time()
                self._timing[sid]["duration"] = self._timing[sid]["end"] - self._timing[sid]["start"]

    def store_error(self, sid: int, error: str, error_type: str = "execution_error"):
        with self._lock:
            self._errors[sid] = {"error": error, "error_type": error_type, "timestamp": time.time()}
            self._status[sid] = SegmentStatus.FAILED
            if sid in self._timing:
                self._timing[sid]["end"] = time.time()
                self._timing[sid]["duration"] = self._timing[sid]["end"] - self._timing[sid]["start"]

    def is_failed(self, sid: int) -> bool:
        with self._lock:
            return self._status.get(sid) in {SegmentStatus.FAILED, SegmentStatus.SKIPPED}

    def get_result(self, sid: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._results.get(sid)

    def build_context_for(self, segment_id: int, depends_on: List[int]) -> Dict[str, Any]:
        """Build context window from upstream results for a dependent segment."""
        with self._lock:
            upstream = {}
            for dep_id in depends_on:
                if dep_id in self._results:
                    upstream[dep_id] = self._results[dep_id]
            return {
                "prompt_id": self._prompt_id,
                "prompt_text": self._prompt_text,
                "segment_id": segment_id,
                "upstream_results": upstream,
                "all_upstream_completed": len(upstream) == len(depends_on),
            }

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            status_counts = {}
            for s in SegmentStatus:
                c = sum(1 for v in self._status.values() if v == s)
                if c > 0:
                    status_counts[s.value] = c
            total_dur = sum(t.get("duration", 0.0) for t in self._timing.values())
            return {
                "prompt_id": self._prompt_id,
                "total_segments": len(self._status),
                "status_counts": status_counts,
                "total_duration": round(total_dur, 4),
                "results_count": len(self._results),
                "errors_count": len(self._errors),
            }


# ═══════════════════════════════════════════════════════════════
#  RESULT CACHE (LRU, thread-safe)
# ═══════════════════════════════════════════════════════════════

class ResultCache:
    """Thread-safe LRU cache with optional TTL."""

    def __init__(self, max_size: int = 1024, ttl_seconds: Optional[float] = None):
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._store: OrderedDict[str, Dict] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._store:
                self._misses += 1
                return None
            entry = self._store[key]
            if self._ttl and (time.time() - entry["ts"]) > self._ttl:
                del self._store[key]
                self._misses += 1
                return None
            self._store.move_to_end(key)
            self._hits += 1
            return entry["result"]

    def put(self, key: str, result: Any):
        with self._lock:
            self._store[key] = {"result": result, "ts": time.time()}
            self._store.move_to_end(key)
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)

    def flush(self):
        with self._lock:
            self._store.clear()

    def reset_stats(self):
        with self._lock:
            self._hits = self._misses = 0

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total else 0.0,
                "size": len(self._store),
            }


# ═══════════════════════════════════════════════════════════════
#  FAILURE HANDLER
# ═══════════════════════════════════════════════════════════════

class FailureHandler:
    """Retry → escalate → fallback failure strategy."""

    FALLBACK_RESPONSE = "[Fallback] Unable to process after multiple attempts."

    def __init__(self, max_retries: int = 2, escalate_weak: bool = True):
        self._max_retries = max_retries
        self._escalate_weak = escalate_weak
        self._attempts: Dict[int, int] = {}
        self._failures: List[Dict] = []

    def handle_failure(self, segment_id: int, error: str,
                       current_tier: str = "strong_model") -> Dict[str, Any]:
        attempt = self._attempts.get(segment_id, 0) + 1
        self._attempts[segment_id] = attempt
        self._failures.append({"segment_id": segment_id, "error": error,
                               "attempt": attempt, "tier": current_tier})

        if attempt <= self._max_retries:
            return {"action": "retry", "new_tier": current_tier, "attempt": attempt}

        if self._escalate_weak and current_tier == "weak_model" and attempt == self._max_retries + 1:
            return {"action": "escalate", "new_tier": "strong_model", "attempt": attempt}

        return {
            "action": "fallback", "attempt": attempt,
            "fallback_result": {
                "segment_id": segment_id,
                "output": self.FALLBACK_RESPONSE,
                "model_name": "fallback", "latency": 0.0,
                "tokens_used": 0, "cost_estimate": 0.0,
                "status": "fallback",
            },
        }

    def should_skip_dependents(self, segment_id: int) -> bool:
        attempt = self._attempts.get(segment_id, 0)
        return attempt > self._max_retries + (1 if self._escalate_weak else 0)

    def get_skip_result(self, segment_id: int, upstream_id: int) -> Dict[str, Any]:
        return {
            "segment_id": segment_id,
            "output": f"[Skipped] Upstream segment {upstream_id} failed.",
            "model_name": "none", "latency": 0.0,
            "tokens_used": 0, "cost_estimate": 0.0, "status": "skipped",
        }

    def stats(self) -> Dict[str, Any]:
        return {
            "total_failures": len(self._failures),
            "segments_with_failures": len(set(f["segment_id"] for f in self._failures)),
            "retries": sum(1 for f in self._failures if self._attempts.get(f["segment_id"], 0) <= self._max_retries),
        }

    def reset(self):
        self._attempts.clear()
        self._failures.clear()


# ═══════════════════════════════════════════════════════════════
#  MODEL BACKENDS
# ═══════════════════════════════════════════════════════════════

class SimulatedWeakModel:
    MODEL_NAME = "weak-sim"
    BASE_LATENCY = 0.02

    def infer(self, text: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        start = time.time()
        h = hashlib.md5(text.encode()).hexdigest()[:8]
        wc = len(text.split())
        output = f"[WeakModel] Processed ({wc} words): \"{text[:60]}\" hash={h}"
        if context and context.get("upstream_results"):
            output += f" | context from {list(context['upstream_results'].keys())}"
        return {
            "output": output, "latency": round(time.time() - start + self.BASE_LATENCY, 4),
            "model_name": self.MODEL_NAME, "tokens_used": wc * 3, "cost_estimate": 0.0001,
        }


class SimulatedStrongModel:
    MODEL_NAME = "strong-sim"
    BASE_LATENCY = 0.05

    def infer(self, text: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        start = time.time()
        h = hashlib.md5(text.encode()).hexdigest()[:8]
        wc = len(text.split())
        output = f"[StrongModel] Deep analysis ({wc} words): \"{text[:60]}\" hash={h}"
        if context and context.get("upstream_results"):
            output += f" | building on {list(context['upstream_results'].keys())}"
        return {
            "output": output, "latency": round(time.time() - start + self.BASE_LATENCY, 4),
            "model_name": self.MODEL_NAME, "tokens_used": wc * 5, "cost_estimate": 0.001,
        }


class GroqBackend:
    """Shared Groq API logic — DRY base for weak/strong models."""

    def __init__(self, model: str, model_name: str,
                 cost_input: float, cost_output: float):
        self._model = model
        self._model_name = model_name
        self._cost_input = cost_input
        self._cost_output = cost_output

    def _build_prompt(self, text: str, context: Optional[Dict]) -> str:
        prompt = text
        if context and context.get("upstream_results"):
            upstream = "\n".join(
                f"[Segment {k}]: {v.get('output', '')[:300]}"
                for k, v in context["upstream_results"].items()
            )
            prompt = f"Context from previous tasks:\n{upstream}\n\nCurrent task: {text}"
        return prompt

    def _call_api(self, prompt: str) -> Dict[str, Any]:
        import os
        import requests

        # 🔥 CACHE KEY (IMPORTANT)
        key = f"{self._model}_{hash(prompt)}"

        # ✅ CACHE HIT
        if key in GLOBAL_CACHE:
            return GLOBAL_CACHE[key]

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 0.7,
        }

        for attempt in range(3):
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )

            data = response.json()

            if "choices" in data:
                # ✅ STORE IN CACHE
                GLOBAL_CACHE[key] = data
                return data

            error_msg = data.get("error", {}).get("message", "Unknown error")

            if response.status_code == 429 and attempt < 2:
                wait = 2 ** (attempt + 1)
                log.warning(f"Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue

            raise RuntimeError(f"Groq API error: {error_msg}")

        raise RuntimeError("Groq API: max retries exceeded")

    def infer(self, text: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        start = time.time()
        prompt = self._build_prompt(text, context)

        try:
            data = self._call_api(prompt)
            output = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            cost = (usage.get("prompt_tokens", 0) * self._cost_input / 1e6 +
                    usage.get("completion_tokens", 0) * self._cost_output / 1e6)
            return {
                "output": output,
                "latency": round(time.time() - start, 4),
                "model_name": self._model_name,
                "tokens_used": usage.get("total_tokens", 0),
                "cost_estimate": round(cost, 6),
            }
        except Exception as e:
            return {
                "output": f"[API Error] {e}",
                "latency": round(time.time() - start, 4),
                "model_name": self._model_name,
                "tokens_used": 0, "cost_estimate": 0.0,
            }

    def generate(self, text: str) -> str:
        """Convenience for aggregator / baseline calls."""
        return self.infer(text)["output"]


class GroqWeakModel(GroqBackend):
    def __init__(self):
        super().__init__(
            model="llama-3.1-8b-instant",
            model_name="groq-llama3-8b",
            cost_input=0.05, cost_output=0.08,
        )


class GroqStrongModel(GroqBackend):
    def __init__(self):
        super().__init__(
            model="llama-3.3-70b-versatile",
            model_name="groq-llama3-70b",
            cost_input=0.59, cost_output=0.79,
        )


# ═══════════════════════════════════════════════════════════════
#  MODEL DISPATCHER
# ═══════════════════════════════════════════════════════════════

class ModelDispatcher:
    """Routes segments to weak/strong/block based on route field."""

    def __init__(self, weak_backend=None, strong_backend=None):
        self._weak = weak_backend or SimulatedWeakModel()
        self._strong = strong_backend or SimulatedStrongModel()
        self._stats = {"weak": 0, "strong": 0, "blocked": 0, "verify": 0,
                       "total_latency": 0.0, "total_cost": 0.0}

    def dispatch(self, segment: Dict[str, Any],
                 context: Optional[Dict] = None) -> Dict[str, Any]:
        sid = segment.get("segment_id", -1)
        text = segment.get("text", "")
        # Accept both "route" (from router.py) and "route_tier" (legacy)
        tier = segment.get("route", segment.get("route_tier", "strong_model"))

        if tier == "safe_block":
            self._stats["blocked"] += 1
            return {
                "segment_id": sid, "text": text, "route_tier": tier,
                "model_name": "none",
                "output": "[BLOCKED] Content flagged as unsafe.",
                "latency": 0.0, "tokens_used": 0, "cost_estimate": 0.0,
                "status": "blocked",
            }

        needs_verify = (tier == "verify_required")
        if tier in ("weak_model", ):
            backend = self._weak
            self._stats["weak"] += 1
        else:
            backend = self._strong
            self._stats["strong"] += 1
            if needs_verify:
                self._stats["verify"] += 1

        result = backend.infer(text, context=context)
        self._stats["total_latency"] += result.get("latency", 0.0)
        self._stats["total_cost"] += result.get("cost_estimate", 0.0)

        return {
            "segment_id": sid, "text": text, "route_tier": tier,
            "model_name": result["model_name"],
            "output": result["output"],
            "latency": result["latency"],
            "tokens_used": result.get("tokens_used", 0),
            "cost_estimate": result.get("cost_estimate", 0.0),
            "status": "completed",
            "needs_verification": needs_verify,
        }

    def stats(self) -> Dict[str, Any]:
        return dict(self._stats)

    def reset_stats(self):
        for k in self._stats:
            self._stats[k] = 0 if isinstance(self._stats[k], int) else 0.0


# ═══════════════════════════════════════════════════════════════
#  EXECUTION ENGINE (main orchestrator)
# ═══════════════════════════════════════════════════════════════

class ExecutionEngine:
    """
    Top-level orchestrator. Runs routed segments through the execution plan.

    Usage:
        engine = ExecutionEngine(backend="simulated")  # or "groq"
        result = engine.execute(routed_segments, execution_plan, prompt_id=1)
        engine.print_report(result)
    """

    def __init__(
        self,
        backend: str = "simulated",
        max_workers: int = 4,
        cache_size: int = 1024,
        max_retries: int = 2,
    ):
        if backend == "groq":
            weak, strong = GroqWeakModel(), GroqStrongModel()
        else:
            weak, strong = SimulatedWeakModel(), SimulatedStrongModel()

        self.dispatcher = ModelDispatcher(weak_backend=weak, strong_backend=strong)
        self.cache = ResultCache(max_size=cache_size)
        self.failure_handler = FailureHandler(max_retries=max_retries)
        self.max_workers = max_workers

    def execute(
        self,
        routed_segments: List[Dict[str, Any]],
        execution_plan: List[Dict[str, Any]],
        prompt_id: int = 0,
        prompt_text: str = "",
    ) -> Dict[str, Any]:
        """
        Execute all segments according to the plan.

        Parameters
        ----------
        routed_segments : list
            From Module 4 (router). Each has segment_id, text, route.
        execution_plan : list
            From decomposer DAG. Each step has step, mode, segment_ids.
        """
        start_time = time.time()
        execution_log = []

        # Flush cache between prompts (segment IDs repeat across prompts)
        self.cache.flush()

        # Build context
        context = ExecutionContext(prompt_id, prompt_text)

        # Index routed segments by ID
        routed_idx: Dict[int, Dict[str, Any]] = {}
        seg_meta: Dict[int, Dict[str, Any]] = {}  # segment_id → {depends_on, text}
        for seg in routed_segments:
            sid = seg.get("segment_id", -1)
            routed_idx[sid] = seg
            context.register_segment(sid)
            seg_meta[sid] = seg

        all_results: List[Dict[str, Any]] = []

        for step in execution_plan:
            step_num = step["step"]
            mode = step["mode"]
            # FIXED: execution_plan has segment_ids (list of ints),
            # not full segment dicts
            segment_ids = step.get("segment_ids", [])

            step_start = time.time()

            if mode == "parallel" and len(segment_ids) > 1:
                step_results = self._execute_parallel(
                    segment_ids, routed_idx, seg_meta, context)
            else:
                step_results = self._execute_sequential(
                    segment_ids, routed_idx, seg_meta, context)

            step_log = {
                "step": step_num,
                "mode": mode,
                "segment_ids": segment_ids,
                "step_latency": round(time.time() - step_start, 4),
                "results": [
                    {"segment_id": r["segment_id"],
                     "status": r.get("status", "unknown"),
                     "model_name": r.get("model_name", "unknown"),
                     "latency": r.get("latency", 0.0)}
                    for r in step_results
                ],
            }
            execution_log.append(step_log)
            all_results.extend(step_results)

        total_time = time.time() - start_time

        return {
            "prompt_id": prompt_id,
            "subtask_outputs": [
                {
                    "segment_id": r["segment_id"],
                    "route_tier": r.get("route_tier", "unknown"),
                    "model_used": r.get("model_name", "unknown"),
                    "output": r.get("output", ""),
                    "latency": r.get("latency", 0.0),
                    "tokens_used": r.get("tokens_used", 0),
                    "cost": r.get("cost_estimate", 0.0),
                }
                for r in all_results
            ],
            "execution_log": execution_log,
            "context_summary": context.summary(),
            "dispatcher_stats": self.dispatcher.stats(),
            "cache_stats": self.cache.stats(),
            "failure_stats": self.failure_handler.stats(),
            "total_time": round(total_time, 4),
            "steps_executed": len(execution_plan),
        }

    # ─────────────────────────────────────────────────────────
    #  PARALLEL EXECUTION
    # ─────────────────────────────────────────────────────────

    def _execute_parallel(
        self,
        segment_ids: List[int],
        routed_idx: Dict[int, Dict],
        seg_meta: Dict[int, Dict],
        context: ExecutionContext,
    ) -> List[Dict[str, Any]]:
        results = []
        futures: Dict[Future, int] = {}

        with ThreadPoolExecutor(max_workers=2) as pool:
            for sid in segment_ids:
                routed = routed_idx.get(sid)
                if not routed:
                    results.append(self._error_result(sid, "Segment not found"))
                    continue

                cache_key = f"seg_{sid}"
                cached = self.cache.get(cache_key)
                if cached:
                    results.append(cached)
                    context.store_result(sid, cached)
                    continue

                depends_on = routed.get("depends_on", [])
                seg_ctx = context.build_context_for(sid, depends_on)
                context.set_status(sid, SegmentStatus.QUEUED)
                future = pool.submit(self._dispatch_with_retry, routed, seg_ctx, context)
                futures[future] = sid

            for future in as_completed(futures):
                sid = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    context.store_result(sid, result)
                    self.cache.put(f"seg_{sid}", result)
                except Exception as e:
                    err = self._error_result(sid, str(e))
                    results.append(err)
                    context.store_error(sid, str(e))

        return results

    # ─────────────────────────────────────────────────────────
    #  SEQUENTIAL EXECUTION
    # ─────────────────────────────────────────────────────────

    def _execute_sequential(
        self,
        segment_ids: List[int],
        routed_idx: Dict[int, Dict],
        seg_meta: Dict[int, Dict],
        context: ExecutionContext,
    ) -> List[Dict[str, Any]]:
        results = []

        for sid in segment_ids:
            routed = routed_idx.get(sid)
            if not routed:
                results.append(self._error_result(sid, "Segment not found"))
                continue

            depends_on = routed.get("depends_on", [])

            # Check upstream failures
            skip = False
            for dep_id in depends_on:
                if context.is_failed(dep_id) and self.failure_handler.should_skip_dependents(dep_id):
                    results.append(self.failure_handler.get_skip_result(sid, dep_id))
                    context.store_error(sid, f"Upstream {dep_id} failed")
                    skip = True
                    break
            if skip:
                continue

            cache_key = f"seg_{sid}"
            cached = self.cache.get(cache_key)
            if cached:
                results.append(cached)
                context.store_result(sid, cached)
                continue

            seg_ctx = context.build_context_for(sid, depends_on)
            result = self._dispatch_with_retry(routed, seg_ctx, context)
            results.append(result)

            if result.get("status") in ("completed", "blocked", "fallback"):
                context.store_result(sid, result)
                self.cache.put(cache_key, result)
            else:
                context.store_error(sid, result.get("output", "Unknown error"))

        return results

    # ─────────────────────────────────────────────────────────
    #  DISPATCH WITH RETRY
    # ─────────────────────────────────────────────────────────

    def _dispatch_with_retry(
        self,
        segment: Dict[str, Any],
        seg_ctx: Optional[Dict],
        exec_ctx: ExecutionContext,
    ) -> Dict[str, Any]:
        sid = segment.get("segment_id", -1)
        current = dict(segment)
        exec_ctx.set_status(sid, SegmentStatus.RUNNING)

        try:
            result = self.dispatcher.dispatch(current, context=seg_ctx)
            result["segment_id"] = sid
            return result
        except Exception as e:
            tier = current.get("route", current.get("route_tier", "strong_model"))
            decision = self.failure_handler.handle_failure(sid, str(e), tier)

            if decision["action"] == "retry":
                return self._dispatch_with_retry(current, seg_ctx, exec_ctx)
            elif decision["action"] == "escalate":
                escalated = dict(current)
                escalated["route"] = "strong_model"
                escalated["route_tier"] = "strong_model"
                return self._dispatch_with_retry(escalated, seg_ctx, exec_ctx)
            elif decision["action"] == "fallback":
                return decision["fallback_result"]
            else:
                return self._error_result(sid, str(e))

    # ─────────────────────────────────────────────────────────
    #  BASELINE COMPARISON
    # ─────────────────────────────────────────────────────────

    def execute_baseline(self, prompt_text: str, prompt_id: int = 0) -> Dict[str, Any]:
        """
        Send the entire prompt to the strong model as a single call.
        Used for latency/cost comparison against the pipeline.
        """
        start = time.time()
        result = self.dispatcher._strong.infer(prompt_text)
        latency = time.time() - start

        return {
            "prompt_id": prompt_id,
            "model_used": result["model_name"],
            "output": result["output"],
            "latency": round(latency, 4),
            "tokens_used": result.get("tokens_used", 0),
            "cost": result.get("cost_estimate", 0.0),
        }

    # ─────────────────────────────────────────────────────────
    #  STATS HELPERS
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def compute_stats(result: Dict[str, Any]) -> Dict[str, Any]:
        """Compute parallelism and cost stats from execution result."""
        log_entries = result.get("execution_log", [])
        outputs = result.get("subtask_outputs", [])

        total_steps = len(log_entries)
        parallel_steps = sum(1 for s in log_entries if s["mode"] == "parallel")
        total_segs = sum(len(s["segment_ids"]) for s in log_entries)
        parallel_segs = sum(len(s["segment_ids"]) for s in log_entries if s["mode"] == "parallel")

        sum_latency = sum(o.get("latency", 0) for o in outputs)
        actual_time = result.get("total_time", 0)
        speedup = sum_latency / actual_time if actual_time > 0 else 1.0

        total_cost = sum(o.get("cost", 0) for o in outputs)
        total_tokens = sum(o.get("tokens_used", 0) for o in outputs)
        weak_calls = sum(1 for o in outputs if "weak" in o.get("model_used", "").lower()
                         or o.get("route_tier") == "weak_model")
        strong_calls = sum(1 for o in outputs if "strong" in o.get("model_used", "").lower()
                           or o.get("route_tier") == "strong_model")

        return {
            "total_steps": total_steps,
            "parallel_steps": parallel_steps,
            "total_segments": total_segs,
            "parallel_segments": parallel_segs,
            "parallelism_ratio": round(parallel_segs / total_segs, 3) if total_segs else 0,
            "sum_individual_latency": round(sum_latency, 4),
            "actual_total_time": round(actual_time, 4),
            "speedup_factor": round(speedup, 3),
            "total_cost": round(total_cost, 6),
            "total_tokens": total_tokens,
            "weak_calls": weak_calls,
            "strong_calls": strong_calls,
        }

    # ─────────────────────────────────────────────────────────
    #  REPORTING
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def print_report(result: Dict[str, Any]):
        print("=" * 70)
        print("  EXECUTION ENGINE — RESULTS")
        print("=" * 70)
        print(f"  Prompt ID:      {result['prompt_id']}")
        print(f"  Steps:          {result['steps_executed']}")
        print(f"  Total time:     {result['total_time']:.4f}s")

        for step in result["execution_log"]:
            icon = "⚡" if step["mode"] == "parallel" else "🔗"
            print(f"\n  {icon} Step {step['step']} ({step['mode']}): "
                  f"segments {step['segment_ids']} [{step['step_latency']:.4f}s]")
            for r in step["results"]:
                si = {"completed": "✅", "blocked": "🚫", "failed": "❌",
                      "skipped": "⏭️"}.get(r["status"], "❓")
                print(f"    {si} [{r['segment_id']}] {r['status']:10s} "
                      f"model={r['model_name']:20s} lat={r['latency']:.4f}s")

        print("\n  --- Outputs ---")
        for o in result["subtask_outputs"]:
            print(f"  [{o['segment_id']}] ({o['route_tier']}, {o['model_used']}):")
            print(f"    {o['output'][:120]}{'…' if len(o['output']) > 120 else ''}")

        stats = ExecutionEngine.compute_stats(result)
        print(f"\n  Speedup:     {stats['speedup_factor']:.2f}x")
        print(f"  Cost:        {stats['total_cost']:.6f}")
        print(f"  Weak/Strong: {stats['weak_calls']}/{stats['strong_calls']}")
        print("=" * 70)

    @staticmethod
    def _error_result(sid: int, error: str) -> Dict[str, Any]:
        return {
            "segment_id": sid, "output": f"[Error] {error}",
            "model_name": "none", "latency": 0.0,
            "tokens_used": 0, "cost_estimate": 0.0, "status": "failed",
        }

    def reset(self):
        self.cache.flush()
        self.cache.reset_stats()
        self.dispatcher.reset_stats()
        self.failure_handler.reset()