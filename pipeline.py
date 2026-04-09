"""
Pipeline Runner — Adaptive Semantic Parallelism
=================================================
End-to-end pipeline with side-by-side baseline comparison.

Shows for each prompt:
  - Pipeline: decomposition → routing → per-segment outputs → aggregated answer
  - Baseline: single strong-model call on full prompt
  - Latency & cost comparison

Usage:
    python pipeline.py --data data/dataset.jsonl --max-samples 50
    GROQ_API_KEY=xxx python pipeline.py --data data/dataset.jsonl --backend groq --max-samples 20
    python pipeline.py --prompt "Solve x+2=5 and translate hello to French" --backend groq
    python pipeline.py --data data/dataset.jsonl --backend groq --max-samples 100 --save-outputs
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
log = logging.getLogger("pipeline")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config.labels import INTENT_LABELS, STRONG_INTENTS, WEAK_INTENTS


class AdaptiveSemanticPipeline:

    def __init__(self, backend="simulated", intent_model_path="models/intent_classifier",
                 router_model_path="models/router_model.txt", max_workers=4):
        log.info("Initializing pipeline...")

        from semantic_decom_dependency.decomposition_dependency import SemanticDecomposer
        self.decomposer = SemanticDecomposer()

        try:
            from Intent_complexity.intent_estimator import IntentComplexityEstimator
            self.estimator = IntentComplexityEstimator(model_path=intent_model_path)
        except Exception as e:
            log.warning(f"  Intent classifier fallback: {e}")
            self.estimator = None

        try:
            from router.router import LearningBasedRouter
            self.router = LearningBasedRouter(model_path=router_model_path)
        except Exception as e:
            log.warning(f"  Router fallback: {e}")
            self.router = None

        from execution_engine.execution_engine import ExecutionEngine
        self.engine = ExecutionEngine(backend=backend, max_workers=max_workers)

        from safety_system.safety_system import SafetySystem
        self.safety = SafetySystem()

        from verification_aggregation.aggregator import VerificationController, FinalAggregator, FeedbackCollector
        self.verifier = VerificationController()
        agg_model = self.engine.dispatcher._weak if backend != "simulated" else None
        self.aggregator = FinalAggregator(model=agg_model)
        self.feedback = FeedbackCollector()
        self.backend = backend
        log.info("  Pipeline ready.")

    def run(self, prompt: str, prompt_id: int = 1) -> Dict[str, Any]:
        result = {"prompt_id": prompt_id, "prompt": prompt, "timings": {}}
        t0 = time.time()

        # 1+2: Decompose + DAG
        t = time.time()
        decomp = self.decomposer.decompose(prompt)
        segments = decomp["segments"]
        execution_plan = decomp["execution_plan"]
        result["timings"]["decompose"] = round(time.time() - t, 4)
        result["n_segments"] = len(segments)
        result["execution_plan"] = execution_plan
        result["dag_stats"] = decomp["stats"]

        # 6a: Pre-safety
        t = time.time()
        segments = self.safety.check_segments_pre(segments)
        result["timings"]["safety_pre"] = round(time.time() - t, 4)

        # 3: Intent + Complexity
        t = time.time()
        for seg in segments:
            if seg.get("unsafe_candidate"):
                seg.update({"intent": "blocked", "complexity_score": 0.0, "intent_confidence": 0.0})
            elif self.estimator:
                seg.update(self.estimator.estimate(seg["text"]))
            else:
                seg.update({"intent": "explanation", "complexity_score": 0.5, "intent_confidence": 0.5})
        result["timings"]["classify"] = round(time.time() - t, 4)

        # 4: Route
        t = time.time()
        routed = []
        for seg in segments:
            if seg.get("unsafe_candidate"):
                r = dict(seg); r.update({"route": "safe_block", "route_confidence": 1.0, "route_method": "safety"})
            elif self.router:
                r = self.router.route(seg)
            else:
                r = dict(seg)
                r.update({"route": "strong_model" if seg.get("intent","") in STRONG_INTENTS else "weak_model",
                          "route_confidence": 0.8, "route_method": "heuristic"})
            routed.append(r)
        result["timings"]["route"] = round(time.time() - t, 4)

        result["segment_details"] = [{
            "segment_id": s["segment_id"], "text": s.get("text",""),
            "intent": s.get("intent","?"), "complexity": s.get("complexity_score",0),
            "route": s.get("route","?"), "confidence": s.get("intent_confidence",0),
        } for s in routed]

        # 5: Execute
        t = time.time()
        exec_result = self.engine.execute(routed, execution_plan, prompt_id, prompt)
        result["timings"]["execute"] = round(time.time() - t, 4)

        # 6b: Post-safety
        t = time.time()
        seg_meta = {s["segment_id"]: s for s in routed}
        exec_result["subtask_outputs"] = self.safety.check_outputs_post(
            exec_result["subtask_outputs"], segment_metadata=seg_meta)
        result["timings"]["safety_post"] = round(time.time() - t, 4)

        result["segment_outputs"] = [{
            "segment_id": o["segment_id"],
            "model_used": o.get("model_used", o.get("route_tier","")),
            "output": o.get("output",""), "latency": o.get("latency",0),
            "cost": o.get("cost",0), "tokens": o.get("tokens_used",0),
        } for o in exec_result["subtask_outputs"]]

        # 7: Verify
        t = time.time()
        verification = self.verifier.verify(exec_result, routed, prompt)
        result["timings"]["verify"] = round(time.time() - t, 4)
        result["verification"] = verification["verification_stats"]

        # 8: Aggregate
        t = time.time()
        agg = self.aggregator.aggregate(verification["verified_outputs"], prompt_text=prompt)
        result["timings"]["aggregate"] = round(time.time() - t, 4)
        result["final_answer"] = agg["final_answer"]
        result["aggregation_method"] = agg["aggregation_method"]

        self.feedback.collect(verification["verified_outputs"], routed)
        result["pipeline_latency"] = round(time.time() - t0, 4)
        result["pipeline_cost"] = sum(o.get("cost",0) for o in exec_result["subtask_outputs"])
        routes = [s.get("route","?") for s in routed]
        result["routing_summary"] = {"strong": routes.count("strong_model"),
                                     "weak": routes.count("weak_model"),
                                     "blocked": routes.count("safe_block")}
        return result

    def run_baseline(self, prompt: str, prompt_id: int = 1) -> Dict[str, Any]:
        return self.engine.execute_baseline(prompt, prompt_id)
    

    def run_comparison(self, prompt: str, prompt_id: int = 1) -> Dict[str, Any]:
        p = self.run(prompt, prompt_id)
        b = self.run_baseline(prompt, prompt_id)
        lat_red = ((b["latency"] - p["pipeline_latency"]) / b["latency"] * 100) if b["latency"] > 0 else 0
        cost_red = ((b["cost"] - p["pipeline_cost"]) / b["cost"] * 100) if b["cost"] > 0 else 0
        return {
            "prompt_id": prompt_id, "prompt": prompt,
            "pipeline": {
                "n_segments": p["n_segments"], "segment_details": p["segment_details"],
                "segment_outputs": p["segment_outputs"], "final_answer": p["final_answer"],
                "aggregation_method": p["aggregation_method"], "latency": p["pipeline_latency"],
                "cost": p["pipeline_cost"], "routing": p["routing_summary"],
                "dag_stats": p["dag_stats"], "timings": p["timings"],
            },
            "baseline": {
                "output": b["output"], "latency": b["latency"], "cost": b["cost"],
                "model_used": b["model_used"], "tokens_used": b.get("tokens_used",0),
            },
            "comparison": {
                "latency_reduction_pct": round(lat_red, 2),
                "cost_reduction_pct": round(cost_red, 2),
                "pipeline_faster": p["pipeline_latency"] < b["latency"],
                "pipeline_cheaper": p["pipeline_cost"] < b["cost"],
            },
        }

    def evaluate(self, dataset_path: str, max_samples: int = 0,
                 output_path: Optional[str] = None, save_outputs: bool = False):
        data = []
        with open(dataset_path) as f:
            for line in f:
                if line.strip(): data.append(json.loads(line.strip()))
        if max_samples > 0: data = data[:max_samples]
        log.info(f"Evaluating {len(data)} prompts...")

        comparisons, lat_reds, cost_reds, errors = [], [], [], []
        for i, item in enumerate(data):
            try:
                comp = self.run_comparison(item["prompt"], item.get("prompt_id", i+1))
                comparisons.append(comp)
                lat_reds.append(comp["comparison"]["latency_reduction_pct"])
                cost_reds.append(comp["comparison"]["cost_reduction_pct"])
            except Exception as e:
                errors.append({"prompt_id": item.get("prompt_id",i+1), "error": str(e)})
            if (i+1) % 10 == 0: log.info(f"  {i+1}/{len(data)}")

        metrics = self._compute_metrics(comparisons, lat_reds, cost_reds, errors)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f: json.dump(metrics, f, indent=2, default=str)

        if save_outputs:
            op = Path(output_path or "evaluation/results.json").with_name("detailed_outputs.json")
            with open(op, "w") as f: json.dump(comparisons[:30], f, indent=2, default=str, ensure_ascii=False)
            log.info(f"Detailed outputs → {op}")

        self._print_eval(metrics)
        self._print_samples(comparisons[:5])
        return metrics

    def _compute_metrics(self, comps, lat_reds, cost_reds, errors):
        n = len(comps)
        if not n: return {"error": "no results"}
        p_lats = [c["pipeline"]["latency"] for c in comps]
        b_lats = [c["baseline"]["latency"] for c in comps]
        p_costs = [c["pipeline"]["cost"] for c in comps]
        b_costs = [c["baseline"]["cost"] for c in comps]
        tot_strong = sum(c["pipeline"]["routing"]["strong"] for c in comps)
        tot_weak = sum(c["pipeline"]["routing"]["weak"] for c in comps)
        tot_segs = sum(c["pipeline"]["n_segments"] for c in comps)
        multi = sum(1 for c in comps if c["pipeline"]["n_segments"] > 1)
        par = [c["pipeline"]["dag_stats"]["parallelism_ratio"] for c in comps]

        tkeys = ["decompose","classify","route","execute","safety_pre","safety_post","verify","aggregate"]
        tavg = {k: round(np.mean([c["pipeline"]["timings"].get(k,0) for c in comps])*1000, 2) for k in tkeys}

        return {
            "n_prompts": n, "n_errors": len(errors), "n_segments_total": tot_segs,
            "multi_task_prompts": multi, "multi_task_pct": round(multi/n*100,1),
            "latency": {
                "pipeline_mean_ms": round(np.mean(p_lats)*1000,2),
                "pipeline_median_ms": round(np.median(p_lats)*1000,2),
                "baseline_mean_ms": round(np.mean(b_lats)*1000,2),
                "baseline_median_ms": round(np.median(b_lats)*1000,2),
                "reduction_mean_pct": round(np.mean(lat_reds),2),
                "reduction_median_pct": round(np.median(lat_reds),2),
                "positive_reduction_pct": round(sum(1 for x in lat_reds if x>0)/len(lat_reds)*100,1),
            },
            "cost": {
                "pipeline_total": round(sum(p_costs),6), "baseline_total": round(sum(b_costs),6),
                "reduction_mean_pct": round(np.mean(cost_reds),2),
                "reduction_total_pct": round((sum(b_costs)-sum(p_costs))/max(sum(b_costs),1e-9)*100,2),
            },
            "routing": {"strong": tot_strong, "weak": tot_weak,
                        "weak_pct": round(tot_weak/(tot_strong+tot_weak)*100,1) if (tot_strong+tot_weak) else 0},
            "parallelism": {"mean_ratio": round(np.mean(par),3)},
            "timing_breakdown_ms": tavg,
            "feedback": self.feedback.stats(), "safety": self.safety.stats(),
        }


    @staticmethod
    def _print_eval(m):
        lat, cost, rt = m["latency"], m["cost"], m["routing"]
        print(f"\n{'='*65}")
        print(f"  ADAPTIVE SEMANTIC PARALLELISM — RESULTS")
        print(f"{'='*65}")
        print(f"  Prompts: {m['n_prompts']}  |  Segments: {m['n_segments_total']}  |  Multi-task: {m['multi_task_pct']}%")
        print(f"\n  {'METRIC':<28} {'PIPELINE':>12} {'BASELINE':>12} {'REDUCTION':>12}")
        print(f"  {'─'*64}")
        print(f"  {'Latency (mean)':<28} {lat['pipeline_mean_ms']:>10.1f}ms {lat['baseline_mean_ms']:>10.1f}ms {lat['reduction_mean_pct']:>+10.1f}%")
        print(f"  {'Latency (median)':<28} {lat['pipeline_median_ms']:>10.1f}ms {lat['baseline_median_ms']:>10.1f}ms {lat['reduction_median_pct']:>+10.1f}%")
        print(f"  {'Cost (total)':<28} ${cost['pipeline_total']:>10.6f} ${cost['baseline_total']:>10.6f} {cost['reduction_total_pct']:>+10.1f}%")
        print(f"\n  Faster on {lat['positive_reduction_pct']:.0f}% of prompts")
        print(f"  Routing: {rt['strong']} strong / {rt['weak']} weak ({rt['weak_pct']}% weak)")
        print(f"  Parallelism ratio: {m['parallelism']['mean_ratio']:.3f}")
        print(f"\n  Timing breakdown (avg/prompt):")
        for k,v in m["timing_breakdown_ms"].items():
            bar = "█" * max(1, int(v/20))
            print(f"    {k:18s} {v:8.1f}ms {bar}")
        print(f"{'='*65}")

    @staticmethod
    def _print_samples(comps):
        print(f"\n{'='*80}")
        print(f"  SAMPLE COMPARISONS: Pipeline vs Baseline")
        print(f"{'='*80}")
        for comp in comps:
            p, b, c = comp["pipeline"], comp["baseline"], comp["comparison"]
            print(f"\n{'─'*80}")
            print(f"  PROMPT: {comp['prompt'][:90]}{'…' if len(comp['prompt'])>90 else ''}")

            print(f"\n  ┌─ PIPELINE ({p['n_segments']} segs, {p['routing']['strong']}S/{p['routing']['weak']}W) ─")
            for s in p["segment_details"]:
                print(f"  │ [{s['segment_id']}] {s['intent']:15s} → {s['route']:12s} \"{s['text'][:55]}{'…' if len(s['text'])>55 else ''}\"")
            for o in p["segment_outputs"]:
                out = o["output"][:120].replace("\n"," ")
                print(f"  │ Output[{o['segment_id']}] ({o['model_used']}, {o['latency']:.3f}s): {out}{'…' if len(o['output'])>120 else ''}")
            fa = p["final_answer"][:200].replace("\n"," ")
            print(f"  │ FINAL ({p['aggregation_method']}): {fa}{'…' if len(p['final_answer'])>200 else ''}")
            print(f"  │ ⏱ {p['latency']*1000:.0f}ms  💰${p['cost']:.6f}")
            print(f"  └─")

            print(f"\n  ┌─ BASELINE ({b['model_used']}) ─")
            bo = b["output"][:250].replace("\n"," ")
            print(f"  │ {bo}{'…' if len(b['output'])>250 else ''}")
            print(f"  │ ⏱ {b['latency']*1000:.0f}ms  💰${b['cost']:.6f}")
            print(f"  └─")

            li = "🟢" if c["pipeline_faster"] else "🔴"
            ci = "🟢" if c["pipeline_cheaper"] else "🔴"
            print(f"\n  {li} Latency: {c['latency_reduction_pct']:+.1f}%  {ci} Cost: {c['cost_reduction_pct']:+.1f}%")
        print(f"\n{'='*80}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/dataset.jsonl")
    parser.add_argument("--backend", default="simulated", choices=["simulated","groq"])
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--output", default="evaluation/pipeline_results.json")
    parser.add_argument("--intent-model", default="models/intent_classifier")
    parser.add_argument("--router-model", default="models/router_model.txt")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save-outputs", action="store_true")
    parser.add_argument("--prompt", default="")
    args = parser.parse_args()

    pipe = AdaptiveSemanticPipeline(args.backend, args.intent_model, args.router_model, args.workers)

    if args.prompt:
        comp = pipe.run_comparison(args.prompt)
        pipe._print_samples([comp])
    else:
        pipe.evaluate(args.data, args.max_samples, args.output, args.save_outputs)

if __name__ == "__main__":
    main()