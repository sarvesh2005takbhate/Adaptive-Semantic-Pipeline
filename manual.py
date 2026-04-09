"""
Interactive Pipeline Runner
=============================
Run prompts interactively and save results as TXT + JSON.

Usage:
    # Interactive mode
    python run_interactive.py --backend groq

    # Single prompt
    python run_interactive.py --backend groq --prompt "Explain RNN and write code for it"

    # Batch with TXT output
    python run_interactive.py --backend groq --data data/dataset.jsonl --max-samples 20
"""

from __future__ import annotations

import argparse
import json
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

sys.path.insert(0, str(Path(__file__).resolve().parent))


def format_comparison_txt(comp: Dict[str, Any]) -> str:
    """Format a single comparison as readable text."""
    p = comp["pipeline"]
    b = comp["baseline"]
    c = comp["comparison"]
    lines = []

    lines.append("=" * 80)
    lines.append(f"PROMPT: {comp['prompt']}")
    lines.append("=" * 80)

    # Pipeline
    lines.append(f"\n--- PIPELINE ({p['n_segments']} segments) ---")
    lines.append(f"Routing: {p['routing']['strong']} strong / {p['routing']['weak']} weak")
    lines.append(f"Aggregation: {p['aggregation_method']}")
    lines.append(f"Latency: {p['latency']*1000:.1f}ms | Cost: ${p['cost']:.6f}")

    lines.append(f"\nSegment Breakdown:")
    for s in p["segment_details"]:
        lines.append(f"  [{s['segment_id']}] Intent: {s['intent']:15s} | Route: {s['route']:12s} | Complexity: {s['complexity']:.2f}")
        lines.append(f"      Text: {s['text']}")

    lines.append(f"\nPer-Segment Outputs:")
    for o in p["segment_outputs"]:
        lines.append(f"  [{o['segment_id']}] Model: {o['model_used']} | Latency: {o['latency']:.3f}s | Cost: ${o['cost']:.6f} | Tokens: {o['tokens']}")
        lines.append(f"      Output: {o['output'][:500]}")
        if len(o["output"]) > 500:
            lines.append(f"      ... ({len(o['output'])} chars total)")

    lines.append(f"\nFinal Answer:")
    lines.append(f"  {p['final_answer']}")

    # Baseline
    lines.append(f"\n--- BASELINE ({b['model_used']}) ---")
    lines.append(f"Latency: {b['latency']*1000:.1f}ms | Cost: ${b['cost']:.6f} | Tokens: {b.get('tokens_used',0)}")
    lines.append(f"\nOutput:")
    lines.append(f"  {b['output']}")

    # Comparison
    lines.append(f"\n--- COMPARISON ---")
    lat_label = "FASTER" if c["pipeline_faster"] else "SLOWER"
    cost_label = "CHEAPER" if c["pipeline_cheaper"] else "MORE EXPENSIVE"
    lines.append(f"  Latency: {c['latency_reduction_pct']:+.1f}% ({lat_label})")
    lines.append(f"  Cost:    {c['cost_reduction_pct']:+.1f}% ({cost_label})")

    # Timing
    lines.append(f"\nTiming Breakdown:")
    for k, v in p.get("timings", {}).items():
        lines.append(f"  {k:20s} {v*1000:8.1f}ms")

    lines.append("")
    return "\n".join(lines)


def format_summary_txt(metrics: Dict[str, Any]) -> str:
    """Format evaluation summary as text."""
    lat = metrics["latency"]
    cost = metrics["cost"]
    rt = metrics["routing"]

    lines = []
    lines.append("=" * 80)
    lines.append("  ADAPTIVE SEMANTIC PARALLELISM — EVALUATION SUMMARY")
    lines.append(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)

    lines.append(f"\n  Total Prompts:         {metrics['n_prompts']}")
    lines.append(f"  Total Segments:        {metrics['n_segments_total']}")
    lines.append(f"  Multi-task:            {metrics['multi_task_prompts']} ({metrics['multi_task_pct']}%)")
    lines.append(f"  Errors:                {metrics['n_errors']}")

    lines.append(f"\n  {'METRIC':<28} {'PIPELINE':>14} {'BASELINE':>14} {'CHANGE':>14}")
    lines.append(f"  {'-'*70}")
    lines.append(f"  {'Latency (mean)':<28} {lat['pipeline_mean_ms']:>12.1f}ms {lat['baseline_mean_ms']:>12.1f}ms {lat['reduction_mean_pct']:>+12.1f}%")
    lines.append(f"  {'Latency (median)':<28} {lat['pipeline_median_ms']:>12.1f}ms {lat['baseline_median_ms']:>12.1f}ms {lat['reduction_median_pct']:>+12.1f}%")
    lines.append(f"  {'Cost (total)':<28} {'$'+str(round(cost['pipeline_total'],6)):>14} {'$'+str(round(cost['baseline_total'],6)):>14} {cost['reduction_total_pct']:>+12.1f}%")

    lines.append(f"\n  Faster on:             {lat['positive_reduction_pct']:.0f}% of prompts")
    lines.append(f"  Strong segments:       {rt['strong']}")
    lines.append(f"  Weak segments:         {rt['weak']} ({rt['weak_pct']}%)")
    lines.append(f"  Parallelism ratio:     {metrics['parallelism']['mean_ratio']:.3f}")

    lines.append(f"\n  Timing Breakdown (avg per prompt):")
    for k, v in metrics.get("timing_breakdown_ms", {}).items():
        bar = "█" * max(1, int(v / 15))
        lines.append(f"    {k:20s} {v:8.1f}ms  {bar}")

    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def save_results(comparisons: List[Dict], metrics: Dict, output_dir: str):
    """Save all results as TXT and JSON."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Summary TXT
    summary_txt = format_summary_txt(metrics)
    with open(out / "evaluation_summary.txt", "w") as f:
        f.write(summary_txt)

    # 2. Per-prompt comparisons TXT
    with open(out / "prompt_comparisons.txt", "w") as f:
        f.write(f"ADAPTIVE SEMANTIC PARALLELISM — DETAILED RESULTS\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total prompts: {len(comparisons)}\n\n")

        for comp in comparisons:
            f.write(format_comparison_txt(comp))
            f.write("\n\n")

    # 3. JSON (machine-readable)
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    with open(out / "detailed_outputs.json", "w") as f:
        json.dump(comparisons, f, indent=2, default=str, ensure_ascii=False)

    print(f"\nResults saved to {out}/:")
    print(f"  evaluation_summary.txt    — summary table")
    print(f"  prompt_comparisons.txt    — per-prompt details with LLM outputs")
    print(f"  metrics.json              — machine-readable metrics")
    print(f"  detailed_outputs.json     — full JSON output")


def run_interactive(pipeline):
    """Interactive loop: type prompts, see results."""
    print("\n" + "=" * 60)
    print("  ADAPTIVE SEMANTIC PARALLELISM — Interactive Mode")
    print("=" * 60)
    print("  Type a prompt and press Enter to see pipeline vs baseline.")
    print("  Type 'quit' or 'exit' to stop.\n")

    history = []
    prompt_id = 1

    while True:
        try:
            prompt = input(f"  [{prompt_id}] Your prompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if not prompt or prompt.lower() in ("quit", "exit", "q"):
            break

        print(f"\n  Processing...\n")
        try:
            comp = pipeline.run_comparison(prompt, prompt_id)
            history.append(comp)

            # Print result
            p = comp["pipeline"]
            b = comp["baseline"]
            c = comp["comparison"]

            print(f"  {'─'*60}")
            print(f"  PIPELINE ({p['n_segments']} segments)")
            print(f"  {'─'*60}")

            for s in p["segment_details"]:
                print(f"    [{s['segment_id']}] {s['intent']:15s} → {s['route']}")
                print(f"        \"{s['text'][:70]}\"")

            print(f"\n  Segment Outputs:")
            for o in p["segment_outputs"]:
                print(f"    [{o['segment_id']}] ({o['model_used']}, {o['latency']:.2f}s):")
                out_lines = o["output"].split("\n")
                for line in out_lines[:5]:
                    print(f"        {line[:80]}")
                if len(out_lines) > 5:
                    print(f"        ... ({len(out_lines)} lines)")

            print(f"\n  Final Answer ({p['aggregation_method']}):")
            fa_lines = p["final_answer"].split("\n")
            for line in fa_lines[:8]:
                print(f"    {line[:80]}")
            if len(fa_lines) > 8:
                print(f"    ... ({len(fa_lines)} lines total)")

            print(f"\n  ⏱ {p['latency']*1000:.0f}ms  💰${p['cost']:.6f}")

            print(f"\n  {'─'*60}")
            print(f"  BASELINE ({b['model_used']})")
            print(f"  {'─'*60}")
            b_lines = b["output"].split("\n")
            for line in b_lines[:8]:
                print(f"    {line[:80]}")
            if len(b_lines) > 8:
                print(f"    ... ({len(b_lines)} lines total)")
            print(f"\n  ⏱ {b['latency']*1000:.0f}ms  💰${b['cost']:.6f}")

            # Comparison
            li = "🟢" if c["pipeline_faster"] else "🔴"
            ci = "🟢" if c["pipeline_cheaper"] else "🔴"
            print(f"\n  {li} Latency: {c['latency_reduction_pct']:+.1f}%  "
                  f"{ci} Cost: {c['cost_reduction_pct']:+.1f}%")
            print()

        except Exception as e:
            print(f"  Error: {e}\n")

        prompt_id += 1

    # Save history if any
    if history:
        save = input("\n  Save results? (y/n): ").strip().lower()
        if save == "y":
            out_dir = input("  Output directory [results/]: ").strip() or "results"
            # Compute quick metrics
            lat_reds = [c["comparison"]["latency_reduction_pct"] for c in history]
            cost_reds = [c["comparison"]["cost_reduction_pct"] for c in history]
            import numpy as np
            metrics = {
                "n_prompts": len(history),
                "n_errors": 0,
                "n_segments_total": sum(c["pipeline"]["n_segments"] for c in history),
                "multi_task_prompts": sum(1 for c in history if c["pipeline"]["n_segments"] > 1),
                "multi_task_pct": round(sum(1 for c in history if c["pipeline"]["n_segments"] > 1) / len(history) * 100, 1),
                "latency": {
                    "pipeline_mean_ms": round(np.mean([c["pipeline"]["latency"] for c in history]) * 1000, 2),
                    "pipeline_median_ms": round(np.median([c["pipeline"]["latency"] for c in history]) * 1000, 2),
                    "baseline_mean_ms": round(np.mean([c["baseline"]["latency"] for c in history]) * 1000, 2),
                    "baseline_median_ms": round(np.median([c["baseline"]["latency"] for c in history]) * 1000, 2),
                    "reduction_mean_pct": round(np.mean(lat_reds), 2),
                    "reduction_median_pct": round(np.median(lat_reds), 2),
                    "positive_reduction_pct": round(sum(1 for x in lat_reds if x > 0) / len(lat_reds) * 100, 1),
                },
                "cost": {
                    "pipeline_total": round(sum(c["pipeline"]["cost"] for c in history), 6),
                    "baseline_total": round(sum(c["baseline"]["cost"] for c in history), 6),
                    "reduction_total_pct": round(
                        (sum(c["baseline"]["cost"] for c in history) - sum(c["pipeline"]["cost"] for c in history))
                        / max(sum(c["baseline"]["cost"] for c in history), 1e-9) * 100, 2),
                    "reduction_mean_pct": round(np.mean(cost_reds), 2),
                },
                "routing": {
                    "strong": sum(c["pipeline"]["routing"]["strong"] for c in history),
                    "weak": sum(c["pipeline"]["routing"]["weak"] for c in history),
                    "weak_pct": 0,
                },
                "parallelism": {"mean_ratio": round(np.mean([c["pipeline"]["dag_stats"]["parallelism_ratio"] for c in history]), 3)},
                "timing_breakdown_ms": {},
            }
            s, w = metrics["routing"]["strong"], metrics["routing"]["weak"]
            metrics["routing"]["weak_pct"] = round(w / (s + w) * 100, 1) if (s + w) else 0

            save_results(history, metrics, out_dir)
            print(f"  Saved to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Interactive Pipeline Runner")
    parser.add_argument("--backend", default="simulated", choices=["simulated", "groq"])
    parser.add_argument("--intent-model", default="models/intent_classifier")
    parser.add_argument("--router-model", default="models/router_model.txt")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prompt", default="", help="Single prompt (non-interactive)")
    parser.add_argument("--data", default="", help="Batch mode on dataset")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--output-dir", default="results", help="Output directory")
    args = parser.parse_args()

    from pipeline import AdaptiveSemanticPipeline
    pipeline = AdaptiveSemanticPipeline(
        backend=args.backend,
        intent_model_path=args.intent_model,
        router_model_path=args.router_model,
        max_workers=args.workers,
    )

    if args.prompt:
        # Single prompt
        comp = pipeline.run_comparison(args.prompt)
        print(format_comparison_txt(comp))

    elif args.data:
        # Batch mode with TXT saving
        import numpy as np
        data = []
        with open(args.data) as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
        if args.max_samples > 0:
            data = data[:args.max_samples]

        comparisons = []
        for i, item in enumerate(data):
            try:
                comp = pipeline.run_comparison(item["prompt"], item.get("prompt_id", i + 1))
                comparisons.append(comp)
            except Exception as e:
                print(f"  Error on {i+1}: {e}")
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(data)}")

        if comparisons:
            lat_reds = [c["comparison"]["latency_reduction_pct"] for c in comparisons]
            cost_reds = [c["comparison"]["cost_reduction_pct"] for c in comparisons]
            p_lats = [c["pipeline"]["latency"] for c in comparisons]
            b_lats = [c["baseline"]["latency"] for c in comparisons]
            metrics = {
                "n_prompts": len(comparisons), "n_errors": len(data) - len(comparisons),
                "n_segments_total": sum(c["pipeline"]["n_segments"] for c in comparisons),
                "multi_task_prompts": sum(1 for c in comparisons if c["pipeline"]["n_segments"] > 1),
                "multi_task_pct": round(sum(1 for c in comparisons if c["pipeline"]["n_segments"] > 1) / len(comparisons) * 100, 1),
                "latency": {
                    "pipeline_mean_ms": round(np.mean(p_lats) * 1000, 2),
                    "pipeline_median_ms": round(np.median(p_lats) * 1000, 2),
                    "baseline_mean_ms": round(np.mean(b_lats) * 1000, 2),
                    "baseline_median_ms": round(np.median(b_lats) * 1000, 2),
                    "reduction_mean_pct": round(np.mean(lat_reds), 2),
                    "reduction_median_pct": round(np.median(lat_reds), 2),
                    "positive_reduction_pct": round(sum(1 for x in lat_reds if x > 0) / len(lat_reds) * 100, 1),
                },
                "cost": {
                    "pipeline_total": round(sum(c["pipeline"]["cost"] for c in comparisons), 6),
                    "baseline_total": round(sum(c["baseline"]["cost"] for c in comparisons), 6),
                    "reduction_total_pct": round(
                        (sum(c["baseline"]["cost"] for c in comparisons) - sum(c["pipeline"]["cost"] for c in comparisons))
                        / max(sum(c["baseline"]["cost"] for c in comparisons), 1e-9) * 100, 2),
                    "reduction_mean_pct": round(np.mean(cost_reds), 2),
                },
                "routing": {
                    "strong": sum(c["pipeline"]["routing"]["strong"] for c in comparisons),
                    "weak": sum(c["pipeline"]["routing"]["weak"] for c in comparisons),
                    "weak_pct": 0,
                },
                "parallelism": {"mean_ratio": round(np.mean([c["pipeline"]["dag_stats"]["parallelism_ratio"] for c in comparisons]), 3)},
                "timing_breakdown_ms": {},
            }
            s, w = metrics["routing"]["strong"], metrics["routing"]["weak"]
            metrics["routing"]["weak_pct"] = round(w / (s + w) * 100, 1) if (s + w) else 0

            save_results(comparisons, metrics, args.output_dir)
    else:
        # Interactive mode
        run_interactive(pipeline)


if __name__ == "__main__":
    main()