#!/usr/bin/env python3
"""
Pipeline Performance Benchmark — Measures latency of each pipeline stage.

Usage:
    python evaluation/benchmark.py --candidates ./candidates.jsonl [--n 5000]

Produces a JSON report with per-stage timing, throughput, and total latency.
Used to populate the Evaluation & Metrics page with real performance data.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from pipeline.loader import load_candidates
from pipeline.jd_parser import load_jd_requirements
from pipeline.hard_filters import apply_hard_filters, passes_hard_filters
from pipeline.honeypot_detector import filter_honeypots, detect_honeypot
from pipeline.feature_scorer import FeatureScorer
from pipeline.behavioral_scorer import compute_behavioral_multiplier, compute_behavioral_additive
from pipeline.ranker import CandidateRanker


def benchmark_pipeline(candidates_path: str, n: int = 5000) -> dict:
    """Run each pipeline stage individually and measure wall-clock time."""

    results = {}

    # ─── Load candidates ────────────────────────────────────────────────
    t0 = time.perf_counter()
    all_candidates = []
    for c in load_candidates(candidates_path):
        all_candidates.append(c)
        if len(all_candidates) >= n:
            break
    t_load = time.perf_counter() - t0
    results["load"] = {
        "stage": "Load & Parse JSONL",
        "candidates": len(all_candidates),
        "latency_s": round(t_load, 3),
        "throughput": round(len(all_candidates) / t_load, 0),
    }
    print(f"  Load: {t_load:.3f}s ({len(all_candidates)} candidates)")

    # ─── JD parsing ─────────────────────────────────────────────────────
    t0 = time.perf_counter()
    jd = load_jd_requirements()
    t_jd = time.perf_counter() - t0
    results["jd_parse"] = {
        "stage": "JD Requirements Parsing",
        "latency_s": round(t_jd, 3),
    }
    print(f"  JD Parse: {t_jd:.3f}s")

    # ─── Stage 1: Hard filters ──────────────────────────────────────────
    t0 = time.perf_counter()
    filtered = apply_hard_filters(all_candidates)
    t_hard = time.perf_counter() - t0
    results["hard_filters"] = {
        "stage": "Stage 1: Hard Filters",
        "input": len(all_candidates),
        "output": len(filtered),
        "filtered_out": len(all_candidates) - len(filtered),
        "filter_rate": round((len(all_candidates) - len(filtered)) / len(all_candidates) * 100, 1),
        "latency_s": round(t_hard, 3),
        "throughput": round(len(all_candidates) / max(t_hard, 0.001), 0),
    }
    print(f"  Hard Filters: {t_hard:.3f}s ({len(all_candidates)} -> {len(filtered)})")

    # ─── Stage 2: Honeypot detection ────────────────────────────────────
    t0 = time.perf_counter()
    clean, honeypots = filter_honeypots(filtered)
    t_honeypot = time.perf_counter() - t0
    results["honeypot"] = {
        "stage": "Stage 2: Honeypot Detection",
        "input": len(filtered),
        "output": len(clean),
        "honeypots_found": len(honeypots),
        "latency_s": round(t_honeypot, 3),
        "throughput": round(len(filtered) / max(t_honeypot, 0.001), 0),
    }
    print(f"  Honeypot: {t_honeypot:.3f}s ({len(filtered)} -> {len(clean)}, {len(honeypots)} honeypots)")

    # ─── Stage 3: Feature scoring ───────────────────────────────────────
    scorer = FeatureScorer(jd)
    t0 = time.perf_counter()
    scored = []
    for c in clean:
        scores = scorer.score_candidate(c)
        scored.append((c, scores))
    t_scoring = time.perf_counter() - t0
    results["feature_scoring"] = {
        "stage": "Stage 3: Feature Scoring (6 dimensions)",
        "candidates_scored": len(clean),
        "latency_s": round(t_scoring, 3),
        "per_candidate_ms": round(t_scoring / max(len(clean), 1) * 1000, 2),
        "throughput": round(len(clean) / max(t_scoring, 0.001), 0),
    }
    print(f"  Feature Scoring: {t_scoring:.3f}s ({len(clean)} candidates, "
          f"{t_scoring / max(len(clean), 1) * 1000:.2f}ms/candidate)")

    # ─── Stage 4: Behavioral multiplier ─────────────────────────────────
    t0 = time.perf_counter()
    final_scored = []
    for c, scores in scored:
        mult = compute_behavioral_multiplier(c)
        add = compute_behavioral_additive(c)
        final = scores["weighted_total"] * mult + add
        final_scored.append((c, final))
    t_behavioral = time.perf_counter() - t0
    results["behavioral"] = {
        "stage": "Stage 4: Behavioral Multiplier",
        "candidates": len(clean),
        "latency_s": round(t_behavioral, 3),
        "throughput": round(len(clean) / max(t_behavioral, 0.001), 0),
    }
    print(f"  Behavioral: {t_behavioral:.3f}s")

    # ─── Stage 5: Sorting (simulating cross-encoder on small pool) ──────
    t0 = time.perf_counter()
    final_scored.sort(key=lambda x: -x[1])
    top_100 = final_scored[:100]
    t_sort = time.perf_counter() - t0
    results["sort_rank"] = {
        "stage": "Stage 5: Sort & Rank (Top-100 extraction)",
        "latency_s": round(t_sort, 3),
    }
    print(f"  Sort & Rank: {t_sort:.3f}s")

    # ─── Total ──────────────────────────────────────────────────────────
    total = t_load + t_jd + t_hard + t_honeypot + t_scoring + t_behavioral + t_sort
    results["total"] = {
        "total_latency_s": round(total, 3),
        "total_candidates": len(all_candidates),
        "candidates_ranked": len(clean),
        "meets_5min_constraint": total < 300,
        "projected_100k_s": round(total / len(all_candidates) * 100000, 1) if len(all_candidates) > 0 else None,
    }
    print(f"\n  TOTAL: {total:.3f}s for {len(all_candidates)} candidates")
    if results["total"]["projected_100k_s"]:
        print(f"  Projected 100K: {results['total']['projected_100k_s']:.1f}s "
              f"({'[PASS]' if results['total']['projected_100k_s'] < 300 else '[FAIL]'} 5-min constraint)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Pipeline Performance Benchmark")
    parser.add_argument("--candidates", required=True, help="Path to candidates JSONL")
    parser.add_argument("--n", type=int, default=5000, help="Number of candidates to benchmark (default: 5000)")
    parser.add_argument("--output", default="evaluation/benchmark_results.json", help="Output JSON path")
    args = parser.parse_args()

    print("=" * 60)
    print("Pipeline Performance Benchmark")
    print(f"  Candidates: {args.candidates}")
    print(f"  Sample size: {args.n}")
    print("=" * 60)

    results = benchmark_pipeline(args.candidates, args.n)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
