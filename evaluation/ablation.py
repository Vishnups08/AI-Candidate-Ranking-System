#!/usr/bin/env python3
"""
Ablation Study Runner — evaluation/ablation.py

Measures the contribution of each pipeline component by disabling it
and reporting the drop in gold-set composite score.

Components tested:
  1. Baseline (full pipeline)
  2. No skill-career coherence gate
  3. No honeypot de-noising
  4. No cross-encoder re-ranking
  5. No behavioral multiplier (fixed at 1.0)
  6. Skills score only (no career_fit, experience, location, education, semantic)

Usage:
    python evaluation/ablation.py --candidates path/to/candidates.jsonl

Output:
    Markdown ablation table (printed to stdout)
    Optional JSON: --json-out ablation_results.json
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import config
from pipeline.loader import load_candidates
from pipeline.jd_parser import load_jd_requirements
from pipeline.honeypot_detector import detect_honeypot
from pipeline.feature_scorer import FeatureScorer
from pipeline.behavioral_scorer import compute_behavioral_multiplier, compute_behavioral_additive

GOLD_PATH = Path(__file__).parent / "gold_labels.json"
EMB_DIR = Path(__file__).parent.parent / "precompute" / "embeddings"


# ── Metric helpers ────────────────────────────────────────────────────────────

def dcg(rels, k):
    rels = rels[:k]
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg(rels, k):
    ideal = dcg(sorted(rels, reverse=True), k)
    return dcg(rels, k) / ideal if ideal > 0 else 0.0


def average_precision(rels, rel_threshold=3):
    hits, sum_prec = 0, 0.0
    total_rel = sum(1 for r in rels if r >= rel_threshold)
    if total_rel == 0:
        return 0.0
    for i, r in enumerate(rels):
        if r >= rel_threshold:
            hits += 1
            sum_prec += hits / (i + 1)
    return sum_prec / total_rel


def precision_at_k(rels, k, rel_threshold=3):
    rels = rels[:k]
    return sum(1 for r in rels if r >= rel_threshold) / k if rels else 0.0


def composite(rels):
    n10 = ndcg(rels, 10)
    n50 = ndcg(rels, 50)
    mp = average_precision(rels)
    p10 = precision_at_k(rels, 10)
    score = 0.50 * n10 + 0.30 * n50 + 0.15 * mp + 0.05 * p10
    return {"ndcg_10": n10, "ndcg_50": n50, "map": mp, "p10": p10, "composite": score}


# ── Embedding loader ─────────────────────────────────────────────────────────

def load_embeddings():
    jd_emb, cand_emb = None, {}
    p = EMB_DIR / "jd_embedding.npy"
    if p.exists():
        jd_emb = np.load(str(p))
    pe = EMB_DIR / "candidate_profiles.npz"
    if pe.exists():
        d = np.load(str(pe), allow_pickle=True)
        for cid, emb in zip(d["candidate_ids"], d["embeddings"]):
            cand_emb.setdefault(str(cid), {})["profile_embedding"] = emb
    re_ = EMB_DIR / "candidate_roles.npz"
    if re_.exists():
        d = np.load(str(re_), allow_pickle=True)
        for cid, r in zip(d["candidate_ids"], d["role_embeddings"]):
            cand_emb.setdefault(str(cid), {})["role_embeddings"] = r
    return jd_emb, cand_emb


# ── Single ablation run ───────────────────────────────────────────────────────

def run_ablation(
    records: dict,
    gold: dict,
    jd,
    jd_emb,
    cand_emb: dict,
    *,
    disable_coherence: bool = False,
    disable_honeypot: bool = False,
    disable_cross_encoder: bool = False,
    disable_behavioral: bool = False,
    skills_only: bool = False,
) -> dict:
    """
    Score the gold set with one component disabled.
    Returns a metrics dict.
    """
    scorer = FeatureScorer(jd, jd_emb, cand_emb)

    # Monkey-patch coherence gate to 1.0 if disabled
    if disable_coherence:
        scorer._skill_career_coherence = lambda candidate: 1.0  # type: ignore

    # Override weights if skills_only
    original_weights = config.WEIGHTS.copy()
    if skills_only:
        config.WEIGHTS = {
            "semantic_similarity": 0.0,
            "career_fit": 0.0,
            "skills_match": 0.95,
            "experience_fit": 0.0,
            "location_logistics": 0.0,
            "education": 0.0,
        }

    scored = []
    for cid, c in records.items():
        is_hp, _ = detect_honeypot(c)
        feats = scorer.score_candidate(c)
        mult = 1.0 if disable_behavioral else compute_behavioral_multiplier(c)
        add  = 0.0 if disable_behavioral else compute_behavioral_additive(c)
        final = feats["weighted_total"] * mult + add
        if disable_honeypot:
            is_hp = False  # treat all as clean
        if is_hp:
            final = -1.0
        scored.append((cid, final, gold[cid], is_hp))

    # Restore weights
    if skills_only:
        config.WEIGHTS = original_weights

    # Optional cross-encoder blend
    if not disable_cross_encoder and config.USE_CROSS_ENCODER:
        try:
            from sentence_transformers import CrossEncoder
            ce_path = config._local_model_path(config.CROSS_ENCODER_MODEL)
            ce = CrossEncoder(ce_path, device="cpu",
                              local_files_only=ce_path != config.CROSS_ENCODER_MODEL)
            q = getattr(jd, "jd_core_text", jd.title)
            pairs, idx = [], []
            for i, (cid, final, tier, is_hp) in enumerate(scored):
                if is_hp:
                    continue
                c = records[cid]
                prof = c.get("profile", {})
                skills_str = ", ".join(s.get("name", "") for s in c.get("skills", [])[:15])
                txt = f"{prof.get('current_title','')}. Skills: {skills_str}. {prof.get('summary','')}"\
                      [:config.EMBEDDING_MAX_TEXT_LENGTH]
                pairs.append([q, txt])
                idx.append(i)
            if pairs:
                ce_scores = 1 / (1 + np.exp(-np.array(ce.predict(pairs))))
                for i, s in zip(idx, ce_scores):
                    cid, final, tier, is_hp = scored[i]
                    blended = (1 - config.CROSS_ENCODER_WEIGHT) * final + config.CROSS_ENCODER_WEIGHT * float(s)
                    scored[i] = (cid, blended, tier, is_hp)
        except Exception:
            pass  # Skip CE if unavailable

    scored.sort(key=lambda x: (-x[1], x[0]))
    rels = [tier for _, _, tier, _ in scored]
    return composite(rels)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Ablation study for the Redrob ranker pipeline")
    ap.add_argument("--candidates", required=True, help="Path to candidates JSONL")
    ap.add_argument("--json-out", default=None, help="Optional path to write JSON results")
    args = ap.parse_args()

    gold = json.loads(GOLD_PATH.read_text())["labels"]
    gold_ids = set(gold)
    print(f"Gold set: {len(gold_ids)} labeled candidates")

    records = {}
    for c in load_candidates(args.candidates):
        if c["candidate_id"] in gold_ids:
            records[c["candidate_id"]] = c
        if len(records) == len(gold_ids):
            break
    print(f"Loaded {len(records)} gold records\n")

    jd = load_jd_requirements()
    jd_emb, cand_emb = load_embeddings()

    # If no precomputed embeddings, embed on the fly
    if not cand_emb:
        print("No precomputed embeddings found — embedding gold set on the fly...")
        try:
            from sentence_transformers import SentenceTransformer
            sys.path.insert(0, str(Path(__file__).parent.parent / "precompute"))
            from build_embeddings import build_profile_text, build_role_texts

            m_path = config._local_model_path(config.EMBEDDING_MODEL)
            model = SentenceTransformer(m_path, device="cpu",
                                        local_files_only=m_path != config.EMBEDDING_MODEL)
            ids = list(records)
            pvecs = model.encode(
                [config.EMBEDDING_PASSAGE_PREFIX + build_profile_text(records[c]) for c in ids],
                batch_size=32, normalize_embeddings=True
            )
            for cid, v in zip(ids, pvecs):
                cand_emb[cid] = {"profile_embedding": v}
            for cid in ids:
                rtexts = build_role_texts(records[cid])
                if rtexts:
                    rvecs = model.encode(rtexts, batch_size=32, normalize_embeddings=True)
                    cand_emb[cid]["role_embeddings"] = rvecs
            print(f"Embedded {len(cand_emb)} candidates\n")
        except ImportError:
            print("WARNING: sentence-transformers not available; semantic scores will be keyword-fallback\n")

    # Define ablation runs
    ablations = [
        ("Baseline (full pipeline)", {}),
        ("No coherence gate",        {"disable_coherence": True}),
        ("No honeypot de-noising",   {"disable_honeypot": True}),
        ("No cross-encoder",         {"disable_cross_encoder": True}),
        ("No behavioral multiplier", {"disable_behavioral": True}),
        ("Skills-only (no semantic)", {"skills_only": True}),
    ]

    results = {}
    for name, kwargs in ablations:
        print(f"Running: {name}...", end=" ", flush=True)
        metrics = run_ablation(records, gold, jd, jd_emb, cand_emb, **kwargs)
        results[name] = metrics
        print(f"composite={metrics['composite']:.4f}")

    # Print Markdown table
    baseline = results["Baseline (full pipeline)"]["composite"]
    print("\n\n" + "=" * 72)
    print("ABLATION TABLE (gold set, independent labels)")
    print("=" * 72)
    print()
    print(f"| Configuration                | NDCG@10 | NDCG@50 | MAP    | P@10   | Composite | Diff vs Baseline |")
    print(f"|------------------------------|---------|---------|--------|--------|-----------|------------------|")
    for name, m in results.items():
        delta = m["composite"] - baseline
        delta_str = f"{delta:+.4f}" if name != "Baseline (full pipeline)" else "-"
        print(
            f"| {name:<28} | {m['ndcg_10']:.4f}  | {m['ndcg_50']:.4f}  | {m['map']:.4f} | {m['p10']:.4f} | {m['composite']:.4f}    | {delta_str:<16} |"
        )

    print()
    print("=" * 72)
    print("Interpretation:")
    for name, m in results.items():
        if name == "Baseline (full pipeline)":
            continue
        delta = m["composite"] - baseline
        arrow = "earns its place" if delta < -0.005 else ("neutral" if abs(delta) <= 0.005 else "may hurt")
        print(f"  {name}: Diff = {delta:+.4f}  =>  {arrow}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2))
        print(f"\nJSON results written to {args.json_out}")


if __name__ == "__main__":
    main()
