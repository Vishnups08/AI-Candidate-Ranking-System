#!/usr/bin/env python3
"""
Gold-set evaluator: the optimization target for tuning.

Runs the REAL scoring pipeline (hard filters -> honeypot -> feature scorer ->
behavioral multiplier -> optional cross-encoder) over the human-labeled gold
sample, ranks those candidates by model score, and reports the same metrics the
competition uses (NDCG@10, NDCG@50, MAP, P@10) plus the composite.

Because the gold set is labeled INDEPENDENTLY of the scoring code (see
evaluation/gold_labels.json), an improvement here is a real signal, not the
self-consistency that evaluation/manual_labels.json would report.

Usage:
  python evaluation/evaluate_gold.py \
      --candidates "../[PUB] .../candidates.jsonl" \
      [--no-cross-encoder]
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--no-cross-encoder", action="store_true")
    args = ap.parse_args()

    gold = json.loads(GOLD_PATH.read_text())["labels"]
    gold_ids = set(gold)
    print(f"Gold set: {len(gold_ids)} labeled candidates")

    # Pull the gold candidates' full records.
    records = {}
    for c in load_candidates(args.candidates):
        if c["candidate_id"] in gold_ids:
            records[c["candidate_id"]] = c
        if len(records) == len(gold_ids):
            break
    print(f"Loaded {len(records)} of {len(gold_ids)} gold records from pool")

    jd = load_jd_requirements()
    jd_emb, cand_emb = load_embeddings()
    print(f"Embeddings: jd={'yes' if jd_emb is not None else 'NO'} "
          f"candidates={len(cand_emb)}")
    scorer = FeatureScorer(jd, jd_emb, cand_emb)

    # Score exactly as rank.py does (minus the top-100 cut).
    scored = []
    for cid, c in records.items():
        is_hp, _ = detect_honeypot(c)
        feats = scorer.score_candidate(c)
        mult = compute_behavioral_multiplier(c)
        add = compute_behavioral_additive(c)
        final = feats["weighted_total"] * mult + add
        # Honeypots are excluded pre-scoring in the real pipeline; emulate by
        # forcing them to the bottom so the metric reflects that behavior.
        if is_hp:
            final = -1.0
        scored.append((cid, final, gold[cid], is_hp))

    # Optional cross-encoder blend (mirrors ranker.py) on this small set.
    if not args.no_cross_encoder and config.USE_CROSS_ENCODER:
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
                skills = ", ".join(s.get("name", "") for s in c.get("skills", [])[:15])
                txt = f"{prof.get('current_title','')}. Skills: {skills}. {prof.get('summary','')}"[:config.EMBEDDING_MAX_TEXT_LENGTH]
                pairs.append([q, txt]); idx.append(i)
            ce_scores = 1 / (1 + np.exp(-np.array(ce.predict(pairs))))
            for i, s in zip(idx, ce_scores):
                cid, final, tier, is_hp = scored[i]
                blended = (1 - config.CROSS_ENCODER_WEIGHT) * final + config.CROSS_ENCODER_WEIGHT * float(s)
                scored[i] = (cid, blended, tier, is_hp)
            print("Cross-encoder blend: applied")
        except Exception as e:
            print(f"Cross-encoder skipped in eval: {e}")

    scored.sort(key=lambda x: (-x[1], x[0]))
    ranked_rels = [tier for _, _, tier, _ in scored]

    n10 = ndcg(ranked_rels, 10)
    n50 = ndcg(ranked_rels, 50)
    mp = average_precision(ranked_rels)
    p10 = precision_at_k(ranked_rels, 10)
    composite = 0.50 * n10 + 0.30 * n50 + 0.15 * mp + 0.05 * p10

    print("\n" + "=" * 56)
    print("GOLD-SET METRICS (independent labels)")
    print("=" * 56)
    print(f"  NDCG@10 : {n10:.4f}")
    print(f"  NDCG@50 : {n50:.4f}")
    print(f"  MAP     : {mp:.4f}")
    print(f"  P@10    : {p10:.4f}")
    print(f"  COMPOSITE (0.5/0.3/0.15/0.05): {composite:.4f}")
    print("=" * 56)

    # Honeypot leakage among model's top-10 (should be 0).
    top10_hp = sum(1 for _, _, _, is_hp in scored[:10] if is_hp)
    print(f"  Honeypots in top-10: {top10_hp}")

    print("\nModel's top-15 (cid | score | gold_tier):")
    for cid, sc, tier, is_hp in scored[:15]:
        flag = " [HP]" if is_hp else ""
        print(f"  {cid}  {sc:7.4f}  tier={tier}{flag}")

    print("\nWhere the high-gold candidates (tier>=3) actually landed:")
    for rnk, (cid, sc, tier, _) in enumerate(scored, 1):
        if tier >= 3:
            print(f"  gold tier {tier}: {cid} ranked #{rnk} by model (score={sc:.4f})")

    return composite


if __name__ == "__main__":
    main()
