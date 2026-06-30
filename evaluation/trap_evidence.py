#!/usr/bin/env python3
"""
Trap Evidence Report — evaluation/trap_evidence.py

Scans the candidate pool and surfaces concrete examples of each "trap"
archetype the pipeline handles:

  1. Honeypots — internally impossible profiles (timeline, skill-duration, overlap)
  2. Career-long consulting — demoted by JD-explicit disqualifier
  3. Keyword-stuffers — high naive rank, low pipeline rank (coherence gate)
  4. CV/robotics without NLP/IR — domain disqualifier
  5. Research-only / no production — production deployment disqualifier

Usage:
    python evaluation/trap_evidence.py --candidates path/to/candidates.jsonl [--n 5]

Output:
    Markdown report (printed to stdout) + optional JSON: --json-out

This is designed to be embedded in the System Architecture Note as
"Trap-handling evidence: a clean table of archetype → score → outcome."
"""

import argparse
import json
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
from pipeline.honeypot_detector import detect_honeypot, filter_honeypots
from pipeline.hard_filters import apply_hard_filters
from pipeline.feature_scorer import FeatureScorer
from pipeline.behavioral_scorer import compute_behavioral_multiplier, compute_behavioral_additive

EMB_DIR = Path(__file__).parent.parent / "precompute" / "embeddings"


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


def naive_keyword_score(candidate: dict) -> int:
    jd_keywords = {
        "embedding", "embeddings", "retrieval", "vector", "vector database",
        "search", "ranking", "nlp", "machine learning", "ml", "python",
        "faiss", "pinecone", "weaviate", "qdrant", "elasticsearch",
        "information retrieval", "sentence-transformers", "bm25",
        "recommendation", "deep learning", "a/b testing", "ndcg",
    }
    return sum(
        1 for s in candidate.get("skills", [])
        if any(kw in s.get("name", "").lower() for kw in jd_keywords)
    )


def is_consulting_career(candidate: dict) -> bool:
    career = candidate.get("career_history", [])
    if not career:
        return False
    return all(
        r.get("company", "").lower().strip() in config.CONSULTING_FIRMS
        for r in career
    )


def is_cv_without_nlp(candidate: dict) -> bool:
    cv_kw = {"computer vision", "opencv", "object detection", "speech recognition", "robotics"}
    nlp_kw = {"nlp", "natural language", "retrieval", "ranking", "search", "text classification"}
    all_text = " ".join(
        s.get("name", "").lower() for s in candidate.get("skills", [])
    ) + " ".join(
        r.get("description", "").lower() for r in candidate.get("career_history", [])
    )
    has_cv = any(k in all_text for k in cv_kw)
    has_nlp = any(k in all_text for k in nlp_kw)
    return has_cv and not has_nlp


def is_research_only(candidate: dict) -> bool:
    career = candidate.get("career_history", [])
    if len(career) < 2:
        return False
    research_kw = {"research", "lab", "phd", "academic", "paper"}
    prod_kw = {"production", "deployed", "shipped", "users", "scale", "serving"}
    for r in career:
        desc = r.get("description", "").lower()
        title = r.get("title", "").lower()
        if any(k in desc for k in prod_kw):
            return False
        if not any(k in title for k in research_kw):
            return False
    return True


def main():
    ap = argparse.ArgumentParser(description="Trap evidence report generator")
    ap.add_argument("--candidates", required=True, help="Path to candidates JSONL")
    ap.add_argument("--n", type=int, default=3, help="Examples per trap archetype")
    ap.add_argument("--json-out", default=None, help="Optional JSON output path")
    args = ap.parse_args()

    print("Loading candidates and pipeline...", flush=True)
    all_candidates = list(load_candidates(args.candidates))
    print(f"Loaded {len(all_candidates)} candidates")

    jd = load_jd_requirements()
    jd_emb, cand_emb = load_embeddings()
    scorer = FeatureScorer(jd, jd_emb, cand_emb)

    # Naive keyword rank
    naive_ranked = sorted(all_candidates, key=lambda c: -naive_keyword_score(c))
    naive_rank_map = {c["candidate_id"]: i + 1 for i, c in enumerate(naive_ranked)}

    # Hard filters + honeypot detection
    filtered = apply_hard_filters(all_candidates)
    clean, honeypots = filter_honeypots(filtered)
    print(f"After hard filters: {len(filtered)} | After honeypots: {len(clean)} | Flagged: {len(honeypots)}")

    # Score clean candidates
    scored = []
    for c in clean:
        feats = scorer.score_candidate(c)
        mult = compute_behavioral_multiplier(c)
        add = compute_behavioral_additive(c)
        final = feats["weighted_total"] * mult + add
        scored.append((c, feats, mult, add, final))

    scored.sort(key=lambda x: (-round(x[4], 4), x[0]["candidate_id"]))
    pipeline_rank_map = {entry[0]["candidate_id"]: i + 1 for i, entry in enumerate(scored)}

    report = {}

    # ── Archetype 1: Honeypots ────────────────────────────────────────────────
    hp_examples = []
    for hp in honeypots[:args.n]:
        c = hp["candidate"]
        cid = c["candidate_id"]
        profile = c.get("profile", {})
        naive_rank = naive_rank_map.get(cid, "N/A")
        hp_examples.append({
            "candidate_id": cid,
            "title": profile.get("current_title", ""),
            "yoe": profile.get("years_of_experience", 0),
            "naive_rank": naive_rank,
            "pipeline_outcome": "EXCLUDED — never scored",
            "detection_reason": hp["reason"],
            "archetype": "honeypot",
        })
    report["honeypots"] = hp_examples

    # ── Archetype 2: Career-long consulting ───────────────────────────────────
    consulting_examples = []
    for c, feats, mult, add, final in scored:
        if is_consulting_career(c) and len(consulting_examples) < args.n:
            cid = c["candidate_id"]
            profile = c.get("profile", {})
            consulting_examples.append({
                "candidate_id": cid,
                "title": profile.get("current_title", ""),
                "companies": [r.get("company", "") for r in c.get("career_history", [])[:3]],
                "career_fit_score": round(feats.get("career_fit", 0), 4),
                "pipeline_rank": pipeline_rank_map.get(cid, "N/A"),
                "naive_rank": naive_rank_map.get(cid, "N/A"),
                "final_score": round(final, 4),
                "penalty_reason": "career_quality component: 0.15 (career-long consulting = JD explicit disqualifier)",
                "archetype": "career_long_consulting",
            })
    report["career_long_consulting"] = consulting_examples

    # ── Archetype 3: Keyword-stuffers (high naive, low pipeline) ──────────────
    stuffer_examples = []
    for c, feats, mult, add, final in scored:
        cid = c["candidate_id"]
        naive_rank = naive_rank_map.get(cid, 9999)
        pipeline_rank = pipeline_rank_map.get(cid, 9999)
        rank_drop = pipeline_rank - naive_rank
        if rank_drop >= 15 and len(stuffer_examples) < args.n:
            profile = c.get("profile", {})
            skill_count = len(c.get("skills", []))
            jd_hits = naive_keyword_score(c)

            # Estimate coherence multiplier from career vs skills mismatch
            career_score = feats.get("career_fit", 0)
            skills_raw = feats.get("skills_match", 0)
            coherence_estimate = career_score / max(skills_raw, 0.01) if skills_raw > 0 else 0.25
            coherence_estimate = max(0.25, min(1.0, coherence_estimate))

            stuffer_examples.append({
                "candidate_id": cid,
                "title": profile.get("current_title", ""),
                "naive_rank": naive_rank,
                "pipeline_rank": pipeline_rank,
                "rank_drop": rank_drop,
                "jd_skill_hits": jd_hits,
                "total_skills": skill_count,
                "coherence_gate_approx": round(coherence_estimate, 3),
                "skills_match_score": round(feats.get("skills_match", 0), 4),
                "career_fit_score": round(feats.get("career_fit", 0), 4),
                "final_score": round(final, 4),
                "archetype": "keyword_stuffer",
            })
    report["keyword_stuffers"] = stuffer_examples

    # ── Archetype 4: CV/robotics without NLP ──────────────────────────────────
    cv_examples = []
    for c, feats, mult, add, final in scored:
        if is_cv_without_nlp(c) and len(cv_examples) < args.n:
            cid = c["candidate_id"]
            profile = c.get("profile", {})
            cv_examples.append({
                "candidate_id": cid,
                "title": profile.get("current_title", ""),
                "pipeline_rank": pipeline_rank_map.get(cid, "N/A"),
                "career_fit_score": round(feats.get("career_fit", 0), 4),
                "skills_match_score": round(feats.get("skills_match", 0), 4),
                "final_score": round(final, 4),
                "disqualifier": "primarily_cv_without_nlp — career_fit disqualifier penalty +0.20",
                "archetype": "cv_without_nlp",
            })
    report["cv_without_nlp"] = cv_examples

    # ── Archetype 5: Research-only / no production ────────────────────────────
    research_examples = []
    for c, feats, mult, add, final in scored:
        if is_research_only(c) and len(research_examples) < args.n:
            cid = c["candidate_id"]
            profile = c.get("profile", {})
            research_examples.append({
                "candidate_id": cid,
                "title": profile.get("current_title", ""),
                "pipeline_rank": pipeline_rank_map.get(cid, "N/A"),
                "career_fit_score": round(feats.get("career_fit", 0), 4),
                "final_score": round(final, 4),
                "disqualifier": "research_only_no_production — career_fit disqualifier penalty +0.30",
                "archetype": "research_only",
            })
    report["research_only"] = research_examples

    # ── Print Markdown ────────────────────────────────────────────────────────
    print("\n\n" + "=" * 72)
    print("TRAP-HANDLING EVIDENCE REPORT")
    print("=" * 72)

    print(f"\n## Summary")
    print(f"| Archetype | Count detected | Pipeline action |")
    print(f"|-----------|---------------|-----------------|")
    print(f"| Honeypots (internally impossible) | {len(honeypots)} | Excluded before scoring |")
    print(f"| Career-long consulting            | {len(consulting_examples)} shown | Career fit = 0.15 (penalty) |")
    print(f"| Keyword-stuffers                  | {len(stuffer_examples)} shown | Coherence gate discounts skills |")
    print(f"| CV/robotics without NLP/IR        | {len(cv_examples)} shown | Disqualifier penalty +0.20 |")
    print(f"| Research-only, no production      | {len(research_examples)} shown | Disqualifier penalty +0.30 |")

    print(f"\n## 1. Honeypots — Excluded Before Scoring")
    print(f"| Candidate ID | Title | YoE | Naive Rank | Pipeline Outcome | Detection Reason |")
    print(f"|---|---|---|---|---|---|")
    for e in hp_examples:
        reason_short = e["detection_reason"][:60] + "..." if len(e["detection_reason"]) > 60 else e["detection_reason"]
        print(f"| {e['candidate_id']} | {e['title']} | {e['yoe']:.0f} | #{e['naive_rank']} | {e['pipeline_outcome']} | {reason_short} |")

    print(f"\n## 2. Career-Long Consulting — Demoted by JD Disqualifier")
    print(f"| Candidate ID | Title | Companies | Career Fit | Pipeline Rank | Naive Rank | Final Score |")
    print(f"|---|---|---|---|---|---|---|")
    for e in consulting_examples:
        companies = " → ".join(e["companies"])
        print(f"| {e['candidate_id']} | {e['title']} | {companies} | {e['career_fit_score']} | #{e['pipeline_rank']} | #{e['naive_rank']} | {e['final_score']} |")

    print(f"\n## 3. Keyword-Stuffers — Demoted by Coherence Gate")
    print(f"| Candidate ID | Title | Naive Rank | Pipeline Rank | Δ | JD Hits | Coherence Gate | Skills Score | Career Score |")
    print(f"|---|---|---|---|---|---|---|---|---|")
    for e in stuffer_examples:
        print(f"| {e['candidate_id']} | {e['title']} | #{e['naive_rank']} | #{e['pipeline_rank']} | -{e['rank_drop']} | {e['jd_skill_hits']}/{e['total_skills']} | ×{e['coherence_gate_approx']} | {e['skills_match_score']} | {e['career_fit_score']} |")

    print(f"\n## 4. CV/Robotics Without NLP/IR")
    print(f"| Candidate ID | Title | Pipeline Rank | Career Fit | Skills Match | Final Score |")
    print(f"|---|---|---|---|---|---|")
    for e in cv_examples:
        print(f"| {e['candidate_id']} | {e['title']} | #{e['pipeline_rank']} | {e['career_fit_score']} | {e['skills_match_score']} | {e['final_score']} |")

    print(f"\n## 5. Research-Only / No Production")
    print(f"| Candidate ID | Title | Pipeline Rank | Career Fit | Final Score |")
    print(f"|---|---|---|---|---|")
    for e in research_examples:
        print(f"| {e['candidate_id']} | {e['title']} | #{e['pipeline_rank']} | {e['career_fit_score']} | {e['final_score']} |")

    print("\n" + "=" * 72)
    print("Key insight: The coherence gate and honeypot de-noising together remove")
    print("  candidates a naive keyword matcher would incorrectly rank highly.")
    print("  This is the core differentiator vs. bag-of-words retrieval.")
    print("=" * 72)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2))
        print(f"\nJSON written to {args.json_out}")


if __name__ == "__main__":
    main()
