#!/usr/bin/env python3
"""
Redrob AI Candidate Ranking System
Main entry point for producing the submission CSV.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""

import argparse
import sys
import time
from pathlib import Path
import os
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import config
from pipeline.loader import load_candidates, load_all_candidates
from pipeline.jd_parser import load_jd_requirements
from pipeline.hard_filters import apply_hard_filters
from pipeline.honeypot_detector import filter_honeypots
from pipeline.ranker import CandidateRanker


def load_embeddings(embeddings_dir: str) -> tuple:
    """Load pre-computed embeddings if available."""
    embeddings_path = Path(embeddings_dir)

    jd_embedding = None
    candidate_embeddings = {}

    # Load JD embedding
    jd_emb_path = embeddings_path / "jd_embedding.npy"
    if jd_emb_path.exists():
        jd_embedding = np.load(str(jd_emb_path))
        print(f"  Loaded JD embedding from {jd_emb_path}")

    # Load candidate profile embeddings
    profile_emb_path = embeddings_path / "candidate_profiles.npz"
    if profile_emb_path.exists():
        data = np.load(str(profile_emb_path), allow_pickle=True)
        candidate_ids = data["candidate_ids"]
        embeddings = data["embeddings"]
        for cid, emb in zip(candidate_ids, embeddings):
            if cid not in candidate_embeddings:
                candidate_embeddings[str(cid)] = {}
            candidate_embeddings[str(cid)]["profile_embedding"] = emb
        print(f"  Loaded {len(candidate_ids)} profile embeddings")

    # Load candidate role embeddings
    role_emb_path = embeddings_path / "candidate_roles.npz"
    if role_emb_path.exists():
        data = np.load(str(role_emb_path), allow_pickle=True)
        candidate_ids = data["candidate_ids"]
        role_emb_lists = data["role_embeddings"]
        for cid, role_embs in zip(candidate_ids, role_emb_lists):
            cid_str = str(cid)
            if cid_str not in candidate_embeddings:
                candidate_embeddings[cid_str] = {}
            candidate_embeddings[cid_str]["role_embeddings"] = role_embs
        print(f"  Loaded role embeddings for {len(candidate_ids)} candidates")

    return jd_embedding, candidate_embeddings


def main():
    parser = argparse.ArgumentParser(
        description="Redrob AI Candidate Ranking System"
    )
    parser.add_argument(
        "--candidates", required=True,
        help="Path to candidates JSONL/JSON file"
    )
    parser.add_argument(
        "--out", required=True,
        help="Path for output CSV file"
    )
    parser.add_argument(
        "--embeddings-dir",
        default="precompute/embeddings",
        help="Directory containing pre-computed embeddings"
    )
    parser.add_argument(
        "--skip-embeddings", action="store_true",
        help="Skip loading embeddings (use keyword fallback for semantic scoring)"
    )
    args = parser.parse_args()

    start_time = time.time()
    print("=" * 60)
    print("Redrob AI Candidate Ranking System")
    print("=" * 60)

    # Step 1: Load JD requirements
    print("\n[1/6] Loading JD requirements...")
    jd = load_jd_requirements()
    print(f"  JD: {jd.title} at {jd.company}")

    # Step 2: Load embeddings
    jd_embedding = None
    candidate_embeddings = {}

    if not args.skip_embeddings:
        print("\n[2/6] Loading pre-computed embeddings...")
        emb_dir = Path(args.embeddings_dir)
        if emb_dir.exists():
            jd_embedding, candidate_embeddings = load_embeddings(str(emb_dir))
        else:
            print(f"  No embeddings directory found at {emb_dir}")
            print("  Using keyword-based fallback for semantic scoring")
            print("  (Run precompute/build_embeddings.py first for better results)")
    else:
        print("\n[2/6] Skipping embeddings (using keyword fallback)")

    # Step 3: Load candidates
    print(f"\n[3/6] Loading candidates from {args.candidates}...")
    candidates = list(load_candidates(args.candidates))
    print(f"  Loaded {len(candidates)} candidates")

    # Step 4: Apply hard filters
    print("\n[4/6] Applying hard filters...")
    filtered = apply_hard_filters(candidates)

    # Step 5: Honeypot detection
    print("\n[5/6] Running honeypot detection...")
    clean, honeypots = filter_honeypots(filtered)

    if honeypots:
        print(f"  Sample honeypot reasons:")
        for hp in honeypots[:3]:
            cid = hp["candidate"]["candidate_id"]
            print(f"    {cid}: {hp['reason'][:100]}...")

    # Step 6: Rank candidates
    print(f"\n[6/6] Ranking {len(clean)} candidates...")
    ranker = CandidateRanker(jd, jd_embedding, candidate_embeddings)
    results = ranker.rank_candidates(clean)

    # Validate
    is_full_run = len(clean) >= config.TOP_K
    print("\nValidating results...")
    errors = CandidateRanker.validate_results(results, strict=is_full_run)
    if errors:
        print(f"  VALIDATION ERRORS:")
        for e in errors:
            print(f"    - {e}")
        sys.exit(1)
    else:
        print("  All validations passed [OK]")

    # Write output
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    CandidateRanker.write_csv(results, str(output_path))

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"COMPLETE in {elapsed:.1f}s")
    print(f"  Total candidates: {len(candidates)}")
    print(f"  After hard filters: {len(filtered)}")
    print(f"  After honeypot removal: {len(clean)}")
    print(f"  Honeypots detected: {len(honeypots)}")
    print(f"  Output: {output_path} ({config.TOP_K} ranked candidates)")
    print("=" * 60)

    # Print top 5 for inspection
    print("\nTop 5 candidates:")
    for r in results[:5]:
        print(f"  Rank {r['rank']}: {r['candidate_id']} "
              f"(score={r['score']:.4f}) — {r['reasoning'][:80]}...")


if __name__ == "__main__":
    main()
