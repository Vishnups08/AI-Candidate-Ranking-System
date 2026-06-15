#!/usr/bin/env python3
"""
Local Self-Evaluation Framework.
Calculates NDCG@10, NDCG@50, MAP, and P@10 of a submission against manual labels.

Usage:
    python evaluation/self_evaluate.py --submission output/submission.csv --labels evaluation/manual_labels.json
"""

import argparse
import json
import math
import sys
import pandas as pd


def dcg_at_k(r, k):
    """Score is discounted cumulative gain at k."""
    r = r[:k]
    if not r:
        return 0.0
    return sum((2 ** rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(r))


def ndcg_at_k(r, k):
    """Score is normalized discounted cumulative gain at k."""
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.0
    return dcg_at_k(r, k) / dcg_max


def average_precision(r):
    """Score is average precision (area under PR curve)."""
    if not r:
        return 0.0
    
    # Calculate precision at each relevant rank
    num_relevant = 0
    sum_precs = 0.0
    for idx, rel in enumerate(r):
        # A candidate is considered relevant if tier is >= 3
        if rel >= 3:
            num_relevant += 1
            sum_precs += num_relevant / (idx + 1)
            
    if num_relevant == 0:
        return 0.0
    return sum_precs / num_relevant


def precision_at_k(r, k, threshold=3):
    """Calculate precision at K based on relevance threshold."""
    r = r[:k]
    if not r:
        return 0.0
    num_relevant = sum(1 for rel in r if rel >= threshold)
    return num_relevant / k


def main():
    parser = argparse.ArgumentParser(description="Evaluate Candidate Ranking Submission")
    parser.add_argument("--submission", required=True, help="Path to submission CSV")
    parser.add_argument("--labels", required=True, help="Path to manual labels JSON")
    args = parser.parse_args()

    # Load labels
    try:
        with open(args.labels, "r", encoding="utf-8") as f:
            labels = json.load(f)
        print(f"Loaded {len(labels)} manual candidate labels.")
    except Exception as e:
        print(f"Error loading labels: {e}")
        sys.exit(1)

    # Load submission CSV
    try:
        sub_df = pd.read_csv(args.submission)
        print(f"Loaded submission with {len(sub_df)} ranked candidates.")
    except Exception as e:
        print(f"Error loading submission: {e}")
        sys.exit(1)

    # Extract ranks
    ranked_ids = sub_df["candidate_id"].tolist()
    
    # Map ranked candidate IDs to relevance scores
    # Unlabeled candidates default to a baseline tier of 1 (unless they are flagged as non-tech/honeypots)
    relevance_scores = []
    labeled_count = 0
    honeypot_count = 0
    
    for cid in ranked_ids:
        if cid in labels:
            score = labels[cid]
            labeled_count += 1
            if score == 0:
                honeypot_count += 1
        else:
            # Default to 1 (irrelevant but clean)
            score = 1
        relevance_scores.append(score)

    # Calculate metrics
    ndcg_10 = ndcg_at_k(relevance_scores, 10)
    ndcg_50 = ndcg_at_k(relevance_scores, 50)
    map_score = average_precision(relevance_scores)
    p_10 = precision_at_k(relevance_scores, 10)
    
    composite = 0.50 * ndcg_10 + 0.30 * ndcg_50 + 0.15 * map_score + 0.05 * p_10

    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  NDCG@10:  {ndcg_10:.4f}")
    print(f"  NDCG@50:  {ndcg_50:.4f}")
    print(f"  MAP:      {map_score:.4f}")
    print(f"  P@10:     {p_10:.4f}")
    print("-" * 60)
    print(f"  COMPOSITE SCORE: {composite:.4f}")
    print("-" * 60)
    print(f"  Labeled candidates ranked: {labeled_count}/{len(ranked_ids)}")
    print(f"  Honeypots in Top 100:     {honeypot_count} (Honeypot rate: {honeypot_count/len(ranked_ids)*100:.1f}%)")
    
    if honeypot_count > 10:
        print("\n  [WARNING] DISQUALIFIED: Honeypot rate > 10% in Top 100!")
    elif honeypot_count > 0:
        print("\n  [WARNING] Honeypots detected in Top 100 list. Try to eliminate them.")
    else:
        print("\n  [SUCCESS] 0% honeypot rate in Top 100!")
    print("=" * 60)


if __name__ == "__main__":
    main()
