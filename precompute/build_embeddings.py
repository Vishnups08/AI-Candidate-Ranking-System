#!/usr/bin/env python3
"""
Pre-compute candidate embeddings using bge-small-en-v1.5.
This is an OFFLINE step that runs once (~15-20 min for 100K candidates).
The ranking step (rank.py) loads pre-computed embeddings and runs in <3 min.
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.loader import load_candidates

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: sentence-transformers not installed.")
    print("Run: pip install sentence-transformers")
    sys.exit(1)

import config


def build_profile_text(candidate: dict) -> str:
    """Build the text to embed for a candidate's profile.
    Includes title, company, industry, headline, summary, skills, and career snippets.
    """
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])[:3]

    parts = [
        # Core identity
        f"{profile.get('current_title', '')} at {profile.get('current_company', '')}",
        f"Industry: {profile.get('current_industry', '')}",
        # Profile text
        profile.get("headline", ""),
        profile.get("summary", ""),
        # Skills
        "Skills: " + ", ".join(s.get("name", "") for s in skills[:15]),
        # Career snippets (first 80 chars of each recent role description)
    ]
    for role in career:
        desc = role.get("description", "")[:80]
        if desc:
            parts.append(f"{role.get('title', '')}: {desc}")

    return " ".join(p for p in parts if p).strip()


def build_role_texts(candidate: dict) -> list[str]:
    """Build text for each career role description, limited to top 3 most recent roles."""
    career = candidate.get("career_history", [])[:3]
    texts = []
    for role in career:
        desc = role.get("description", "").strip()
        title = role.get("title", "").strip()
        if desc:
            texts.append(f"{title}: {desc[:config.EMBEDDING_MAX_TEXT_LENGTH]}")
        elif title:
            texts.append(title)
    return texts


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute candidate embeddings"
    )
    parser.add_argument(
        "--candidates", required=True,
        help="Path to candidates JSONL/JSON file"
    )
    parser.add_argument(
        "--output-dir", default="precompute/embeddings",
        help="Output directory for embeddings"
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Batch size for encoding"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Building Candidate Embeddings")
    print(f"Model: {config.EMBEDDING_MODEL}")
    print("=" * 60)

    # Load model
    print("\n[1/4] Loading embedding model...")
    start = time.time()
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    print(f"  Model loaded in {time.time() - start:.1f}s")

    # Load candidates
    print(f"\n[2/4] Loading candidates from {args.candidates}...")
    candidates = list(load_candidates(args.candidates))
    print(f"  Loaded {len(candidates)} candidates")

    # Apply hard filters and honeypot detection first to keep only clean candidates
    print("  Applying filters and honeypot detection...")
    from pipeline.hard_filters import apply_hard_filters
    from pipeline.honeypot_detector import filter_honeypots
    filtered = apply_hard_filters(candidates)
    clean, honeypots = filter_honeypots(filtered)
    print(f"  Only encoding {len(clean)} clean candidates after filtering.")

    # Build profile embeddings
    print(f"\n[3/4] Building profile embeddings (batch_size={args.batch_size})...")
    start = time.time()

    candidate_ids = []
    profile_texts = []

    for candidate in clean:
        cid = candidate.get("candidate_id", "")
        text = build_profile_text(candidate)[:config.EMBEDDING_MAX_TEXT_LENGTH]
        # Add BGE instruction prefix for passage encoding
        text = config.EMBEDDING_PASSAGE_PREFIX + text
        candidate_ids.append(cid)
        profile_texts.append(text)

    # Encode in batches
    profile_embeddings = model.encode(
        profile_texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    elapsed = time.time() - start
    print(f"  Profile embeddings: {profile_embeddings.shape} in {elapsed:.1f}s")

    # Save profile embeddings
    profile_path = output_dir / "candidate_profiles.npz"
    np.savez_compressed(
        str(profile_path),
        candidate_ids=np.array(candidate_ids),
        embeddings=profile_embeddings,
    )
    print(f"  Saved to {profile_path} ({profile_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Build role embeddings
    print(f"\n[4/4] Building role embeddings...")
    start = time.time()

    role_candidate_ids = []
    role_counts = []
    role_texts_flat = []

    for candidate in clean:
        cid = candidate.get("candidate_id", "")
        role_candidate_ids.append(cid)
        texts = build_role_texts(candidate)
        role_counts.append(len(texts))
        role_texts_flat.extend(texts)

    print(f"  Encoding {len(role_texts_flat)} role descriptions in batches...")
    if role_texts_flat:
        role_embeddings_flat = model.encode(
            role_texts_flat,
            batch_size=args.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
    else:
        role_embeddings_flat = np.array([])

    role_embedding_lists = []
    curr_idx = 0
    for count in role_counts:
        if count > 0:
            role_embs = role_embeddings_flat[curr_idx : curr_idx + count]
            curr_idx += count
        else:
            role_embs = np.empty((0, config.EMBEDDING_DIM))
        role_embedding_lists.append(role_embs)

    elapsed = time.time() - start
    print(f"  Role embeddings built in {elapsed:.1f}s")

    # Save role embeddings (using object arrays since variable-length)
    role_path = output_dir / "candidate_roles.npz"
    np.savez_compressed(
        str(role_path),
        candidate_ids=np.array(role_candidate_ids),
        role_embeddings=np.array(role_embedding_lists, dtype=object),
    )
    print(f"  Saved to {role_path} ({role_path.stat().st_size / 1024 / 1024:.1f} MB)")

    print("\n" + "=" * 60)
    print("EMBEDDING PRE-COMPUTATION COMPLETE")
    print(f"  Profile embeddings: {profile_path}")
    print(f"  Role embeddings: {role_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
