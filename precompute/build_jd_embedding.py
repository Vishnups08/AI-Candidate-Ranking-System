#!/usr/bin/env python3
"""
Pre-compute JD embedding using bge-small-en-v1.5.
Quick run (~5 seconds). Produces a single .npy file.
"""

import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Error: sentence-transformers not installed.")
    print("Run: pip install sentence-transformers")
    sys.exit(1)

import config
from pipeline.jd_parser import load_jd_requirements, get_jd_embedding_text


def main():
    output_dir = Path("precompute/embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building JD embedding...")

    # Load model (prefer the repo-local snapshot for offline reproducibility)
    _model_path = config._local_model_path(config.EMBEDDING_MODEL)
    model = SentenceTransformer(_model_path, device="cpu")

    # Get JD text
    jd = load_jd_requirements()
    jd_text = get_jd_embedding_text(jd)
    print(f"  JD text ({len(jd_text)} chars): {jd_text[:100]}...")

    # Encode with BGE instruction prefix for query-side
    jd_text_with_prefix = config.EMBEDDING_QUERY_PREFIX + jd_text
    start = time.time()
    jd_embedding = model.encode(jd_text_with_prefix, normalize_embeddings=True)
    elapsed = time.time() - start

    # Save
    output_path = output_dir / "jd_embedding.npy"
    np.save(str(output_path), jd_embedding)

    print(f"  JD embedding shape: {jd_embedding.shape}")
    print(f"  Saved to {output_path} in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
