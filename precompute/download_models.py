#!/usr/bin/env python3
"""
One-time OFFLINE-prep step: download the two models the pipeline uses into a
repo-local ./models directory.

Why this exists
---------------
At Stage 3 the organizers reproduce the ranking step inside a sandbox with the
network OFF. If model weights are not present locally, sentence-transformers
would try to reach the Hugging Face Hub, fail, and (previously) the pipeline
silently fell back to a worse path -> the reproduced CSV would not match the
submitted one. Pre-downloading here, and loading with local_files_only=True in
rank.py, makes the ranking step provably network-free and reproducible.

Run once, with network ON, before ranking:
    python precompute/download_models.py

This downloads:
    - BAAI/bge-base-en-v1.5                (bi-encoder, embeddings)
    - cross-encoder/ms-marco-MiniLM-L-6-v2 (cross-encoder, re-rank)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)


MODELS = [config.EMBEDDING_MODEL, config.CROSS_ENCODER_MODEL]


def main():
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print(f"Downloading models into {config.MODELS_DIR}")
    print("=" * 60)

    for repo_id in MODELS:
        dest = config.MODELS_DIR / repo_id.replace("/", "__")
        print(f"\n[{repo_id}] -> {dest}")
        if dest.exists() and any(dest.iterdir()):
            print("  Already present, skipping.")
            continue
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(dest),
            # Skip redundant/large files we never load on CPU inference.
            ignore_patterns=["*.onnx", "*.safetensors.index.json", "openvino/*", "*.h5"],
        )
        print("  Done.")

    print("\n" + "=" * 60)
    print("All models cached locally. Ranking step can now run network-off.")
    print("=" * 60)


if __name__ == "__main__":
    main()
