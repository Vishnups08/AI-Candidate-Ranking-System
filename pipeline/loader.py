"""
Streaming JSONL loader for candidate data.
Memory-efficient: processes one candidate at a time.
"""

import json
import gzip
from pathlib import Path
from typing import Generator


def load_candidates(filepath: str) -> Generator[dict, None, None]:
    """
    Yield candidate dicts one at a time from a JSONL or gzipped JSONL file.
    Handles both .jsonl and .jsonl.gz transparently.
    """
    path = Path(filepath)

    if path.suffix == ".gz":
        opener = lambda: gzip.open(path, "rt", encoding="utf-8")
    elif path.suffix == ".jsonl":
        opener = lambda: open(path, "r", encoding="utf-8")
    elif path.suffix == ".json":
        # Handle plain JSON array (like sample_candidates.json)
        with open(path, "r", encoding="utf-8") as f:
            candidates = json.load(f)
        for candidate in candidates:
            yield candidate
        return
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Expected .jsonl, .jsonl.gz, or .json")

    with opener() as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed JSON at line {line_num}: {e}")
                continue


def load_all_candidates(filepath: str) -> list[dict]:
    """Load all candidates into memory at once. Use for smaller files."""
    return list(load_candidates(filepath))


def count_candidates(filepath: str) -> int:
    """Count total candidates without loading them all into memory."""
    count = 0
    for _ in load_candidates(filepath):
        count += 1
    return count


def load_candidate_ids(filepath: str) -> set[str]:
    """Load just candidate IDs for validation purposes."""
    ids = set()
    for candidate in load_candidates(filepath):
        ids.add(candidate["candidate_id"])
    return ids
