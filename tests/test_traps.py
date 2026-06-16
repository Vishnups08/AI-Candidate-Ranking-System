"""
Trap-archetype regression tests.

These encode the JD's "read between the lines" intent as assertions on real
candidates from the pool, identified by reading their full profiles
(see evaluation/gold_sample.txt). They guard against regressions where a tuning
change re-breaks a trap, and they double as defensible interview artifacts.

Run:  python -m pytest tests/test_traps.py -v
      (or: python tests/test_traps.py  for a no-pytest summary)

Requires the candidates file. Set REDROB_CANDIDATES or rely on the default
relative path to the hackathon bundle.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pytest
except ImportError:  # allow the __main__ summary runner without pytest installed
    class _PytestStub:
        """Minimal no-op stand-in so decorators import cleanly without pytest."""
        def fixture(self, *a, **k):
            def deco(fn): return fn
            return deco

        def skip(self, msg):
            raise RuntimeError(msg)

        class mark:
            @staticmethod
            def parametrize(*a, **k):
                def deco(fn): return fn
                return deco

    pytest = _PytestStub()

from pipeline.loader import load_candidates
from pipeline.honeypot_detector import detect_honeypot
from pipeline.hard_filters import passes_hard_filters
from pipeline.feature_scorer import FeatureScorer
from pipeline.behavioral_scorer import compute_behavioral_multiplier, compute_behavioral_additive
from pipeline.jd_parser import load_jd_requirements

DEFAULT_CANDIDATES = (Path(__file__).parent.parent.parent /
                      "[PUB] India_runs_data_and_ai_challenge" /
                      "India_runs_data_and_ai_challenge" / "candidates.jsonl")

# Archetype exemplars identified by manual profile reading (gold_sample.txt).
GENUINE_FITS = ["CAND_0002025", "CAND_0000031"]          # tier 4: real retrieval/recsys
STRONG_FITS = ["CAND_0000273", "CAND_0001707", "CAND_0001131"]  # tier 3: product ML
KEYWORD_STUFFERS = ["CAND_0000074", "CAND_0000722", "CAND_0000821", "CAND_0002356"]
# Genuine impossibility honeypots: advanced/expert skills with 0 months used,
# and they PASS hard filters (tech titles) so they could reach scoring.
HONEYPOTS = ["CAND_0003582", "CAND_0016000", "CAND_0033972", "CAND_0055792"]
CONSULTING_JUNIORS = ["CAND_0000047", "CAND_0000098", "CAND_0000056"]

NEEDED = set(GENUINE_FITS + STRONG_FITS + KEYWORD_STUFFERS + HONEYPOTS + CONSULTING_JUNIORS)


def _candidates_path():
    p = os.environ.get("REDROB_CANDIDATES", str(DEFAULT_CANDIDATES))
    if not Path(p).exists():
        pytest.skip(f"candidates file not found at {p}")
    return p


@pytest.fixture(scope="module")
def records():
    path = _candidates_path()
    recs = {}
    for c in load_candidates(path):
        if c["candidate_id"] in NEEDED:
            recs[c["candidate_id"]] = c
        if len(recs) == len(NEEDED):
            break
    return recs


@pytest.fixture(scope="module")
def scorer():
    return FeatureScorer(load_jd_requirements())


def _final_score(scorer, c):
    feats = scorer.score_candidate(c)
    return feats["weighted_total"] * compute_behavioral_multiplier(c) + compute_behavioral_additive(c)


# --- Trap 1: keyword-stuffers score below every genuine strong fit ---
# (Relative, not an arbitrary constant: the behavioral multiplier can inflate
# absolute scores, but a padded non-tech profile must never reach a real fit.)
@pytest.mark.parametrize("cid", KEYWORD_STUFFERS)
def test_keyword_stuffer_below_strong_fits(records, scorer, cid):
    stuffer = _final_score(scorer, records[cid])
    min_strong = min(_final_score(scorer, records[c]) for c in STRONG_FITS if c in records)
    assert stuffer < min_strong, (
        f"{cid} keyword-stuffer ({stuffer:.3f}) reached a strong fit "
        f"({min_strong:.3f})")


# --- Trap 2: genuine product-ML fits outscore every keyword-stuffer ---
def test_genuine_fits_beat_stuffers(records, scorer):
    fit_scores = [_final_score(scorer, records[c]) for c in GENUINE_FITS if c in records]
    stuffer_scores = [_final_score(scorer, records[c]) for c in KEYWORD_STUFFERS if c in records]
    assert min(fit_scores) > max(stuffer_scores), (
        f"a genuine fit ({min(fit_scores):.3f}) did not beat the best "
        f"keyword-stuffer ({max(stuffer_scores):.3f})")


# --- Trap 3: honeypots are detected and excluded ---
@pytest.mark.parametrize("cid", HONEYPOTS)
def test_honeypot_detected_or_scored_low(records, scorer, cid):
    c = records[cid]
    is_hp, _ = detect_honeypot(c)
    # Either flagged as honeypot, or (if subtle) scored clearly out of contention.
    if not is_hp:
        assert _final_score(scorer, c) < 0.35, f"{cid} neither flagged nor low-scored"


# --- Trap 4: career-long consulting juniors are penalized below strong fits ---
def test_consulting_juniors_below_strong_fits(records, scorer):
    strong = [_final_score(scorer, records[c]) for c in STRONG_FITS if c in records]
    consult = [_final_score(scorer, records[c]) for c in CONSULTING_JUNIORS if c in records]
    assert max(consult) < min(strong), (
        f"a consulting junior ({max(consult):.3f}) outscored a strong fit "
        f"({min(strong):.3f})")


# --- Trap 5: hard filters keep genuine fits, drop obvious non-tech stuffers ---
def test_hard_filters_keep_fits(records):
    for cid in GENUINE_FITS + STRONG_FITS:
        if cid in records:
            assert passes_hard_filters(records[cid]), f"{cid} wrongly hard-filtered"


if __name__ == "__main__":
    # Lightweight no-pytest runner for a quick summary.
    path = _candidates_path() if "REDROB_CANDIDATES" in os.environ or DEFAULT_CANDIDATES.exists() else None
    if not path:
        print("candidates file not found; set REDROB_CANDIDATES")
        sys.exit(1)
    recs = {}
    for c in load_candidates(path):
        if c["candidate_id"] in NEEDED:
            recs[c["candidate_id"]] = c
        if len(recs) == len(NEEDED):
            break
    sc = FeatureScorer(load_jd_requirements())
    print(f"Loaded {len(recs)}/{len(NEEDED)} exemplars\n")
    for label, ids in [("GENUINE_FIT", GENUINE_FITS), ("STRONG_FIT", STRONG_FITS),
                       ("KEYWORD_STUFFER", KEYWORD_STUFFERS), ("HONEYPOT", HONEYPOTS),
                       ("CONSULTING_JR", CONSULTING_JUNIORS)]:
        for cid in ids:
            if cid in recs:
                c = recs[cid]
                hp = detect_honeypot(c)[0]
                print(f"  {label:16s} {cid}  score={_final_score(sc, c):.3f}  honeypot={hp}")
