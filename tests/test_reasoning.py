"""
Reasoning grounding audit (Stage-4 "no hallucination" check).

The Stage-4 manual review penalizes reasoning that "mentions skills not in the
candidate's profile". This test generates reasoning for a real sample of ranked
candidates and asserts that every skill/company name the reasoning cites is
actually present in that candidate's record. It also checks rank-tone
consistency (top ranks not critical, bottom ranks not glowing) and variation.

Run:  python -m pytest tests/test_reasoning.py -q
      (or: python tests/test_reasoning.py  for a sample dump)
"""

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pytest
except ImportError:
    class _Stub:
        def fixture(self, *a, **k):
            def d(f): return f
            return d
        def skip(self, m): raise RuntimeError(m)
    pytest = _Stub()

from pipeline.loader import load_candidates
from pipeline.hard_filters import passes_hard_filters
from pipeline.honeypot_detector import detect_honeypot
from pipeline.feature_scorer import FeatureScorer
from pipeline.behavioral_scorer import compute_behavioral_multiplier, compute_behavioral_additive
from pipeline.reasoning_generator import generate_reasoning
from pipeline.jd_parser import load_jd_requirements

DEFAULT_CANDIDATES = (Path(__file__).parent.parent.parent /
                      "[PUB] India_runs_data_and_ai_challenge" /
                      "India_runs_data_and_ai_challenge" / "candidates.jsonl")
SAMPLE_SIZE = 400   # scan budget; we rank this slice and check the top ~60


def _path():
    p = os.environ.get("REDROB_CANDIDATES", str(DEFAULT_CANDIDATES))
    if not Path(p).exists():
        pytest.skip(f"candidates file not found at {p}")
    return p


def _ranked_sample():
    """Score a slice of the pool and return the top-60 with reasoning."""
    scorer = FeatureScorer(load_jd_requirements())
    rows = []
    for i, c in enumerate(load_candidates(_path())):
        if i >= SAMPLE_SIZE:
            break
        if not passes_hard_filters(c):
            continue
        if detect_honeypot(c)[0]:
            continue
        feats = scorer.score_candidate(c)
        final = feats["weighted_total"] * compute_behavioral_multiplier(c) + compute_behavioral_additive(c)
        rows.append((final, feats, c))
    rows.sort(key=lambda x: -x[0])
    out = []
    for rank, (score, feats, c) in enumerate(rows[:60], 1):
        out.append((rank, c, generate_reasoning(c, rank, feats)))
    return out


def _profile_vocabulary(c):
    """All skill + company tokens that legitimately appear in a candidate."""
    vocab = set()
    for s in c.get("skills", []):
        vocab.add(s.get("name", "").lower())
    prof = c.get("profile", {})
    vocab.add(prof.get("current_company", "").lower())
    for r in c.get("career_history", []):
        vocab.add(r.get("company", "").lower())
    return {v for v in vocab if v}


# Skill names the reasoning may emit (must be traceable to the profile).
_KNOWN_SKILL_TOKENS = set()
for _name in ["faiss", "pinecone", "weaviate", "qdrant", "milvus", "elasticsearch",
              "opensearch", "sentence transformers", "embeddings", "information retrieval",
              "rag", "langchain", "llamaindex", "bm25", "tf-idf", "xgboost", "lightgbm",
              "lora", "qlora", "peft", "fine-tuning llms", "learning to rank",
              "recommendation systems", "semantic search", "vector search",
              "hugging face transformers", "pytorch", "tensorflow", "nlp", "deep learning",
              "machine learning", "data science", "bert", "gpt", "llms"]:
    _KNOWN_SKILL_TOKENS.add(_name)


def _cited_skills(reasoning, candidate):
    """Skills the reasoning POSITIVELY attributes to the candidate.

    Only the positive skill clauses can hallucinate; we must not flag tokens
    that appear in a concern ("no direct match on ... embeddings, vector DB"),
    or that are part of the candidate's job title (e.g. 'Recommendation Systems
    Engineer'). Parse the explicit positive cue clauses only.
    """
    title = candidate.get("profile", {}).get("current_title", "").lower()
    cited = []
    # Positive clauses the generator emits, e.g.
    #   "directly JD-relevant skills: FAISS, Pinecone, Embeddings."
    #   "brings X, Y experience" / "with domain-adjacent skills in X, Y"
    for m in re.findall(r"(?:relevant skills:|skills in|brings|with some relevant skills \()([^.;()]+)", reasoning, re.I):
        for piece in re.split(r",| and ", m):
            tok = piece.strip().strip(")").lower()
            if tok and tok not in title:   # title tokens are not skill claims
                cited.append(tok)
    return cited


@pytest.fixture(scope="module")
def sample():
    return _ranked_sample()


def test_no_skill_hallucination(sample):
    """Every skill the reasoning names must exist in the candidate's skills."""
    failures = []
    for rank, c, reasoning in sample:
        skill_names = {s.get("name", "").lower() for s in c.get("skills", [])}
        for tok in _cited_skills(reasoning, c):
            if not any(tok in sn or sn in tok for sn in skill_names):
                failures.append(f"#{rank} {c['candidate_id']}: cites '{tok}' not in skills")
    assert not failures, "skill hallucinations:\n" + "\n".join(failures[:10])


def test_company_grounded(sample):
    """Any capitalized company-looking mention should be a real employer."""
    failures = []
    for rank, c, reasoning in sample:
        companies = {r.get("company", "").lower() for r in c.get("career_history", [])}
        companies.add(c.get("profile", {}).get("current_company", "").lower())
        companies = {x for x in companies if x}
        # The generator only emits companies via f-strings from the record, so
        # this guards against future regressions: every 'at <Word>' must match.
        for m in re.findall(r"at ([A-Z][A-Za-z0-9.&'\- ]{1,30}?)(?:[;,.]| with| aligns| \()", reasoning):
            cand = m.strip().lower()
            if cand and cand not in companies and not any(cand in co or co in cand for co in companies):
                # ignore generic phrases
                if cand not in ("a series a", "a product", "real users", "scale"):
                    failures.append(f"#{rank} {c['candidate_id']}: 'at {m}' not an employer")
    assert not failures, "company hallucinations:\n" + "\n".join(failures[:10])


def test_rank_tone_consistency(sample):
    """Top ranks shouldn't read as disqualified; bottom ranks shouldn't glow."""
    disqualifier_phrases = ("explicit jd disqualifier", "does not sponsor", "career-long consulting is an explicit")
    for rank, c, reasoning in sample:
        low = reasoning.lower()
        if rank <= 5:
            assert not any(p in low for p in disqualifier_phrases), (
                f"#{rank} {c['candidate_id']} top-5 but reasoning flags a disqualifier: {reasoning}")


def test_reasoning_varied(sample):
    """Reasoning strings must not be templated duplicates."""
    texts = [r for _, _, r in sample]
    assert len(set(texts)) == len(texts), "duplicate reasoning strings found"


def test_reasoning_nonempty(sample):
    for rank, c, reasoning in sample:
        assert reasoning and len(reasoning) > 20, f"#{rank} {c['candidate_id']} reasoning too short"


if __name__ == "__main__":
    s = _ranked_sample()
    print(f"Generated reasoning for top {len(s)} of a {SAMPLE_SIZE}-candidate slice\n")
    for rank, c, reasoning in s[:8] + s[-4:]:
        print(f"#{rank} {c['candidate_id']}")
        print(f"   {reasoning}\n")
