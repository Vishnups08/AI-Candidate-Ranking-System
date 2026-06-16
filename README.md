# Redrob AI — Candidate Ranking System

**India Runs Hackathon · Track 1: Data & AI Challenge**

Ranks 100,000 candidates against a job description the way a recruiter would —
by understanding *who actually fits the role*, not by counting keywords. The
JD deliberately hides its intent ("a candidate with all the AI keywords but the
title *Marketing Manager* is not a fit"; "a Tier-5 who built a recommendation
system but never says *RAG* is a fit"), and the dataset is seeded with
keyword-stuffers, plain-language fits, behavioral decoys, and ~80 honeypots.
This system is built and tuned specifically to read through those traps.

The ranking step runs **CPU-only, network-off, in under 5 minutes** per the
competition's production constraints.

---

## Pipeline

```
100K candidates
   │  Stage 1  Hard filters          experience band, non-tech-only careers → ~28K
   │  Stage 2  Honeypot removal      internal-impossibility checks (skills/timeline/overlap)
   │  Stage 3  Feature scoring       6 dimensions, 0–1 each (see below)
   │  Stage 4  Behavioral multiplier 0.6–1.3× from Redrob engagement signals
   │  Stage 5  Cross-encoder re-rank top-300 (MS-MARCO MiniLM) + grounded reasoning
   ▼
Top-100 ranked CSV
```

### Scoring dimensions (Stage 3)

| Dimension | Weight | What it captures |
|---|---|---|
| Semantic similarity | 0.25 | `bge-small-en-v1.5` embedding of the **coherent** profile fields (title, headline, summary, skills) vs. a JD embedding. Surfaces plain-language fits. |
| Career fit | 0.25 | Title relevance, product-vs-consulting, role-description relevance, stability, and the JD's six explicit disqualifiers. |
| Skills match | 0.20 | Must-have / nice-to-have / domain skills **gated by skill–career coherence** (padded AI skills on a non-tech career are discounted). |
| Experience fit | 0.10 | Optimal 5–9 yrs per JD, graceful decay outside the band. |
| Location & logistics | 0.10 | India / preferred cities, notice period, salary realism, work mode. |
| Education | 0.05 | Field relevance + institution tier (lowest weight — JD says skills > pedigree). |
| **Behavioral** | ×0.6–1.3 | Availability, recency, response rate, interview completion, verification. Applied as a **multiplier**: a perfect-on-paper candidate inactive for 6 months with a 5% response rate is down-weighted, exactly as the JD instructs. |

---

## Quick start

```bash
pip install -r requirements.txt

# 1. One-time, network ON: cache models locally (~1.1 GB into ./models)
python precompute/download_models.py

# 2. One-time, offline-prep: embed the JD and the candidate pool
python precompute/build_jd_embedding.py
python precompute/build_embeddings.py --candidates ./candidates.jsonl   # ~2 h CPU

# 3. Produce the submission (CPU-only, network-off, < 5 min)
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# 4. Validate format before uploading
python validate_submission.py submission.csv
```

### Reproduce the submission (single command)

With models + embeddings already precomputed, the ranking step that produces the
CSV is one command (the command organizers reproduce at Stage 3):

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

`rank.py` sets `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` and loads both models
from the repo-local `./models` snapshot with `local_files_only=True`, so the
ranking step makes **no network calls** and the reproduced CSV matches the
submitted one.

---

## Evaluation — how we know it works (and where we don't)

There is no public leaderboard, so we built our own signal **independent of the
scoring code** (the bundled heuristic labels would be circular — they share logic
with the ranker):

- `evaluation/sample_for_gold.py` draws a stratified sample across every JD trap
  archetype (genuine fits, plain-language fits, keyword-stuffers, consulting-only,
  inactive, honeypots, random).
- `evaluation/gold_labels.json` — **98 candidates hand-labeled 0–4 by reading the
  full profiles** (see `gold_sample.txt`), not by running the model.
- `evaluation/evaluate_gold.py` runs the **real pipeline** over them and reports
  the competition metrics.

Current gold-set result (`composite = 0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10`):

| NDCG@10 | NDCG@50 | MAP | P@10 | Composite | Honeypots in top-10 |
|---|---|---|---|---|---|
| 0.916 | 0.986 | 0.877 | 0.700 | **0.920** | 0 |

The two genuine tier-4 fits rank #1 and #2. **Honest caveat:** this is a
98-candidate proxy, not the hidden 100K ground truth — it measures *direction*,
not the final score. Per-change deltas are logged in [RESULTS.md](RESULTS.md).

Trap behavior is locked in with regression tests:

```bash
python -m pytest tests/ -q     # 16 tests: trap archetypes + reasoning grounding
```

---

## Key design decisions

1. **We read the data before trusting it.** The career-role *descriptions* are
   scrambled filler across the whole synthetic dataset (an Operations Manager's
   role reads "mechanical engineering design"). So we embed and reason over the
   **coherent** fields (title, summary, skills) and never quote raw descriptions.

2. **Skill–career coherence gate.** The JD's central trap is the keyword-stuffer.
   Listed skills only count when the candidate's titles and summary actually
   support them; a non-tech career with a padded AI skill list is discounted to
   25%.

3. **Honeypots = internal impossibility, not description mismatch.** An earlier
   detector flagged 58% of the pool on title↔description mismatch — a property of
   the scrambled data, not a honeypot signal. We rely only on internal
   impossibilities (advanced/expert skill with 0 months, tenure ≫ experience,
   overlapping full-time roles), flagging ~113 (spec says ~80).

4. **Behavioral signals as a multiplier, not a feature.** Availability gates
   fit; it doesn't average with it.

5. **Latency–quality tradeoff (the JD asks for this).** `bge-small` reaches ~99%
   of `bge-base`'s gold composite at ~3× the CPU speed, keeping precompute
   re-runnable. `bge-base` is a one-line config upgrade.

---

## Compute compliance

| Constraint | Status |
|---|---|
| Ranking ≤ 5 min wall-clock | ✅ (precompute is a separate offline step) |
| Memory ≤ 16 GB | ✅ |
| CPU only | ✅ no GPU code path |
| Network off during ranking | ✅ enforced via `HF_HUB_OFFLINE` + `local_files_only` |
| Honeypot rate in top-100 | ✅ 0 in gold top-10; detector excludes pre-scoring |

## Project layout

```
rank.py                      # entry point (offline-enforced)
config.py                    # weights, thresholds, model paths
pipeline/                    # loader, jd_parser, hard_filters, honeypot_detector,
                             #   feature_scorer, behavioral_scorer, ranker, reasoning_generator
precompute/                  # download_models, build_jd_embedding, build_embeddings
evaluation/                  # sample_for_gold, gold_labels.json, evaluate_gold, self_evaluate
tests/                       # test_traps.py, test_reasoning.py
RESULTS.md                   # tuning log (change → gold-composite delta)
app.py                       # Streamlit sandbox demo
```

## License
MIT
