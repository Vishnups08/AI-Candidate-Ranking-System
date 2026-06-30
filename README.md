---
title: AI Candidate Ranking System Backend
emoji: 🎯
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

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
#    Auto-generates both submission.xlsx (portal) and submission.csv
python rank.py --candidates ./candidates.jsonl --out ./submission.xlsx

# 4. Validate format before uploading
python validate_submission.py submission.csv
```

### Reproduce the submission (single command)

With models + embeddings already precomputed, the ranking step that produces the
submission is one command (the command organizers reproduce at Stage 3):

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.xlsx
```

This auto-generates **both** `submission.xlsx` (portal upload format) and
`submission.csv` (spec-referenced format) in a single run.

`rank.py` sets `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` and loads both models
from the repo-local `./models` snapshot with `local_files_only=True`, so the
ranking step makes **no network calls** and the reproduced output matches the
submitted one.

## Demo UI

Two ways to explore the ranked results interactively:

### Option A — Standalone HTML demo (for judges)

Runs a lightweight Flask API + opens a pure HTML/JS frontend. No Streamlit install needed.

```bash
pip install flask flask-cors

# Start the API (runs the full pipeline once at startup)
python demo_server.py --candidates ./candidates.jsonl

# Then open in your browser:
open demo/index.html
```

The demo shows:
- Ranked top-N with tier badges and score formula
- Expandable per-candidate score card (6 dims + behavioral + evidence citations)
- **vs Naive Keyword Match toggle** — shows how much each candidate moved up/down
- **Contrast cards** — one honeypot (excluded) + one keyword-stuffer (demoted) with why-not reasoning

### Option B — Streamlit dashboard (internal tuning)

```bash
streamlit run app.py
```

Includes live weight sliders, radar chart, career timeline, behavioral signals, contrast cases tab.

### Option C — Next.js React Dashboard (Standalone UI)

A modern, highly responsive Next.js frontend built with React and Tailwind CSS that connects to the Python Flask API backend.

```bash
# 1. Start the Flask API server
python demo_server.py --candidates ./candidates.jsonl --port 5050

# 2. Run the Next.js dev server
npm run dev --prefix frontend
```

Then open `http://localhost:3000` to inspect candidate shortlist tables, details tabs, and run dynamic neural re-ranking tests.

---


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
python -m pytest tests/ -q     # trap archetypes + reasoning grounding
```

### Ablation study

Run to verify every pipeline component earns its place:

```bash
python evaluation/ablation.py --candidates ./candidates.jsonl
```

Expected output (approximate — exact numbers depend on gold-set run):

| Configuration | Composite | Δ vs Baseline |
|---|---|---|
| Baseline (full pipeline) | ~0.920 | — |
| No coherence gate | ~0.880 | -0.040 |
| No honeypot de-noising | ~0.860 | -0.060 |
| No cross-encoder | ~0.900 | -0.020 |
| No behavioral multiplier | ~0.905 | -0.015 |
| Skills-only (no semantic) | ~0.820 | -0.100 |

### Trap evidence report

Surfaces concrete examples of each archetype from the actual pool:

```bash
python evaluation/trap_evidence.py --candidates ./candidates.jsonl
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

6. **Why grounded hybrid, not agentic LLM calls?** See [ARCHITECTURE.md](ARCHITECTURE.md)
   for the full rationale. Short version: accuracy + explainability + reproducibility +
   offline constraint. Every rank is traceable to a specific profile field; it is
   architecturally impossible to hallucinate a skill the candidate doesn't have.

---

## India-Context Design

These signals are deliberately India-aware (not incidentally):

| Signal | India-specific tuning |
|--------|----------------------|
| Location scoring | Pune, Noida, NCR = 1.0; Hyderabad, Mumbai, Bangalore = 0.7 |
| Salary realism | ≤50 LPA = ideal, 50–70 = stretch, >70 = concern (Series A budget) |
| Notice period | ≤30 days preferred (+0.10), standard Indian 90 days = small penalty |
| Consulting penalty | TCS, Infosys, Wipro, Cognizant, HCL etc. = career quality score 0.15 |
| Institution tiers | IIT/BITS/NIT = tier_1 (+0.30 education bonus) |
| Service vs product | Product-company bias per JD's explicit preference |

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
demo_server.py               # Flask API backend for standalone demo
demo/index.html              # Standalone HTML/JS demo UI
pipeline/                    # loader, jd_parser, hard_filters, honeypot_detector,
                             #   feature_scorer, behavioral_scorer, ranker, reasoning_generator
precompute/                  # download_models, build_jd_embedding, build_embeddings
evaluation/                  # sample_for_gold, gold_labels.json, evaluate_gold,
                             #   ablation.py, trap_evidence.py
tests/                       # test_traps.py, test_reasoning.py
ARCHITECTURE.md              # System architecture note (required deliverable)
RESULTS.md                   # tuning log (change → gold-composite delta)
app.py                       # Streamlit dashboard (internal tuning)
```

## License
MIT
