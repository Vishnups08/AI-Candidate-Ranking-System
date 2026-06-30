# Redrob AI — Intelligent Candidate Discovery & Ranking
**Track 1: Data & AI Challenge — Approach & Methodology**

> Team: GroundTruth · Lead: Vishnu · Repo: https://github.com/Vishnups08/AI-Candidate-Ranking-System · Sandbox: https://groundtruth-ai-candidate-ranking-system.streamlit.app/

---

## Slide 1: The Problem, Read Carefully

The JD is written to punish keyword matching. It says so explicitly:

- "A candidate with all the AI keywords but the title *Marketing Manager* is **not** a fit."
- "A Tier-5 who built a recommendation system but never says *RAG* **is** a fit."
- "A perfect-on-paper candidate who hasn't logged in for 6 months ... is **not actually available**."

The dataset backs this up with **keyword-stuffers, plain-language fits, behavioral
decoys, and ~80 honeypots** with internally impossible profiles. The task is to
rank like a recruiter who reads between the lines — under a hard production budget
(**CPU-only, network-off, < 5 min** for the ranking step).

---

## Slide 2: We Read the Data Before Trusting It

The single most important finding, from manually reading ~100 profiles:

> **The career-role descriptions are scrambled filler.** An "Operations Manager"
> has a role described as "mechanical engineering design"; an "AI Specialist" has
> one reading "gradient-boosted trees." Descriptions are recycled across
> candidates and do **not** reliably match the person.

Consequences that shaped every later decision:
- **Embed and reason over the coherent fields** (title, headline, summary, skills) —
  never quote raw role descriptions.
- The **summary** is where genuine candidates describe retrieval/ranking work in
  plain language → it's the highest-value semantic signal.
- A "title↔description mismatch" is a property of the *dataset*, not a honeypot.

---

## Slide 3: 5-Stage Hybrid Pipeline

```
100K → [1] Hard filters → [2] Honeypot removal → [3] 6-dim scoring
     → [4] Behavioral multiplier → [5] Cross-encoder re-rank + reasoning → Top-100
```

| Stage | What it does |
|---|---|
| 1 Hard filters | Experience band + drop careers that are non-tech end-to-end (100K → ~28K) |
| 2 Honeypot removal | Internal-impossibility checks, *before* scoring |
| 3 Feature scoring | 6 dimensions, 0–1 each |
| 4 Behavioral | ×0.6–1.3 multiplier from Redrob engagement signals |
| 5 Re-rank + reason | Cross-encoder on top-300; grounded per-candidate reasoning |

Hybrid by design: rules catch hard signals (title, experience, location),
embeddings catch plain-language fits, the cross-encoder sharpens the top.

---

## Slide 4: Scoring Dimensions (Stage 3)

| Dimension | Weight | Captures |
|---|---|---|
| Semantic similarity | 0.25 | `bge-small` embedding of coherent fields vs. JD |
| Career fit | 0.25 | Title, product-vs-consulting, stability, 6 JD disqualifiers |
| Skills match | 0.20 | Skills **× skill–career coherence gate** |
| Experience fit | 0.10 | 5–9 yrs optimal, graceful decay |
| Location & logistics | 0.10 | India/preferred cities, notice, salary, work mode |
| Education | 0.05 | Field + tier (lowest — JD: skills > pedigree) |
| Behavioral | ×0.6–1.3 | Availability **gates** fit; it doesn't average with it |

**Coherence gate** (the JD's central trap): listed skills only count when the
candidate's titles + summary support them. A non-tech career with a padded AI
skill list is discounted to 25%.

---

## Slide 5: Honeypots = Internal Impossibility

Our first detector flagged **58% of the pool** — because it keyed on
title↔description mismatch, which (per Slide 2) is just scrambled data.

We rebuilt it to detect only **internal impossibilities**, the patterns the spec
describes:
- advanced/expert proficiency in a skill with **0 months** used,
- total tenure **far exceeding** stated years of experience,
- **overlapping** full-time roles,
- non-tech title claiming many expert AI skills.

Result: **57,823 → 113 flagged** (spec says ~80), genuine fits untouched,
**0 honeypots in the gold top-10**. Honeypots are removed *before* scoring.

---

## Slide 6: Explainable, Grounded Reasoning

Each candidate gets a rank-aware, 1–2 sentence reasoning composed only from
**facts in their record** — never templated, never quoting scrambled descriptions.

Guardrails (enforced by `tests/test_reasoning.py`, 16 tests passing):
- **No skill hallucination** — every cited skill exists in the profile.
- **Employer-grounded** — every named company is a real employer.
- **Rank-tone consistency** — top ranks aren't flagged as disqualified; low ranks
  aren't glowing.
- **Variation** — no two reasonings are identical.

---

## Slide 7: How We Know It Works (Honest Evaluation)

No public leaderboard → we built our **own signal, independent of the scoring code**
(the bundled heuristic labels share logic with the ranker → circular):

- Stratified sample across **every trap archetype**.
- **98 candidates hand-labeled 0–4 by reading full profiles** — not by the model.
- The **real pipeline** is scored against them.

| NDCG@10 | NDCG@50 | MAP | P@10 | **Composite** | Honeypots@10 |
|---|---|---|---|---|---|
| 0.916 | 0.986 | 0.877 | 0.700 | **0.920** | 0 |

Both genuine tier-4 fits rank #1 and #2. **Honest caveat:** this is a
98-candidate proxy, not the hidden 100K ground truth — it measures *direction*.
Every tuning change is logged with its delta in `RESULTS.md`.

---

## Slide 8: Engineering for the Production Constraint

The JD explicitly wants engineers who reason about **latency vs. quality**:

- **`bge-small` over `bge-base`:** ~99% of the gold composite (0.920 vs 0.926) at
  **~3× faster** CPU precompute → the embedding step stays re-runnable.
  bge-base is a one-line config upgrade.
- **Precompute / rank split:** embeddings are built offline; the ranking step
  loads them and finishes in minutes.
- **Provably offline:** models are snapshotted to a repo-local `./models` and
  loaded with `local_files_only`; `HF_HUB_OFFLINE` is forced in `rank.py`, so the
  reproduced CSV matches the submitted one.

| Constraint | Status |
|---|---|
| Ranking ≤ 5 min, CPU-only, 16 GB | ✅ |
| Network off during ranking | ✅ enforced in code |
| Honeypot rate in top-100 | ✅ 0 in gold top-10 |

---

## Slide 9: Stack & Why

- **sentence-transformers + PyTorch (CPU):** `BAAI/bge-small-en-v1.5` bi-encoder +
  `cross-encoder/ms-marco-MiniLM-L-6-v2` re-ranker — strong local semantics, no API.
- **NumPy:** vectorized cosine similarity over precomputed embeddings.
- **pytest:** trap + reasoning regression tests as living documentation.
- **Streamlit:** hosted sandbox to run the ranker on a small sample and inspect
  per-dimension scores.

---

## Slide 10: Submission Assets

- **Repo:** https://github.com/Vishnups08/AI-Candidate-Ranking-System — clean, tested, single-command reproduction.
- **Sandbox:** https://groundtruth-ai-candidate-ranking-system.streamlit.app/ — runs the ranker on a ≤100-candidate sample.
- **Reproduce:** `python rank.py --candidates ./candidates.jsonl --out ./submission.xlsx`
- **Output:** `submission.xlsx` (and `submission.csv`) — exactly 100 rows + header, validated.
- **Evidence of real engineering:** independent gold set, `RESULTS.md` tuning log
  with reverted experiments, 16 passing regression tests, iterative git history.
