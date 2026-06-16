# Tuning Results Log

Each entry records a change and its effect on the independent gold-set composite
(`python evaluation/evaluate_gold.py`) and the trap exemplar scores
(`python tests/test_traps.py`). The gold labels are assigned by manual profile
reading (evaluation/gold_labels.json), independent of the scoring code, so an
improvement here is a real signal rather than self-consistency.

Composite = 0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10

---

## Baseline (keyword-fallback semantic; embeddings rebuilding)

Trap exemplar scores (`tests/test_traps.py`):

| archetype | candidate | score | honeypot |
|---|---|---|---|
| genuine fit | CAND_0002025 | 1.036 | – |
| genuine fit | CAND_0000031 | 0.778 | – |
| strong fit | CAND_0001707 | 0.864 | – |
| strong fit | CAND_0000273 | 0.825 | – |
| strong fit | CAND_0001131 | 0.801 | – |
| keyword stuffer | CAND_0000821 | **0.596** | – |
| keyword stuffer | CAND_0002356 | **0.500** | – |
| keyword stuffer | CAND_0000722 | 0.389 | – |
| keyword stuffer | CAND_0000074 | 0.353 | – |
| honeypot | CAND_0000004 | 0.398 | True |
| honeypot | CAND_0000005 | 0.373 | True |
| honeypot | CAND_0000009 | 0.367 | True |
| consulting jr | CAND_0000098 | 0.276 | – |
| consulting jr | CAND_0000047 | 0.216 | – |
| consulting jr | CAND_0000056 | 0.211 | – |

**Issues identified:**
1. Keyword-stuffer CAND_0000821 (Mechanical Engineer + 8 padded AI skills)
   scores 0.596 — the skills-match dimension rewards the padded AI skill list
   without checking that the title/career are non-tech. This is the JD's
   central trap and must be fixed.
2. CAND_0002356 (Project Manager, padded) at 0.500 is too close to real
   candidates.
3. Honeypot detector only flags ~10 of the expected ~80 in the full pool
   (recall too low) — Phase 2.5.

**Next:** add a skill–title/career coherence gate so AI skills only count when
the candidate's actual work supports them; re-measure on full embeddings.

---

## R1 — Real gold composite with BGE embeddings (on-the-fly gold embedding)

First measurement against the independent gold set (98 labels: 80×t0, 3×t1,
4×t2, 9×t3, 2×t4) using BGE-base embeddings on the signal-focused profile text,
+ cross-encoder blend. Gold candidates embedded on the fly so tuning is
decoupled from the slow ~2 h full-pool precompute.

| metric | value |
|---|---|
| NDCG@10 | 0.9192 |
| NDCG@50 | 0.9867 |
| MAP | 0.8929 |
| P@10 | 0.7000 |
| **COMPOSITE** | **0.9245** |
| honeypots in top-10 | 0 |

Top-2 (both genuine tier-4) ranked #1, #2 correctly. NDCG@50 near-perfect.

**Issue R1.1 (drives NDCG@10 down):** three tier-2 candidates outrank three
genuine tier-3 product-ML candidates:
- tier-2 CAND_0002706 (#8), 0002037 (#9), 0002120 (#10) sit above
- tier-3 CAND_0000666 (#11), 0001651 (#13), 0001494 (#14), 0000981 (#16).
The tier-2s are over-credited; need to separate genuine product-ML fit (real
retrieval/recsys in summary) from generic "applied ML" profiles.

**Issue R1.2:** tier-0 keyword-stuffer CAND_0002220 reaches #15 — skills-match
still rewards padded AI skills on a non-tech (Content Writer) profile. Needs
the skill–title coherence gate.

---

## R2 — Skill–career coherence gate (the JD's central trap)

Added `FeatureScorer._skill_career_coherence`: skills-match is multiplied by a
[0.25, 1.0] factor based on whether the candidate's titles + summary/career
actually support having those skills. Non-tech title + no ML work evidence →
0.25 (skills are padding). Excludes mechanical/civil/etc. from "engineer".

| metric | R1 | R2 | Δ |
|---|---|---|---|
| NDCG@10 | 0.9192 | 0.9192 | — |
| NDCG@50 | 0.9867 | 0.9882 | +0.0015 |
| MAP | 0.8929 | 0.8970 | +0.0041 |
| P@10 | 0.7000 | 0.7000 | — |
| **COMPOSITE** | **0.9245** | **0.9256** | **+0.0011** |

Keyword-stuffer trap scores fell: CAND_0000074 0.353→0.289, CAND_0000821
0.596→0.525, CAND_0002220 dropped out of the top-15. Genuine fits unchanged.

## R3 — Attempted: reward recognized product companies (REVERTED)

Tried adding a PRODUCT_COMPANIES bonus to company_quality to lift tier-3
product-ML fits above tier-2s. Composite *fell* 0.9256→0.9245: the tier-2
candidates also work at product companies (Zoho etc.), so the bonus lifted them
too. This confirmed the tier-2/tier-3 boundary is label noise on near-identical
templated profiles — over-tuning it would overfit the 98-sample gold set and
likely hurt on the hidden ground truth. Reverted; kept only the coherence gate.

**Decision:** stop micro-tuning the t2/t3 boundary. The top-2 (genuine tier-4)
are correctly #1/#2, NDCG@50 is 0.988, and the remaining top-10 churn is within
label-noise of my own gold set. Further gold-set gains here are not trustworthy.
