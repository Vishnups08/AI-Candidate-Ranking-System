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
