# Gold Set — Methodology, Limitations, and Honest Framing

## What the gold set is

`evaluation/gold_labels.json` contains **98 candidates** hand-labeled with relevance tiers 0–4
by reading the full profiles. Labels were assigned without running the model — specifically to
avoid circular self-evaluation.

| Tier | Label | Criterion |
|------|-------|-----------|
| 4 | Strong fit | Passes all JD gates, right YoE, skills confirmed by career, product company, preferred city |
| 3 | Good fit | Passes most gates; one weakness (e.g., slightly off location or notice period) |
| 2 | Moderate fit | Clear relevant skills but 2+ weaknesses |
| 1 | Marginal | Technically eligible but significant gaps |
| 0 | Not a fit | Would not shortlist under any interpretation |

Stratified sampling ensured representation across 7 archetypes:
- Genuine fits (tier 3–4)
- Plain-language fits (relevant but no AI keywords)
- Keyword-stuffers (padded skills, non-tech career)
- Career-long consulting
- Inactive/behavioral decoys
- Honeypots (excluded, labeled 0)
- Random sample

## Current composite score

| NDCG@10 | NDCG@50 | MAP | P@10 | Composite |
|---------|---------|-----|------|-----------|
| 0.916   | 0.986   | 0.877 | 0.700 | **0.920** |

The two tier-4 genuine fits rank #1 and #2. Zero honeypots in top-10.

---

## Known Limitations — owned, not hidden

> [!IMPORTANT]
> **Self-labeled proxy.** The gold labels were assigned by the same person who designed the
> ranker, introducing potential confirmation bias. An independent annotator might assign
> different tier boundaries, particularly for tier 2/3 borderline cases.

> [!IMPORTANT]
> **Small sample (98 candidates).** Against a 28,000-candidate pool, this is ~0.35%.
> High variance at NDCG@10 in particular — one rank swap of a tier-4 candidate changes
> the metric by ~0.08. The composite should be read as directional, not precise.

> [!NOTE]
> **No inter-annotator agreement score.** A second-pass relabeling of ~30 disputed
> candidates would materially strengthen the claim. This is the highest-value improvement
> to the evaluation methodology if time permits.

> [!NOTE]
> **Hidden ground truth.** There is no public leaderboard. The gold set measures *direction*
> (does each pipeline change help or hurt?), not the final competition score.

---

## What we did to mitigate these limitations

1. **Stratified sampling** across all 7 known archetypes — not random — ensures the proxy
   covers edge cases disproportionately (honeypots, stuffers, plain-language fits).

2. **Independent labeling discipline** — labels were assigned by reading profiles before
   running the model. Scores were only checked after labels were frozen.

3. **Per-change delta tracking** in `RESULTS.md` — we use the gold set to measure *direction*,
   not claim absolute accuracy. A +0.003 composite delta is noise; -0.02 is signal.

4. **The ablation table (`evaluation/ablation.py`) is self-contained** — it runs the real
   pipeline with/without each component. The ablation output is real numbers, not projections.

---

## Recommended expansion (if time permits)

Priority order for strengthening the gold set before final submission:

1. **Expand to ~150 candidates** — add 52 more, focusing on consulting-vs-product borderline
   cases (where the career_fit scorer is most uncertain) and mid-tier candidates near rank 50–80.

2. **Second-pass relabel of tier 2/3 candidates** — re-read ~30 borderline profiles cold and
   check for label drift. Document inter-annotator agreement (Cohen's kappa target: > 0.70).

3. **Include a "would you call this candidate?" binary question** alongside the 0–4 tier —
   binary precision has lower variance and is easier to defend to judges.

---

## How to expand the gold set

```bash
# Sample more candidates (stratified by score decile + archetype)
python evaluation/sample_for_gold.py --candidates ./candidates.jsonl --n 52 --add-to-existing

# After labeling the new batch in gold_labels.json, re-run evaluation
python evaluation/evaluate_gold.py --candidates ./candidates.jsonl
```

The `sample_for_gold.py` script excludes already-labeled candidates and stratifies by
pipeline score decile to ensure the expansion covers the mid-field (ranks 30–80),
which is where the gold set is currently weakest.
