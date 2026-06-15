# Redrob AI Candidate Ranking System

**India Runs Hackathon — Track 1: Data & AI Challenge**

An intelligent candidate ranking system that goes beyond keyword matching to understand who genuinely fits a role. Uses a hybrid pipeline combining rule-based scoring, semantic embeddings, behavioral signal analysis, and honeypot detection to produce a recruiter-trustworthy shortlist.

## Architecture

```
100K Candidates → Hard Filters → Honeypot Detection → Feature Scoring → Behavioral Multiplier → Top 100 CSV
                  (~5-15K pass)   (~80 traps removed)   (7 dimensions)    (availability/engagement)
```

### Scoring Dimensions
| Dimension | Weight | What It Measures |
|-----------|--------|------------------|
| Semantic Similarity | 0.25 | Per-role-description embedding match against JD using `bge-small-en-v1.5` |
| Career Fit | 0.25 | Title relevance, product vs consulting, career progression, description relevance |
| Skills Match | 0.20 | Must-have/nice-to-have skill matching with credibility cross-check + assessment scores |
| Experience Fit | 0.10 | Optimal band: 5-9 years (per JD) with graceful decay |
| Location & Logistics | 0.10 | India location, notice period, salary realism, work mode |
| Education | 0.05 | Field relevance, institution tier (lowest weight — JD says skills > pedigree) |
| Behavioral Multiplier | 0.6x-1.3x | Activity recency, response rate, interview completion, GitHub, verifications |

## Quick Start

### Prerequisites
- Python 3.10+
- 16GB RAM recommended
- No GPU required

### Setup
```bash
git clone https://github.com/YOUR_USERNAME/redrob-ranker.git
cd redrob-ranker
pip install -r requirements.txt
```

### Step 1: Pre-compute Embeddings (one-time, ~15-20 min)
```bash
python precompute/build_jd_embedding.py
python precompute/build_embeddings.py --candidates ./candidates.jsonl
```

### Step 2: Generate Submission (< 5 min, CPU only)
```bash
python rank.py --candidates ./candidates.jsonl --out ./output/submission.csv
```

### Step 3: Validate
```bash
python validate_submission.py output/submission.csv
```

## Reproduce the Submission CSV

**Single command** (assumes embeddings are pre-computed):
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

If embeddings haven't been computed yet:
```bash
python precompute/build_jd_embedding.py && python precompute/build_embeddings.py --candidates ./candidates.jsonl && python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

## Project Structure
```
redrob-ranker/
├── rank.py                       # Main entry point (CLI)
├── config.py                     # All weights, thresholds, constants
├── requirements.txt              # Dependencies
├── submission_metadata.yaml      # Competition metadata
├── pipeline/
│   ├── loader.py                 # Streaming JSONL loader
│   ├── jd_parser.py              # JD requirements extraction
│   ├── hard_filters.py           # Stage 1: Fast disqualifiers
│   ├── honeypot_detector.py      # Stage 2: Anomaly detection
│   ├── feature_scorer.py         # Stage 3: 7-dimension scoring
│   ├── behavioral_scorer.py      # Stage 4: Behavioral multiplier
│   ├── ranker.py                 # Stage 5: Composite ranking
│   └── reasoning_generator.py    # Per-candidate reasoning
├── precompute/
│   ├── build_embeddings.py       # Offline: candidate embeddings
│   └── build_jd_embedding.py     # Offline: JD embedding
├── app.py                        # Streamlit sandbox demo
└── output/
    └── submission.csv            # Final output
```

## Key Design Decisions

1. **Hybrid rule + embedding approach**: Rules catch obvious signals (title, experience, location); embeddings catch "plain-language Tier 5" candidates whose descriptions demonstrate fit without using buzzwords.

2. **Honeypot detection before scoring**: ~80 honeypot candidates with impossible profiles are detected and excluded before they can pollute rankings.

3. **Per-role-description semantic scoring**: Instead of concatenating all text, we embed each career role separately and score with recency decay. A recent role building "hybrid retrieval systems" scores higher than the same words in a 5-year-old role.

4. **Skill credibility cross-check**: Self-reported "expert" proficiency is verified against endorsement count, duration, and platform assessment scores. This catches keyword-stuffing traps.

5. **Behavioral signals as multiplier**: A perfect-on-paper candidate who hasn't logged in for 6 months with 5% response rate gets ~0.65x multiplier, exactly as the JD instructs.

## Compute Compliance
| Constraint | Status |
|-----------|--------|
| Runtime ≤ 5 min | ✅ ~2-3 min for ranking step |
| Memory ≤ 16GB | ✅ ~4-6GB peak |
| CPU only | ✅ No GPU during ranking |
| No network | ✅ No API calls during ranking |
| Disk ≤ 5GB | ✅ ~200MB for embeddings |

## License
MIT
