"""
Stage 5: Composite Ranker.
Combines feature scores with behavioral multiplier, sorts, and outputs CSV.
"""

import csv
import time
from typing import Optional

import numpy as np

import config
from pipeline.jd_parser import JDRequirements, load_jd_requirements
from pipeline.feature_scorer import FeatureScorer
from pipeline.behavioral_scorer import compute_behavioral_multiplier, compute_behavioral_additive
from pipeline.reasoning_generator import generate_reasoning, generate_structured_explanation


class CandidateRanker:
    """Ranks candidates and produces the final CSV output."""

    def __init__(self, jd: JDRequirements,
                 jd_embedding: Optional[np.ndarray] = None,
                 candidate_embeddings: Optional[dict] = None):
        self.jd = jd
        self.scorer = FeatureScorer(jd, jd_embedding, candidate_embeddings)

    def rank_candidates(self, candidates: list[dict]) -> list[dict]:
        """
        Score all candidates and return top 100 ranked.
        Each result contains: candidate_id, rank, score, reasoning, details.
        """
        print(f"  Scoring {len(candidates)} candidates...")

        # Score all candidates
        scored = []
        for i, candidate in enumerate(candidates):
            if (i + 1) % 1000 == 0:
                print(f"    Scored {i + 1}/{len(candidates)}...")

            # Feature scores
            feature_scores = self.scorer.score_candidate(candidate)
            weighted_total = feature_scores["weighted_total"]

            # Behavioral multiplier
            multiplier = compute_behavioral_multiplier(candidate)
            additive = compute_behavioral_additive(candidate)

            # Final composite score
            final_score = weighted_total * multiplier + additive

            scored.append({
                "candidate": candidate,
                "feature_scores": feature_scores,
                "behavioral_multiplier": multiplier,
                "behavioral_additive": additive,
                "final_score": final_score,
            })

        # Sort by final_score descending (rounded to 4 decimals to match output precision), tie-breaking by candidate_id ascending
        scored.sort(key=lambda x: (-round(x["final_score"], 4), x["candidate"]["candidate_id"]))

        # Optional Cross-Encoder Re-Ranking
        if config.USE_CROSS_ENCODER and len(scored) > 0:
            print(f"  Re-ranking top {min(len(scored), config.CROSS_ENCODER_TOP_K)} candidates with Cross-Encoder...")
            ce_start = time.time()
            top_for_rerank = scored[:config.CROSS_ENCODER_TOP_K]

            try:
                from sentence_transformers import CrossEncoder
                # CPU only, loaded from the repo-local snapshot so it works
                # network-off. local_files_only=True only when the snapshot
                # exists, so a fresh clone can still pull it once.
                ce_path = config._local_model_path(config.CROSS_ENCODER_MODEL)
                ce_is_local = ce_path != config.CROSS_ENCODER_MODEL
                ce_model = CrossEncoder(ce_path, device="cpu", local_files_only=ce_is_local)

                query_text = self.jd.jd_core_text if hasattr(self.jd, 'jd_core_text') else f"{self.jd.title} {' '.join(self.jd.must_have_skills)}"

                # Build pairs
                pairs = []
                for entry in top_for_rerank:
                    c = entry["candidate"]
                    prof = c.get("profile", {})
                    # Build rich document representation for candidate
                    title = prof.get("current_title", "")
                    skills = ", ".join(s.get("name", "") for s in c.get("skills", [])[:15])
                    summary = prof.get("summary", "")
                    cand_text = f"{title}. Skills: {skills}. {summary}"[:config.EMBEDDING_MAX_TEXT_LENGTH]
                    pairs.append([query_text, cand_text])

                # Score pairs
                ce_scores = ce_model.predict(pairs)

                # Normalize cross-encoder scores to [0, 1] for blending
                # Sigmoid is typical for CrossEncoders trained on MS MARCO
                ce_scores_norm = 1 / (1 + np.exp(-ce_scores))

                # Blend scores
                for entry, ce_score in zip(top_for_rerank, ce_scores_norm):
                    orig_score = entry["final_score"]
                    blended = (1 - config.CROSS_ENCODER_WEIGHT) * orig_score + config.CROSS_ENCODER_WEIGHT * ce_score
                    entry["bi_encoder_score"] = orig_score
                    entry["cross_encoder_score"] = float(ce_score)
                    entry["final_score"] = blended

                # Re-sort top_for_rerank based on blended score
                top_for_rerank.sort(key=lambda x: (-round(x["final_score"], 4), x["candidate"]["candidate_id"]))

                # Merge back
                scored[:config.CROSS_ENCODER_TOP_K] = top_for_rerank
                print(f"    Cross-encoder applied to {len(pairs)} candidates "
                      f"(local={ce_is_local}) in {time.time() - ce_start:.1f}s")

            except ImportError as e:
                # A silent fallback would make the reproduced CSV differ from the
                # submitted one. Fail loudly so the discrepancy can never happen.
                raise RuntimeError(
                    "sentence-transformers is required for cross-encoder re-ranking "
                    "but is not installed. Install it, or set config.USE_CROSS_ENCODER=False "
                    "and regenerate the submission so the CSV matches the code path."
                ) from e

        # Take top 100
        top_k = scored[:config.TOP_K]

        # Assign ranks and generate reasoning
        results = []
        for rank, entry in enumerate(top_k, start=1):
            candidate = entry["candidate"]
            score = entry["final_score"]
            feature_scores = entry["feature_scores"]

            # Generate reasoning (plain text for CSV)
            reasoning = generate_reasoning(candidate, rank, feature_scores)

            # Generate structured explanation (rich breakdown for UI/demo)
            explanation = generate_structured_explanation(
                candidate,
                rank,
                feature_scores,
                entry["behavioral_multiplier"],
                entry["behavioral_additive"],
            )

            results.append({
                "candidate_id": candidate["candidate_id"],
                "rank": rank,
                "score": round(score, 4),
                "reasoning": reasoning,
                "explanation": explanation,
                "details": {
                    "feature_scores": feature_scores,
                    "behavioral_multiplier": entry["behavioral_multiplier"],
                    "behavioral_additive": entry["behavioral_additive"],
                },
            })

        return results

    @staticmethod
    def _build_dataframe(results: list[dict]):
        """Build a pandas DataFrame from ranked results."""
        import pandas as pd

        data = []
        for result in results:
            data.append({
                config.OUTPUT_COLUMNS[0]: result["candidate_id"],
                config.OUTPUT_COLUMNS[1]: result["rank"],
                config.OUTPUT_COLUMNS[2]: result["score"],
                config.OUTPUT_COLUMNS[3]: result["reasoning"],
            })
        return pd.DataFrame(data)

    @staticmethod
    def write_xlsx(results: list[dict], output_path: str):
        """Write results to Excel (XLSX) in submission format."""
        df = CandidateRanker._build_dataframe(results)
        df.to_excel(output_path, index=False, engine='openpyxl')
        print(f"  Written {len(results)} rows to {output_path}")

    @staticmethod
    def write_csv(results: list[dict], output_path: str):
        """Write results to CSV in submission format."""
        df = CandidateRanker._build_dataframe(results)
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"  Written {len(results)} rows to {output_path}")

    @staticmethod
    def write_output(results: list[dict], output_path: str):
        """Auto-detect format from extension and write both CSV + XLSX.

        The submission spec references CSV while the portal upload field
        expects XLSX. We always produce both so either can be submitted.
        """
        from pathlib import Path

        p = Path(output_path)
        stem = p.stem
        parent = p.parent

        xlsx_path = parent / f"{stem}.xlsx"
        csv_path = parent / f"{stem}.csv"

        CandidateRanker.write_xlsx(results, str(xlsx_path))
        CandidateRanker.write_csv(results, str(csv_path))
        print(f"  Both formats ready: {xlsx_path}  &  {csv_path}")

    @staticmethod
    def validate_results(results: list[dict], strict: bool = True) -> list[str]:
        """
        Run basic validation on results before writing.
        strict=True enforces exactly 100 results (for real submission).
        strict=False allows fewer (for testing on sample data).
        """
        errors = []
        n = len(results)

        if strict and n != config.TOP_K:
            errors.append(f"Expected {config.TOP_K} results, got {n}")
        elif n == 0:
            errors.append("No results produced")
            return errors

        # Check ranks are 1..N
        ranks = [r["rank"] for r in results]
        expected_ranks = list(range(1, n + 1))
        if sorted(ranks) != expected_ranks:
            errors.append(f"Ranks don't cover 1-{n} exactly")

        # Check candidate_id uniqueness
        ids = [r["candidate_id"] for r in results]
        if len(set(ids)) != len(ids):
            errors.append("Duplicate candidate_ids found")

        # Check score monotonicity
        scores = [r["score"] for r in results]
        for i in range(len(scores) - 1):
            if scores[i] < scores[i + 1]:
                errors.append(
                    f"Score not non-increasing: rank {i+1} ({scores[i]}) < rank {i+2} ({scores[i+1]})"
                )
                break

        # Check reasoning not empty
        empty_reasoning = sum(1 for r in results if not r.get("reasoning"))
        if empty_reasoning > 0:
            errors.append(f"{empty_reasoning} candidates have empty reasoning")

        return errors
