"""
Stage 3: Multi-Dimensional Feature Scoring.
7 scoring dimensions (0-1 each) combined with configurable weights.
"""

import math
import numpy as np
from typing import Optional

import config
from pipeline.jd_parser import JDRequirements


class FeatureScorer:
    """Scores candidates across 7 dimensions against JD requirements."""

    def __init__(self, jd: JDRequirements, jd_embedding: Optional[np.ndarray] = None,
                 candidate_embeddings: Optional[dict] = None):
        self.jd = jd
        self.jd_embedding = jd_embedding
        self.candidate_embeddings = candidate_embeddings or {}

    def score_candidate(self, candidate: dict) -> dict:
        """
        Score a candidate across all dimensions.
        Returns dict with dimension scores and weighted total.
        """
        scores = {
            "career_fit": self._score_career_fit(candidate),
            "skills_match": self._score_skills_match(candidate),
            "experience_fit": self._score_experience_fit(candidate),
            "location_logistics": self._score_location_logistics(candidate),
            "education": self._score_education(candidate),
            "semantic_similarity": self._score_semantic_similarity(candidate),
        }

        # Compute weighted sum
        weighted_total = sum(
            scores[dim] * config.WEIGHTS[dim]
            for dim in scores
        )

        scores["weighted_total"] = weighted_total
        return scores

    # =========================================================================
    # Career Fit Score (Weight: 0.25)
    # =========================================================================
    def _score_career_fit(self, candidate: dict) -> float:
        profile = candidate.get("profile", {})
        career_history = candidate.get("career_history", [])

        components = []

        # 1. Title relevance (current + history)
        title_score = self._compute_title_relevance(profile, career_history)
        components.append(("title_relevance", title_score, 0.30))

        # 2. Product vs consulting experience
        company_score = self._compute_company_quality(career_history)
        components.append(("company_quality", company_score, 0.25))

        # 3. Career description relevance
        desc_score = self._compute_description_relevance(career_history)
        components.append(("description_relevance", desc_score, 0.25))

        # 4. Career stability & progression
        stability_score = self._compute_career_stability(career_history, profile)
        components.append(("stability", stability_score, 0.10))

        # 5. Disqualifier checks
        disqualifier_penalty = self._compute_disqualifier_penalty(profile, career_history)
        components.append(("disqualifier_penalty", 1.0 - disqualifier_penalty, 0.10))

        # Weighted sub-score
        total = sum(score * weight for _, score, weight in components)
        return min(max(total, 0.0), 1.0)

    def _compute_title_relevance(self, profile: dict, career_history: list) -> float:
        """Score based on how relevant the candidate's titles are."""
        current_title = profile.get("current_title", "").lower().strip()
        current_score = config.TITLE_RELEVANCE.get(current_title, 0.15)

        # Also consider best historical title
        best_history_score = 0.0
        for role in career_history:
            title = role.get("title", "").lower().strip()
            score = config.TITLE_RELEVANCE.get(title, 0.15)
            if score > best_history_score:
                best_history_score = score

        # Current title matters more (0.7) than best historical (0.3)
        return current_score * 0.7 + best_history_score * 0.3

    def _compute_company_quality(self, career_history: list) -> float:
        """Score based on product vs consulting company experience."""
        if not career_history:
            return 0.3

        consulting_roles = 0
        product_roles = 0

        for role in career_history:
            company = role.get("company", "").lower().strip()
            if company in config.CONSULTING_FIRMS:
                consulting_roles += 1
            else:
                product_roles += 1

        total_roles = consulting_roles + product_roles
        if total_roles == 0:
            return 0.3

        # Career-long consulting only = heavy penalty per JD
        if consulting_roles == total_roles:
            return 0.15  # JD explicit: "we've had bad fit experiences"

        # Mix is fine
        product_ratio = product_roles / total_roles
        return 0.3 + product_ratio * 0.7  # 0.3 to 1.0

    def _compute_description_relevance(self, career_history: list) -> float:
        """Score based on how relevant career role descriptions are."""
        if not career_history:
            return 0.0

        relevance_keywords = {
            # High-value keywords (strong match)
            "ranking": 3, "retrieval": 3, "search": 2, "recommendation": 3,
            "embedding": 3, "vector": 3, "nlp": 2, "information retrieval": 3,
            "machine learning": 2, "ml model": 2, "deep learning": 2,
            "production": 2, "deployed": 2, "scale": 1, "real users": 2,
            "a/b test": 3, "evaluation": 2, "metrics": 1,
            "fine-tun": 2, "lora": 2, "transformer": 2, "bert": 2,
            "llm": 2, "large language model": 2,
            "bm25": 3, "tf-idf": 2, "cosine": 2,
            "pinecone": 3, "weaviate": 3, "qdrant": 3, "milvus": 3,
            "faiss": 3, "elasticsearch": 2, "opensearch": 2,
            "python": 1, "pytorch": 2, "tensorflow": 2,
            "api": 1, "backend": 1, "microservice": 1,
            "pipeline": 1, "data pipeline": 1,
            "recruiter": 2, "hiring": 2, "candidate": 2, "talent": 2,

            # Low-value keywords (weak match)
            "software": 0.5, "code": 0.5, "develop": 0.5,
        }

        total_score = 0.0
        max_possible = 0.0

        for i, role in enumerate(career_history):
            desc = role.get("description", "").lower()
            # Recency decay: most recent role = 1.0, older roles decay
            recency_weight = 1.0 / (1 + i * 0.3)

            role_score = 0.0
            for keyword, value in relevance_keywords.items():
                if keyword in desc:
                    role_score += value

            total_score += role_score * recency_weight
            max_possible += sum(relevance_keywords.values()) * recency_weight * 0.3

        if max_possible == 0:
            return 0.0

        normalized = total_score / max_possible
        return min(normalized, 1.0)

    def _compute_career_stability(self, career_history: list, profile: dict) -> float:
        """Score career stability (penalize excessive job-hopping)."""
        if len(career_history) <= 1:
            return 0.7  # Too little data

        durations = [r.get("duration_months", 0) for r in career_history]
        avg_duration = sum(durations) / len(durations) if durations else 0

        if avg_duration < 12:
            return 0.2  # Very short tenures
        elif avg_duration < 18:
            return 0.4  # JD warns about "title chasers"
        elif avg_duration < 24:
            return 0.7  # Acceptable
        elif avg_duration < 48:
            return 1.0  # Ideal
        else:
            return 0.85  # Long tenure, slight concern about adaptability

    def _compute_disqualifier_penalty(self, profile: dict, career_history: list) -> float:
        """Compute penalty for JD-explicit disqualifiers. Returns 0-1 penalty.
        
        The JD lists 6 explicit disqualifiers:
        1. Pure research career with no production deployment
        2. Career-long consulting only (TCS, Infosys, Wipro, etc.)
        3. Only recent LLM experience (<12 months, LangChain-era only)
        4. Hasn't written production code in 18+ months (architecture/manager roles)
        5. Primarily CV/speech/robotics without NLP/IR exposure
        6. Title-chaser with frequent job hopping (<18 month tenures)
        """
        penalty = 0.0
        skills = [s.get("name", "").lower() for s in profile.get("skills", [])] if "skills" in profile else []

        # --- 1. Research-only career with no production ---
        all_research = True
        for role in career_history:
            desc = role.get("description", "").lower()
            title = role.get("title", "").lower()
            prod_keywords = {"production", "deployed", "shipped", "users", "real-world", "scale", "serving"}
            research_keywords = {"research", "paper", "published", "academic", "thesis", "lab"}
            
            has_production = any(kw in desc for kw in prod_keywords)
            is_research = any(kw in title for kw in research_keywords)
            
            if has_production or not is_research:
                all_research = False
                break
        
        if all_research and len(career_history) > 1:
            penalty += 0.3

        # --- 2. Career-long consulting only ---
        all_consulting = all(
            role.get("company", "").lower().strip() in config.CONSULTING_FIRMS
            for role in career_history
        ) if career_history else False
        
        if all_consulting:
            penalty += 0.3

        # --- 3. Only recent LLM experience (LangChain-era, no pre-2023 ML) ---
        current_title = profile.get("current_title", "").lower()
        yoe = profile.get("years_of_experience", 0)
        
        llm_era_only_skills = {"langchain", "llamaindex", "llama index", "prompt engineering",
                               "chatgpt", "gpt-4", "openai api", "anthropic", "claude"}
        pre_llm_skills = {"scikit-learn", "sklearn", "tensorflow", "pytorch", "xgboost",
                          "random forest", "svm", "regression", "classification",
                          "word2vec", "glove", "fasttext", "spacy", "nltk"}
        
        # Check if candidate's skill set is dominated by LLM-era tools
        candidate_skills = set()
        for role in career_history:
            for s_name in [role.get("title", "")]:
                candidate_skills.add(s_name.lower())

        has_llm_era = any(kw in " ".join(candidate_skills) for kw in llm_era_only_skills)
        has_pre_llm = any(kw in " ".join(candidate_skills) for kw in pre_llm_skills)
        
        if has_llm_era and not has_pre_llm and yoe < 2:
            penalty += 0.25

        # --- 4. No recent coding — moved into management/architecture ---
        management_titles = {"engineering manager", "vp engineering", "vp of engineering",
                           "director of engineering", "cto", "chief technology officer",
                           "head of engineering", "principal architect"}
        
        if current_title in management_titles:
            # Check if recent role descriptions mention coding
            recent_role = career_history[0] if career_history else {}
            recent_desc = recent_role.get("description", "").lower()
            coding_keywords = {"code", "implement", "develop", "build", "python", "wrote", "programming"}
            
            if not any(kw in recent_desc for kw in coding_keywords):
                penalty += 0.15

        # --- 5. Primarily CV/speech/robotics without NLP/IR ---
        cv_robotics_skills = {"computer vision", "opencv", "image classification", "object detection",
                             "speech recognition", "asr", "tts", "text-to-speech",
                             "robotics", "ros", "slam", "autonomous", "lidar"}
        nlp_ir_skills = {"nlp", "natural language", "information retrieval", "search",
                        "retrieval", "ranking", "text mining", "text classification", 
                        "sentiment", "ner", "named entity"}
        
        all_skill_text = " ".join(s.get("name", "").lower() for s in profile.get("skills", []) 
                                  if isinstance(s, dict))
        for role in career_history:
            all_skill_text += " " + role.get("description", "").lower()
        
        has_cv_robotics = any(kw in all_skill_text for kw in cv_robotics_skills)
        has_nlp_ir = any(kw in all_skill_text for kw in nlp_ir_skills)
        
        if has_cv_robotics and not has_nlp_ir:
            penalty += 0.2

        # --- 6. Title-chaser / frequent job hopping ---
        if len(career_history) >= 3:
            durations = [r.get("duration_months", 24) for r in career_history]
            avg_tenure = sum(durations) / len(durations)
            # JD: "switching companies every 1.5 years"
            if avg_tenure < 18 and len(career_history) >= 4:
                penalty += 0.15

        return min(penalty, 1.0)

    # =========================================================================
    # Skills Match Score (Weight: 0.20)
    # =========================================================================
    def _score_skills_match(self, candidate: dict) -> float:
        skills = candidate.get("skills", [])
        signals = candidate.get("redrob_signals", {})

        if not skills:
            return 0.0

        # Build skill lookup
        skill_map = {}
        for skill in skills:
            name = skill.get("name", "").lower().strip()
            skill_map[name] = skill

        # 1. Must-have skills matching (50% of skills score)
        must_have_score = self._match_skill_tier(skill_map, config.MUST_HAVE_SKILLS)

        # 2. Nice-to-have skills (30%)
        nice_score = self._match_skill_tier(skill_map, config.NICE_TO_HAVE_SKILLS)

        # 3. Domain-adjacent skills (20%)
        adjacent_score = self._match_skill_tier(skill_map, config.DOMAIN_ADJACENT_SKILLS)

        # Combined match score
        match_score = must_have_score * 0.50 + nice_score * 0.30 + adjacent_score * 0.20

        # 4. Skill credibility cross-check
        credibility = self._compute_skill_credibility(skills)

        # 5. Skill assessment scores (if available)
        assessment_boost = self._compute_assessment_boost(signals)

        if assessment_boost > 0:
            # Verified assessments are highly reliable
            final_score = match_score * 0.5 + credibility * 0.2 + assessment_boost * 0.3
        else:
            final_score = match_score * 0.7 + credibility * 0.3

        # Skill–career coherence gate (the JD's central trap). "A candidate with
        # all the AI keywords as skills but whose title is Marketing Manager is
        # not a fit." Listed skills only count if the candidate's titles/work
        # support them; incoherent profiles get their skill score discounted.
        final_score *= self._skill_career_coherence(candidate)

        return min(max(final_score, 0.0), 1.0)

    def _skill_career_coherence(self, candidate: dict) -> float:
        """Multiplier in [0.25, 1.0]: how well the candidate's titles/career
        support their claimed skills. Padded skill lists on a non-tech career
        are heavily discounted."""
        profile = candidate.get("profile", {})
        career = candidate.get("career_history", [])

        current_title = profile.get("current_title", "").lower().strip()
        titles = [current_title] + [r.get("title", "").lower().strip() for r in career]

        tech_title_kw = ("engineer", "scientist", "developer", "ml", "ai",
                         "data", "research", "architect", "analyst", "programmer",
                         "nlp", "machine learning", "software")
        # "engineer" alone is too broad — mechanical/civil/etc. are not tech.
        non_sw_eng = ("mechanical", "civil", "electrical", "chemical",
                      "industrial", "structural", "hardware", "manufacturing")

        def _is_tech_title(t):
            if not t or any(d in t for d in non_sw_eng):
                return False
            return any(kw in t for kw in tech_title_kw)

        has_tech_title = any(_is_tech_title(t) for t in titles)

        strong_tech_kw = ("ml engineer", "ai engineer", "machine learning",
                          "data scientist", "research engineer", "applied scientist",
                          "software engineer", "backend engineer", "nlp engineer",
                          "deep learning", "research scientist")
        has_strong_tech_title = any(
            any(kw in t for kw in strong_tech_kw) and not any(d in t for d in non_sw_eng)
            for t in titles if t)

        # The summary is the most reliable coherent field: does the candidate
        # *describe* ML/AI/retrieval work, not just list it as a skill?
        text = profile.get("summary", "").lower() + " " + profile.get("headline", "").lower()
        for r in career[:3]:
            text += " " + r.get("description", "").lower()
        work_kw = ("machine learning", "ml model", "deep learning", "nlp",
                   "retrieval", "ranking", "recommendation", "embedding",
                   "model", "data pipeline", "neural", "classifier", "forecasting",
                   "predictive", "data science", "applied ml")
        describes_ml_work = any(kw in text for kw in work_kw)

        if has_strong_tech_title and describes_ml_work:
            return 1.0
        if has_strong_tech_title:
            return 0.9
        if has_tech_title and describes_ml_work:
            return 0.8
        if has_tech_title:
            return 0.6
        if describes_ml_work:
            return 0.5
        # Non-tech title, no ML work evidence: skills are almost certainly padding.
        return 0.25

    # Synonym mapping for common skill aliases
    SKILL_SYNONYMS = {
        "machine learning": {"ml"},
        "natural language processing": {"nlp"},
        "deep learning": {"dl"},
        "artificial intelligence": {"ai"},
        "scikit-learn": {"sklearn", "scikit learn"},
        "sentence-transformers": {"sbert", "sentence transformers"},
        "large language models": {"llm", "llms"},
        "information retrieval": {"ir"},
        "hugging face transformers": {"huggingface", "hf transformers"},
        "recommendation systems": {"recsys", "recommender systems"},
        "a/b testing": {"ab testing", "split testing"},
        "tensorflow": {"tf"},
        "pytorch": {"torch"},
        "elasticsearch": {"elastic search", "elastic"},
        "opensearch": {"open search"},
    }

    def _match_skill_tier(self, skill_map: dict, target_skills: set) -> float:
        """Compute match score against target skills with word-boundary and synonym awareness."""
        if not target_skills:
            return 0.0

        matched = 0
        for target in target_skills:
            target_lower = target.lower().strip()
            # 1. Exact match
            if target_lower in skill_map:
                matched += 1
                continue

            # 2. Synonym match
            synonym_matched = False
            for canonical, aliases in self.SKILL_SYNONYMS.items():
                if target_lower == canonical or target_lower in aliases:
                    # Check if any synonym or canonical form is in skill_map
                    all_forms = aliases | {canonical}
                    if any(form in skill_map for form in all_forms):
                        matched += 1
                        synonym_matched = True
                        break
            if synonym_matched:
                continue

            # 3. Word-boundary-aware fuzzy match (prevents "ml" matching "html")
            for skill_name in skill_map:
                if self._word_boundary_match(target_lower, skill_name):
                    matched += 1
                    break

        # Threshold: need 50% match for perfect score (was 30%, too generous)
        return min(matched / max(len(target_skills) * 0.5, 1), 1.0)

    @staticmethod
    def _word_boundary_match(target: str, skill_name: str) -> bool:
        """Check if target matches skill_name with word boundary awareness.
        Prevents 'ml' from matching 'html', but allows 'ml' to match 'applied ml'.
        """
        import re
        # Target as whole word in skill_name
        if re.search(r'\b' + re.escape(target) + r'\b', skill_name):
            return True
        # Skill_name as whole word in target
        if re.search(r'\b' + re.escape(skill_name) + r'\b', target):
            return True
        return False

    def _compute_skill_credibility(self, skills: list[dict]) -> float:
        """Cross-check proficiency × endorsements × duration for credibility."""
        if not skills:
            return 0.0

        proficiency_weights = {
            "beginner": 0.2, "intermediate": 0.5, "advanced": 0.8, "expert": 1.0
        }

        total_credibility = 0.0
        count = 0

        for skill in skills:
            prof = skill.get("proficiency", "beginner")
            endorsements = skill.get("endorsements", 0)
            duration = skill.get("duration_months", 0)

            prof_weight = proficiency_weights.get(prof, 0.2)
            # Credibility: high proficiency should correlate with endorsements and duration
            endorsement_signal = min(math.log1p(endorsements) / 4, 1.0)
            duration_signal = min(math.sqrt(duration) / 8, 1.0)

            credibility = prof_weight * 0.4 + endorsement_signal * 0.3 + duration_signal * 0.3
            total_credibility += credibility
            count += 1

        return total_credibility / count if count > 0 else 0.0

    def _compute_assessment_boost(self, signals: dict) -> float:
        """Compute boost from Redrob skill assessment scores."""
        assessments = signals.get("skill_assessment_scores", {})
        if not assessments:
            return 0.0

        # Only count assessments for relevant skills
        relevant_scores = []
        for skill_name, score in assessments.items():
            skill_lower = skill_name.lower()
            # Check if this assessed skill is relevant to the JD
            is_relevant = any(
                kw in skill_lower
                for kw in {"python", "ml", "machine learning", "nlp", "deep learning",
                          "ai", "data", "algorithm", "ranking", "retrieval",
                          "embedding", "search", "neural", "statistics"}
            )
            if is_relevant:
                relevant_scores.append(score)

        if not relevant_scores:
            # Even irrelevant assessment scores show engagement
            all_scores = list(assessments.values())
            if all_scores:
                return min(sum(all_scores) / (len(all_scores) * 100), 1.0) * 0.3
            return 0.0

        # Average of relevant assessment scores
        return min(sum(relevant_scores) / (len(relevant_scores) * 100), 1.0)

    # =========================================================================
    # Experience Fit Score (Weight: 0.10)
    # =========================================================================
    def _score_experience_fit(self, candidate: dict) -> float:
        yoe = candidate.get("profile", {}).get("years_of_experience", 0)

        for min_y, max_y, score in config.EXPERIENCE_BANDS:
            if min_y <= yoe <= max_y:
                return score

        return 0.15  # Outside all bands

    # =========================================================================
    # Location & Logistics Score (Weight: 0.10)
    # =========================================================================
    def _score_location_logistics(self, candidate: dict) -> float:
        profile = candidate.get("profile", {})
        signals = candidate.get("redrob_signals", {})

        location = profile.get("location", "").lower()
        country = profile.get("country", "").lower()
        willing_to_relocate = signals.get("willing_to_relocate", False)
        notice_days = signals.get("notice_period_days", 90)
        salary_range = signals.get("expected_salary_range_inr_lpa", {})
        work_mode = signals.get("preferred_work_mode", "")

        # Base location score
        is_india = country in config.INDIA_COUNTRY or "india" in location

        if is_india:
            # Check preferred cities
            in_preferred = any(city in location for city in config.PREFERRED_CITIES)
            in_good = any(city in location for city in config.GOOD_CITIES)

            if in_preferred:
                location_score = 1.0
            elif in_good and willing_to_relocate:
                location_score = 0.9
            elif in_good:
                location_score = 0.7
            elif willing_to_relocate:
                location_score = 0.7
            else:
                location_score = 0.4
        else:
            if willing_to_relocate:
                location_score = 0.25
            else:
                location_score = 0.1

        # Notice period modifier
        if notice_days <= 30:
            notice_mod = 0.10
        elif notice_days <= 60:
            notice_mod = 0.0
        elif notice_days <= 90:
            notice_mod = -0.05
        elif notice_days <= 120:
            notice_mod = -0.10
        else:
            notice_mod = -0.15

        # Salary range fit
        salary_max = salary_range.get("max", 30)
        if salary_max <= config.SALARY_IDEAL_MAX_LPA:
            salary_fit = 1.0
        elif salary_max <= config.SALARY_STRETCH_MAX_LPA:
            salary_fit = 0.7
        else:
            salary_fit = 0.4

        # Work mode compatibility (hybrid preferred)
        work_mode_scores = {
            "hybrid": 1.0, "flexible": 0.9, "onsite": 0.8, "remote": 0.5
        }
        work_mode_score = work_mode_scores.get(work_mode, 0.6)

        # Combined
        final = (
            location_score * 0.50
            + salary_fit * 0.20
            + work_mode_score * 0.10
            + notice_mod
        )

        return min(max(final, 0.0), 1.0)

    # =========================================================================
    # Education Score (Weight: 0.05)
    # =========================================================================
    def _score_education(self, candidate: dict) -> float:
        education = candidate.get("education", [])
        certifications = candidate.get("certifications", [])
        career_history = candidate.get("career_history", [])

        if not education and not certifications:
            return 0.2  # Baseline — JD says skills > pedigree

        best_score = 0.2

        relevant_fields = {
            "computer science", "cs", "machine learning", "artificial intelligence",
            "data science", "information technology", "it",
            "mathematics", "statistics", "applied mathematics",
            "electronics", "electrical engineering",
            "computational linguistics",
        }

        for edu in education:
            score = 0.2  # baseline
            field = edu.get("field_of_study", "").lower()
            tier = edu.get("tier", "unknown")
            degree = edu.get("degree", "").lower()

            # Field relevance
            if any(rf in field for rf in relevant_fields):
                score += 0.3

            # Institution tier
            tier_bonuses = {"tier_1": 0.3, "tier_2": 0.2, "tier_3": 0.1, "tier_4": 0.0}
            score += tier_bonuses.get(tier, 0.0)

            # Advanced degree
            if any(d in degree for d in ["m.tech", "mtech", "ph.d", "phd", "m.s", "m.sc"]):
                score += 0.15

            best_score = max(best_score, min(score, 1.0))

        # Certification boost — relevant AI/ML certs add credibility
        if certifications:
            relevant_cert_keywords = {
                "machine learning", "ml", "deep learning", "ai", "data science",
                "aws machine learning", "google cloud ml", "tensorflow",
                "pytorch", "nlp", "natural language", "deeplearning.ai",
            }
            cert_boost = 0.0
            for cert in certifications:
                cert_name = cert.get("name", "").lower()
                if any(kw in cert_name for kw in relevant_cert_keywords):
                    cert_boost += 0.05  # Each relevant cert adds a small boost
            best_score = min(best_score + min(cert_boost, 0.15), 1.0)  # Cap at +0.15

        # Company size culture-fit signal (JD: Series A startup, 1-50 people)
        # Candidates from smaller companies are more likely culture fits
        startup_experience = False
        for role in career_history:
            size = role.get("company_size", "")
            if size in {"1-10", "11-50", "51-200"}:
                startup_experience = True
                break
        
        if startup_experience:
            best_score = min(best_score + 0.05, 1.0)

        return best_score

    # =========================================================================
    # Semantic Similarity Score (Weight: 0.25)
    # =========================================================================
    def _score_semantic_similarity(self, candidate: dict) -> float:
        """
        Compute semantic similarity between candidate profile and JD.
        Uses pre-computed embeddings for speed.
        """
        cid = candidate.get("candidate_id", "")

        if self.jd_embedding is None or cid not in self.candidate_embeddings:
            # Fallback: keyword-based similarity if embeddings not available
            return self._fallback_semantic_score(candidate)

        candidate_data = self.candidate_embeddings[cid]

        # Profile embedding similarity
        profile_emb = candidate_data.get("profile_embedding")
        if profile_emb is not None:
            profile_sim = self._cosine_similarity(self.jd_embedding, profile_emb)
        else:
            profile_sim = 0.0

        # Per-role description similarities (recency-weighted)
        role_embeddings = candidate_data.get("role_embeddings")
        if role_embeddings is not None and len(role_embeddings) > 0:
            role_sims = []
            for i, role_emb in enumerate(role_embeddings):
                sim = self._cosine_similarity(self.jd_embedding, role_emb)
                recency_weight = 1.0 / (1 + i * 0.4)  # Decay for older roles
                role_sims.append(sim * recency_weight)

            # Use max weighted similarity (best matching role)
            best_role_sim = max(role_sims) if role_sims else 0.0
        else:
            best_role_sim = 0.0

        # Combine: role descriptions matter more than general profile
        semantic_score = best_role_sim * 0.6 + profile_sim * 0.4

        # Normalize to 0-1 (cosine sim can be negative)
        semantic_score = max(0.0, min(semantic_score, 1.0))

        return semantic_score

    def _fallback_semantic_score(self, candidate: dict) -> float:
        """Keyword-based fallback when embeddings aren't available."""
        profile = candidate.get("profile", {})
        career_history = candidate.get("career_history", [])

        # Concatenate all text
        text = " ".join([
            profile.get("headline", ""),
            profile.get("summary", ""),
        ] + [
            role.get("description", "") for role in career_history
        ]).lower()

        # Key concepts from JD
        concepts = {
            "ranking": 2, "retrieval": 2, "search": 1.5, "recommendation": 2,
            "embedding": 2, "vector": 2, "hybrid": 1.5,
            "production": 1.5, "deployed": 1.5, "scale": 1,
            "evaluation": 1.5, "a/b test": 2, "ndcg": 2,
            "recruiter": 1.5, "candidate": 1, "matching": 1.5,
            "nlp": 1, "information retrieval": 2,
            "bm25": 2, "fine-tun": 1.5, "llm": 1, "transformer": 1,
        }

        score = 0
        for concept, weight in concepts.items():
            if concept in text:
                score += weight

        max_score = sum(concepts.values()) * 0.25
        return min(score / max_score if max_score > 0 else 0, 1.0)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
