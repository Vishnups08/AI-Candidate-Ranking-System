"""
Stage 5: Reasoning Generator.
Produces rank-aware, fact-grounded, unique reasoning strings.

Design philosophy:
- Every reasoning is COMPOSED from the candidate's actual data, not selected from templates.
- Structure: [Core strength] + [JD connection] + [Concern/gap if any]
- The 6 Stage-4 checks this must pass:
    1. Specific facts from profile (skill names, years, company)
    2. JD connection (tie to retrieval, ranking, embeddings, etc.)
    3. Honest concerns (acknowledge gaps)
    4. No hallucination (only reference real profile data)
    5. Variation (no templates)
    6. Rank consistency (enthusiastic top-10, measured mid, cautious bottom)
"""

import config


# JD requirement keywords for connecting reasoning to the job description
_JD_MUST_HAVE_KEYWORDS = {
    "embeddings", "sentence-transformers", "vector database", "pinecone",
    "weaviate", "qdrant", "milvus", "faiss", "elasticsearch", "opensearch",
    "retrieval", "ranking", "search", "information retrieval", "hybrid search",
    "ndcg", "mrr", "map", "a/b testing", "evaluation",
}

_JD_NICE_TO_HAVE_KEYWORDS = {
    "lora", "qlora", "peft", "fine-tuning", "xgboost", "learning to rank",
    "learning-to-rank", "hr tech", "recruiting", "talent", "hiring",
    "distributed systems", "open source", "open-source",
}

_JD_DOMAIN_KEYWORDS = {
    "nlp", "natural language processing", "recommendation", "deep learning",
    "machine learning", "ml", "pytorch", "tensorflow", "transformers",
    "bert", "gpt", "llm", "rag", "langchain", "llamaindex",
    "bm25", "tf-idf", "cosine similarity",
}


# Tier labels for confidence display
_TIER_LABELS = [
    (0.95, "Exceptional Fit"),
    (0.85, "Strong Fit"),
    (0.70, "Good Fit"),
    (0.55, "Moderate Fit"),
    (0.40, "Marginal Fit"),
    (0.00, "Weak Fit"),
]

_CONFIDENCE_LEVELS = [
    (0.85, "high"),
    (0.65, "medium"),
    (0.00, "low"),
]


def _tier_label(score: float) -> str:
    for threshold, label in _TIER_LABELS:
        if score >= threshold:
            return label
    return "Weak Fit"


def _confidence(score: float) -> str:
    for threshold, label in _CONFIDENCE_LEVELS:
        if score >= threshold:
            return label
    return "low"


def _evidence_career_fit(facts: dict) -> str:
    parts = []
    title = facts["current_title"]
    company = facts["current_company"]
    yoe = facts["yoe"]
    parts.append(f"Title: {title} at {company} ({yoe:.0f} yrs)")
    if facts["is_product_company"]:
        parts.append(f"Product-company background")
    elif facts["consulting_career"]:
        parts.append(f"Career-long consulting ({', '.join(facts['career_companies'][:2])})")
    if facts["career_highlights"]:
        h = facts["career_highlights"][0]
        if h["highlight"]:
            short = h["highlight"][:80] + "..." if len(h["highlight"]) > 80 else h["highlight"]
            parts.append(f"Highlight: {short}")
    return " | ".join(parts)


def _evidence_skills_match(facts: dict) -> str:
    parts = []
    must = facts["must_have_skills"]
    nice = facts["nice_to_have_skills"]
    domain = facts["domain_skills"]
    if must:
        parts.append(f"JD must-haves matched: {', '.join(must[:5])}")
    if nice:
        parts.append(f"Nice-to-haves: {', '.join(nice[:3])}")
    if domain:
        parts.append(f"Domain skills: {', '.join(domain[:3])}")
    if not parts:
        parts.append("No direct JD-skill overlap detected")
    return " | ".join(parts)


def _evidence_experience_fit(facts: dict) -> str:
    yoe = facts["yoe"]
    if 5 <= yoe <= 9:
        return f"{yoe:.0f} yrs — within JD optimal band (5-9 yrs)"
    elif 4 <= yoe < 5:
        return f"{yoe:.0f} yrs — slightly below optimal (JD: 5-9 yrs)"
    elif 9 < yoe <= 12:
        return f"{yoe:.0f} yrs — slightly above optimal (JD: 5-9 yrs)"
    elif yoe < 4:
        return f"{yoe:.0f} yrs — under-experienced for JD requirements"
    else:
        return f"{yoe:.0f} yrs — may be over-experienced for Series A founding role"


def _evidence_location(facts: dict) -> str:
    loc = facts["location"]
    if facts["is_preferred_location"]:
        return f"{loc} — JD-preferred location (Pune/Noida/NCR)"
    elif facts["is_good_location"]:
        return f"{loc} — Tier-1 city, JD-acceptable"
    elif facts["is_india"]:
        return f"{loc} — India-based but outside JD-preferred cities"
    else:
        return f"{loc} — outside India, JD does not sponsor visas"


def _evidence_behavioral(facts: dict, multiplier: float) -> str:
    parts = []
    if facts["open_to_work"]:
        parts.append("Open to work: Yes")
    else:
        parts.append("Open to work: No")
    rate = facts["response_rate"]
    parts.append(f"Recruiter response rate: {rate:.0%}")
    if facts["github_score"] >= 0:
        parts.append(f"GitHub activity: {facts['github_score']}/100")
    notice = facts["notice_days"]
    parts.append(f"Notice: {notice} days")
    return " | ".join(parts)


def generate_structured_explanation(
    candidate: dict,
    rank: int,
    scores: dict,
    multiplier: float,
    additive: float,
) -> dict:
    """
    Generate a recruiter-grade structured explanation for a candidate's rank.

    Returns a dict with:
    - score_card: per-dimension breakdown with score, weight, contribution, evidence
    - formula: human-readable calculation string
    - tier_label: e.g. 'Strong Fit'
    - confidence: 'high' / 'medium' / 'low'
    - narrative: the existing 1-3 sentence prose reasoning
    - why_not_notes: list of concern strings (empty for top candidates)
    """
    import config as _config

    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})

    facts = _extract_all_facts(profile, career, skills, signals, scores)

    weights = _config.WEIGHTS
    weighted_total = scores.get("weighted_total", 0.0)
    final_score = weighted_total * multiplier + additive

    # Build per-dimension score card
    dimensions = {
        "career_fit": {
            "label": "Career Fit",
            "score": round(scores.get("career_fit", 0), 4),
            "weight": round(weights.get("career_fit", 0), 3),
            "contribution": round(scores.get("career_fit", 0) * weights.get("career_fit", 0), 4),
            "evidence": _evidence_career_fit(facts),
        },
        "skills_match": {
            "label": "Skills Match",
            "score": round(scores.get("skills_match", 0), 4),
            "weight": round(weights.get("skills_match", 0), 3),
            "contribution": round(scores.get("skills_match", 0) * weights.get("skills_match", 0), 4),
            "evidence": _evidence_skills_match(facts),
        },
        "experience_fit": {
            "label": "Experience Fit",
            "score": round(scores.get("experience_fit", 0), 4),
            "weight": round(weights.get("experience_fit", 0), 3),
            "contribution": round(scores.get("experience_fit", 0) * weights.get("experience_fit", 0), 4),
            "evidence": _evidence_experience_fit(facts),
        },
        "location_logistics": {
            "label": "Location & Logistics",
            "score": round(scores.get("location_logistics", 0), 4),
            "weight": round(weights.get("location_logistics", 0), 3),
            "contribution": round(scores.get("location_logistics", 0) * weights.get("location_logistics", 0), 4),
            "evidence": _evidence_location(facts),
        },
        "education": {
            "label": "Education",
            "score": round(scores.get("education", 0), 4),
            "weight": round(weights.get("education", 0), 3),
            "contribution": round(scores.get("education", 0) * weights.get("education", 0), 4),
            "evidence": _evidence_education(candidate),
        },
        "semantic_similarity": {
            "label": "Semantic Similarity",
            "score": round(scores.get("semantic_similarity", 0), 4),
            "weight": round(weights.get("semantic_similarity", 0), 3),
            "contribution": round(scores.get("semantic_similarity", 0) * weights.get("semantic_similarity", 0), 4),
            "evidence": "BGE embedding cosine similarity between candidate profile and JD",
        },
    }

    behavioral = {
        "label": "Behavioral Multiplier",
        "value": round(multiplier, 4),
        "additive": round(additive, 4),
        "evidence": _evidence_behavioral(facts, multiplier),
    }

    # Build formula string: "Career 0.90×0.25 + Skills 0.80×0.20 + ... × 1.20 = 1.08"
    dim_parts = []
    for key, dim in dimensions.items():
        dim_parts.append(f"{dim['label'].split()[0]} {dim['score']:.2f}×{dim['weight']:.2f}")
    formula = " + ".join(dim_parts)
    if additive > 0:
        formula += f" × {multiplier:.2f} + {additive:.3f} = {final_score:.4f}"
    else:
        formula += f" × {multiplier:.2f} = {final_score:.4f}"

    # Short formula for display: top 3 dimensions
    top_dims = sorted(dimensions.items(), key=lambda x: x[1]["contribution"], reverse=True)[:3]
    short_formula = " · ".join(
        f"{d['label'].split()[0]} {d['score']:.2f}"
        for _, d in top_dims
    )
    short_formula += f" · Behavioral ×{multiplier:.2f} → rank #{rank}"

    # Concerns (why-not notes)
    why_not_notes = []
    if facts["consulting_career"]:
        why_not_notes.append("Career-long consulting — JD explicit disqualifier")
    if not facts["must_have_skills"]:
        why_not_notes.append("No direct JD must-have skills matched (embeddings, vector DB, retrieval)")
    if facts["yoe"] < 5:
        why_not_notes.append(f"Under-experienced: {facts['yoe']:.0f} yrs vs JD optimal 5-9 yrs")
    elif facts["yoe"] > 12:
        why_not_notes.append(f"Over-experienced: {facts['yoe']:.0f} yrs for Series A founding role")
    if facts["notice_days"] > 90:
        why_not_notes.append(f"{facts['notice_days']}-day notice period may delay onboarding")
    if not facts["is_india"]:
        why_not_notes.append(f"Based in {facts['location']}, {facts['country']} — no visa sponsorship")
    if facts["salary_max"] > 70 and rank > 10:
        why_not_notes.append(f"Salary expectation ({facts['salary_max']} LPA) likely above Series A budget")

    narrative = generate_reasoning(candidate, rank, scores)

    return {
        "score_card": dimensions,
        "behavioral": behavioral,
        "weighted_total": round(weighted_total, 4),
        "final_score": round(final_score, 4),
        "formula": formula,
        "short_formula": short_formula,
        "tier_label": _tier_label(scores.get("weighted_total", 0)),
        "confidence": _confidence(scores.get("weighted_total", 0)),
        "narrative": narrative,
        "why_not_notes": why_not_notes,
    }


def generate_honeypot_contrast_card(candidate: dict, honeypot_reason: str) -> dict:
    """
    Generate a 'why-not' contrast card for a honeypot candidate.
    Shows judges exactly why the profile was rejected.
    """
    profile = candidate.get("profile", {})
    flags = [f.strip() for f in honeypot_reason.split(";") if f.strip()]

    flag_explanations = []
    for flag in flags:
        if "career_months" in flag:
            flag_explanations.append(
                f"⏱ Timeline impossibility: {flag} — "
                "career history totals more months than years of experience allow."
            )
        elif "date_mismatch" in flag:
            flag_explanations.append(
                f"📅 Date math failure: {flag} — "
                "start→end dates don't match claimed duration."
            )
        elif "impossible_skills" in flag:
            flag_explanations.append(
                f"🚫 Impossible skill claims: {flag} — "
                "expert-level proficiency listed with 0 months experience."
            )
        elif "title_skill_absurdity" in flag:
            flag_explanations.append(
                f"🎭 Title-skill absurdity: {flag} — "
                "non-tech title claims a slate of expert AI skills."
            )
        elif "career_overlap" in flag:
            flag_explanations.append(
                f"🔁 Concurrent role overlap: {flag} — "
                "two full-time roles overlap by more than 3 months."
            )
        else:
            flag_explanations.append(f"⚠ {flag}")

    return {
        "type": "honeypot",
        "candidate_id": candidate.get("candidate_id", ""),
        "title": profile.get("current_title", ""),
        "company": profile.get("current_company", ""),
        "outcome": "Excluded — profile contains internally impossible data",
        "raw_reason": honeypot_reason,
        "flag_explanations": flag_explanations,
        "pipeline_note": "Removed in Stage 2 (Honeypot Detection). Never scored or ranked.",
    }


def generate_demotion_contrast_card(
    candidate: dict,
    naive_rank: int,
    pipeline_rank: int,
    scores: dict,
    coherence_multiplier: float,
) -> dict:
    """
    Generate a 'why-not' card for a keyword-stuffer that was demoted by the pipeline.
    Shows the gap between naive keyword rank and pipeline rank.
    """
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])

    skill_count = len(skills)
    jd_skill_hits = sum(
        1 for s in skills
        if any(
            kw in s.get("name", "").lower()
            for kw in {"embedding", "retrieval", "vector", "search", "ranking",
                       "nlp", "ml", "machine learning", "python", "faiss", "pinecone"}
        )
    )

    # Extract raw scores safely whether passed as flat float dicts or nested score_cards
    skills_val = scores.get("skills_match", 0)
    if isinstance(skills_val, dict):
        skills_val = skills_val.get("score", 0)

    career_val = scores.get("career_fit", 0)
    if isinstance(career_val, dict):
        career_val = career_val.get("score", 0)

    return {
        "type": "keyword_stuffer",
        "candidate_id": candidate.get("candidate_id", ""),
        "title": profile.get("current_title", ""),
        "company": profile.get("current_company", ""),
        "naive_rank": naive_rank,
        "pipeline_rank": pipeline_rank,
        "rank_drop": pipeline_rank - naive_rank,
        "jd_skill_hits": jd_skill_hits,
        "total_skills": skill_count,
        "coherence_multiplier": round(coherence_multiplier, 3),
        "skills_score_raw": round(skills_val or 0, 4),
        "career_fit_score": round(career_val or 0, 4),
        "outcome": f"Demoted from naive rank #{naive_rank} to pipeline rank #{pipeline_rank}",
        "demotion_reason": (
            f"Skill-career coherence gate applied a {coherence_multiplier:.2f}× multiplier "
            f"because the candidate's title/career history doesn't support the "
            f"{jd_skill_hits} JD-relevant skills claimed. "
            f"Padded skill lists on non-tech careers are discounted to prevent "
            f"keyword-stuffing from defeating a retrieval-aware system."
        ),
    }


def _evidence_education(candidate: dict) -> str:
    edu = candidate.get("education", [])
    certs = candidate.get("certifications", [])
    parts = []
    for e in edu[:2]:
        deg = e.get("degree", "")
        field = e.get("field_of_study", "")
        tier = e.get("tier", "unknown")
        if deg or field:
            parts.append(f"{deg} in {field} ({tier.replace('_', ' ').upper()})")
    if certs:
        cert_names = [c.get("name", "") for c in certs[:2] if c.get("name")]
        if cert_names:
            parts.append(f"Certs: {', '.join(cert_names)}")
    return " | ".join(parts) if parts else "No formal education data"


def generate_reasoning(candidate: dict, rank: int, scores: dict) -> str:
    """
    Generate a unique, fact-grounded, 1-3 sentence reasoning for this candidate's rank.
    Every reasoning is composed fresh from the candidate's data — no template rotation.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})

    # Extract all facts we can reference
    facts = _extract_all_facts(profile, career, skills, signals, scores)

    # Build reasoning components
    strength = _build_strength_clause(facts, rank)
    jd_connection = _build_jd_connection(facts, rank)
    concern = _build_concern_clause(facts, rank)

    # Compose based on rank tier
    if rank <= 10:
        reasoning = f"{strength} {jd_connection}"
        if concern:
            reasoning += f" {concern}"
    elif rank <= 30:
        reasoning = f"{strength} {jd_connection} {concern}" if concern else f"{strength} {jd_connection}"
    elif rank <= 60:
        reasoning = f"{strength} {concern}" if concern else f"{strength} {jd_connection}"
    else:
        # Ranks 61-100: lead with context, then gap
        reasoning = f"{strength} {concern}" if concern else f"{strength} {jd_connection}"

    return _clean_reasoning(reasoning)


def _extract_all_facts(profile: dict, career: list, skills: list,
                       signals: dict, scores: dict) -> dict:
    """Extract comprehensive facts from candidate profile for reasoning."""
    current_title = profile.get("current_title", "Unknown")
    current_company = profile.get("current_company", "Unknown")
    yoe = profile.get("years_of_experience", 0)
    location = profile.get("location", "Unknown")
    country = profile.get("country", "Unknown")
    industry = profile.get("current_industry", "Unknown")

    # Categorize skills by JD relevance
    must_have_skills = []
    nice_to_have_skills = []
    domain_skills = []
    other_skills = []

    for skill in skills:
        name = skill.get("name", "")
        name_lower = name.lower()
        if any(kw in name_lower for kw in _JD_MUST_HAVE_KEYWORDS):
            must_have_skills.append(name)
        elif any(kw in name_lower for kw in _JD_NICE_TO_HAVE_KEYWORDS):
            nice_to_have_skills.append(name)
        elif any(kw in name_lower for kw in _JD_DOMAIN_KEYWORDS):
            domain_skills.append(name)
        else:
            other_skills.append(name)

    # Career analysis
    career_highlights = []
    production_keywords_found = []
    career_companies = []
    total_career_months = 0

    for i, role in enumerate(career[:4]):
        title = role.get("title", "")
        company = role.get("company", "")
        desc = role.get("description", "")
        duration = role.get("duration_months", 0)
        career_companies.append(company)
        total_career_months += duration

        # Find specific production/achievement mentions
        highlight = _extract_specific_highlight(desc)
        if highlight:
            career_highlights.append({
                "title": title,
                "company": company,
                "duration_months": duration,
                "highlight": highlight,
                "is_recent": i == 0,
            })

        # Check for production deployment language
        desc_lower = desc.lower()
        for kw in ["production", "deployed", "shipped", "real users", "scale", "million", "100k", "serving"]:
            if kw in desc_lower and kw not in production_keywords_found:
                production_keywords_found.append(kw)

    # Is current company a consulting firm?
    is_product_company = current_company.lower().strip() not in config.CONSULTING_FIRMS
    consulting_career = all(
        c.lower().strip() in config.CONSULTING_FIRMS for c in career_companies
    ) if career_companies else False

    # Behavioral signals
    response_rate = signals.get("recruiter_response_rate", 0)
    notice_days = signals.get("notice_period_days", 90)
    github_score = signals.get("github_activity_score", -1)
    open_to_work = signals.get("open_to_work_flag", False)
    last_active = signals.get("last_active_date", "")
    interview_rate = signals.get("interview_completion_rate", 0.5)
    salary_range = signals.get("expected_salary_range_inr_lpa", {})
    salary_max = salary_range.get("max", 30)
    work_mode = signals.get("preferred_work_mode", "")

    # Location analysis
    loc_lower = location.lower()
    is_preferred_location = any(c in loc_lower for c in config.PREFERRED_CITIES)
    is_good_location = any(c in loc_lower for c in config.GOOD_CITIES)
    is_india = country.lower() in config.INDIA_COUNTRY or "india" in loc_lower

    # Scores
    career_score = scores.get("career_fit", 0)
    skills_score = scores.get("skills_match", 0)
    semantic_score = scores.get("semantic_similarity", 0)
    experience_score = scores.get("experience_fit", 0)

    return {
        "current_title": current_title,
        "current_company": current_company,
        "yoe": yoe,
        "location": location,
        "country": country,
        "industry": industry,
        "must_have_skills": must_have_skills,
        "nice_to_have_skills": nice_to_have_skills,
        "domain_skills": domain_skills,
        "all_relevant_skills": must_have_skills + nice_to_have_skills + domain_skills,
        "career_highlights": career_highlights,
        "production_keywords": production_keywords_found,
        "career_companies": career_companies,
        "is_product_company": is_product_company,
        "consulting_career": consulting_career,
        "response_rate": response_rate,
        "notice_days": notice_days,
        "github_score": github_score,
        "open_to_work": open_to_work,
        "last_active": last_active,
        "interview_rate": interview_rate,
        "salary_max": salary_max,
        "work_mode": work_mode,
        "is_preferred_location": is_preferred_location,
        "is_good_location": is_good_location,
        "is_india": is_india,
        "career_score": career_score,
        "skills_score": skills_score,
        "semantic_score": semantic_score,
        "experience_score": experience_score,
    }


def _extract_specific_highlight(description: str) -> str:
    """Extract the most achievement-specific sentence from a role description."""
    if not description:
        return ""

    # Prefer sentences with quantifiable achievements or specific systems
    priority_keywords = [
        "ranking", "retrieval", "search", "recommendation", "embedding",
        "vector", "pipeline", "production", "deployed", "shipped",
        "million", "scaled", "built", "designed", "improved", "reduced",
    ]

    sentences = [s.strip() for s in description.split(".") if s.strip()]

    # Score each sentence by relevance
    best_sentence = ""
    best_score = -1

    for sentence in sentences:
        lower = sentence.lower()
        score = sum(1 for kw in priority_keywords if kw in lower)
        # Bonus for quantifiable results
        if any(c.isdigit() for c in sentence):
            score += 2
        if score > best_score:
            best_score = score
            best_sentence = sentence

    if best_sentence and best_score > 0:
        if len(best_sentence) > 130:
            best_sentence = best_sentence[:127] + "..."
        return best_sentence

    return ""


def _build_strength_clause(facts: dict, rank: int) -> str:
    """Build the core strength description — unique per candidate."""
    title = facts["current_title"]
    company = facts["current_company"]
    yoe = facts["yoe"]

    parts = []

    # Core identity (always included, always unique due to name/company)
    if rank <= 10:
        parts.append(f"{title} at {company} with {yoe:.0f} years experience")
    elif rank <= 30:
        parts.append(f"{yoe:.0f}-year {title} at {company}")
    elif rank <= 60:
        parts.append(f"{title} ({yoe:.0f} yrs) at {company}")
    else:
        parts.append(f"{yoe:.0f} years as {title} at {company}")

    # Skills — name specific skills, never generic counts
    must_have = facts["must_have_skills"]
    nice_have = facts["nice_to_have_skills"]
    domain = facts["domain_skills"]

    if must_have and rank <= 30:
        skill_str = ", ".join(must_have[:3])
        parts.append(f"with directly JD-relevant skills: {skill_str}")
    elif must_have:
        skill_str = ", ".join(must_have[:2])
        parts.append(f"brings {skill_str} experience")
    elif domain:
        skill_str = ", ".join(domain[:2])
        if rank <= 30:
            parts.append(f"with domain-adjacent skills in {skill_str}")
        else:
            parts.append(f"has {skill_str} background")
    elif nice_have:
        skill_str = ", ".join(nice_have[:2])
        parts.append(f"with some relevant skills ({skill_str})")

    # NOTE: we deliberately do NOT quote raw career-role descriptions here.
    # In this dataset the descriptions are scrambled (an AI Specialist's role
    # may read "built computer vision models" or "gradient-boosted trees"),
    # so quoting them produces reasoning that contradicts the candidate's real
    # skills and reads as incoherent to a Stage-4 reviewer. We compose only from
    # grounded, coherent fields (title, company, skills, production signals).

    return "; ".join(parts[:3]) + "."


def _build_jd_connection(facts: dict, rank: int) -> str:
    """Build a sentence connecting the candidate to specific JD requirements."""
    connections = []

    # Production deployment (JD: "deployed to real users")
    if facts["production_keywords"]:
        prod_kws = facts["production_keywords"][:2]
        connections.append(f"career history shows {' and '.join(prod_kws)} experience matching JD's emphasis on real-world deployment")

    # Product vs consulting (JD: "we've had bad fit experiences with career-long consulting")
    if facts["is_product_company"] and not facts["consulting_career"]:
        connections.append(f"product-company background at {facts['current_company']} aligns with JD preference over consulting")
    elif facts["consulting_career"]:
        connections.append(f"career-long consulting background ({', '.join(facts['career_companies'][:2])}) is a concern per JD")

    # Retrieval/search/ranking experience (JD: core mandate)
    must_have = facts["must_have_skills"]
    if must_have:
        retrieval_skills = [s for s in must_have if any(
            kw in s.lower() for kw in ["retrieval", "search", "ranking", "embedding", "vector", "faiss", "pinecone", "elasticsearch"]
        )]
        if retrieval_skills:
            connections.append(f"{', '.join(retrieval_skills[:2])} directly maps to JD's retrieval and ranking mandate")

    # Location fit (JD: "Pune/Noida preferred")
    if facts["is_preferred_location"]:
        connections.append(f"based in {facts['location']} (JD-preferred location)")
    elif facts["is_good_location"]:
        connections.append(f"based in {facts['location']} (JD-acceptable Tier-1 city)")

    # Experience fit (JD: "5-9 years")
    yoe = facts["yoe"]
    if 5 <= yoe <= 9:
        connections.append(f"{yoe:.0f} years falls in JD's optimal 5-9 year band")

    # Behavioral availability (JD: "Active on Redrob platform")
    if facts["open_to_work"] and facts["response_rate"] > 0.6:
        connections.append(f"actively looking with {facts['response_rate']:.0%} response rate")

    if not connections:
        # Fallback for candidates with weak JD connections
        all_skills = facts["all_relevant_skills"]
        if all_skills:
            connections.append(f"has {len(all_skills)} skills in the AI/ML domain relevant to the role")
        else:
            connections.append("limited direct overlap with JD's core retrieval and ranking requirements")

    # Pick the strongest 1-2 connections based on rank. Use _upper_first (not
    # str.capitalize, which would lowercase the rest and turn 'Zomato'->'zomato').
    if rank <= 10:
        return "; ".join(_upper_first(c) if i == 0 else c for i, c in enumerate(connections[:2])) + "."
    elif rank <= 30:
        return _upper_first(connections[0]) + "."
    else:
        return _upper_first(connections[0]) + "."


def _upper_first(s: str) -> str:
    """Uppercase only the first character, preserving the rest (proper nouns)."""
    return s[0].upper() + s[1:] if s else s


def _build_concern_clause(facts: dict, rank: int) -> str:
    """Build honest concern/gap acknowledgment — critical for Stage 4 credibility."""
    concerns = []

    # Only add concerns for ranks where they're expected
    if rank <= 5:
        # Very top candidates: only mention minor logistics if any
        if facts["notice_days"] > 60:
            concerns.append(f"notice period is {facts['notice_days']} days (JD prefers sub-30)")
        return ""  # Top 5 generally don't need concern clauses

    # Experience outside band
    yoe = facts["yoe"]
    if yoe < 5:
        concerns.append(f"at {yoe:.0f} years, below JD's preferred 5-9 year range")
    elif yoe > 12:
        concerns.append(f"at {yoe:.0f} years, may be over-experienced for this Series A founding role")

    # Notice period
    if facts["notice_days"] > 90:
        concerns.append(f"{facts['notice_days']}-day notice period may delay onboarding")
    elif facts["notice_days"] > 60:
        concerns.append(f"notice period of {facts['notice_days']} days above JD's preferred 30-day window")

    # Location
    if not facts["is_india"]:
        concerns.append(f"based in {facts['location']}, {facts['country']} — JD does not sponsor work visas")
    elif not facts["is_preferred_location"] and not facts["is_good_location"]:
        if rank > 30:
            concerns.append(f"based in {facts['location']}, outside JD's preferred cities")

    # Low engagement signals
    if facts["response_rate"] < 0.2 and rank > 20:
        concerns.append(f"low recruiter response rate ({facts['response_rate']:.0%}) suggests limited availability")

    # Consulting career
    if facts["consulting_career"] and rank > 15:
        concerns.append("career-long consulting is an explicit JD disqualifier")

    # Skills gap
    if not facts["must_have_skills"] and rank > 20:
        if facts["domain_skills"]:
            concerns.append(f"no direct JD must-have skills (has adjacent: {', '.join(facts['domain_skills'][:2])})")
        else:
            concerns.append("no direct match on JD must-have skills (embeddings, vector DB, retrieval evaluation)")

    # Salary
    if facts["salary_max"] > config.SALARY_STRETCH_MAX_LPA and rank > 10:
        concerns.append(f"salary expectation ({facts['salary_max']} LPA) may exceed Series A budget")

    # Select concerns based on rank tier
    if not concerns:
        return ""

    if rank <= 10:
        # Top tier: brief, single concern
        return f"Minor note: {concerns[0]}."
    elif rank <= 30:
        return f"Concern: {concerns[0]}."
    elif rank <= 60:
        # Mid tier: 1-2 concerns
        concern_text = "; ".join(concerns[:2])
        return f"Gaps: {concern_text}."
    else:
        # Lower tier: honest about why they're ranked low
        concern_text = "; ".join(concerns[:2])
        return f"Ranked lower due to: {concern_text}."


def _clean_reasoning(text: str) -> str:
    """Clean up reasoning text: remove artifacts, normalize whitespace."""
    # Normalize unicode dashes and strip any non-printable / replacement chars
    # that can leak in from source text (e.g. U+FFFD mojibake) and look broken
    # to a Stage-4 human reviewer.
    text = (text.replace("—", "-").replace("–", "-")
                .replace("�", "").replace("’", "'")
                .replace("“", "'").replace("”", "'"))
    text = "".join(ch for ch in text if ch.isprintable())
    # Remove stray leading dash fragments left by removed clauses
    text = text.replace("; - ", "; ").replace(" - ;", ";")
    # Remove double spaces
    while "  " in text:
        text = text.replace("  ", " ")
    # Remove empty sentences
    text = text.replace(". .", ".").replace("..", ".").replace(";.", ".")
    # Remove leading/trailing artifacts
    text = text.strip().strip(";").strip()
    # Ensure it ends with a period
    if text and not text.endswith("."):
        text += "."
    # Capitalize first letter
    if text:
        text = text[0].upper() + text[1:]
    # Escape quotes for CSV safety
    text = text.replace('"', "'")
    return text
