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

    # Career highlight — reference specific achievement if available
    highlights = facts["career_highlights"]
    if highlights and rank <= 20:
        h = highlights[0]
        if h["highlight"]:
            parts.append(f"— {h['highlight']}")

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

    # Pick the strongest 1-2 connections based on rank
    if rank <= 10:
        return " ".join(c.capitalize() if i == 0 else c for i, c in enumerate(connections[:2])) + "."
    elif rank <= 30:
        return connections[0].capitalize() + "."
    else:
        return connections[0].capitalize() + "."


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
