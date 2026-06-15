"""
Stage 1: Hard Filters.
Fast, generous elimination of obviously unfit candidates.
Reduces 100K → ~5K-15K candidates.
"""

import config


def passes_hard_filters(candidate: dict) -> bool:
    """
    Return True if candidate passes all hard filters.
    Filters are deliberately generous to avoid eliminating hidden gems.
    """
    profile = candidate.get("profile", {})
    career_history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})

    # Filter 1: Experience band (generous: 2-25 years)
    yoe = profile.get("years_of_experience", 0)
    if yoe < config.MIN_EXPERIENCE_YEARS or yoe > config.MAX_EXPERIENCE_YEARS:
        return False

    # Filter 2: Non-tech career check
    # Eliminate candidate if current title is non-tech AND they have never held a tech title
    current_title = profile.get("current_title", "").lower().strip()
    is_current_non_tech = current_title in config.NON_TECH_TITLES

    if is_current_non_tech:
        has_any_tech_role = False
        for role in career_history:
            role_title = role.get("title", "").lower().strip()
            if role_title not in config.NON_TECH_TITLES:
                has_any_tech_role = True
                break
        
        if not has_any_tech_role:
            return False

    # Filter 3: Zero skill relevance
    # Only eliminate if truly zero overlap with broad domain list
    # AND no tech in career descriptions
    skill_names = {s.get("name", "").lower().strip() for s in skills}
    has_skill_overlap = bool(skill_names & config.RELEVANT_SKILL_DOMAINS)

    if not has_skill_overlap:
        # Check career descriptions as fallback
        has_tech_career = _has_tech_career_descriptions(career_history)
        if not has_tech_career:
            return False

    return True


def _has_any_relevant_skill(skills: list[dict]) -> bool:
    """Check if candidate has any relevant skill."""
    for skill in skills:
        name = skill.get("name", "").lower().strip()
        if name in config.RELEVANT_SKILL_DOMAINS:
            return True
        # Fuzzy match: check if any domain keyword is IN the skill name
        for domain in config.RELEVANT_SKILL_DOMAINS:
            if domain in name or name in domain:
                return True
    return False


def _has_tech_career_descriptions(career_history: list[dict]) -> bool:
    """Check if any career description mentions tech/ML concepts."""
    tech_keywords = {
        "machine learning", "ml", "ai", "artificial intelligence",
        "data science", "deep learning", "neural", "model",
        "python", "algorithm", "software", "engineering",
        "pipeline", "api", "backend", "production", "deployment",
        "embedding", "vector", "search", "ranking", "retrieval",
        "recommendation", "nlp", "natural language",
    }
    for role in career_history:
        desc = role.get("description", "").lower()
        if any(kw in desc for kw in tech_keywords):
            return True
    return False


def apply_hard_filters(candidates: list[dict]) -> list[dict]:
    """Apply hard filters to a list of candidates, returning those that pass."""
    passed = []
    filtered_count = 0
    for candidate in candidates:
        if passes_hard_filters(candidate):
            passed.append(candidate)
        else:
            filtered_count += 1
    
    print(f"  Hard filters: {len(passed)} passed, {filtered_count} filtered out")
    return passed
