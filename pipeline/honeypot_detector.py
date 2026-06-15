"""
Stage 2: Honeypot Detection.
Identifies candidates with subtly impossible profiles.
~80 honeypots exist in the dataset. >10% honeypot rate in top 100 = DQ.
Run BEFORE scoring so honeypots never pollute rankings.
"""

from datetime import datetime, date
import config


def detect_honeypot(candidate: dict) -> tuple[bool, str]:
    """
    Check if a candidate is a honeypot.
    Returns (is_honeypot, reason).
    """
    profile = candidate.get("profile", {})
    career_history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    flags = []

    # Check 1: Timeline impossibility
    # Total career months should not vastly exceed years_of_experience
    yoe = profile.get("years_of_experience", 0)
    total_career_months = sum(
        role.get("duration_months", 0) for role in career_history
    )
    max_reasonable_months = yoe * config.HONEYPOT_CAREER_MONTHS_MULTIPLIER
    if total_career_months > max_reasonable_months and yoe > 0:
        flags.append(
            f"career_months ({total_career_months}) > {max_reasonable_months} "
            f"(yoe={yoe} × {config.HONEYPOT_CAREER_MONTHS_MULTIPLIER})"
        )

    # Check 2: Date math validation
    # start_date to end_date should roughly match duration_months
    for role in career_history:
        start_str = role.get("start_date")
        end_str = role.get("end_date")
        duration = role.get("duration_months", 0)

        if start_str and end_str:
            try:
                start = _parse_date(start_str)
                end = _parse_date(end_str)
                actual_months = (end.year - start.year) * 12 + (end.month - start.month)
                diff = abs(actual_months - duration)
                if diff > config.HONEYPOT_DATE_DIFF_TOLERANCE_MONTHS:
                    flags.append(
                        f"date_mismatch in role '{role.get('title', 'unknown')}': "
                        f"dates say {actual_months}mo, claimed {duration}mo"
                    )
            except (ValueError, TypeError):
                pass

    # Check 3: Impossible skill claims
    # Expert/advanced proficiency with 0 months duration
    impossible_skills = 0
    for skill in skills:
        prof = skill.get("proficiency", "")
        dur = skill.get("duration_months", 0)
        if prof in ("expert", "advanced") and dur == 0:
            impossible_skills += 1

    if impossible_skills >= config.HONEYPOT_IMPOSSIBLE_SKILL_COUNT:
        flags.append(
            f"impossible_skills: {impossible_skills} advanced/expert skills with 0 months"
        )

    # Check 4: Title-skill absurdity
    # Non-tech title + excessive number of advanced/expert AI skills
    current_title = profile.get("current_title", "").lower().strip()
    is_non_tech = current_title in config.NON_TECH_TITLES

    if is_non_tech:
        ai_skill_keywords = {
            "ml", "machine learning", "ai", "deep learning", "nlp",
            "neural", "embedding", "transformers", "pytorch", "tensorflow",
            "bert", "gpt", "llm", "rag", "fine-tuning", "lora",
            "vector database", "pinecone", "weaviate", "faiss",
            "computer vision", "reinforcement learning", "gans",
            "sentence-transformers", "huggingface",
        }
        advanced_ai_count = 0
        for skill in skills:
            name = skill.get("name", "").lower()
            prof = skill.get("proficiency", "")
            if prof in ("advanced", "expert"):
                if any(kw in name for kw in ai_skill_keywords):
                    advanced_ai_count += 1

        if advanced_ai_count >= config.HONEYPOT_NONTECH_AI_SKILL_THRESHOLD:
            flags.append(
                f"title_skill_absurdity: {current_title} with {advanced_ai_count} advanced AI skills"
            )

    # Check 5: Career description vs title inconsistency
    # If MOST roles have descriptions that don't match their titles at all (only for non-tech roles)
    mismatch_count = 0
    checked_count = 0
    for role in career_history:
        title = role.get("title", "").lower()
        desc = role.get("description", "").lower()
        is_role_non_tech = any(k in title for k in [
            "hr", "accountant", "marketing", "sales", "mechanical", "civil",
            "graphic designer", "customer support", "operations", "content writer"
        ])
        if is_role_non_tech:
            checked_count += 1
            if _title_description_mismatch(title, desc):
                mismatch_count += 1

    if checked_count > 0 and mismatch_count >= checked_count * 0.5:
        flags.append(
            f"title_desc_mismatch: {mismatch_count}/{checked_count} non-tech roles "
            f"have descriptions that don't match their titles"
        )

    # Check 6: Career overlap impossibility
    # Two concurrent non-current roles with significant date overlap
    overlap_found = _check_career_overlaps(career_history)
    if overlap_found:
        flags.append("career_overlap: overlapping full-time roles detected")

    # Classification logic:
    # High confidence honeypots:
    # 1. title_skill_absurdity
    # 2. impossible_skills (expert/advanced with 0 duration)
    # 3. title_desc_mismatch (if current title is non-tech)
    # 4. career_overlap
    
    current_title = profile.get("current_title", "").lower().strip()
    is_current_non_tech = current_title in config.NON_TECH_TITLES
    
    has_high_conf = False
    for f in flags:
        if "title_skill_absurdity" in f or "impossible_skills" in f or "career_overlap" in f:
            has_high_conf = True
        if "title_desc_mismatch" in f and is_current_non_tech:
            has_high_conf = True

    is_honeypot = has_high_conf or (len(flags) >= 2)
    reason = "; ".join(flags) if flags else ""

    return is_honeypot, reason


def _parse_date(date_str: str) -> date:
    """Parse a date string in YYYY-MM-DD format."""
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def _title_description_mismatch(title: str, description: str) -> bool:
    """
    Check if a role's description is completely unrelated to its title.
    Returns True if there's a significant mismatch.
    """
    # Map of title keywords to expected description keywords
    title_desc_expectations = {
        "hr": {"hr", "human resources", "hiring", "recruitment", "people", "talent", "onboarding"},
        "accountant": {"accounting", "financial", "audit", "tax", "ledger", "budget", "compliance", "gaap"},
        "marketing": {"marketing", "brand", "campaign", "seo", "content", "audience", "growth"},
        "sales": {"sales", "revenue", "pipeline", "quota", "client", "deal", "crm"},
        "mechanical": {"mechanical", "cad", "manufacturing", "design", "solidworks", "fea", "hardware"},
        "civil": {"civil", "construction", "structural", "site", "concrete", "building"},
        "graphic designer": {"design", "graphic", "visual", "creative", "adobe", "figma", "brand"},
        "customer support": {"support", "customer", "ticket", "helpdesk", "service"},
        "operations": {"operations", "process", "logistics", "supply chain", "warehouse", "fulfillment"},
        "content writer": {"writing", "content", "editorial", "article", "blog", "copy"},
    }

    for title_key, expected_desc_words in title_desc_expectations.items():
        if title_key in title:
            # Check if description has expected words
            desc_has_expected = any(word in description for word in expected_desc_words)
            if not desc_has_expected:
                return True
            break

    return False


def _check_career_overlaps(career_history: list[dict]) -> bool:
    """Check for impossible overlapping full-time roles."""
    dated_roles = []
    for role in career_history:
        start_str = role.get("start_date")
        end_str = role.get("end_date")
        if start_str:
            try:
                start = _parse_date(start_str)
                end = _parse_date(end_str) if end_str else date.today()
                dated_roles.append((start, end, role.get("title", "")))
            except (ValueError, TypeError):
                pass

    # Check all pairs for overlap > 3 months
    for i in range(len(dated_roles)):
        for j in range(i + 1, len(dated_roles)):
            s1, e1, _ = dated_roles[i]
            s2, e2, _ = dated_roles[j]

            overlap_start = max(s1, s2)
            overlap_end = min(e1, e2)
            if overlap_start < overlap_end:
                overlap_months = (
                    (overlap_end.year - overlap_start.year) * 12
                    + (overlap_end.month - overlap_start.month)
                )
                if overlap_months > 3:
                    return True

    return False


def filter_honeypots(candidates: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Split candidates into clean and honeypot lists.
    Returns (clean_candidates, honeypots).
    """
    clean = []
    honeypots = []
    for candidate in candidates:
        is_hp, reason = detect_honeypot(candidate)
        if is_hp:
            honeypots.append({"candidate": candidate, "reason": reason})
        else:
            clean.append(candidate)

    print(f"  Honeypot detection: {len(clean)} clean, {len(honeypots)} flagged as honeypots")
    return clean, honeypots
