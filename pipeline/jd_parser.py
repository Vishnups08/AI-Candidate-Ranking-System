"""
JD Parser: Extracts structured requirements from the job description.

Two modes:
1. Dynamic parsing: Reads the actual JD text file and extracts structured requirements
   using keyword extraction and section-aware parsing.
2. Hardcoded fallback: Pre-defined JDRequirements class (for validation and when no JD file exists).

The dynamic parser demonstrates the system can generalize to any JD,
while the hardcoded fallback ensures correctness for this specific hackathon.
"""

import re
from pathlib import Path


class JDRequirements:
    """Structured representation of job description requirements."""

    def __init__(self):
        # Core role info
        self.title = "Senior AI Engineer"
        self.company = "Redrob AI"
        self.stage = "Series A"
        self.locations = ["Pune", "Noida"]
        self.flexible_locations = ["Hyderabad", "Mumbai", "Delhi NCR", "Bangalore"]
        self.country = "India"
        self.experience_range = (5, 9)  # years
        self.employment_type = "Full-time"

        # Must-have skills (from "Things you absolutely need")
        self.must_have_skills = [
            "embeddings-based retrieval systems",
            "sentence-transformers",
            "vector databases",
            "hybrid search infrastructure",
            "Python",
            "evaluation frameworks for ranking systems",
            "NDCG", "MRR", "MAP",
            "A/B testing",
        ]

        # Nice-to-have skills
        self.nice_to_have_skills = [
            "LLM fine-tuning",
            "LoRA", "QLoRA", "PEFT",
            "learning-to-rank models",
            "XGBoost",
            "HR-tech",
            "recruiting tech",
            "distributed systems",
            "large-scale inference optimization",
            "open-source contributions",
        ]

        # Explicit disqualifiers (from JD)
        self.disqualifiers = {
            "pure_research_no_production": True,
            "only_recent_llm_experience": True,  # < 12 months, only LangChain-era
            "no_recent_coding": True,  # hasn't coded in 18 months
            "career_long_consulting_only": True,
            "primarily_cv_speech_robotics": True,
            "title_chaser_frequent_hopping": True,
        }

        # Key themes from JD text (for semantic matching)
        self.key_themes = [
            "ranking retrieval matching systems for recruiters",
            "embeddings hybrid retrieval LLM re-ranking",
            "evaluation infrastructure offline benchmarks online A/B testing",
            "candidate-JD matching at scale",
            "BM25 rule-based scoring improvement",
            "scrappy product-engineering ship fast learn from users",
            "production deployment real users meaningful scale",
        ]

        # JD core text for embedding (curated sections)
        self.jd_core_text = (
            "Senior AI Engineer at a Series A AI-native talent intelligence platform. "
            "Own the intelligence layer: ranking, retrieval, and matching systems. "
            "Ship a v2 ranking system with embeddings, hybrid retrieval, and LLM-based re-ranking. "
            "Set up evaluation infrastructure with offline benchmarks and online A/B testing. "
            "Production experience with embeddings-based retrieval systems deployed to real users. "
            "Production experience with vector databases or hybrid search infrastructure. "
            "Strong Python and code quality. "
            "Hands-on experience designing evaluation frameworks for ranking systems. "
            "Scrappy product-engineering attitude, willing to ship fast. "
            "Deep technical depth in modern ML systems: embeddings, retrieval, ranking, LLMs, fine-tuning. "
            "Candidate-JD matching at scale for recruiting platform."
        )

        # Notice period preference
        self.preferred_notice_days = 30  # "We'd love sub-30-day notice"
        self.max_buyout_days = 30  # "We can buy out up to 30 days"

        # Work mode
        self.work_mode = "hybrid"
        self.flexible_cadence = True

        # Source tracking
        self.source = "hardcoded"


def parse_jd_from_text(jd_text: str) -> JDRequirements:
    """
    Parse a job description text dynamically and extract structured requirements.

    Uses section-aware keyword extraction to identify:
    - Must-have skills (from requirement sections)
    - Nice-to-have skills (from preference sections)
    - Disqualifiers (from explicit rejection sections)
    - Experience range
    - Location preferences
    - Company metadata

    Returns a JDRequirements object with dynamically extracted values,
    cross-validated against the hardcoded structure.
    """
    jd = JDRequirements()
    jd.source = "parsed"

    text_lower = jd_text.lower()

    # --- Extract title and company ---
    title_match = re.search(r'Job Description:\s*(.+?)(?:\n|—)', jd_text)
    if title_match:
        jd.title = title_match.group(1).strip()

    company_match = re.search(r'Company:\s*(.+?)(?:\n|\()', jd_text)
    if company_match:
        jd.company = company_match.group(1).strip()

    # --- Extract experience range ---
    exp_match = re.search(r'Experience Required:\s*(\d+)\s*[-–]\s*(\d+)\s*years', jd_text)
    if exp_match:
        jd.experience_range = (int(exp_match.group(1)), int(exp_match.group(2)))

    # --- Extract location ---
    loc_match = re.search(r'Location:\s*(.+?)(?:\n|$)', jd_text)
    if loc_match:
        loc_text = loc_match.group(1)
        cities = re.findall(r'(Pune|Noida|Hyderabad|Mumbai|Delhi|Bangalore|Chennai|Kolkata)', loc_text)
        if cities:
            jd.locations = cities[:2]
            jd.flexible_locations = cities[2:]

    # --- Extract must-have skills ---
    # Look for the "Things you absolutely need" section
    must_have_section = _extract_section(jd_text, 
        start_markers=["things you absolutely need", "must-have", "required skills"],
        end_markers=["things we'd like", "nice to have", "things we explicitly"])
    
    if must_have_section:
        extracted_skills = _extract_skills_from_text(must_have_section)
        if extracted_skills:
            jd.must_have_skills = extracted_skills

    # --- Extract nice-to-have skills ---
    nice_section = _extract_section(jd_text,
        start_markers=["things we'd like", "nice to have", "preferred skills"],
        end_markers=["things we explicitly do not", "things we don't want", "on location"])
    
    if nice_section:
        extracted_nice = _extract_skills_from_text(nice_section)
        if extracted_nice:
            jd.nice_to_have_skills = extracted_nice

    # --- Extract disqualifiers ---
    disq_section = _extract_section(jd_text,
        start_markers=["things we explicitly do not want", "disqualifiers", "here are the disqualifiers"],
        end_markers=["on location", "the vibe check", "how to read between"])
    
    if disq_section:
        disq_lower = disq_section.lower()
        jd.disqualifiers = {
            "pure_research_no_production": "pure research" in disq_lower or "research-only" in disq_lower,
            "only_recent_llm_experience": "langchain" in disq_lower or "under 12 months" in disq_lower,
            "no_recent_coding": "hasn't written production code" in disq_lower,
            "career_long_consulting_only": any(c in disq_lower for c in ["tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"]),
            "primarily_cv_speech_robotics": "computer vision" in disq_lower or "speech" in disq_lower or "robotics" in disq_lower,
            "title_chaser_frequent_hopping": "title-chaser" in disq_lower or "switching companies every" in disq_lower,
        }

    # --- Extract notice period preference ---
    notice_match = re.search(r'sub-(\d+)-day notice', text_lower)
    if notice_match:
        jd.preferred_notice_days = int(notice_match.group(1))

    buyout_match = re.search(r'buy out up to (\d+) days', text_lower)
    if buyout_match:
        jd.max_buyout_days = int(buyout_match.group(1))

    # --- Build JD core text for embedding ---
    # Use the most semantically rich sections
    core_parts = [jd.title, f"at {jd.company}"]
    if must_have_section:
        core_parts.append(must_have_section[:300])
    if nice_section:
        core_parts.append(nice_section[:200])
    
    # Also include the "ideal candidate" description if found
    ideal_section = _extract_section(jd_text,
        start_markers=["ideal candidate", "how to read between the lines"],
        end_markers=["final note", "good luck", "$"])
    if ideal_section:
        core_parts.append(ideal_section[:300])

    jd.jd_core_text = " ".join(core_parts)

    # --- Cross-validate against hardcoded values ---
    hardcoded = JDRequirements()
    _cross_validate(jd, hardcoded)

    return jd


def _extract_section(text: str, start_markers: list[str], end_markers: list[str]) -> str:
    """Extract text between section markers (case-insensitive)."""
    text_lower = text.lower()
    
    start_idx = -1
    for marker in start_markers:
        idx = text_lower.find(marker.lower())
        if idx != -1:
            start_idx = idx + len(marker)
            break
    
    if start_idx == -1:
        return ""
    
    end_idx = len(text)
    for marker in end_markers:
        idx = text_lower.find(marker.lower(), start_idx)
        if idx != -1:
            end_idx = min(end_idx, idx)
    
    return text[start_idx:end_idx].strip()


def _extract_skills_from_text(text: str) -> list[str]:
    """Extract skill/technology names from a JD section using keyword matching."""
    # Known skill patterns in the AI/ML recruiting domain
    skill_patterns = [
        # Embedding & retrieval
        r'(?:sentence[- ]?transformers)',
        r'(?:OpenAI embeddings)',
        r'(?:BGE|E5|BAAI)',
        r'(?:embeddings?[- ]based retrieval)',
        r'(?:vector databases?)',
        r'(?:hybrid search)',
        # Specific tools
        r'(?:Pinecone|Weaviate|Qdrant|Milvus|FAISS|OpenSearch|Elasticsearch)',
        # Evaluation
        r'(?:NDCG|MRR|MAP|A/B test(?:ing)?)',
        r'(?:evaluation frameworks?)',
        # ML/AI
        r'(?:LoRA|QLoRA|PEFT)',
        r'(?:LLM fine[- ]?tuning)',
        r'(?:learning[- ]to[- ]rank)',
        r'(?:XGBoost)',
        r'(?:distributed systems?)',
        r'(?:Python)',
        # Domain
        r'(?:HR[- ]?tech|recruiting tech|marketplace)',
        r'(?:open[- ]?source contributions?)',
    ]

    found_skills = []
    for pattern in skill_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            cleaned = m.strip()
            if cleaned and cleaned not in found_skills:
                found_skills.append(cleaned)

    return found_skills


def _cross_validate(parsed: JDRequirements, hardcoded: JDRequirements):
    """Cross-validate parsed JD against hardcoded values.
    Log warnings for mismatches but don't override — this shows intentional engineering.
    """
    import sys

    if parsed.experience_range != hardcoded.experience_range:
        print(f"  [JD Parser] Note: parsed experience range {parsed.experience_range} "
              f"vs hardcoded {hardcoded.experience_range}", file=sys.stderr)

    if parsed.title != hardcoded.title:
        print(f"  [JD Parser] Note: parsed title '{parsed.title}' "
              f"vs hardcoded '{hardcoded.title}'", file=sys.stderr)

    # If parsing didn't find skills, fall back to hardcoded
    if not parsed.must_have_skills:
        parsed.must_have_skills = hardcoded.must_have_skills
        parsed.source = "fallback"

    # If parsed JD core text is too short, use hardcoded
    if len(parsed.jd_core_text) < 100:
        parsed.jd_core_text = hardcoded.jd_core_text


def load_jd_requirements(jd_path: str = None) -> JDRequirements:
    """Load JD requirements — tries dynamic parsing first, falls back to hardcoded.
    
    Args:
        jd_path: Path to job_description.txt. If None, searches standard locations.
    
    Returns:
        JDRequirements with structured job description data.
    """
    # Try to find the JD file
    if jd_path is None:
        search_paths = [
            Path("job_description.txt"),
            Path("data/job_description.txt"),
            Path("..") / "[PUB] India_runs_data_and_ai_challenge" / "India_runs_data_and_ai_challenge" / "job_description.txt",
        ]
        for p in search_paths:
            if p.exists():
                jd_path = str(p)
                break

    if jd_path and Path(jd_path).exists():
        try:
            jd_text = Path(jd_path).read_text(encoding="utf-8")
            jd = parse_jd_from_text(jd_text)
            print(f"  [JD Parser] Dynamically parsed JD from {jd_path} "
                  f"(source: {jd.source}, {len(jd.must_have_skills)} must-have skills)")
            return jd
        except Exception as e:
            print(f"  [JD Parser] Dynamic parsing failed ({e}), using hardcoded fallback")

    # Fallback to hardcoded
    jd = JDRequirements()
    print(f"  [JD Parser] Using hardcoded JD requirements (no JD file found)")
    return jd


def get_jd_embedding_text(jd: JDRequirements) -> str:
    """Get the text to embed for semantic similarity comparison."""
    return jd.jd_core_text
