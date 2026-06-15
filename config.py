"""
Configuration for the Redrob AI Candidate Ranking Pipeline.
All weights, thresholds, and constants in one place for easy tuning.
"""

# =============================================================================
# Embedding Model
# =============================================================================
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM = 768

# BGE models use instruction prefixes for better retrieval quality
EMBEDDING_QUERY_PREFIX = "Represent this job description for retrieval: "
EMBEDDING_PASSAGE_PREFIX = "Represent this professional profile for retrieval: "

# Max text length for embeddings (chars) — increased from 256 for richer signal
EMBEDDING_MAX_TEXT_LENGTH = 512

# =============================================================================
# Cross-Encoder Re-Ranking
# =============================================================================
USE_CROSS_ENCODER = True
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_TOP_K = 500
CROSS_ENCODER_WEIGHT = 0.3  # Weight of cross-encoder score in final rank (remaining is bi-encoder composite)

# =============================================================================
# Feature Scoring Weights (must sum to 1.0)
# =============================================================================
WEIGHTS = {
    "semantic_similarity": 0.25,
    "career_fit": 0.25,
    "skills_match": 0.20,
    "experience_fit": 0.10,
    "location_logistics": 0.10,
    "education": 0.05,
    # Behavioral is applied as a multiplier (0.6–1.3), not additive
}

# Sanity check
_additive_sum = sum(WEIGHTS.values())
assert abs(_additive_sum - 0.95) < 0.01, f"Additive weights must sum to 0.95, got {_additive_sum}"

# Remaining 0.05 is a small additive behavioral bonus for very strong signals
BEHAVIORAL_ADDITIVE_WEIGHT = 0.05

# =============================================================================
# Behavioral Multiplier Range
# =============================================================================
BEHAVIORAL_MULTIPLIER_MIN = 0.6
BEHAVIORAL_MULTIPLIER_MAX = 1.3

# =============================================================================
# Hard Filter Thresholds
# =============================================================================
MIN_EXPERIENCE_YEARS = 2.0
MAX_EXPERIENCE_YEARS = 25.0

# Non-tech titles that disqualify if ALL career history is also non-tech
NON_TECH_TITLES = {
    "accountant", "hr manager", "human resources manager", "marketing manager",
    "sales executive", "content writer", "graphic designer", "customer support",
    "operations manager", "civil engineer", "mechanical engineer",
    "financial analyst", "supply chain manager", "procurement manager",
    "legal counsel", "event manager", "recruiter", "office manager",
    "administrative assistant", "receptionist", "project manager",
    "business analyst", "product manager", "scrum master", "delivery manager",
}

# Tech-adjacent titles that pass filters
TECH_TITLES = {
    "ai engineer", "ml engineer", "machine learning engineer",
    "senior ai engineer", "senior ml engineer", "senior machine learning engineer",
    "junior ml engineer", "junior ai engineer",
    "data scientist", "senior data scientist", "lead data scientist",
    "research engineer", "research scientist", "applied scientist",
    "software engineer", "senior software engineer", "staff software engineer",
    "backend engineer", "senior backend engineer",
    "data engineer", "senior data engineer", "platform engineer",
    "nlp engineer", "computer vision engineer", "deep learning engineer",
    "full stack engineer", "devops engineer", "site reliability engineer",
    "technical lead", "engineering manager", "tech lead",
    "product engineer", "solutions architect",
}

# Relevant skill domains (for hard filter — generous)
RELEVANT_SKILL_DOMAINS = {
    "python", "machine learning", "ml", "deep learning", "ai",
    "artificial intelligence", "nlp", "natural language processing",
    "data science", "neural networks", "tensorflow", "pytorch",
    "scikit-learn", "sklearn", "pandas", "numpy",
    "embeddings", "sentence-transformers", "transformers", "huggingface",
    "vector database", "pinecone", "weaviate", "qdrant", "milvus", "faiss",
    "elasticsearch", "opensearch", "solr",
    "information retrieval", "search", "ranking", "recommendation",
    "llm", "large language models", "gpt", "bert", "fine-tuning",
    "lora", "qlora", "peft", "rag",
    "xgboost", "lightgbm", "random forest",
    "sql", "spark", "hadoop", "kafka", "airflow",
    "docker", "kubernetes", "aws", "gcp", "azure",
    "java", "scala", "go", "rust", "c++",
    "backend", "distributed systems", "microservices",
    "statistics", "statistical modeling", "bayesian",
    "computer vision", "image classification", "object detection",
    "reinforcement learning", "time series",
    "feature engineering", "model deployment", "mlops",
    "a/b testing", "experimentation", "ndcg", "mrr",
    "bm25", "tf-idf", "cosine similarity",
    "flask", "fastapi", "django",
    "git", "linux", "bash",
    "data pipeline", "etl", "data warehouse",
    "nlu", "text classification", "sentiment analysis",
    "speech recognition", "tts", "gans",
    "milvus", "chroma", "langchain", "llamaindex",
    "bentoml", "mlflow", "weights & biases", "wandb",
    "opencv", "spacy", "nltk",
}

# =============================================================================
# Career Fit Configuration
# =============================================================================

# Title relevance tiers
TITLE_RELEVANCE = {
    # Tier 1: Perfect match (1.0)
    "ai engineer": 1.0, "ml engineer": 1.0, "machine learning engineer": 1.0,
    "senior ai engineer": 1.0, "senior ml engineer": 1.0,
    "senior machine learning engineer": 1.0,
    "nlp engineer": 1.0, "deep learning engineer": 1.0,
    "junior ai engineer": 0.85, "junior ml engineer": 0.85,

    # Tier 2: Strong match (0.85)
    "data scientist": 0.85, "senior data scientist": 0.85,
    "lead data scientist": 0.85, "applied scientist": 0.85,
    "research engineer": 0.80, "research scientist": 0.75,

    # Tier 3: Good match (0.70)
    "software engineer": 0.70, "senior software engineer": 0.70,
    "staff software engineer": 0.70, "backend engineer": 0.70,
    "senior backend engineer": 0.70, "full stack engineer": 0.60,

    # Tier 4: Adjacent (0.50)
    "data engineer": 0.50, "senior data engineer": 0.50,
    "platform engineer": 0.50, "devops engineer": 0.40,
    "solutions architect": 0.45, "technical lead": 0.55,
    "tech lead": 0.55, "engineering manager": 0.45,
    "product engineer": 0.55,
}

# Consulting firms (career-long penalty per JD)
CONSULTING_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "hcl technologies", "tech mahindra", "mindtree", "ltimindtree",
    "mphasis", "l&t infotech", "lti", "persistent systems",
    "cyient", "hexaware", "niit technologies", "zensar",
    "cts",  # Cognizant abbreviation
    "deloitte", "kpmg", "ey", "pwc",  # Big 4 consulting
}

# =============================================================================
# Skills Configuration
# =============================================================================

# Must-have skills (from JD "Things you absolutely need")
MUST_HAVE_SKILLS = {
    "embeddings", "sentence-transformers", "openai embeddings", "bge", "e5",
    "vector database", "pinecone", "weaviate", "qdrant", "milvus", "faiss",
    "elasticsearch", "opensearch",
    "python",
    "ndcg", "mrr", "map", "a/b testing", "evaluation", "ranking metrics",
    "information retrieval", "search", "ranking", "retrieval",
}

# Nice-to-have skills
NICE_TO_HAVE_SKILLS = {
    "lora", "qlora", "peft", "fine-tuning", "fine-tuning llms",
    "xgboost", "lightgbm", "learning to rank", "learning-to-rank",
    "hr tech", "recruiting", "talent", "hiring",
    "distributed systems", "large-scale", "inference optimization",
    "open source", "open-source",
}

# Domain-adjacent skills
DOMAIN_ADJACENT_SKILLS = {
    "nlp", "natural language processing", "nlu",
    "recommendation", "recommendation systems",
    "deep learning", "neural networks", "transformers",
    "pytorch", "tensorflow", "huggingface",
    "bert", "gpt", "llm", "large language models",
    "text classification", "sentiment analysis",
    "rag", "langchain", "llamaindex",
    "bm25", "tf-idf",
    "machine learning", "ml", "data science",
    "spark", "hadoop", "kafka",
}

# =============================================================================
# Location Configuration
# =============================================================================

# Preferred cities (from JD)
PREFERRED_CITIES = {
    "pune", "noida", "delhi", "delhi ncr", "new delhi", "gurgaon", "gurugram",
    "faridabad", "ghaziabad",
}

# Good cities (from JD "welcome to apply")
GOOD_CITIES = {
    "hyderabad", "mumbai", "bangalore", "bengaluru", "chennai", "kolkata",
}

# India (for country detection)
INDIA_COUNTRY = {"india", "in"}

# =============================================================================
# Experience Scoring Bands
# =============================================================================
EXPERIENCE_BANDS = [
    # (min_years, max_years, score)
    (5.0, 9.0, 1.0),    # Optimal per JD
    (4.0, 4.99, 0.75),  # Near-optimal (low end)
    (9.01, 12.0, 0.75), # Near-optimal (high end)
    (3.0, 3.99, 0.45),  # Extended (low)
    (12.01, 15.0, 0.45),# Extended (high)
    (2.0, 2.99, 0.20),  # Minimum
    (15.01, 25.0, 0.20),# Over-experienced
]

# =============================================================================
# Salary Configuration
# =============================================================================
# Series A in India — realistic salary range for Senior AI Engineer
SALARY_IDEAL_MAX_LPA = 50  # Under 50 LPA = realistic
SALARY_STRETCH_MAX_LPA = 70  # 50-70 = stretch but possible

# =============================================================================
# Honeypot Detection Thresholds
# =============================================================================
HONEYPOT_CAREER_MONTHS_MULTIPLIER = 14  # career_months > yoe * 14 = suspicious
HONEYPOT_DATE_DIFF_TOLERANCE_MONTHS = 6
HONEYPOT_IMPOSSIBLE_SKILL_COUNT = 3  # >=3 expert skills with 0 months
HONEYPOT_TITLE_DESC_SIMILARITY_THRESHOLD = 0.15
HONEYPOT_NONTECH_AI_SKILL_THRESHOLD = 8  # Non-tech title + 8+ advanced AI skills

# =============================================================================
# Reasoning Configuration
# =============================================================================
TOP_TIER_RANKS = range(1, 11)       # Rank 1-10: Confident tone
STRONG_TIER_RANKS = range(11, 31)   # Rank 11-30: Solid with notes
MID_TIER_RANKS = range(31, 61)      # Rank 31-60: Balanced
LOWER_TIER_RANKS = range(61, 101)   # Rank 61-100: Cautious

# =============================================================================
# Output Configuration
# =============================================================================
TOP_K = 100
OUTPUT_COLUMNS = ["candidate_id", "rank", "score", "reasoning"]
