# Production Roadmap — Weeks 9 to 12

*This document outlines the engineering path to take this hackathon prototype to a production-grade enterprise system, directly answering the JD's requirement for "Weeks 9 to 12: Productionization."*

---

## 1. Data Pipeline & Embedding Updates

Currently, candidate embeddings are pre-computed offline. In production, candidate profiles are continuously added and updated.

### Planned Architecture:
- **Event-Driven Updates**: Listen to Kafka/Kinesis streams for candidate profile updates (e.g., `ProfileUpdatedEvent`).
- **Asynchronous Embedding**: A Celery worker consumes these events, extracts the semantic text, calls the embedding model (`bge-small-en-v1.5`), and upserts the vector into the database.
- **Vector Database**: Migrate from in-memory numpy arrays (`.npz`) to a proper vector database (Milvus, Qdrant, or pgvector) to support horizontal scaling, metadata filtering (e.g., filtering out those without 'Python' before vector search), and fast ANN (Approximate Nearest Neighbors) retrieval.

## 2. Low-Latency API Scaling

The 5-minute batch constraint for 100K candidates is met (~1 minute on CPU), but interactive API requests (like the `/api/rank-jd` Custom Scan) need to support multiple concurrent recruiters.

### Planned Architecture:
- **Stateless Rankers**: Deploy the Flask/FastAPI backend behind a load balancer (Nginx/ALB) across multiple Kubernetes pods.
- **Caching Layer**: Use Redis to cache the pre-computed feature scores (Experience Fit, Education, etc.) since they only change when the candidate updates their profile, not when a new JD is searched.
- **GPU Inference for Cross-Encoder**: The Stage 5 cross-encoder re-ranking is currently simulated/lightweight for CPU. In production, move the MS-MARCO cross-encoder to a dedicated GPU inference server (e.g., Triton Inference Server or vLLM) to re-rank the top 500 candidates in <50ms.

## 3. Online Evaluation & Feedback Loops

Offline metrics (NDCG, MAP) on a 98-candidate gold set prove the system works, but production systems need to learn from actual recruiter behavior.

### Planned Architecture:
- **Telemetry**: Instrument the frontend to capture implicit feedback:
  - `CandidateViewed`: The recruiter clicked to expand the profile.
  - `CandidateShortlisted`: The recruiter moved them to the next stage.
  - `CandidateRejected`: The recruiter rejected them (with reason code).
- **A/B Testing Framework**: Deploy competing ranker versions (e.g., v1 with 0.25 Semantic weight vs. v2 with 0.35 Semantic weight) using a traffic router. Measure which version leads to higher `CandidateShortlisted` rates.
- **Active Learning**: Periodically sample edge cases (where the ranker was uncertain) and route them to human recruiters for explicit grading to continuously expand the gold set.

## 4. Security, Privacy & Fairness

Candidate data is highly sensitive and ranking models must not introduce bias.

### Planned Architecture:
- **PII Stripping**: Ensure names and identifying information are stripped *before* being fed into any external models or logs.
- **Bias Auditing**: Automatically run the ranker against synthetic profiles that differ only by inferred gender/ethnicity markers to mathematically prove the scoring function remains invariant.
- **Explainability API**: Enhance the `reasoning_generator.py` to output SHAP values (feature importance) for every candidate, so recruiters can see exactly *why* a candidate scored 0.85 (e.g., "+0.2 from FAISS experience, -0.1 from missing Kubernetes").

---

*This roadmap transitions the system from a static, batch-processing script into a live, learning, and horizontally scalable enterprise service.*
