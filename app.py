import streamlit as st
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os
import sys

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).parent))

import config
from pipeline.loader import load_candidates
from pipeline.jd_parser import load_jd_requirements
from pipeline.hard_filters import apply_hard_filters
from pipeline.honeypot_detector import filter_honeypots, detect_honeypot
from pipeline.ranker import CandidateRanker
from pipeline.reasoning_generator import (
    generate_structured_explanation,
    generate_honeypot_contrast_card,
    generate_demotion_contrast_card,
)
from pipeline.behavioral_scorer import compute_behavioral_multiplier, compute_behavioral_additive


# Set page config
st.set_page_config(
    page_title="Redrob AI Ranker Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom premium styling (Sleek Dark Mode & Glassmorphism)
st.markdown(
    """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Global theme overrides */
    .stApp {
        background: radial-gradient(circle at top right, #1a1a2e, #0f0f1b);
        color: #e2e8f0;
    }
    
    /* Glassmorphism containers */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        padding: 24px;
        border: 1px rgba(255, 255, 255, 0.08) solid;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        margin-bottom: 20px;
    }
    
    .glass-metric {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.01) 100%);
        border-radius: 12px;
        padding: 16px;
        border: 1px rgba(255, 255, 255, 0.05) solid;
        text-align: center;
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.2);
    }
    
    /* Header Gradient */
    .gradient-text {
        background: linear-gradient(135deg, #a78bfa 0%, #3b82f6 50%, #60a5fa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    /* Accent borders */
    .accent-border-violet {
        border-left: 4px solid #8b5cf6;
    }
    
    .accent-border-blue {
        border-left: 4px solid #3b82f6;
    }

    .accent-border-emerald {
        border-left: 4px solid #10b981;
    }

    .accent-border-rose {
        border-left: 4px solid #f43f5e;
    }
    
    /* Interactive button overrides */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6);
    }
    
    /* Clean sidebar headers */
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 15px;
        margin-bottom: 10px;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        padding-bottom: 5px;
    }
    
    </style>
    """,
    unsafe_allow_html=True,
)

# App Title & Hero Header
st.markdown(
    """
    <div style='text-align: center; padding: 20px 0px 10px 0px;'>
        <h1 style='font-size: 2.8rem; margin-bottom: 5px;'><span class='gradient-text'>Redrob AI</span> Candidate discovery</h1>
        <p style='color: #94a3b8; font-size: 1.1rem; max-width: 700px; margin: 0 auto;'>
            A premium, multi-dimensional candidate discovery and ranking system powered by BGE Semantic Embeddings and verified behavioral signals.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# Initialize JD
jd = load_jd_requirements()

# Sidebar: Interactive Tuning
st.sidebar.markdown("<div class='sidebar-header'>🎯 WEIGHTS & SCORING</div>", unsafe_allow_html=True)
st.sidebar.caption("Fine-tune scoring dimensions live. The total sum is normalized to 0.95, and the remaining 0.05 is an additive behavioral reward.")

# Config Weight Sliders
sem_weight = st.sidebar.slider("Semantic Similarity Weight", 0.0, 0.60, 0.25, 0.05)
career_weight = st.sidebar.slider("Career Fit Weight", 0.0, 0.60, 0.25, 0.05)
skills_weight = st.sidebar.slider("Skills Match Weight", 0.0, 0.60, 0.20, 0.05)
exp_weight = st.sidebar.slider("Experience Fit Weight", 0.0, 0.40, 0.10, 0.05)
loc_weight = st.sidebar.slider("Location & Logistics Weight", 0.0, 0.40, 0.10, 0.05)
edu_weight = st.sidebar.slider("Education Weight", 0.0, 0.20, 0.05, 0.01)

# Normalize sliders to sum to 0.95
total_sliders = sem_weight + career_weight + skills_weight + exp_weight + loc_weight + edu_weight
if total_sliders > 0:
    scale_factor = 0.95 / total_sliders
    config.WEIGHTS = {
        "semantic_similarity": sem_weight * scale_factor,
        "career_fit": career_weight * scale_factor,
        "skills_match": skills_weight * scale_factor,
        "experience_fit": exp_weight * scale_factor,
        "location_logistics": loc_weight * scale_factor,
        "education": edu_weight * scale_factor,
    }
else:
    # Fallback default
    config.WEIGHTS = {
        "semantic_similarity": 0.25,
        "career_fit": 0.25,
        "skills_match": 0.20,
        "experience_fit": 0.10,
        "location_logistics": 0.10,
        "education": 0.05,
    }

# Display normalized values in sidebar
st.sidebar.markdown("<div class='sidebar-header'>⚖️ NORMALIZED WEIGHTS</div>", unsafe_allow_html=True)
for dim, weight in config.WEIGHTS.items():
    st.sidebar.markdown(f"**{dim.replace('_', ' ').title()}**: `{weight:.3f}`")

# India-context callout
st.sidebar.markdown("<div class='sidebar-header'>🇮🇳 INDIA-CONTEXT DESIGN</div>", unsafe_allow_html=True)
st.sidebar.markdown("""
<div style='font-size:0.8rem;color:#94a3b8;line-height:1.6;'>
<b>Deliberately India-aware signals:</b><br/>
• <b>Preferred cities:</b> Pune, Noida, NCR (1.0×), Tier-1 cities (0.7×)<br/>
• <b>INR LPA realism:</b> ≤50 LPA = ideal, 50–70 = stretch, >70 = concern<br/>
• <b>Notice period:</b> ≤30 days = +0.10, 90+ = −0.05 to −0.15<br/>
• <b>Consulting penalty:</b> TCS/Infosys/Wipro career = 0.15 career quality<br/>
• <b>Institution tier:</b> IIT/BITS/NIT = tier_1 bonus (+0.30)<br/>
• <b>Service vs product:</b> Product-company bias per JD<br/>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<div class='sidebar-header'>📥 CANDIDATES DATA</div>", unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Upload candidates file (JSON/JSONL)", type=["json", "jsonl"])

# Load candidates
candidates_source = None
if uploaded_file is not None:
    # Save to temp file
    temp_path = Path("temp_candidates.json")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    candidates = list(load_candidates(str(temp_path)))
    candidates_source = uploaded_file.name
    # Cleanup temp file
    if temp_path.exists():
        os.remove(temp_path)
else:
    # Fallback to sample data
    sample_path = Path("..") / "[PUB] India_runs_data_and_ai_challenge" / "India_runs_data_and_ai_challenge" / "sample_candidates.json"
    if not sample_path.exists():
        sample_path = Path("precompute") / "sample_candidates.json"  # alternative path
    
    if sample_path.exists():
        candidates = list(load_candidates(str(sample_path)))
        candidates_source = "sample_candidates.json (Demo)"
    else:
        st.error("Sample candidates file not found. Please upload a JSON or JSONL file.")
        st.stop()

# Caching helper to load embedding model on the fly (offline-first, downloads if missing)
@st.cache_resource
def get_embedding_model():
    from sentence_transformers import SentenceTransformer
    _model_path = config._local_model_path(config.EMBEDDING_MODEL)
    return SentenceTransformer(_model_path, device="cpu")

# Load embeddings
jd_embedding = None
candidate_embeddings = {}
emb_dir = Path("precompute/embeddings")

# Check if we have precomputed files
jd_emb_path = emb_dir / "jd_embedding.npy"
profile_emb_path = emb_dir / "candidate_profiles.npz"
role_emb_path = emb_dir / "candidate_roles.npz"
has_precomputed = (
    "sample" in candidates_source.lower()
    and jd_emb_path.exists()
    and profile_emb_path.exists()
    and role_emb_path.exists()
)

if has_precomputed:
    # Load from precomputed files
    jd_embedding = np.load(str(jd_emb_path))
    
    data = np.load(str(profile_emb_path), allow_pickle=True)
    for cid, emb in zip(data["candidate_ids"], data["embeddings"]):
        if str(cid) not in candidate_embeddings:
            candidate_embeddings[str(cid)] = {}
        candidate_embeddings[str(cid)]["profile_embedding"] = emb
        
    data = np.load(str(role_emb_path), allow_pickle=True)
    for cid, role_embs in zip(data["candidate_ids"], data["role_embeddings"]):
        cid_str = str(cid)
        if cid_str not in candidate_embeddings:
            candidate_embeddings[cid_str] = {}
        candidate_embeddings[cid_str]["role_embeddings"] = role_embs
else:
    # Compute on the fly! (For cloud deployment or custom uploads)
    with st.spinner("Initializing BGE-small semantic model and embedding candidates on-the-fly..."):
        try:
            model = get_embedding_model()
            
            # Embed JD
            jd_text = jd.jd_core_text
            jd_embedding = model.encode(config.EMBEDDING_QUERY_PREFIX + jd_text, normalize_embeddings=True)
            
            # Embed candidates
            from precompute.build_embeddings import build_profile_text, build_role_texts
            
            profile_texts = [build_profile_text(c) for c in candidates]
            profile_embs = model.encode(
                [config.EMBEDDING_PASSAGE_PREFIX + t for t in profile_texts],
                show_progress_bar=False,
                normalize_embeddings=True
            )
            
            for c, emb in zip(candidates, profile_embs):
                cid = str(c["candidate_id"])
                candidate_embeddings[cid] = {"profile_embedding": emb}
                
                # Embed roles
                role_texts = build_role_texts(c)
                if role_texts:
                    role_embs = model.encode(
                        [config.EMBEDDING_PASSAGE_PREFIX + rt for rt in role_texts],
                        show_progress_bar=False,
                        normalize_embeddings=True
                    )
                    candidate_embeddings[cid]["role_embeddings"] = role_embs
                else:
                    candidate_embeddings[cid]["role_embeddings"] = []
            st.success(f"Successfully generated neural embeddings for {len(candidates)} candidates on-the-fly.")
        except Exception as e:
            st.warning(f"On-the-fly embedding failed: {e}. Falling back to keyword-based semantic matching.")

# Pipeline execution
filtered = apply_hard_filters(candidates)
clean, honeypots = filter_honeypots(filtered)

# Naive keyword score for each candidate (for vs-naive toggle)
def _naive_keyword_score(candidate: dict) -> int:
    jd_kw = {
        "embedding", "embeddings", "retrieval", "vector", "search", "ranking",
        "nlp", "machine learning", "ml", "python", "faiss", "pinecone",
        "weaviate", "qdrant", "elasticsearch", "bm25", "recommendation",
        "information retrieval", "sentence-transformers", "a/b testing", "ndcg",
    }
    return sum(
        1 for s in candidate.get("skills", [])
        if any(kw in s.get("name", "").lower() for kw in jd_kw)
    )

naive_scored = sorted(clean, key=lambda c: -_naive_keyword_score(c))
naive_rank_map = {c["candidate_id"]: i + 1 for i, c in enumerate(naive_scored)}

# Rank
ranker = CandidateRanker(jd, jd_embedding, candidate_embeddings)
results = ranker.rank_candidates(clean)

# Metric layout
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f"""
        <div class='glass-metric accent-border-violet'>
            <div style='font-size: 0.85rem; color: #a78bfa; font-weight: 600; text-transform: uppercase;'>Total Candidates</div>
            <div style='font-size: 1.8rem; font-weight: 800; margin-top: 5px;'>{len(candidates)}</div>
            <div style='font-size: 0.75rem; color: #64748b; margin-top: 3px;'>Source: {candidates_source}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""
        <div class='glass-metric accent-border-blue'>
            <div style='font-size: 0.85rem; color: #60a5fa; font-weight: 600; text-transform: uppercase;'>Passed Filters</div>
            <div style='font-size: 1.8rem; font-weight: 800; margin-top: 5px;'>{len(filtered)}</div>
            <div style='font-size: 0.75rem; color: #10b981; margin-top: 3px;'>Pass rate: {len(filtered)/len(candidates)*100:.1f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"""
        <div class='glass-metric accent-border-rose'>
            <div style='font-size: 0.85rem; color: #f43f5e; font-weight: 600; text-transform: uppercase;'>Honeypots Blocked</div>
            <div style='font-size: 1.8rem; font-weight: 800; margin-top: 5px;'>{len(honeypots)}</div>
            <div style='font-size: 0.75rem; color: #f43f5e; margin-top: 3px;'>Excluded from score</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c4:
    top_score = results[0]["score"] if results else 0.0
    st.markdown(
        f"""
        <div class='glass-metric accent-border-emerald'>
            <div style='font-size: 0.85rem; color: #34d399; font-weight: 600; text-transform: uppercase;'>Top Match Score</div>
            <div style='font-size: 1.8rem; font-weight: 800; margin-top: 5px;'>{top_score:.3f}</div>
            <div style='font-size: 0.75rem; color: #64748b; margin-top: 3px;'>Max possible: 1.350</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# Layout: Left side (ranks list), Right side (selected candidate details)
left_col, right_col = st.columns([1, 1])

with left_col:
    st.markdown("<h3 style='margin-bottom: 15px;'>🏆 Shortlist & Ranking</h3>", unsafe_allow_html=True)
    
    # Create pandas dataframe for results
    records = []
    for r in results:
        records.append({
            "Rank": r["rank"],
            "Candidate ID": r["candidate_id"],
            "Score": f"{r['score']:.3f}",
            "Reasoning": r["reasoning"]
        })
    df_results = pd.DataFrame(records)
    
    # Custom Selection UI
    selected_rank = st.selectbox(
        "Select candidate to inspect details:",
        options=df_results["Rank"].tolist(),
        format_func=lambda x: f"Rank {x}: {df_results[df_results['Rank'] == x]['Candidate ID'].values[0]} (Score: {df_results[df_results['Rank'] == x]['Score'].values[0]})"
    )
    
    # Shortlist Table Display
    st.dataframe(
        df_results,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "Candidate ID": st.column_config.TextColumn("Candidate ID", width="medium"),
            "Score": st.column_config.TextColumn("Score", width="small"),
            "Reasoning": st.column_config.TextColumn("Summary Reasoning", width="large"),
        },
        hide_index=True,
        use_container_width=True
    )
    
    # Export csv
    csv_data = df_results.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Shortlist CSV",
        data=csv_data,
        file_name="submission_shortlist.csv",
        mime="text/csv",
    )

# Get selected candidate details
selected_candidate_id = df_results[df_results["Rank"] == selected_rank]["Candidate ID"].values[0]
cand_detail = next(c for c in candidates if c.get("candidate_id") == selected_candidate_id)
cand_result = next(r for r in results if r["candidate_id"] == selected_candidate_id)

with right_col:
    st.markdown(f"<h3>👤 Candidate Profile: <span class='gradient-text'>{selected_candidate_id}</span></h3>", unsafe_allow_html=True)
    
    # Glassmorphism Profile details card
    profile = cand_detail.get("profile", {})
    signals = cand_detail.get("redrob_signals", {})
    career = cand_detail.get("career_history", [])
    skills = cand_detail.get("skills", [])
    edu = cand_detail.get("education", [])
    
    # Layout within card
    st.markdown(
        f"""
        <div class='glass-card accent-border-blue'>
            <h4 style='margin-top: 0px; margin-bottom: 5px; color: #f8fafc;'>{profile.get('anonymized_name', 'Anonymous Candidate')}</h4>
            <div style='color: #60a5fa; font-weight: 600; font-size: 0.95rem; margin-bottom: 10px;'>{profile.get('current_title', 'Developer')} @ {profile.get('current_company', 'Tech Company')}</div>
            <div style='color: #e2e8f0; font-style: italic; font-size: 0.9rem; margin-bottom: 15px;'>"{profile.get('headline', '')}"</div>
            <p style='color: #94a3b8; font-size: 0.9rem; line-height: 1.5; margin-bottom: 15px;'>{profile.get('summary', '')}</p>
            <div style='display: flex; gap: 20px; font-size: 0.85rem; color: #cbd5e1;'>
                <div>📍 <b>Location:</b> {profile.get('location', 'India')}</div>
                <div>💼 <b>Experience:</b> {profile.get('years_of_experience', 0)} years</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Score Breakdown Tab, Career history Tab, Skills Tab, Behavioral Tab
    t1, t2, t3, t4, t5 = st.tabs(["📊 Score Breakdown", "💼 Career History", "🛠️ Skills & Pedigree", "📈 Behavioral Signals", "🔍 Contrast Cases"])
    
    with t1:
        # Score Breakdown — using structured explanation
        expl = cand_result.get("explanation", {})
        sc = expl.get("score_card", {})
        behavioral = expl.get("behavioral", {})

        # Tier + confidence banner
        tier_label = expl.get("tier_label", "")
        confidence = expl.get("confidence", "")
        col_tier, col_conf = st.columns(2)
        with col_tier:
            tier_color = {
                "Exceptional Fit": "#8b5cf6", "Strong Fit": "#3b82f6",
                "Good Fit": "#06b6d4", "Moderate Fit": "#f59e0b",
                "Marginal Fit": "#f43f5e", "Weak Fit": "#64748b",
            }.get(tier_label, "#64748b")
            st.markdown(
                f"<div style='padding:8px 16px;border-radius:8px;background:rgba(255,255,255,0.04);"
                f"border-left:4px solid {tier_color};margin-bottom:12px;'>"
                f"<span style='color:{tier_color};font-weight:700;'>{tier_label}</span></div>",
                unsafe_allow_html=True,
            )
        with col_conf:
            st.markdown(
                f"<div style='padding:8px 16px;border-radius:8px;background:rgba(255,255,255,0.04);"
                f"border-left:4px solid #64748b;margin-bottom:12px;'>"
                f"<span style='color:#94a3b8;font-size:0.85rem;'>Confidence: </span>"
                f"<span style='font-weight:600;color:#e2e8f0;'>{confidence}</span></div>",
                unsafe_allow_html=True,
            )

        st.markdown("##### Dimension Score Breakdown")

        dim_order = ["career_fit", "skills_match", "experience_fit",
                     "location_logistics", "education", "semantic_similarity"]
        dim_labels = {
            "career_fit": "Career Fit", "skills_match": "Skills Match",
            "experience_fit": "Experience Fit", "location_logistics": "Location & Logistics",
            "education": "Education", "semantic_similarity": "Semantic Similarity",
        }
        for dim in dim_order:
            dim_data = sc.get(dim, {})
            dim_score = dim_data.get("score", 0.0)
            dim_weight = dim_data.get("weight", 0.0)
            dim_contrib = dim_data.get("contribution", 0.0)
            evidence = dim_data.get("evidence", "")
            st.markdown(
                f"""
                <div style='display:flex;justify-content:space-between;font-size:0.85rem;margin-bottom:2px;'>
                    <span style='color:#e2e8f0;font-weight:500;'>{dim_labels[dim]}</span>
                    <span><b style='color:#06b6d4;'>{dim_score:.4f}</b>
                    <span style='color:#64748b;font-size:0.75rem;'> ×{dim_weight:.3f} = {dim_contrib:.4f}</span></span>
                </div>
                <div style='font-size:0.75rem;color:#64748b;background:rgba(255,255,255,0.02);
                    padding:3px 8px;border-radius:4px;margin-bottom:6px;'>{evidence}</div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(dim_score)

        # Behavioral block
        mult = cand_result.get("details", {}).get("behavioral_multiplier", 1.0)
        additive = cand_result.get("details", {}).get("behavioral_additive", 0.0)
        beh_evidence = behavioral.get("evidence", "")
        weighted_total = expl.get("weighted_total", sc.get("weighted_total", 0.0))

        st.markdown(
            f"""
            <div style='margin-top:12px;padding:12px;background:rgba(99,102,241,0.08);
                border-radius:10px;border:1px solid rgba(99,102,241,0.2);'>
                <div style='font-size:0.75rem;color:#8b5cf6;font-weight:600;
                    text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px;'>
                    Score Computation
                </div>
                <div style='display:flex;justify-content:space-between;font-size:0.82rem;margin-bottom:4px;'>
                    <span style='color:#94a3b8;'>Base weighted total</span>
                    <span style='font-family:monospace;'>{weighted_total:.4f}</span>
                </div>
                <div style='display:flex;justify-content:space-between;font-size:0.82rem;margin-bottom:2px;'>
                    <span style='color:#94a3b8;'>Behavioral multiplier</span>
                    <span style='font-family:monospace;color:#a78bfa;'>×{mult:.4f}</span>
                </div>
                <div style='font-size:0.72rem;color:#64748b;margin-bottom:6px;'>{beh_evidence}</div>
                {'<div style="display:flex;justify-content:space-between;font-size:0.82rem;margin-bottom:4px;"><span style="color:#94a3b8;">Behavioral bonus</span><span style="font-family:monospace;">+'+str(round(additive,4))+'</span></div>' if additive > 0 else ''}
                <div style='display:flex;justify-content:space-between;font-size:0.95rem;
                    font-weight:700;border-top:1px solid rgba(255,255,255,0.08);
                    padding-top:8px;margin-top:4px;'>
                    <span>Final Composite Score</span>
                    <span style='color:#60a5fa;'>{cand_result['score']:.4f}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Formula line
        short_formula = expl.get("short_formula", "")
        full_formula = expl.get("formula", "")
        if short_formula:
            st.markdown("##### 🧮 Score Formula")
            st.code(short_formula, language=None)

        # Why-not notes
        why_not = expl.get("why_not_notes", [])
        if why_not:
            st.markdown("##### ⚠️ Concerns")
            for note in why_not:
                st.markdown(
                    f"<div style='font-size:0.82rem;color:#f59e0b;padding:5px 10px;"
                    f"border-left:2px solid #f59e0b;background:rgba(245,158,11,0.07);"
                    f"border-radius:0 6px 6px 0;margin-bottom:5px;'>⚠ {note}</div>",
                    unsafe_allow_html=True,
                )

        # Vs Naive comparison
        naive_rank = naive_rank_map.get(selected_candidate_id)
        if naive_rank:
            pipeline_rank = cand_result["rank"]
            delta = naive_rank - pipeline_rank
            arrow = "▲ promoted" if delta > 0 else ("▼ demoted" if delta < 0 else "same")
            color = "#10b981" if delta > 0 else ("#f43f5e" if delta < 0 else "#64748b")
            st.markdown("##### ⚡ vs Naive Keyword Ranking")
            st.markdown(
                f"<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px;'>"
                f"<div style='text-align:center;padding:12px;background:rgba(255,255,255,0.03);"
                f"border-radius:8px;border:1px solid rgba(255,255,255,0.06);'>"
                f"<div style='font-size:0.7rem;color:#64748b;'>Naive Rank</div>"
                f"<div style='font-size:1.6rem;font-weight:800;color:#64748b;'>#{naive_rank}</div></div>"
                f"<div style='text-align:center;padding:12px;background:rgba(99,102,241,0.08);"
                f"border-radius:8px;border:1px solid rgba(99,102,241,0.3);'>"
                f"<div style='font-size:0.7rem;color:#8b5cf6;'>Pipeline Rank</div>"
                f"<div style='font-size:1.6rem;font-weight:800;color:#06b6d4;'>#{pipeline_rank}</div></div>"
                f"</div>"
                f"<div style='font-size:0.82rem;color:{color};padding:6px 12px;"
                f"border-left:3px solid {color};background:rgba(255,255,255,0.02);"
                f"border-radius:0 6px 6px 0;'>{arrow} {abs(delta)} places</div>",
                unsafe_allow_html=True,
            )

        st.markdown("##### 📝 System Reasoning")
        st.markdown(
            f"""
            <div style='background:rgba(255,255,255,0.02);padding:15px;border-radius:8px;
                border:1px solid rgba(255,255,255,0.05);font-size:0.9rem;line-height:1.5;'>
                {cand_result['reasoning']}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with t2:
        st.markdown("##### Career Timeline")
        for r in career:
            company = r.get("company", "Unknown")
            title = r.get("title", "Developer")
            start = r.get("start_date", "")
            end = r.get("end_date") or "Present"
            dur = r.get("duration_months", 0)
            desc = r.get("description", "")
            
            st.markdown(
                f"""
                <div style='padding-left: 15px; border-left: 2px solid #3b82f6; margin-bottom: 20px;'>
                    <div style='font-weight: 700; font-size: 0.95rem; color: #f8fafc;'>{title}</div>
                    <div style='font-size: 0.85rem; color: #60a5fa; margin-bottom: 5px;'>{company} | {start} to {end} ({dur} mo)</div>
                    <div style='font-size: 0.85rem; color: #94a3b8;'>{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with t3:
        st.markdown("##### Skills Profile")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("**Skills & Proficiency**")
            for s in skills:
                name = s.get("name", "")
                prof = s.get("proficiency", "beginner").title()
                dur = s.get("duration_months", 0)
                st.markdown(f"- **{name}** (`{prof}`, {dur} mo)")
        with col_s2:
            st.markdown("**Academic Background**")
            for e in edu:
                inst = e.get("institution", "University")
                deg = e.get("degree", "Degree")
                field = e.get("field_of_study", "Field")
                grade = e.get("grade", "")
                tier = e.get("tier", "unknown").upper().replace("_", " ")
                st.markdown(
                    f"""
                    <div style='margin-bottom: 10px;'>
                        <div style='font-weight: 600; font-size: 0.85rem;'>{deg} in {field}</div>
                        <div style='font-size: 0.8rem; color: #94a3b8;'>{inst} ({tier})</div>
                        <div style='font-size: 0.8rem; color: #94a3b8;'>Grade: {grade}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with t4:
        st.markdown("##### Observable Behavioral Signals")
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.markdown(f"**Open to Work:** `{'Yes' if signals.get('open_to_work_flag') else 'No'}`")
            st.markdown(f"**Notice Period:** `{signals.get('notice_period_days')} days`")
            st.markdown(f"**Expected Salary:** `{signals.get('expected_salary_range_inr_lpa', {}).get('min')} - {signals.get('expected_salary_range_inr_lpa', {}).get('max')} LPA`")
            st.markdown(f"**Verified Email/Phone:** `{'Email' if signals.get('verified_email') else ''} {'Phone' if signals.get('verified_phone') else ''}`")
            st.markdown(f"**LinkedIn Connected:** `{'Yes' if signals.get('linkedin_connected') else 'No'}`")
        with col_b2:
            st.markdown(f"**Recruiter Response Rate:** `{signals.get('recruiter_response_rate')*100:.1f}%`")
            st.markdown(f"**Avg Response Time:** `{signals.get('avg_response_time_hours')} hours`")
            st.markdown(f"**Interview Attendance Rate:** `{signals.get('interview_completion_rate')*100:.1f}%`")
            st.markdown(f"**GitHub Activity Score:** `{signals.get('github_activity_score')}/100` (-1 if none)")
            st.markdown(f"**Saved by Recruiters (30d):** `{signals.get('saved_by_recruiters_30d')}`")

    with t5:
        st.markdown("##### 🔍 Why Our Pipeline Beats Naive Keyword Matching")
        st.markdown(
            "<div style='font-size:0.85rem;color:#94a3b8;margin-bottom:16px;'>"
            "These 'why-not' cards show candidates a naive keyword matcher would rank highly — "
            "and exactly why this pipeline demoted or excluded them."
            "</div>",
            unsafe_allow_html=True,
        )

        cc1, cc2 = st.columns(2)

        with cc1:
            st.markdown("**🛡 Honeypot Excluded**")
            if honeypots:
                hp = honeypots[0]
                card = generate_honeypot_contrast_card(hp["candidate"], hp["reason"])
                hp_profile = hp["candidate"].get("profile", {})
                st.markdown(
                    f"""
                    <div style='background:rgba(244,63,94,0.07);border:1px solid rgba(244,63,94,0.3);
                        border-radius:12px;padding:16px;'>
                        <div style='font-weight:700;font-size:0.9rem;color:#f8fafc;margin-bottom:4px;'>
                            {hp_profile.get('current_title','Unknown')} @ {hp_profile.get('current_company','Unknown')}
                        </div>
                        <div style='font-size:0.72rem;color:#f43f5e;font-weight:600;margin-bottom:10px;'>
                            🚫 {card['outcome']}
                        </div>
                        <div style='font-size:0.75rem;color:#64748b;margin-bottom:6px;'>Detection flags:</div>
                        {"".join(f"<div style='font-size:0.78rem;color:#94a3b8;padding:4px 8px;border-left:2px solid #f43f5e;background:rgba(255,255,255,0.02);border-radius:0 4px 4px 0;margin-bottom:4px;'>{f}</div>" for f in card['flag_explanations'])}
                        <div style='font-size:0.7rem;color:#475569;margin-top:8px;font-style:italic;'>
                            {card['pipeline_note']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info("No honeypots detected in this dataset.")

        with cc2:
            st.markdown("**📉 Keyword-Stuffer Demoted**")
            # Find the most-demoted candidate
            demoted = None
            for r in results:
                cid = r["candidate_id"]
                n_rank = naive_rank_map.get(cid, r["rank"])
                drop = r["rank"] - n_rank
                if drop >= 5:
                    expl_d = r.get("explanation", {})
                    sc_d = expl_d.get("score_card", {})
                    skills_s = sc_d.get("skills_match", {}).get("score", 0)
                    career_s = sc_d.get("career_fit", {}).get("score", 0)
                    coh_est = max(0.25, min(1.0, career_s / max(skills_s, 0.01)))
                    cand_d = next((c for c in candidates if c.get("candidate_id") == cid), None)
                    if cand_d:
                        demoted = generate_demotion_contrast_card(
                            cand_d, n_rank, r["rank"], sc_d, coh_est
                        )
                        break

            if demoted:
                st.markdown(
                    f"""
                    <div style='background:rgba(245,158,11,0.06);border:1px solid rgba(245,158,11,0.25);
                        border-radius:12px;padding:16px;'>
                        <div style='font-weight:700;font-size:0.9rem;color:#f8fafc;margin-bottom:4px;'>
                            {demoted['title']} @ {demoted['company']}
                        </div>
                        <div style='font-size:0.72rem;color:#f59e0b;font-weight:600;margin-bottom:10px;'>
                            ⚠ {demoted['outcome']}
                        </div>
                        <div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px;'>
                            <div style='text-align:center;padding:10px;background:rgba(0,0,0,0.2);border-radius:8px;'>
                                <div style='font-size:0.7rem;color:#64748b;'>Naive Rank</div>
                                <div style='font-size:1.4rem;font-weight:800;color:#64748b;'>#{demoted['naive_rank']}</div>
                            </div>
                            <div style='text-align:center;padding:10px;background:rgba(245,158,11,0.1);border-radius:8px;'>
                                <div style='font-size:0.7rem;color:#f59e0b;'>Pipeline Rank</div>
                                <div style='font-size:1.4rem;font-weight:800;color:#06b6d4;'>#{demoted['pipeline_rank']}</div>
                            </div>
                        </div>
                        <div style='font-size:0.78rem;color:#94a3b8;line-height:1.5;margin-bottom:8px;'>
                            {demoted['demotion_reason']}
                        </div>
                        <div style='font-size:0.72rem;color:#475569;'>
                            JD-skill hits: {demoted['jd_skill_hits']}/{demoted['total_skills']} skills |
                            Coherence gate: ×{demoted['coherence_multiplier']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info("No significantly demoted candidates found in top results.")

