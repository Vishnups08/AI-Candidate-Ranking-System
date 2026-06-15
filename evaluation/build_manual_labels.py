import json
import sys
import re
from pathlib import Path
from collections import Counter

# Ground-truth heuristic labeling script
# Replaces the arbitrary 50 labels with programmatic labels based on JD rules

def score_candidate_heuristic(c):
    score = 1 # Base score for non-honeypots

    profile = c.get("profile", {})
    title = profile.get("current_title", "").lower()
    yoe = profile.get("years_of_experience", 0)
    skills = [s.get("name", "").lower() for s in c.get("skills", [])]
    career = c.get("career_history", [])
    
    # 1. Title/Role Fit
    if "ai engineer" in title or "machine learning" in title or "ml engineer" in title:
        score += 1
    elif "data scientist" in title or "nlp engineer" in title:
        score += 0.5
        
    # 2. Key Skills Match
    core_skills = ["sentence-transformers", "pinecone", "weaviate", "qdrant", "milvus", "faiss", "elasticsearch", "hybrid search", "ndcg", "mrr", "map", "a/b test"]
    nice_skills = ["lora", "qlora", "peft", "learning-to-rank", "xgboost"]
    
    matched_core = sum(1 for s in skills if any(cs in s for cs in core_skills))
    matched_nice = sum(1 for s in skills if any(ns in s for ns in nice_skills))
    
    if matched_core >= 2:
        score += 1
    if matched_nice >= 1:
        score += 0.5
        
    # 3. YOE Fit (5-9 years preferred)
    if 5 <= yoe <= 9:
        score += 1
    elif 3 <= yoe < 5 or 9 < yoe <= 12:
        score += 0.5
        
    # 4. Disqualifiers (Cap score to 0 or 1)
    # Research only?
    all_research = True
    for role in career:
        desc = role.get("description", "").lower()
        if "production" in desc or "deployed" in desc or "shipped" in desc:
            all_research = False
            break
    if all_research and len(career) > 1:
        return 0 # Disqualified
        
    # Career long consulting?
    consulting_firms = {"tcs", "infosys", "wipro", "cognizant", "accenture", "capgemini", "ibm", "tech mahindra", "hcl"}
    all_consulting = all(
        role.get("company", "").lower().strip() in consulting_firms
        for role in career
    ) if career else False
    if all_consulting:
        return 0
        
    # Title chaser (avg tenure < 18mo across >= 4 jobs)
    if len(career) >= 4:
        avg_tenure = sum(r.get("duration_months", 24) for r in career) / len(career)
        if avg_tenure < 18:
            return 0
            
    # Primarily CV/Robotics
    all_skill_text = " ".join(skills)
    if ("computer vision" in all_skill_text or "robotics" in all_skill_text) and "nlp" not in all_skill_text:
        return 0
        
    # Round to int and cap at 5
    final_score = min(5, round(score))
    return final_score

def build_labels(dataset_path, output_path):
    print(f"Loading candidates from {dataset_path}...")
    candidates = []
    
    if dataset_path.endswith(".jsonl"):
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    candidates.append(json.loads(line))
    else:
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            candidates = data.get("candidates", []) if isinstance(data, dict) else data
            
    print(f"Loaded {len(candidates)} candidates.")
    
    labels = {}
    for c in candidates:
        cid = c.get("candidate_id")
        if cid:
            labels[cid] = score_candidate_heuristic(c)
            
    print(f"Generated {len(labels)} labels.")
    
    # Distribution
    dist = Counter(labels.values())
    print("Label distribution:")
    for score in sorted(dist.keys()):
        print(f"  Score {score}: {dist[score]} candidates")
        
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    import os
    # Find dataset
    ds_path = r"c:\Users\Vishnu\Music\New folder\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\candidates.jsonl"
    
    if not os.path.exists(ds_path):
        print(f"Dataset not found at {ds_path}")
        sys.exit(1)
        
    build_labels(ds_path, r"c:\Users\Vishnu\Music\New folder\redrob-ranker\evaluation\manual_labels.json")
