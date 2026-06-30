"""
test_api.py — Integration tests for the Flask API endpoints.

Proves the backend server routes are functioning correctly and returning
the expected schema to the frontend.
"""

import sys
import os
import json
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the Flask app
try:
    from demo_server import app, _init
except ImportError:
    pytest.skip("Could not import demo_server, skipping API tests", allow_module_level=True)

DEFAULT_CANDIDATES = (Path(__file__).parent.parent.parent /
                      "[PUB] India_runs_data_and_ai_challenge" /
                      "India_runs_data_and_ai_challenge" / "candidates.jsonl")

@pytest.fixture(scope="module")
def client():
    """Create a Flask test client with the app initialized."""
    app.config["TESTING"] = True
    
    # Initialize state if not already done
    p = os.environ.get("REDROB_CANDIDATES", str(DEFAULT_CANDIDATES))
    if not Path(p).exists():
        pytest.skip(f"candidates file not found at {p}")
        
    _init(p)
    
    with app.test_client() as client:
        yield client

def test_api_status(client):
    """Test the /api/status endpoint returns system health."""
    response = client.get("/api/status")
    assert response.status_code == 200
    
    data = response.get_json()
    assert "status" in data
    assert data["status"] == "ok"
    assert "candidates_loaded" in data
    assert data["candidates_loaded"] > 0
    assert "default_jd_title" in data

def test_api_evaluation(client):
    """Test the /api/evaluation endpoint returns benchmarking and gold metrics."""
    response = client.get("/api/evaluation")
    assert response.status_code == 200
    
    data = response.get_json()
    assert "gold_metrics" in data
    assert "composite" in data["gold_metrics"]
    assert data["gold_metrics"]["composite"] == 0.920
    
    assert "ablation" in data
    assert len(data["ablation"]) > 0
    
    assert "benchmark" in data
    assert data["benchmark"]["meets_constraint"] is True
    assert len(data["benchmark"]["stages"]) == 5

def test_api_rank_jd_error_handling(client):
    """Test the /api/rank-jd endpoint handles missing payload gracefully."""
    # Missing jd_text
    response = client.post("/api/rank-jd", json={"n": 10})
    assert response.status_code == 400
    assert "error" in response.get_json()
    
    # Malformed JSON
    response = client.post("/api/rank-jd", data="not json", content_type="application/json")
    assert response.status_code in [400, 500]  # Flask might return 400 or 500 depending on handler
