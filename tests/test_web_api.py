import pytest
from fastapi.testclient import TestClient
from archaeologist.web.server import app

client = TestClient(app)

def test_get_repositories():
    response = client.get("/api/repos")
    assert response.status_code == 200
    data = response.json()
    assert "repositories" in data
    repo_ids = [r["id"] for r in data["repositories"]]
    assert "requests" in repo_ids
    assert "flask" in repo_ids
    assert "mss" in repo_ids

def test_get_repository_details():
    response = client.get("/api/repos/flask")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "pallets/flask"
    assert data["total_commits"] == 673
    assert len(data["starter_questions"]) > 0

def test_get_causal_knowledge_graph():
    response = client.get("/api/graph/flask?limit=20")
    assert response.status_code == 200
    data = response.json()
    assert data["repo_id"] == "flask"
    assert "nodes" in data
    assert "edges" in data
    assert len(data["nodes"]) > 0
    # Verify node structure
    first_node = data["nodes"][0]
    assert "id" in first_node
    assert "type" in first_node
    assert "label" in first_node

def test_validate_key_empty():
    response = client.post("/api/validate-key", json={"api_key": ""})
    assert response.status_code == 400

def test_validate_key_invalid():
    response = client.post("/api/validate-key", json={"api_key": "invalid_key_12345"})
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is False

def test_get_hotspots():
    response = client.get("/api/hotspots/flask?top_n=5")
    assert response.status_code == 200
    data = response.json()
    assert data["repo_id"] == "flask"
    assert "hotspots" in data
    assert len(data["hotspots"]) <= 5

def test_get_ownership():
    response = client.get("/api/ownership/flask")
    assert response.status_code == 200
    data = response.json()
    assert data["repo_id"] == "flask"
    assert "ownership" in data
    assert "bus_factor_risk" in data["ownership"]

def test_get_coupling():
    response = client.get("/api/coupling/flask?top_n=5")
    assert response.status_code == 200
    data = response.json()
    assert data["repo_id"] == "flask"
    assert "couplings" in data

def test_get_symbols():
    response = client.get("/api/symbols/flask?top_n=5")
    assert response.status_code == 200
    data = response.json()
    assert data["repo_id"] == "flask"
    assert "symbols" in data

def test_get_leaderboard():
    response = client.get("/api/eval/leaderboard")
    assert response.status_code == 200
    data = response.json()
    assert "leaderboard" in data
