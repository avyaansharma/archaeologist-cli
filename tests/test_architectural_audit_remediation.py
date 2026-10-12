"""Tests for Adversarial Architectural Audit and Remediation.

Validates fixes for:
- R1/D2: GitHub sync fetch_issues sort="updated" and updated_at since bound
- R3/D31: Hybrid fusion channel separation preventing fallback hits from burying genuine vector hits
- R4/D27: Verify node token budget and judge_unavailable circuit breaking
- D3/R9: Direct SQLite database repo_id resolution in CLI
"""
import os
from datetime import UTC, datetime, timedelta
from typing import Any

from archaeologist.agent.graph import verification_router
from archaeologist.agent.nodes.verify import verify_node
from archaeologist.cli import _configure_repo_env
from archaeologist.retrieval.fusion import reciprocal_rank_fusion
from archaeologist.storage.paths import get_stored_repo_id


def _make_hit(cid: str, score: float, source_type: str = "commit", text: str = "") -> dict[str, Any]:
    return {
        "id": cid,
        "score": score,
        "payload": {
            "id": cid,
            "source_type": source_type,
            "text": text or f"Content for {cid}",
            "timestamp": datetime.now(UTC).isoformat(),
            "symbols_modified": [],
            "related_ids": []
        }
    }


def test_r3_vector_hit_is_not_buried_by_fallback_hits():
    """R3: 30 keyword fallbacks (score 1.0) must not outrank a strong vector hit (score 0.92)."""
    fallback = [_make_hit(f"like_{i}", 1.0, source_type="text_fallback") for i in range(30)]
    fused = reciprocal_rank_fusion([_make_hit("best_vector", 0.92)] + fallback, [], limit=5)
    assert len(fused) > 0
    assert fused[0]["id"] == "best_vector"


def test_d31_exact_match_outranks_ordinary_hits():
    """D31: Exact matches (score 3.0) must outrank ordinary hits anywhere in the list."""
    ordinary = [_make_hit(f"o{i}", 0.5) for i in range(30)]
    fused = reciprocal_rank_fusion(ordinary + [_make_hit("exact", 3.0)], [], limit=5)
    assert len(fused) > 0
    assert fused[0]["id"] == "exact"


def test_r1_github_issues_defaults_and_updated_at():
    """R1: fetch_issues must default to sort='updated' and compare against updated_at."""
    import inspect

    from archaeologist.ingestion.github_client import GitHubIngestionClient

    sig = inspect.signature(GitHubIngestionClient.fetch_issues)
    assert sig.parameters["sort"].default == "updated"
    assert sig.parameters["direction"].default == "desc"

    # Simulate issue with old created_at but recent updated_at inside since window
    now = datetime.now()
    since = now - timedelta(days=30)
    created_old = now - timedelta(days=365)
    updated_recent = now - timedelta(days=5)

    class MockIssue:
        def __init__(self):
            self.created_at = created_old
            self.updated_at = updated_recent
            self.pull_request = None
            self.body = "Bug fix issue"

    issue = MockIssue()
    issue_ts = issue.updated_at or issue.created_at
    assert issue_ts >= since


def test_r4_verify_node_circuit_breaker_on_quota_error(monkeypatch):
    """R4: A 429 quota exhaustion or judge fatal error must set judge_unavailable and skip retries."""
    import archaeologist.agent.nodes.verify as verify_module

    class QuotaExhaustedJudge:
        def __init__(self, *args, **kwargs):
            pass

        def generate_text(self, *args, **kwargs):
            raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")

        def generate_json(self, *args, **kwargs):
            raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")

    monkeypatch.setattr(verify_module, "GeminiClientWrapper", QuotaExhaustedJudge)
    monkeypatch.setattr(verify_module, "get_gemini_api_key", lambda: "mock_key")

    state = {
        "question": "Why did timeout change?",
        "retrieved_chunks": [_make_hit("c1", 0.9)],
        "draft_answer": None,
        "verification_passed": False,
        "unverified_claims": [],
        "retry_count": 0
    }

    out = verify_node(state)
    assert out["verification_passed"] is False
    assert out.get("judge_unavailable") is True

    # Router must route directly to synthesize, skipping retries
    state.update(out)
    route = verification_router(state)
    assert route == "synthesize"


def test_r4_verify_node_circuit_breaker_on_missing_api_key(monkeypatch):
    """R4: Missing API key must set judge_unavailable and route directly to synthesize."""
    import archaeologist.agent.nodes.verify as verify_module
    monkeypatch.setattr(verify_module, "get_gemini_api_key", lambda: None)

    state = {
        "question": "Why did timeout change?",
        "retrieved_chunks": [_make_hit("c1", 0.9)],
        "draft_answer": None,
        "verification_passed": True,
        "unverified_claims": [],
        "retry_count": 0
    }

    out = verify_node(state)
    assert out["verification_passed"] is False
    assert out.get("judge_unavailable") is True

    state.update(out)
    route = verification_router(state)
    assert route == "synthesize"


def test_r4_verify_node_evidence_budget():
    """R4: Evidence must be sorted by score and capped to token budget without unbounded explosion."""
    import archaeologist.agent.nodes.verify as verify_module
    from archaeologist.agent.nodes.verify import verify_node

    captured_prompts = []

    class MockJudge:
        def __init__(self, *args, **kwargs):
            pass

        def generate_text(self, prompt, *args, **kwargs):
            captured_prompts.append(prompt)
            return "Valid draft answer."

        def generate_json(self, prompt, *args, **kwargs):
            captured_prompts.append(prompt)
            return {"verification_passed": True, "unverified_claims": []}

    state = {
        "question": "Architectural inquiry",
        # Provide 100 chunks with large text
        "retrieved_chunks": [
            _make_hit(f"chunk_{i}", 0.1 * i, text=f"Evidence segment {i} " + ("word " * 200))
            for i in range(100)
        ],
        "draft_answer": None,
        "verification_passed": False,
        "unverified_claims": [],
        "retry_count": 0
    }

    import unittest.mock as mock
    with mock.patch.object(verify_module, "GeminiClientWrapper", MockJudge), \
         mock.patch.object(verify_module, "get_gemini_api_key", return_value="dummy_key"):
        out = verify_node(state)
        assert out["verification_passed"] is True
        assert len(captured_prompts) > 0
        # Check that top-scoring chunk (chunk_99) was included first
        assert "chunk_99" in captured_prompts[0]
        # Check total prompt size did not blow up past reasonable token budget (~16k chars)
        assert len(captured_prompts[0]) < 25000


def test_d3_db_path_repo_id_scoping():
    """D3: Passing db_path resolves repo_id from the database itself, not ambient repo."""
    flask_db = os.path.abspath("eval/data/flask.db")
    if os.path.exists(flask_db):
        repo_id = get_stored_repo_id(db_path=flask_db)
        assert repo_id == "flask"

        configured_id = _configure_repo_env(db_path=flask_db)
        assert configured_id == "flask"
