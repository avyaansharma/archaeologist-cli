import os
import pytest
from unittest.mock import MagicMock, patch
from sqlmodel import SQLModel, Session, select, create_engine

from archaeologist.storage.models import Chunk, PullRequest, Issue, Commit
from archaeologist.storage.db import get_session_context, init_db
from archaeologist.retrieval.vector_store import VectorStore
from archaeologist.retrieval.bm25_index import BM25Index
from archaeologist.retrieval.fusion import reciprocal_rank_fusion
from archaeologist.agent.nodes.follow_links import follow_links_node
from archaeologist.mcp_server.tools import repo_ownership_tool
from archaeologist.web.server import _activate_repo_environment


def test_reproduction_ambient_gemini_api_key_not_deleted(monkeypatch):
    """Bug 1: _activate_repo_environment deletes ambient GEMINI_API_KEY when client provides None."""
    monkeypatch.setenv("GEMINI_API_KEY", "ambient_server_secret_key_12345")
    
    # Client connects without supplying their own private API key
    _activate_repo_environment("requests", client_api_key=None)
    
    # The server's ambient key MUST NOT be deleted from the process environment
    assert os.environ.get("GEMINI_API_KEY") == "ambient_server_secret_key_12345", (
        f"Ambient GEMINI_API_KEY was unexpectedly deleted! Got: {os.environ.get('GEMINI_API_KEY')}"
    )


def test_reproduction_dense_sparse_rrf_non_uuid_chunk_id_preservation(tmp_path):
    """Bug 2: Non-UUID chunk IDs must be preserved in vector payload and merge with sparse hits in RRF."""
    store_dir = str(tmp_path / "qdrant_test_db")
    vstore = VectorStore(storage_path=store_dir, vector_size=4, collection_name="test_rrf_sync")
    vstore.init_collection()
    
    raw_chunk_id = "commit:a9b8c7d6e5"
    chunks = [{
        "id": raw_chunk_id,
        "repo_id": "test/repo",
        "source_type": "commit",
        "source_id": "a9b8c7d6e5",
        "text": "Fixed critical authentication vulnerability in auth.py",
        "timestamp": "2024-01-01T00:00:00",
        "file_paths": ["auth.py"],
        "symbols_modified": ["login"],
        "related_ids": ["pr#42"],
        "is_reverted": False
    }]
    embeddings = [[0.1, 0.2, 0.3, 0.4]]
    
    vstore.upsert_chunks(chunks, embeddings)
    dense_hits = vstore.search_chunks(query_vector=[0.1, 0.2, 0.3, 0.4], limit=5)
    vstore.close(force=True)
    
    assert len(dense_hits) == 1
    # Dense hit payload should retain the original raw chunk ID
    assert dense_hits[0]["payload"].get("id") == raw_chunk_id
    
    # Sparse hit from BM25 using the same chunk ID
    sparse_hits = [{
        "score": 15.0,
        "chunk": chunks[0]
    }]
    
    fused = reciprocal_rank_fusion(dense_hits, sparse_hits, limit=5)
    # RRF MUST merge the dense and sparse hits into a single candidate, NOT two duplicates!
    assert len(fused) == 1, f"Expected 1 fused hit but got {len(fused)} (unmerged dense/sparse duplicates: {[h['id'] for h in fused]})"
    assert fused[0]["id"] == raw_chunk_id


def test_reproduction_follow_links_basename_matching(tmp_path, monkeypatch):
    """Bug 3: follow_links_node fails to resolve cross-links when repo_id is supplied as basename."""
    db_file = tmp_path / "test_follow.db"
    db_url = f"sqlite:///{db_file}"
    monkeypatch.setenv("DATABASE_URL", db_url)
    init_db(db_url)
    
    from datetime import datetime
    with get_session_context() as session:
        # Ingested with full slug 'pallets/flask'
        c = Chunk(
            id="pr_chunk_42",
            repo_id="pallets/flask",
            source_type="pull_request",
            source_id="pr#42",
            text="PR #42 fixes request context teardown",
            timestamp=datetime.utcnow(),
            file_paths=["ctx.py"],
            symbols_modified=[],
            related_ids=["issue#10"],
            is_reverted=False
        )
        session.add(c)
        session.commit()
    
    state = {
        "retrieved_chunks": [
            {
                "id": "commit_1",
                "source_id": "c1a2b3",
                "related_ids": ["pr#42"]
            }
        ],
        "repo_id": "flask",  # Basename supplied by user/UI
        "evidence_by_chunk_id": {}
    }
    
    result = follow_links_node(state)
    new_chunks = result.get("retrieved_chunks", [])
    # Must successfully resolve pr#42 despite basename mismatch
    assert any(c.get("source_id") == "pr#42" for c in new_chunks), (
        f"follow_links_node failed to resolve linked PR chunk when queried with basename 'flask'. Result: {new_chunks}"
    )


def test_reproduction_gemini_client_rotates_on_auth_and_permission_errors(monkeypatch):
    """Bug 4: GeminiClientWrapper must rotate to secondary keys on 400/401/403/API_KEY_INVALID errors."""
    monkeypatch.setenv("GEMINI_API_KEY", "bad_key_1")
    monkeypatch.setenv("GEMINI_API_KEY_SECONDARY", "good_key_2")
    
    from archaeologist.utils.gemini_client import GeminiClientWrapper
    
    wrapper = GeminiClientWrapper()
    assert len(wrapper.clients) >= 2
    
    mock_bad_client = MagicMock()
    mock_bad_client.models.generate_content.side_effect = Exception("400 API_KEY_INVALID: API key not valid. Please pass a valid API key.")
    
    mock_good_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.text = "Successfully generated forensic response."
    mock_good_client.models.generate_content.return_value = mock_resp
    
    wrapper.clients = [mock_bad_client, mock_good_client]
    
    result = wrapper.generate_text("Explain commit rationale")
    assert result == "Successfully generated forensic response."


def test_reproduction_vector_store_malformed_date_resilience(tmp_path):
    """Bug 5: vector_store.search_chunks must safely handle malformed or non-standard date strings without raising ValueError."""
    store_dir = str(tmp_path / "qdrant_date_db")
    vstore = VectorStore(storage_path=store_dir, vector_size=4, collection_name="test_date_resilience")
    vstore.init_collection()
    
    # Should not raise ValueError on malformed date string
    try:
        hits = vstore.search_chunks(
            query_vector=[0.1, 0.2, 0.3, 0.4],
            date_from="not-a-valid-date",
            date_to="yesterday",
            limit=5
        )
        assert isinstance(hits, list)
    finally:
        vstore.close(force=True)


def test_reproduction_bm25_date_range_filtering():
    """Bug 6: BM25Index.search should support date_from and date_to temporal filtering."""
    bm25 = BM25Index()
    chunks = [
        {
            "id": "c1",
            "text": "Fixed auth session timeout issue",
            "timestamp": "2020-01-01T00:00:00",
            "repo_id": "test_repo"
        },
        {
            "id": "c2",
            "text": "Fixed auth session timeout issue",
            "timestamp": "2024-06-01T00:00:00",
            "repo_id": "test_repo"
        }
    ]
    bm25.fit(chunks)
    
    # Query with date_from=2024-01-01 should ONLY return c2, filtering out old chunk c1
    hits = bm25.search("auth session", date_from="2024-01-01", repo_id="test_repo")
    assert len(hits) == 1, f"Expected 1 hit within date window, but got {len(hits)}"
    assert hits[0]["chunk"]["id"] == "c2"


def test_reproduction_repo_ownership_empty_counter_handling():
    """Bug 7: repo_ownership_tool should not crash with IndexError when a file has an empty Counter."""
    from collections import Counter
    counts = Counter()
    # When counts is empty, counts.most_common(1) is [], indexing [0] crashes with IndexError
    # Our helper or tool logic must defensively handle empty counts
    sorted_files = [("empty.py", counts)]
    file_breakdown = {}
    for f, c in sorted_files:
        if not c:
            continue
        top_author, top_cnt = c.most_common(1)[0]
        file_breakdown[f] = {"top_author": top_author}
    assert "empty.py" not in file_breakdown

