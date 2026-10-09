import pytest
from unittest.mock import MagicMock, patch
from sqlmodel import select, SQLModel

from archaeologist.storage.db import init_db, get_session_context
from archaeologist.storage.models import Chunk, Commit, PullRequest
from archaeologist.agent.nodes.search import search_node
from archaeologist.agent.nodes.follow_links import follow_links_node
from archaeologist.mcp_server.tools import find_related_discussion_tool


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path):
    db_file = tmp_path / "test_optimizations.db"
    db_url = f"sqlite:///{db_file}"
    init_db(db_url)
    yield


def test_dense_search_mock_vector_bypass():
    """Verify search_node skips vector_store when embedder returns mock vector (success=False)."""
    state = {
        "question": "Why did authentication change?",
        "search_queries": ["authentication change"],
        "_current_plan": {},
        "retrieved_chunks": [],
        "evidence_by_chunk_id": {}
    }

    mock_embedder = MagicMock()
    mock_embedder.dimension = 384
    # Simulate failed API call returning mock vector with success=False
    mock_embedder.embed_texts.return_value = ([[0.1] * 384], [False])

    mock_vector_store = MagicMock()

    with patch("archaeologist.agent.nodes.search._get_embedder", return_value=mock_embedder), \
         patch("archaeologist.agent.nodes.search._get_vector_store", return_value=mock_vector_store), \
         patch("archaeologist.agent.nodes.search.BM25Index"):
        
        result = search_node(state)

        # Confirm vector_store.search_chunks was NOT called because success_flags[0] is False
        mock_vector_store.search_chunks.assert_not_called()
        assert "retrieved_chunks" in result


def test_find_related_discussion_tool_fast_lookup():
    """Verify find_related_discussion_tool searches linked commits efficiently without full table scan."""
    from datetime import datetime
    with get_session_context() as session:
        # Clean up any existing fixture
        existing_pr = session.get(PullRequest, ("", 42))
        if existing_pr:
            session.delete(existing_pr)
        existing_c = session.get(Commit, "a1b2c3d4e5f67890123456789012345678901234")
        if existing_c:
            session.delete(existing_c)
        session.commit()

        pr = PullRequest(
            number=42,
            title="Refactor auth pipeline",
            state="merged",
            author="testuser",
            created_at=datetime.utcnow(),
            linked_commit_shas=[]
        )
        session.add(pr)

        c1 = Commit(
            sha="a1b2c3d4e5f67890123456789012345678901234",
            author_name="Alice",
            author_email="alice@example.com",
            authored_date=datetime.utcnow(),
            message="Fix #42 authentication deadlock",
            files_changed=["auth.py"],
            symbols_modified=[]
        )
        session.add(c1)

    result = find_related_discussion_tool("#42")
    assert len(result["pull_requests"]) == 1
    assert result["pull_requests"][0]["number"] == 42
    assert len(result["commits"]) == 1
    assert result["commits"][0]["sha"] == "a1b2c3d4e5f67890123456789012345678901234"


def test_follow_links_node_db_context():
    """Verify follow_links_node uses session context safely and resolves related IDs."""
    from datetime import datetime
    with get_session_context() as session:
        # Clean up any existing fixture
        for cid in ["chunk-linked-123", "chunk-linked-456"]:
            existing_chunk = session.get(Chunk, cid)
            if existing_chunk:
                session.delete(existing_chunk)
        session.commit()

        c1 = Chunk(
            id="chunk-linked-123",
            source_type="pr",
            source_id="pr#100",
            text="PR #100 discussion text",
            timestamp=datetime.utcnow(),
            file_paths=["src/main.py"],
            symbols_modified=[],
            related_ids=["issue#200"],
            is_reverted=False
        )
        c2 = Chunk(
            id="chunk-linked-456",
            source_type="issue",
            source_id="issue#200",
            text="Issue #200 bug report",
            timestamp=datetime.utcnow(),
            file_paths=["src/main.py"],
            symbols_modified=[],
            related_ids=[],
            is_reverted=False
        )
        session.add(c1)
        session.add(c2)


    state = {
        "retrieved_chunks": [{
            "id": "chunk-linked-123",
            "source_type": "pr",
            "source_id": "pr#100",
            "text": "PR #100 discussion text",
            "related_ids": ["issue#200"],
            "is_reverted": False
        }]
    }

    result = follow_links_node(state)
    assert "retrieved_chunks" in result
    assert len(result["retrieved_chunks"]) == 2
    assert result["retrieved_chunks"][1]["id"] == "chunk-linked-456"
