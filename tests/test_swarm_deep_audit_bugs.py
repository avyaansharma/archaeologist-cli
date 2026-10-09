import os
import sys
import tempfile
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from sqlmodel import select

from archaeologist.storage.db import init_db, get_session_context
from archaeologist.storage.models import Commit, Chunk, SymbolIndex
from archaeologist.mcp_server.tools import symbol_history_tool, blame_explain_tool
from archaeologist.ingestion.symbol_parser import extract_modified_line_numbers_from_diff
from archaeologist.ingestion.chunker import chunk_commit
from archaeologist.retrieval.fusion import reciprocal_rank_fusion
from archaeologist.agent.nodes.follow_links import follow_links_node


@pytest.fixture
def clean_db(tmp_path):
    db_file = tmp_path / "test_swarm_deep.db"
    db_url = f"sqlite:///{db_file.as_posix()}"
    old_url = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = db_url
    init_db(db_url)
    yield db_url
    if old_url:
        os.environ["DATABASE_URL"] = old_url
    else:
        os.environ.pop("DATABASE_URL", None)


def test_symbol_history_tool_avoids_full_orm_entity_hydration(clean_db, monkeypatch):
    """Bug 1: symbol_history_tool must NOT execute unconstrained select(Commit).all(),
    which hydrates entire commit diffs and messages into memory. It must use SQL LIKE
    filtering and column projections.
    """
    now = datetime.now(timezone.utc)
    with get_session_context() as session:
        for i in range(10):
            session.add(Commit(
                sha=f"sha_{i:040x}",
                repo_id="repo_sym",
                author_name=f"Author {i}",
                author_email="a@example.com",
                authored_date=now,
                message=f"Commit message {i}",
                diff_summary="Large diff summary " * 20,
                files_changed=["app.py"],
                symbols_modified=["app.py:main"] if i == 0 else ["app.py:other"]
            ))

    executed_stmts = []
    real_get_session_context = get_session_context

    from contextlib import contextmanager
    @contextmanager
    def spy_session():
        with real_get_session_context() as sess:
            real_exec = sess.exec
            def spy_exec(stmt, *args, **kwargs):
                executed_stmts.append(stmt)
                return real_exec(stmt, *args, **kwargs)
            sess.exec = spy_exec
            yield sess

    monkeypatch.setattr("archaeologist.mcp_server.tools.get_session_context", spy_session)

    res = symbol_history_tool("main", repo_id="repo_sym")
    assert len(res) == 1
    assert res[0]["sha"] == f"sha_{0:040x}"

    for stmt in executed_stmts:
        sql_str = str(stmt).lower()
        if "commit" in sql_str:
            assert "diff_summary" not in sql_str, "Memory leak: symbol_history_tool should not load diff_summary into memory!"
            assert "where" in sql_str, "Unbounded table scan: symbol_history_tool missing WHERE clause!"


def test_diff_parser_handles_quoted_file_paths_with_spaces():
    """Bug 2: extract_modified_line_numbers_from_diff drops files whose paths contain spaces
    because git quotes them in diff headers (e.g. --- \"a/path with space/file.py\").
    The parser must strip enclosing quotes.
    """
    diff_text = (
        '--- "a/src/components with spaces/Button.py"\n'
        '+++ "b/src/components with spaces/Button.py"\n'
        '@@ -10,3 +10,4 @@\n'
        ' def render():\n'
        '+    print("clicked")\n'
        '     return True\n'
    )
    res = extract_modified_line_numbers_from_diff(diff_text)
    assert "src/components with spaces/Button.py" in res, (
        "Diff parser dropped quoted file path containing spaces!"
    )
    assert 11 in res["src/components with spaces/Button.py"]["added"]


def test_chunk_commit_subchunks_preserve_causal_revert_links():
    """Bug 3: When a commit has squash-merge bullets or paragraphs, sub-chunks lose
    causal links because chunker.py passed commit.get('related_ids') instead of the
    enriched related_ids containing reverts_sha and superseded_by_sha.
    """
    commit_data = {
        "sha": "c1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0",
        "author_name": "Alice",
        "authored_date": datetime.now(timezone.utc),
        "message": "Revert accidental breaking change\n\n* Fix auth handler crash\n* Restore legacy token parser",
        "files_changed": ["auth.py"],
        "symbols_modified": ["auth.py:login"],
        "is_revert": True,
        "reverts_sha": "0000111122223333444455556666777788889999",
        "superseded_by_sha": None,
        "related_ids": ["issue#100"]
    }

    chunks = chunk_commit(commit_data, diff_summary="Reverted auth crash", repo_id="test_repo")
    assert len(chunks) >= 2, "Expected head chunk + subchunks for bullet items"

    for idx, c in enumerate(chunks):
        assert "0000111122223333444455556666777788889999" in c["related_ids"], (
            f"Chunk {idx} ({c['id']}) lost causal revert link! Sub-chunks must inherit enriched related_ids."
        )


def test_blame_explain_safe_offline_fallback(tmp_path, monkeypatch):
    """Bug 4: blame_explain_tool crashes with ValueError if GEMINI_API_KEY is not set.
    It must provide a graceful offline fallback with extracted commit history rather than crashing.
    """
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY_SECONDARY", raising=False)

    # Call blame_explain_tool with an empty or mock repo path
    res = blame_explain_tool(
        repo_path=str(tmp_path),
        file_path="non_existent.py",
        line_start=1,
        line_end=5
    )
    assert isinstance(res, dict)
    assert "explanation" in res
    assert "file_path" in res


def test_fusion_handles_none_payload_gracefully():
    """Bug 5: reciprocal_rank_fusion raises TypeError: argument of type 'NoneType' is not iterable
    if dense_hits contains an item where payload is None.
    """
    dense_hits = [
        {"id": "hit_1", "score": 0.9, "payload": None}
    ]
    sparse_hits = []

    # Should not raise TypeError or AttributeError
    res = reciprocal_rank_fusion(dense_hits, sparse_hits, limit=5)
    assert len(res) == 1
    assert res[0]["id"] == "hit_1"
    assert isinstance(res[0]["payload"], dict)


def test_follow_links_synchronizes_evidence_by_chunk_id(clean_db):
    """Bug 6: follow_links_node updates retrieved_chunks but fails to update evidence_by_chunk_id.
    This causes subsequent sub-question searches to re-retrieve and duplicate the same chunks.
    """
    now = datetime.now(timezone.utc)
    with get_session_context() as session:
        session.add(Chunk(
            id="linked_chunk_1",
            repo_id="test_repo",
            source_type="issue",
            source_id="issue#42",
            text="Issue 42 bug report",
            timestamp=now,
            file_paths=["app.py"],
            symbols_modified=[],
            related_ids=[]
        ))

    state = {
        "repo_id": "test_repo",
        "retrieved_chunks": [{
            "id": "root_chunk",
            "source_type": "commit",
            "source_id": "sha123",
            "text": "Fix issue",
            "related_ids": ["issue#42"]
        }],
        "evidence_by_chunk_id": {
            "root_chunk": {"id": "root_chunk"}
        }
    }

    result = follow_links_node(state)
    assert "retrieved_chunks" in result
    assert "evidence_by_chunk_id" in result, (
        "follow_links_node did not return evidence_by_chunk_id! This causes duplicate chunk retrieval in later sub-questions."
    )
    assert "linked_chunk_1" in result["evidence_by_chunk_id"]
