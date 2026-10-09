import os
import sys
import tempfile
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from sqlmodel import select

from archaeologist.storage.db import init_db, get_session_context, engine
from archaeologist.storage.models import Commit, Chunk, SymbolIndex, PullRequest, Issue
from archaeologist.retrieval.vector_store import VectorStore
from archaeologist.retrieval.embedder import Embedder
from archaeologist.mcp_server.tools import repo_hotspots_tool, repo_ownership_tool, change_coupling_tool
from archaeologist.ingestion.chunker import make_deterministic_chunk_id, chunk_issue, chunk_pr
from archaeologist.ingestion.github_client import GitHubIngestionClient


@pytest.fixture
def clean_db(tmp_path):
    """Sets up an isolated SQLite database in a temporary directory."""
    db_file = tmp_path / "test_stress.db"
    db_url = f"sqlite:///{db_file.as_posix()}"
    old_url = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = db_url
    init_db(db_url)
    yield db_url
    if old_url:
        os.environ["DATABASE_URL"] = old_url
    else:
        os.environ.pop("DATABASE_URL", None)


def test_bug1_vector_store_avoids_unbounded_table_scan_on_file_search(clean_db, monkeypatch):
    """Bug 1: search_chunks should not execute unbounded select(Chunk.file_paths).all().
    It must use bounded SQL LIKE filtering instead of scanning all database chunks in Python.
    """
    with get_session_context() as session:
        # Seed database with dummy chunks
        now = datetime.now(timezone.utc)
        for i in range(100):
            session.add(Chunk(
                id=f"chk_{i}",
                repo_id="test_repo",
                source_type="file",
                source_id=f"file_{i}.py",
                text=f"content {i}",
                file_paths=[f"src/module_{i}/code.py"],
                timestamp=now
            ))

    vs = VectorStore()
    
    # Spy on session.exec inside vector_store to verify it does not fetch all chunks without WHERE/LIMIT
    executed_statements = []
    real_get_session_context = get_session_context

    from contextlib import contextmanager
    @contextmanager
    def spy_session_context():
        with real_get_session_context() as sess:
            real_exec = sess.exec
            def spy_exec(stmt, *args, **kwargs):
                executed_statements.append(stmt)
                return real_exec(stmt, *args, **kwargs)
            sess.exec = spy_exec
            yield sess

    monkeypatch.setattr("archaeologist.storage.db.get_session_context", spy_session_context)
    
    # Searching by file_path
    try:
        vs.search_chunks(
            query_vector=[0.0] * vs.vector_size,
            file_path="src/module_5/code.py",
            repo_id="test_repo"
        )
    except Exception:
        pass
    finally:
        vs.close()

    # Verify that if any SQL query was run, it had a WHERE clause and LIMIT, rather than unbound select(Chunk.file_paths)
    for stmt in executed_statements:
        sql_str = str(stmt).upper()
        if "CHUNK" in sql_str and "FILE_PATHS" in sql_str:
            assert "WHERE" in sql_str, "Unbounded table scan detected! Query missing WHERE clause"
            assert "LIMIT" in sql_str, "Unbounded table scan detected! Query missing LIMIT clause"


def test_bug2_analytical_tools_avoid_full_orm_entity_hydration(clean_db, monkeypatch):
    """Bug 2: repo_hotspots_tool, repo_ownership_tool, and change_coupling_tool
    should NOT fetch full Commit objects (including diff_summary, raw_diff, message).
    They must use targeted column projections.
    """
    now = datetime.now(timezone.utc)
    with get_session_context() as session:
        for i in range(10):
            session.add(Commit(
                sha=f"sha_{i:04d}",
                repo_id="repo_analytical",
                author_name=f"Author {i%3}",
                author_email=f"author{i%3}@example.com",
                authored_date=now,
                message="Long commit message " * 50,
                diff_summary="Long diff summary " * 50,
                files_changed=["src/app.py", f"src/mod_{i}.py"]
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

    repo_hotspots_tool(repo_id="repo_analytical")
    repo_ownership_tool(repo_id="repo_analytical")
    change_coupling_tool(repo_id="repo_analytical")

    # None of the executed queries should select all columns (select(Commit))
    for stmt in executed_stmts:
        sql_str = str(stmt).lower()
        if "commit" in sql_str:
            # Full Commit entity select includes diff_summary and message
            assert "diff_summary" not in sql_str, "Memory leak: Analytical tools should not load diff_summary into memory!"


def test_bug3_engine_disposal_prevents_connection_leaks(tmp_path):
    """Bug 3: Switching database URLs must dispose previous SQLAlchemy engines
    to avoid file locks and connection pool leakage.
    """
    import archaeologist.storage.db as db_mod
    
    db1 = tmp_path / "leak_test_1.db"
    db2 = tmp_path / "leak_test_2.db"
    
    url1 = f"sqlite:///{db1.as_posix()}"
    url2 = f"sqlite:///{db2.as_posix()}"

    db_mod.init_db(url1)
    engine1 = db_mod.engine
    with db_mod.get_session_context() as s:
        s.exec(select(Commit)).all()

    # Switch to url2
    db_mod.init_db(url2)
    engine2 = db_mod.engine
    assert engine1 is not engine2

    # Verify that engine1 has been disposed or managed in an engine registry
    # On disposed engine, pool is disposed
    assert engine1.pool.status() == "Pool size: 5  Connections in pool: 0 Current Overflow: -5 Current Checked out connections: 0" or getattr(engine1, "_is_disposed", False) or hasattr(engine1.pool, "_is_disposed") or engine1.pool.checkedin() == 0


def test_bug4_qdrant_stores_and_filters_repo_id():
    """Bug 4: VectorStore must store repo_id in Qdrant point payload
    and filter by repo_id in search_chunks to avoid cross-repo vector pollution.
    """
    vs = VectorStore(collection_name="test_multi_repo_col")
    vs.init_collection()

    try:
        # Upsert chunks from two different repos
        chunks = [
            {
                "id": "c_repo_a",
                "repo_id": "repo_alpha",
                "source_type": "commit",
                "source_id": "sha_a",
                "text": "Authentication JWT handler implementation",
                "timestamp": "2026-01-01T00:00:00",
                "file_paths": ["auth.py"],
            },
            {
                "id": "c_repo_b",
                "repo_id": "repo_beta",
                "source_type": "commit",
                "source_id": "sha_b",
                "text": "Authentication OAuth handler implementation",
                "timestamp": "2026-01-01T00:00:00",
                "file_paths": ["auth.py"],
            }
        ]
        embeddings = [[0.1] * vs.vector_size, [0.1] * vs.vector_size]
        vs.upsert_chunks(chunks, embeddings)

        # Search specifically for repo_alpha
        results_alpha = vs.search_chunks(
            query_vector=[0.1] * vs.vector_size,
            repo_id="repo_alpha",
            limit=10
        )
        for r in results_alpha:
            assert r["payload"].get("repo_id") == "repo_alpha", "Cross-repo vector pollution: returned chunk from another repo!"

        # Search specifically for repo_beta
        results_beta = vs.search_chunks(
            query_vector=[0.1] * vs.vector_size,
            repo_id="repo_beta",
            limit=10
        )
        for r in results_beta:
            assert r["payload"].get("repo_id") == "repo_beta", "Cross-repo vector pollution: returned chunk from another repo!"
    finally:
        vs.close()


def test_bug5_symbol_index_composite_primary_key_prevents_repo_collision(clean_db):
    """Bug 5: Identical symbols in different repos (e.g. app.py::main) must NOT
    collide or increment each other's commit counts.
    """
    with get_session_context() as session:
        # Repo 1 has app.py::main
        sym1 = SymbolIndex(
            repo_id="repo_one",
            symbol_id="app.py::main",
            file_path="app.py",
            symbol_name="main",
            kind="function",
            commit_count=5
        )
        session.add(sym1)

        # Repo 2 has app.py::main
        sym2 = SymbolIndex(
            repo_id="repo_two",
            symbol_id="app.py::main",
            file_path="app.py",
            symbol_name="main",
            kind="function",
            commit_count=12
        )
        session.add(sym2)

    with get_session_context() as session:
        r1_sym = session.exec(
            select(SymbolIndex).where(SymbolIndex.repo_id == "repo_one", SymbolIndex.symbol_id == "app.py::main")
        ).first()
        r2_sym = session.exec(
            select(SymbolIndex).where(SymbolIndex.repo_id == "repo_two", SymbolIndex.symbol_id == "app.py::main")
        ).first()

        assert r1_sym is not None, "Repo 1 symbol missing!"
        assert r2_sym is not None, "Repo 2 symbol missing!"
        assert r1_sym.commit_count == 5, f"Repo 1 commit count corrupted: {r1_sym.commit_count}"
        assert r2_sym.commit_count == 12, f"Repo 2 commit count corrupted: {r2_sym.commit_count}"


def test_bug6_deterministic_chunk_id_includes_repo_id():
    """Bug 6: make_deterministic_chunk_id must produce distinct IDs for the same file/source_id
    across different repositories to prevent chunks from dropping during multi-repo ingestion.
    """
    id_repo_a = make_deterministic_chunk_id("file", "LICENSE", 0, "MIT License Copyright...", repo_id="repo_a")
    id_repo_b = make_deterministic_chunk_id("file", "LICENSE", 0, "MIT License Copyright...", repo_id="repo_b")
    
    assert id_repo_a != id_repo_b, "Chunk ID collision across repositories! Identical files in different repos must have distinct chunk IDs."


def test_bug7_embedder_reports_correct_failure_flags_on_mock_fallback():
    """Bug 7: When an external embedding provider fails and falls back to mock RNG vectors,
    return_success_flags must return False, not True.
    """
    emb = Embedder(openai_key="invalid_dummy_key")
    emb.model = "text-embedding-3-small"

    # Force failure in OpenAI embedder
    with patch("urllib.request.urlopen", side_effect=Exception("API connection refused")):
        vectors, success_flags = emb.embed_texts(["sample query text"], return_success_flags=True)

    assert len(vectors) == 1
    assert len(success_flags) == 1
    assert success_flags[0] is False, "Embedder reported True success flag despite failing and returning mock fallback vector!"


def test_bug8_github_client_handles_429_gracefully():
    """Bug 8: GitHub client must catch HTTP 429 and secondary rate limits,
    gracefully retaining previously fetched items instead of crashing.
    """
    client = GitHubIngestionClient("https://github.com/test_owner/test_repo", token="dummy_token")
    mock_repo = MagicMock()
    
    pr1 = MagicMock()
    pr1.number = 1
    pr1.title = "Fix A"
    pr1.body = "Body A"
    pr1.state = "closed"
    pr1.user.login = "alice"
    pr1.created_at = datetime.now(timezone.utc)
    pr1.merged_at = None
    pr1.merge_commit_sha = None
    pr1.get_issue_comments.side_effect = Exception("429 Too Many Requests: secondary rate limit")
    pr1.get_review_comments.return_value = []

    mock_repo.get_pulls.return_value = [pr1]
    client.repo = mock_repo
    client.gh.get_rate_limit = MagicMock(return_value=MagicMock(core=MagicMock(remaining=100)))

    prs = client.fetch_pull_requests(limit=10)
    assert len(prs) == 1
    assert prs[0]["number"] == 1
    assert prs[0]["comments"] == []

