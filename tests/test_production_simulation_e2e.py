import os
import sys
import tempfile
import threading
from datetime import datetime
from unittest.mock import patch, MagicMock
import pytest
from sqlmodel import select

from archaeologist.storage.db import init_db, get_session_context
from archaeologist.storage.models import Commit, PullRequest, Issue, Chunk, SymbolIndex
from archaeologist.web.server import get_causal_knowledge_graph
from archaeologist.ingestion.link_resolver import update_cross_links
from archaeologist.retrieval.fusion import reciprocal_rank_fusion
from archaeologist.mcp_server.tools import search_history_tool


@pytest.fixture
def isolated_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db_url = f"sqlite:///{path.replace(os.sep, '/')}"
    init_db(db_url)
    yield db_url
    try:
        os.remove(path)
    except Exception:
        pass


def test_knowledge_graph_modifies_symbol_edges(isolated_db):
    """Bug 48: Ensures MODIFIES_SYMBOL edges are generated between commits and AST symbols
    even when Commit.symbols_modified contains path prefixes (e.g. 'src/auth.py:login').
    """
    repo_id = "test-prod-repo"
    with get_session_context() as session:
        # Seed AST Symbol
        sym = SymbolIndex(
            symbol_id="src/auth.py:login",
            symbol_name="login",
            file_path="src/auth.py",
            repo_id=repo_id,
            commit_count=3,
            kind="function"
        )
        session.add(sym)

        # Seed Commit modifying the symbol
        commit = Commit(
            sha="a1b2c3d4e5f678901234567890abcdef12345678",
            repo_id=repo_id,
            author_name="Dev",
            author_email="dev@example.com",
            authored_date=datetime.utcnow(),
            message="Improve login authentication",
            files_changed=["src/auth.py"],
            symbols_modified=["src/auth.py:login"],
            insertions=10,
            deletions=2,
            is_revert=False
        )
        session.add(commit)
        session.commit()

    import asyncio
    graph_res = asyncio.run(get_causal_knowledge_graph(repo_id=repo_id, limit=50))

    node_ids = {n["id"] for n in graph_res["nodes"]}
    assert "symbol:login" in node_ids
    assert "commit:a1b2c3d" in node_ids

    modifies_edges = [e for e in graph_res["edges"] if e["type"] == "modifies"]
    assert len(modifies_edges) >= 1
    edge = modifies_edges[0]
    assert edge["source"] == "commit:a1b2c3d"
    assert edge["target"] == "symbol:login"
    assert edge["label"] == "MODIFIES_SYMBOL"


def test_update_cross_links_scoped_to_repo_id(isolated_db):
    """Bug 49: Ensures update_cross_links with repo_id does not process PRs/Issues from other repos."""
    with get_session_context() as session:
        # Repo Alpha: PR #1 mentions #10
        pr_a = PullRequest(
            number=1,
            repo_id="repo-alpha",
            title="Fix login bug",
            body="Fixes #10",
            state="merged",
            author="alice",
            created_at=datetime.utcnow()
        )
        iss_a = Issue(
            number=10,
            repo_id="repo-alpha",
            title="Login crash",
            body="Bug details",
            state="closed",
            author="bob",
            created_at=datetime.utcnow()
        )
        # Repo Beta: PR #2 mentions #20
        pr_b = PullRequest(
            number=2,
            repo_id="repo-beta",
            title="Add dark mode",
            body="Fixes #20",
            state="open",
            author="charlie",
            created_at=datetime.utcnow()
        )
        iss_b = Issue(
            number=20,
            repo_id="repo-beta",
            title="Theme support",
            body="Feature request",
            state="open",
            author="dana",
            created_at=datetime.utcnow()
        )
        session.add_all([pr_a, iss_a, pr_b, iss_b])
        session.commit()

    # Run cross-links scoped ONLY to repo-alpha
    with get_session_context() as session:
        update_cross_links(session, repo_id="repo-alpha")

    # Verify repo-alpha was updated, but repo-beta was untouched
    with get_session_context() as session:
        refetched_pr_a = session.exec(select(PullRequest).where(PullRequest.repo_id == "repo-alpha")).first()
        refetched_iss_a = session.exec(select(Issue).where(Issue.repo_id == "repo-alpha")).first()
        refetched_pr_b = session.exec(select(PullRequest).where(PullRequest.repo_id == "repo-beta")).first()
        refetched_iss_b = session.exec(select(Issue).where(Issue.repo_id == "repo-beta")).first()

        assert 10 in refetched_pr_a.linked_issue_numbers
        assert 1 in refetched_iss_a.linked_pr_numbers
        # repo-beta must NOT have been updated in this run
        assert len(refetched_pr_b.linked_issue_numbers) == 0
        assert len(refetched_iss_b.linked_pr_numbers) == 0


def test_reciprocal_rank_fusion_preserves_repo_id():
    """Bug 50: Ensures sparse hits in reciprocal_rank_fusion retain repo_id in their payloads."""
    sparse_hits = [
        {
            "rank": 1,
            "score": 5.2,
            "chunk": {
                "id": "sparse_chk_1",
                "repo_id": "production-repo",
                "source_type": "source_code",
                "source_id": "src/main.py",
                "text": "def calculate_tax(): pass",
                "timestamp": "2026-01-01T00:00:00",
                "file_paths": ["src/main.py"],
                "symbols_modified": ["calculate_tax"],
                "related_ids": [],
                "is_reverted": False
            }
        }
    ]
    dense_hits = []

    fused = reciprocal_rank_fusion(dense_hits, sparse_hits, limit=5)
    assert len(fused) == 1
    assert fused[0]["payload"].get("repo_id") == "production-repo"


def test_search_history_tool_passes_query_to_rrf(isolated_db):
    """Bug 51: Ensures search_history_tool passes the query parameter to reciprocal_rank_fusion."""
    with patch("archaeologist.mcp_server.tools.reciprocal_rank_fusion") as mock_rrf:
        mock_rrf.return_value = []
        with patch("archaeologist.mcp_server.tools.BM25Index") as mock_bm25_cls, \
             patch("archaeologist.mcp_server.tools.Embedder") as mock_embedder_cls, \
             patch("archaeologist.mcp_server.tools.VectorStore") as mock_vs_cls:
            mock_bm25 = MagicMock()
            mock_bm25.search.return_value = []
            mock_bm25_cls.load.return_value = mock_bm25

            mock_emb = MagicMock()
            mock_emb.dimension = 768
            mock_emb.embed_texts.return_value = [[0.1] * 768]
            mock_embedder_cls.return_value = mock_emb

            mock_vs = MagicMock()
            mock_vs.search_chunks.return_value = []
            mock_vs_cls.return_value = mock_vs

            search_history_tool(query="why did we revert auth?", repo_id="test-repo")

            assert mock_rrf.called
            call_kwargs = mock_rrf.call_args[1]
            assert call_kwargs.get("query") == "why did we revert auth?"


def test_concurrent_production_simulation(isolated_db):
    """Production Simulation: Simulates concurrent threads performing reads and writes simultaneously."""
    repo_id = "concurrent-repo"
    
    # Pre-seed base data
    with get_session_context() as session:
        for i in range(10):
            session.add(Commit(
                sha=f"c0ffee{i:03d}00000000000000000000000000000000",
                repo_id=repo_id,
                author_name=f"Dev {i}",
                author_email=f"dev{i}@example.com",
                authored_date=datetime.utcnow(),
                message=f"Commit message {i}",
                files_changed=[f"file_{i}.py"],
                symbols_modified=[f"func_{i}"],
                insertions=5,
                deletions=1,
                is_revert=False
            ))
            session.add(SymbolIndex(
                symbol_id=f"file_{i}.py:func_{i}",
                symbol_name=f"func_{i}",
                file_path=f"file_{i}.py",
                repo_id=repo_id,
                commit_count=1,
                kind="function"
            ))
        session.commit()

    errors = []

    def worker_reader(worker_id):
        try:
            import asyncio
            for _ in range(5):
                res = asyncio.run(get_causal_knowledge_graph(repo_id=repo_id, limit=20))
                assert res["nodes_count"] > 0
        except Exception as e:
            errors.append(f"Reader worker {worker_id} failed: {e}")

    def worker_writer(worker_id):
        try:
            with get_session_context() as session:
                session.add(Commit(
                    sha=f"beef{worker_id:04d}00000000000000000000000000000000",
                    repo_id=repo_id,
                    author_name=f"Worker {worker_id}",
                    author_email=f"w{worker_id}@example.com",
                    authored_date=datetime.utcnow(),
                    message=f"Worker commit {worker_id}",
                    files_changed=[f"worker_{worker_id}.py"],
                    symbols_modified=[],
                    insertions=1,
                    deletions=0,
                    is_revert=False
                ))
        except Exception as e:
            errors.append(f"Writer worker {worker_id} failed: {e}")

    threads = []
    # Launch 5 concurrent readers and 5 concurrent writers
    for i in range(5):
        t_r = threading.Thread(target=worker_reader, args=(i,))
        t_w = threading.Thread(target=worker_writer, args=(i,))
        threads.extend([t_r, t_w])

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert len(errors) == 0, f"Concurrent workers encountered errors: {errors}"
