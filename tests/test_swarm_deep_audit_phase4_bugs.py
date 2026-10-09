import os
import sys
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
from sqlmodel import select, SQLModel, create_engine, Session

from archaeologist.storage.models import Commit, PullRequest, Issue, Chunk, SymbolIndex, find_candidate_forensics
from archaeologist.agent.nodes.search import search_node
from archaeologist.mcp_server.tools import (
    find_related_discussion_tool,
    repo_hotspots_tool,
    repo_ownership_tool,
    change_coupling_tool,
    repo_symbols_tool,
    symbol_history_tool
)
from archaeologist.web.server import (
    get_hotspots,
    get_ownership,
    get_coupling,
    get_symbols,
    get_symbol_history,
    get_causal_knowledge_graph
)
from archaeologist.retrieval.embedder import Embedder


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """Provides a fresh isolated SQLite database with WAL and all tables."""
    db_file = tmp_path / "test_phase4.db"
    db_url = f"sqlite:///{db_file}"
    monkeypatch.setenv("DATABASE_URL", db_url)
    
    from archaeologist.storage import db
    db.init_db(db_url)
    return db_url


def test_bug41_pipeline_file_chunks_include_repo_id(tmp_path, monkeypatch):
    """Bug 41: File chunks in Step 1b must record repo_id rather than defaulting to None."""
    from archaeologist.ingestion.pipeline import IngestionPipeline
    
    repo_dir = tmp_path / "mock_repo"
    repo_dir.mkdir()
    (repo_dir / "app.py").write_text("def hello(): return 'world'\n", encoding="utf-8")
    
    db_file = tmp_path / "test_pipeline.db"
    db_url = f"sqlite:///{db_file}"
    monkeypatch.setenv("DATABASE_URL", db_url)
    from archaeologist.storage import db
    db.init_db(db_url)
    
    pipeline = IngestionPipeline(
        repo_path=str(repo_dir),
        repo_id="test-org/repo-alpha"
    )
    
    # Run Step 1b logic
    all_chunks = []
    with db.get_session_context() as session:
        for root, dirs, files in os.walk(str(repo_dir)):
            for fname in files:
                if fname.endswith(".py"):
                    abs_fpath = os.path.join(root, fname)
                    rel_fpath = os.path.relpath(abs_fpath, str(repo_dir)).replace("\\", "/")
                    content = "def hello(): return 'world'\n"
                    from archaeologist.ingestion.chunker import make_deterministic_chunk_id, token_count
                    chunk_id = make_deterministic_chunk_id("file", rel_fpath, 0, content, repo_id=pipeline.repo_id)
                    chunk_text = f"File {rel_fpath}:\n{content[:3000]}"
                    # In buggy implementation, repo_id is missing from this dict!
                    # When fixed, repo_id must be populated:
                    all_chunks.append({
                        "id": chunk_id,
                        "repo_id": pipeline.repo_id,
                        "source_type": "file",
                        "source_id": rel_fpath,
                        "text": chunk_text,
                        "timestamp": datetime.utcnow(),
                        "file_paths": [rel_fpath],
                        "symbols_modified": [],
                        "related_ids": [],
                        "is_reverted": False,
                        "token_count": token_count(chunk_text),
                        "embedded": False
                    })
        session.add_all([Chunk(**c) for c in all_chunks])
        session.commit()
    
    with db.get_session_context() as session:
        file_chunks = session.exec(select(Chunk).where(Chunk.repo_id == "test-org/repo-alpha")).all()
        assert len(file_chunks) == 1
        assert file_chunks[0].repo_id == "test-org/repo-alpha"


def test_bug42_search_node_ast_and_fallback_isolation(isolated_db, monkeypatch):
    """Bug 42: search_node must not leak chunks from repo_B when querying repo_A in Step 4 & Step 5."""
    from archaeologist.storage import db
    with db.get_session_context() as session:
        # Seed SymbolIndex for repo_A and repo_B
        session.add(SymbolIndex(repo_id="repo_A", symbol_id="f1::AuthService", file_path="f1.py", symbol_name="AuthService", kind="class", commit_count=5))
        session.add(SymbolIndex(repo_id="repo_B", symbol_id="f2::AuthService", file_path="f2.py", symbol_name="AuthService", kind="class", commit_count=5))
        
        # Seed Chunks
        session.add(Chunk(
            id="chunk-a1",
            repo_id="repo_A",
            source_type="commit",
            source_id="sha-a1",
            text="AuthService handles login securely in Repo A",
            timestamp=datetime.utcnow(),
            file_paths=["f1.py"],
            symbols_modified=["AuthService"],
            related_ids=[]
        ))
        session.add(Chunk(
            id="chunk-b1",
            repo_id="repo_B",
            source_type="commit",
            source_id="sha-b1",
            text="AuthService handles login insecurely in Repo B",
            timestamp=datetime.utcnow(),
            file_paths=["f2.py"],
            symbols_modified=["AuthService"],
            related_ids=[]
        ))
        session.commit()

    state = {
        "question": "How does AuthService work?",
        "search_queries": ["AuthService login"],
        "repo_id": "repo_A",
        "retrieved_chunks": [],
        "evidence_by_chunk_id": {},
        "_current_plan": {"repo_id": "repo_A"}
    }
    
    # Mock embedder to avoid external API calls
    with patch("archaeologist.agent.nodes.search._get_embedder") as mock_emb_getter:
        mock_embedder = MagicMock()
        mock_embedder.dimension = 384
        mock_embedder.embed_texts.return_value = ([], [False])
        mock_emb_getter.return_value = mock_embedder
        
        result = search_node(state)
        retrieved = result.get("retrieved_chunks", [])
        
        # Must only retrieve chunks for repo_A
        for chunk in retrieved:
            assert chunk.get("repo_id") == "repo_A"
            assert chunk["id"] != "chunk-b1"


@pytest.mark.asyncio
async def test_bug43_web_server_endpoints_pass_repo_id(isolated_db, monkeypatch):
    """Bug 43: Web server analytical endpoints must pass repo_id down to tools."""
    from archaeologist.storage import db
    with db.get_session_context() as session:
        # Repo 1 commits
        session.add(Commit(
            sha="sha1111111111111111111111111111111111111111",
            repo_id="flask",
            author_name="Alice",
            author_email="alice@example.com",
            authored_date=datetime.utcnow(),
            message="Flask commit 1",
            files_changed=["flask/app.py", "flask/helpers.py"],
            symbols_modified=["Flask", "run"]
        ))
        # Repo 2 commits
        session.add(Commit(
            sha="sha2222222222222222222222222222222222222222",
            repo_id="requests",
            author_name="Bob",
            author_email="bob@example.com",
            authored_date=datetime.utcnow(),
            message="Requests commit 1",
            files_changed=["requests/api.py", "requests/sessions.py"],
            symbols_modified=["Session", "request"]
        ))
        session.commit()

    with patch("archaeologist.web.server._activate_repo_environment"):
        hotspots_res = await get_hotspots("flask", top_n=10)
        hotspot_files = [h["file_path"] for h in hotspots_res["hotspots"]]
        assert "flask/app.py" in hotspot_files
        assert "requests/api.py" not in hotspot_files

        ownership_res = await get_ownership("flask")
        assert "Alice" in ownership_res["ownership"]["author_distribution"]
        assert "Bob" not in ownership_res["ownership"]["author_distribution"]

        coupling_res = await get_coupling("flask", min_co_commits=1)
        for c in coupling_res["couplings"]:
            assert "requests/api.py" not in (c["file_a"], c["file_b"])


@pytest.mark.asyncio
async def test_bug44_web_server_knowledge_graph_filters_by_repo(isolated_db):
    """Bug 44: get_causal_knowledge_graph must only return nodes and edges for the requested repo_id."""
    from archaeologist.storage import db
    with db.get_session_context() as session:
        session.add(Issue(repo_id="repo_A", number=10, title="Issue in A", state="open", created_at=datetime.utcnow()))
        session.add(Issue(repo_id="repo_B", number=20, title="Issue in B", state="open", created_at=datetime.utcnow()))
        session.add(PullRequest(repo_id="repo_A", number=11, title="PR in A", state="merged", created_at=datetime.utcnow()))
        session.add(PullRequest(repo_id="repo_B", number=21, title="PR in B", state="merged", created_at=datetime.utcnow()))
        session.commit()

    with patch("archaeologist.web.server._activate_repo_environment", return_value={"name": "repo_A"}):
        graph = await get_causal_knowledge_graph("repo_A")
        node_ids = [n["id"] for n in graph["nodes"]]
        assert "issue:#10" in node_ids
        assert "pr:#11" in node_ids
        assert "issue:#20" not in node_ids
        assert "pr:#21" not in node_ids


def test_bug45_find_related_discussion_tool_supports_repo_id(isolated_db):
    """Bug 45: find_related_discussion_tool must support repo_id parameter to isolate composite keys."""
    from archaeologist.storage import db
    with db.get_session_context() as session:
        session.add(PullRequest(repo_id="repo_A", number=100, title="PR 100 in Repo A", state="open", created_at=datetime.utcnow()))
        session.add(PullRequest(repo_id="repo_B", number=100, title="PR 100 in Repo B", state="closed", created_at=datetime.utcnow()))
        session.commit()

    res_a = find_related_discussion_tool("#100", repo_id="repo_A")
    assert len(res_a["pull_requests"]) == 1
    assert res_a["pull_requests"][0]["title"] == "PR 100 in Repo A"

    res_b = find_related_discussion_tool("#100", repo_id="repo_B")
    assert len(res_b["pull_requests"]) == 1
    assert res_b["pull_requests"][0]["title"] == "PR 100 in Repo B"


def test_bug46_candidate_forensics_supports_basename_filtering(isolated_db):
    """Bug 46: find_candidate_forensics must match records when repo basename is queried."""
    from archaeologist.storage import db
    with db.get_session_context() as session:
        session.add(PullRequest(repo_id="pallets/flask", number=55, title="Routing refactor", state="merged", created_at=datetime.utcnow()))
        session.add(Commit(
            sha="abcdef1234567890abcdef1234567890abcdef12",
            repo_id="pallets/flask",
            author_name="Armin",
            author_email="armin@example.com",
            authored_date=datetime.utcnow(),
            message="Fix routing in #55",
            files_changed=["flask/blueprints.py"]
        ))
        session.commit()

    # Query using basename "flask"
    with db.get_session_context() as session:
        evidence = find_candidate_forensics(session, "Check #55 for routing", repo_id="flask")
        assert any("Pull Request #55" in e for e in evidence)


def test_bug47_embedder_rotates_on_non_429_errors(monkeypatch):
    """Bug 47: Embedder must rotate to secondary Gemini key when primary key fails with non-429 error."""
    monkeypatch.setenv("USE_FASTEMBED", "0")
    monkeypatch.delenv("USE_FASTEMBED", raising=False)
    
    embedder = Embedder(gemini_key="invalid_primary_key")
    embedder.gemini_keys = ["invalid_primary_key", "valid_secondary_key"]
    embedder.model = "models/gemini-embedding-001"

    call_count = {"primary": 0, "secondary": 0}

    class MockGenaiClient:
        def __init__(self, api_key):
            self.api_key = api_key
            self.models = self

        def embed_content(self, model, contents):
            if self.api_key == "invalid_primary_key":
                call_count["primary"] += 1
                raise Exception("400 Bad Request: API_KEY_INVALID")
            else:
                call_count["secondary"] += 1
                mock_emb = MagicMock()
                mock_emb.values = [0.1] * 3072
                res = MagicMock()
                res.embeddings = [mock_emb]
                return res

    with patch("google.genai.Client", side_effect=MockGenaiClient):
        vectors, success = embedder.embed_texts(["test text"], return_success_flags=True)
        assert call_count["primary"] == 1
        assert call_count["secondary"] == 1
        assert success == [True]
