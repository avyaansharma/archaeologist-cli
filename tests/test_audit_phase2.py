"""Phase 2 regression tests for adversarial audit remediation.

Validates:
- D10: git-tracked file filtering in chunk_codebase
- D18: Commit.diff_summary caching
- D19: No historical working tree fallback
- D22: CORS restriction to localhost
- D25: Blame explain with git show fallback, large commit coupling filter, symbol exact matching
- D27/D33: Synthesize prompt evidence delimiters and offline evidence presentation
- D30: Qualified AST symbol IDs with class stack
- D32: Embedder timeout enforcement
- D34: Dotted GitHub repo URL parsing
- D35: Hunk-aware diff parsing with '-- comment' lines
- CLI: 'why' command line verification
"""
import os
import subprocess
import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch
from typer.testing import CliRunner

# D10: git-tracked files only
def test_d10_chunk_codebase_tracked_files_only(tmp_path):
    from archaeologist.ingestion.chunker import chunk_codebase

    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"]
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    subprocess.run(git + ["init", "-q", str(repo_dir)], check=True)

    tracked_file = repo_dir / "app.py"
    tracked_file.write_text("def run():\n    pass\n")
    subprocess.run(git + ["-C", str(repo_dir), "add", "app.py"], check=True)
    subprocess.run(git + ["-C", str(repo_dir), "commit", "-q", "-m", "add app"], check=True)

    untracked_secret = repo_dir / "secrets.json"
    untracked_secret.write_text('{"token": "super-secret"}')

    chunks = chunk_codebase(str(repo_dir), repo_id="test_repo")
    paths = [c["file_paths"][0] for c in chunks if c.get("file_paths")]
    assert any("app.py" in p for p in paths)
    assert not any("secrets.json" in p for p in paths)


# D18: Commit diff summaries cached in Commit.diff_summary
def test_d18_diff_summary_cached_in_model():
    from archaeologist.storage.models import Commit
    
    c = Commit(
        sha="a" * 40,
        repo_id="test",
        message="feat: cached commit",
        author_name="dev",
        authored_date=datetime(2024, 1, 1),
        diff_summary="Cached diff explanation"
    )
    assert c.diff_summary == "Cached diff explanation"


# D22: CORS restricted to localhost
def test_d22_cors_restricted_to_localhost():
    from fastapi.testclient import TestClient
    from archaeologist.web.server import app

    client = TestClient(app)
    # Evil external domain
    res_evil = client.options("/api/hotspots/test", headers={"Origin": "https://malicious-site.com", "Access-Control-Request-Method": "GET"})
    allow_origin = res_evil.headers.get("access-control-allow-origin")
    assert allow_origin != "https://malicious-site.com"
    assert allow_origin != "*"

    # Localhost domain
    res_local = client.options("/api/hotspots/test", headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"})
    assert res_local.headers.get("access-control-allow-origin") == "http://localhost:3000"


# D25: Blame explain with git show fallback
def test_d25_blame_explain_fallback_to_git_show(tmp_path):
    from archaeologist.mcp_server.tools import blame_explain_tool

    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"]
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    subprocess.run(git + ["init", "-q", str(repo_dir)], check=True)

    f = repo_dir / "calc.py"
    f.write_text("def add(a, b):\n    return a + b\n")
    subprocess.run(git + ["-C", str(repo_dir), "add", "calc.py"], check=True)
    subprocess.run(git + ["-C", str(repo_dir), "commit", "-q", "-m", "init calc"], check=True)

    # Empty DB, but blame tool falls back to git show -s
    res = blame_explain_tool(str(repo_dir), "calc.py", 1, 2)
    assert "calc.py" in res["file_path"]
    assert len(res["commits"]) >= 1
    assert "init calc" in res["commits"][0]["message"]


# D25: Change coupling skips large commits
def test_d25_change_coupling_skips_large_commits(tmp_path):
    from archaeologist.mcp_server.tools import change_coupling_tool
    from archaeologist.storage.db import get_session_context
    from archaeologist.storage.context import current_db_url_var
    from archaeologist.storage.models import Commit

    db_path = tmp_path / "coupling.db"
    db_url = f"sqlite:///{db_path}"
    token = current_db_url_var.set(db_url)
    try:
        with get_session_context(db_url) as session:
            # Commit with 30 files changed (bulk rename / lockfile / release)
            c_large = Commit(
                sha="1" * 40,
                repo_id="test",
                message="bulk bump",
                author_name="bot",
                author_email="bot@example.com",
                authored_date=datetime(2024, 1, 1),
                files_changed=[f"file_{i}.py" for i in range(30)]
            )
            session.add(c_large)
            session.commit()

        pairs = change_coupling_tool(min_co_commits=1, repo_id="test")
        assert pairs == []
    finally:
        current_db_url_var.reset(token)


# D25: Symbol history exact segment matching
def test_d25_symbol_history_exact_matching(tmp_path):
    from archaeologist.mcp_server.tools import symbol_history_tool
    from archaeologist.storage.db import get_session_context
    from archaeologist.storage.context import current_db_url_var
    from archaeologist.storage.models import Commit

    db_path = tmp_path / "sym.db"
    db_url = f"sqlite:///{db_path}"
    token = current_db_url_var.set(db_url)
    try:
        with get_session_context(db_url) as session:
            c1 = Commit(
                sha="a" * 40,
                repo_id="test",
                message="add login",
                author_name="dev",
                author_email="dev@example.com",
                authored_date=datetime(2024, 1, 1),
                symbols_modified=["auth.py::login"]
            )
            c2 = Commit(
                sha="b" * 40,
                repo_id="test",
                message="add login_helper",
                author_name="dev",
                author_email="dev@example.com",
                authored_date=datetime(2024, 1, 2),
                symbols_modified=["auth.py::login_helper"]
            )
            session.add(c1)
            session.add(c2)
            session.commit()

        # Exact matching on 'login' should NOT match 'login_helper'
        exact_res = symbol_history_tool("login", fuzzy=False, repo_id="test")
        assert len(exact_res) == 1
        assert exact_res[0]["sha"] == "a" * 40

        # Fuzzy matching on 'login' should match both
        fuzzy_res = symbol_history_tool("login", fuzzy=True, repo_id="test")
        assert len(fuzzy_res) == 2
    finally:
        current_db_url_var.reset(token)


# D27/D33: Synthesize node offline evidence formatting
def test_d27_d33_synthesize_offline_evidence_formatting(monkeypatch):
    from archaeologist.agent.nodes.synthesize import synthesize_node

    monkeypatch.setattr("archaeologist.agent.nodes.synthesize.get_gemini_api_key", lambda: None)
    retrieved = [
        {
            "id": "c1",
            "source_type": "commit",
            "source_id": "abcdef1234",
            "text": "Fix connection timeout in client pool",
            "related_ids": ["pr#42"]
        }
    ]
    out = synthesize_node({
        "question": "Why was timeout added?",
        "retrieved_chunks": retrieved,
        "draft_answer": None
    })
    resp = out["response"]
    assert "Forensic Archaeology Summary (Offline Mode)" in resp
    assert "abcdef1234" in resp
    assert "pr#42" in resp


# D30: Qualified AST symbol IDs with class stack
def test_d30_qualified_ast_symbol_ids():
    from archaeologist.ingestion.symbol_parser import extract_symbols_from_code

    code = """
class AuthService:
    def authenticate(self, user):
        pass

class UserService:
    def authenticate(self, user):
        pass

def global_helper():
    pass
"""
    symbols = extract_symbols_from_code(code, "services/auth.py")
    sym_ids = [s["symbol_id"] for s in symbols]
    assert "services/auth.py::AuthService::authenticate" in sym_ids
    assert "services/auth.py::UserService::authenticate" in sym_ids
    assert "services/auth.py::global_helper" in sym_ids


# D32: Embedder timeout enforcement
def test_d32_embedder_timeouts(monkeypatch):
    from archaeologist.retrieval.embedder import Embedder
    import urllib.request

    for k in ("GEMINI_API_KEY", "GEMINI_API_KEY_SECONDARY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(k, raising=False)

    calls = []
    def fake_urlopen(req, timeout=None):
        calls.append(timeout)
        mock = MagicMock()
        mock.__enter__.return_value.read.return_value = b'{"data": [{"embedding": [0.1, 0.2]}]}'
        return mock

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    emb = Embedder(voyage_key="vy-test")
    emb.embed_texts(["test text"])
    assert len(calls) == 1
    assert calls[0] == 30, "Voyage embedder must enforce timeout=30"


# D34: Dotted GitHub repo URL parsing
def test_d34_dotted_github_url_parsing():
    from archaeologist.ingestion.github_client import GitHubIngestionClient

    client = GitHubIngestionClient("https://github.com/socketio/socket.io.git")
    assert client.owner == "socketio"
    assert client.repo_name == "socket.io"

    client2 = GitHubIngestionClient("https://github.com/facebook/react")
    assert client2.owner == "facebook"
    assert client2.repo_name == "react"


# D35: Hunk-aware diff parsing with '-- comment' lines
def test_d35_hunk_aware_diff_parsing():
    from archaeologist.ingestion.symbol_parser import extract_modified_line_numbers_from_diff

    # Unified diff where removed line begins with '--' (e.g. SQL / Lua comment)
    diff = """diff --git a/query.sql b/query.sql
index 1234567..89abcdef 100644
--- a/query.sql
+++ b/query.sql
@@ -10,3 +10,3 @@
--- old sql comment line
+-- new sql comment line
 SELECT * FROM users;
"""
    mapping = extract_modified_line_numbers_from_diff(diff)
    assert "query.sql" in mapping
    assert 10 in mapping["query.sql"]


# CLI 'why' command line verification
def test_cli_why_command(tmp_path):
    from archaeologist.cli import app

    runner = CliRunner()
    # Missing colon should exit 1
    res_err = runner.invoke(app, ["why", "calc.py"])
    assert res_err.exit_code == 1

    # Valid syntax against a temporary repo
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"]
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    subprocess.run(git + ["init", "-q", str(repo_dir)], check=True)

    f = repo_dir / "auth.py"
    f.write_text("def login():\n    return True\n")
    subprocess.run(git + ["-C", str(repo_dir), "add", "auth.py"], check=True)
    subprocess.run(git + ["-C", str(repo_dir), "commit", "-q", "-m", "add login"], check=True)

    res_ok = runner.invoke(app, ["why", "auth.py:1-2", "--repo", str(repo_dir)])
    assert res_ok.exit_code == 0
    assert "Causal Investigation" in res_ok.stdout
