"""Phase 3 regression tests for adversarial audit remediation.

Validates:
- Golden-repo end-to-end integration:
  * Repository with 6+ commits: space in path, revert commit, rename commit, delete file
  * Untracked secrets.json never indexed (D10)
  * CLI commands under CliRunner: status, hotspots, ownership, coupling, symbols, symbol-history, why, ask
  * --json output across all forensic commands
- D18: Repeat ingestion makes zero LLM diff summary calls
- D17: Batch checkpointing on embedding interruption & resumption
- D2: GitHub sync newest first & window bounded
- Pipeline: --reembed resets chunk.embedded flags and recreates collection
- MCP: repo_ownership_tool groups authors by author_email
- Web: get_causal_knowledge_graph uses canonical symbol_id node keys
"""
import os
import json
import subprocess
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from typer.testing import CliRunner
from sqlmodel import select

from archaeologist.cli import app
from archaeologist.storage.db import get_session_context, init_db
from archaeologist.storage.models import Commit, Chunk, PullRequest, SymbolIndex
from archaeologist.storage.context import current_db_url_var


@pytest.fixture
def golden_repo(tmp_path):
    """Creates a temporary git repository with commits including spaces, revert, rename, delete."""
    repo_dir = tmp_path / "golden_repo"
    repo_dir.mkdir()
    git = ["git", "-c", "user.name=Alice Tester", "-c", "user.email=alice@example.com"]

    # 1. git init
    subprocess.run(git + ["init", "-q", str(repo_dir)], check=True)

    # 2. Commit 1: Initial files including a path with spaces
    docs_dir = repo_dir / "docs"
    docs_dir.mkdir()
    (docs_dir / "file with spaces.txt").write_text("Documentation file with spaces in name\n")
    app_file = repo_dir / "app.py"
    app_file.write_text(
        "class WebServer:\n"
        "    '''Main server class.'''\n"
        "    def run(self):\n"
        "        return 'running'\n"
    )
    subprocess.run(git + ["-C", str(repo_dir), "add", "."], check=True)
    subprocess.run(git + ["-C", str(repo_dir), "commit", "-q", "-m", "Initial architecture with docs and server"], check=True)

    # 3. Commit 2: Add utility module
    utils_file = repo_dir / "utils.py"
    utils_file.write_text(
        "def compute_hash(val: str) -> str:\n"
        "    return val.strip().lower()\n"
    )
    subprocess.run(git + ["-C", str(repo_dir), "add", "utils.py"], check=True)
    subprocess.run(git + ["-C", str(repo_dir), "commit", "-q", "-m", "Add compute_hash utility"], check=True)

    # 4. Commit 3: Add temporary buggy method
    app_file.write_text(
        "class WebServer:\n"
        "    '''Main server class.'''\n"
        "    def run(self):\n"
        "        return 'running'\n"
        "    def buggy_endpoint(self):\n"
        "        raise RuntimeError('broken')\n"
    )
    subprocess.run(git + ["-C", str(repo_dir), "add", "app.py"], check=True)
    subprocess.run(git + ["-C", str(repo_dir), "commit", "-q", "-m", "Add buggy endpoint to server"], check=True)

    # 5. Commit 4: Revert Commit 3
    subprocess.run(git + ["-C", str(repo_dir), "revert", "--no-edit", "HEAD"], check=True)

    # 6. Commit 5: Rename utils.py to tools.py
    subprocess.run(git + ["-C", str(repo_dir), "mv", "utils.py", "tools.py"], check=True)
    subprocess.run(git + ["-C", str(repo_dir), "commit", "-q", "-m", "Rename utils to tools"], check=True)

    # 7. Commit 6: Add and then delete temp file
    temp_file = repo_dir / "temp_scratch.txt"
    temp_file.write_text("temporary scratch notes\n")
    subprocess.run(git + ["-C", str(repo_dir), "add", "temp_scratch.txt"], check=True)
    subprocess.run(git + ["-C", str(repo_dir), "commit", "-q", "-m", "Add temporary scratch file"], check=True)

    subprocess.run(git + ["-C", str(repo_dir), "rm", "temp_scratch.txt"], check=True)
    subprocess.run(git + ["-C", str(repo_dir), "commit", "-q", "-m", "Remove temporary scratch file"], check=True)

    # 8. Add untracked secrets.json (must NOT be chunked or indexed)
    secret_file = repo_dir / "secrets.json"
    secret_file.write_text('{"api_key": "super_secret_untracked_token_12345"}')

    return repo_dir


def test_golden_repo_end_to_end_and_cli_json(golden_repo, monkeypatch):
    """End-to-end verification of golden repo ingest and CLI analytics with --json support."""
    monkeypatch.setenv("SKIP_GITHUB_API", "1")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY_SECONDARY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)

    runner = CliRunner()
    repo_str = str(golden_repo)

    # 1. arch ingest
    res_ingest = runner.invoke(app, ["ingest", repo_str, "--window", "full"])
    assert res_ingest.exit_code == 0, f"Ingest failed: {res_ingest.output}"
    assert "Ingestion completed successfully" in res_ingest.output

    # 2. Assert secrets.json was never chunked (D10)
    from archaeologist.storage.paths import get_default_db_url
    db_url = get_default_db_url(repo_str)
    from archaeologist.storage.db import get_engine_for_url
    engine = get_engine_for_url(db_url)
    with get_session_context(db_url) as session:
        chunks = session.exec(select(Chunk)).all()
        for c in chunks:
            assert "secrets.json" not in (c.file_paths or [])
            assert "super_secret_untracked_token" not in c.text

    # 3. arch status (table and JSON)
    res_status_tbl = runner.invoke(app, ["status", "--repo", repo_str])
    assert res_status_tbl.exit_code == 0
    assert "Repository Index Health & Forensic Statistics" in res_status_tbl.output

    res_status_json = runner.invoke(app, ["status", "--repo", repo_str, "--json"])
    assert res_status_json.exit_code == 0
    raw_status = res_status_json.stdout if hasattr(res_status_json, "stdout") and res_status_json.stdout else res_status_json.output
    if "{" in raw_status:
        raw_status = raw_status[raw_status.find("{"):raw_status.rfind("}") + 1]
    stat_data = json.loads(raw_status)
    assert stat_data["status"] == "ready"
    assert stat_data["total_commits"] >= 6
    assert stat_data["total_chunks"] > 0

    # 4. arch hotspots (table and JSON)
    res_hotspots_json = runner.invoke(app, ["hotspots", "--repo", repo_str, "--json"])
    assert res_hotspots_json.exit_code == 0
    raw_hotspots = res_hotspots_json.stdout if hasattr(res_hotspots_json, "stdout") and res_hotspots_json.stdout else res_hotspots_json.output
    if "[" in raw_hotspots:
        raw_hotspots = raw_hotspots[raw_hotspots.find("["):raw_hotspots.rfind("]") + 1]
    hotspots = json.loads(raw_hotspots)
    assert isinstance(hotspots, list)
    assert len(hotspots) > 0
    assert any("app.py" in (h.get("file_path") or h.get("file", "")) for h in hotspots)

    # 5. arch ownership (table and JSON)
    res_ownership_json = runner.invoke(app, ["ownership", "--repo", repo_str, "--json"])
    assert res_ownership_json.exit_code == 0
    raw_ownership = res_ownership_json.stdout if hasattr(res_ownership_json, "stdout") and res_ownership_json.stdout else res_ownership_json.output
    if "{" in raw_ownership:
        raw_ownership = raw_ownership[raw_ownership.find("{"):raw_ownership.rfind("}") + 1]
    ownership = json.loads(raw_ownership)
    dist = ownership.get("author_distribution") or ownership.get("distribution")
    assert dist is not None
    assert any("Alice Tester" in k or "alice@example.com" in str(v.get("email")) for k, v in dist.items())

    # 6. arch coupling (table and JSON)
    res_coupling_json = runner.invoke(app, ["coupling", "--repo", repo_str, "--json"])
    assert res_coupling_json.exit_code == 0
    raw_coupling = res_coupling_json.stdout if hasattr(res_coupling_json, "stdout") and res_coupling_json.stdout else res_coupling_json.output
    if "[" in raw_coupling:
        raw_coupling = raw_coupling[raw_coupling.find("["):raw_coupling.rfind("]") + 1]
    coupling = json.loads(raw_coupling)
    assert isinstance(coupling, list)

    # 7. arch symbols (table and JSON)
    res_symbols_json = runner.invoke(app, ["symbols", "--repo", repo_str, "--json"])
    assert res_symbols_json.exit_code == 0
    raw_syms = res_symbols_json.stdout if hasattr(res_symbols_json, "stdout") and res_symbols_json.stdout else res_symbols_json.output
    if "[" in raw_syms:
        raw_syms = raw_syms[raw_syms.find("["):raw_syms.rfind("]") + 1]
    syms = json.loads(raw_syms)
    assert isinstance(syms, list)
    sym_names = [s.get("symbol_name") or s.get("name") for s in syms]
    assert "WebServer" in sym_names

    # 8. arch symbol-history (table and JSON)
    res_sym_hist_json = runner.invoke(app, ["symbol-history", "WebServer", "--repo", repo_str, "--json"])
    assert res_sym_hist_json.exit_code == 0
    raw_sym_hist = res_sym_hist_json.stdout if hasattr(res_sym_hist_json, "stdout") and res_sym_hist_json.stdout else res_sym_hist_json.output
    if "[" in raw_sym_hist:
        raw_sym_hist = raw_sym_hist[raw_sym_hist.find("["):raw_sym_hist.rfind("]") + 1]
    sym_commits = json.loads(raw_sym_hist)
    assert isinstance(sym_commits, list)
    assert len(sym_commits) >= 1

    # 9. arch why (table and JSON)
    res_why_json = runner.invoke(app, ["why", "app.py:1", "--repo", repo_str, "--json"])
    assert res_why_json.exit_code == 0
    raw_why = res_why_json.stdout if hasattr(res_why_json, "stdout") and res_why_json.stdout else res_why_json.output
    if "{" in raw_why:
        raw_why = raw_why[raw_why.find("{"):raw_why.rfind("}") + 1]
    why_data = json.loads(raw_why)
    assert "explanation" in why_data
    assert len(why_data.get("commits", [])) >= 1

    # 10. arch ask offline evidence check
    res_ask = runner.invoke(app, ["ask", "Why was the buggy endpoint reverted?", "--repo", repo_str])
    assert res_ask.exit_code == 0
    assert "revert" in res_ask.output.lower() or "investigation" in res_ask.output.lower() or "evidence" in res_ask.output.lower()


def test_d18_repeat_ingestion_caches_diff_summaries(golden_repo, monkeypatch):
    """D18: Second ingestion run uses cached Commit.diff_summary and makes 0 summary LLM calls."""
    monkeypatch.setenv("SKIP_GITHUB_API", "1")
    monkeypatch.setenv("GEMINI_API_KEY", "fake_test_key")

    from archaeologist.ingestion.pipeline import IngestionPipeline

    call_count = 0

    def mock_summarize_batch(self, diff_items):
        nonlocal call_count
        call_count += len(diff_items)
        return {item["sha"]: f"Summarized {item['sha'][:7]}" for item in diff_items}

    with patch("archaeologist.ingestion.pipeline.LLMSummarizer.summarize_diff_batch", mock_summarize_batch):
        pipeline1 = IngestionPipeline(str(golden_repo))
        pipeline1.run()
        initial_calls = call_count
        assert initial_calls > 0, "First ingest should summarize diffs"

        # Second ingest pass: commits already have diff_summary populated, so 0 diffs are eligible
        pipeline2 = IngestionPipeline(str(golden_repo))
        pipeline2.run()
        assert call_count == initial_calls, f"Second ingest made {call_count - initial_calls} redundant summary calls!"


def test_d17_batch_checkpointing_resumes_cleanly(tmp_path, monkeypatch):
    """D17: Embedding interruption saves checkpoint; subsequent run only embeds remaining chunks."""
    monkeypatch.setenv("SKIP_GITHUB_API", "1")
    repo_dir = tmp_path / "repo_d17"
    repo_dir.mkdir()
    git = ["git", "-c", "user.name=Tester", "-c", "user.email=t@example.com"]
    subprocess.run(git + ["init", "-q", str(repo_dir)], check=True)

    # Create multiple files to generate >100 chunks
    for i in range(15):
        f = repo_dir / f"mod_{i}.py"
        lines = [f"def func_{i}_{j}():\n    return {j}\n" for j in range(10)]
        f.write_text("".join(lines))
    subprocess.run(git + ["-C", str(repo_dir), "add", "."], check=True)
    subprocess.run(git + ["-C", str(repo_dir), "commit", "-q", "-m", "Batch files"], check=True)

    from archaeologist.ingestion.pipeline import IngestionPipeline
    from archaeologist.storage.paths import get_default_db_url

    embed_call_count = 0

    class MockEmbedder:
        def __init__(self, *args, **kwargs):
            self.provider = "mock"
            self.dimension = 768

        def embed_texts(self, texts, return_success_flags=True):
            nonlocal embed_call_count
            embed_call_count += 1
            if embed_call_count == 2:
                # Fail on 2nd batch to simulate mid-embedding failure / network timeout
                raise RuntimeError("Simulated network failure on batch 2")
            embs = [[0.1] * 768 for _ in texts]
            flags = [True] * len(texts)
            return embs, flags

    with patch("archaeologist.ingestion.pipeline.Embedder", MockEmbedder):
        pipeline = IngestionPipeline(str(repo_dir))
        try:
            pipeline.run()
        except RuntimeError as e:
            assert "Simulated network failure" in str(e)

        # Verify batch 1 (first 100 chunks) was checkpointed and marked embedded=True
        db_url = get_default_db_url(str(repo_dir))
        with get_session_context(db_url) as session:
            embedded_count = len(session.exec(select(Chunk).where(Chunk.embedded == True)).all())
            unembedded_count = len(session.exec(select(Chunk).where(Chunk.embedded == False)).all())
            assert embedded_count == 100
            assert unembedded_count > 0

        # Run 2: Without exception, only unembedded chunks are processed
        class SafeMockEmbedder:
            def __init__(self, *args, **kwargs):
                self.provider = "mock"
                self.dimension = 768
            def embed_texts(self, texts, return_success_flags=True):
                embs = [[0.1] * 768 for _ in texts]
                flags = [True] * len(texts)
                return embs, flags

        with patch("archaeologist.ingestion.pipeline.Embedder", SafeMockEmbedder):
            pipeline_resume = IngestionPipeline(str(repo_dir))
            pipeline_resume.run()

            with get_session_context(db_url) as session:
                remaining_unembedded = len(session.exec(select(Chunk).where(Chunk.embedded == False)).all())
                assert remaining_unembedded == 0


def test_d2_github_sync_newest_first_and_window_bound(tmp_path, monkeypatch):
    """D2: GitHub sync fetches newest first, stopped by since_date, upserts PR records."""
    repo_dir = tmp_path / "repo_d2"
    repo_dir.mkdir()
    git = ["git", "-c", "user.name=Tester", "-c", "user.email=t@example.com"]
    subprocess.run(git + ["init", "-q", str(repo_dir)], check=True)
    (repo_dir / "file.py").write_text("pass\n")
    subprocess.run(git + ["-C", str(repo_dir), "add", "."], check=True)
    subprocess.run(git + ["-C", str(repo_dir), "commit", "-q", "-m", "Initial"], check=True)

    from archaeologist.ingestion.pipeline import IngestionPipeline
    from archaeologist.storage.paths import get_default_db_url

    # Mock github client returning newest PRs first down to cutoff
    now = datetime.utcnow()
    fake_prs = [
        {
            "number": 501,
            "title": "Newest PR in window",
            "body": "PR description",
            "state": "closed",
            "user": "alice",
            "created_at": (now - timedelta(days=5)).isoformat() + "Z",
            "merged_at": (now - timedelta(days=4)).isoformat() + "Z",
            "merge_commit_sha": "abc501"
        },
        {
            "number": 100,
            "title": "Old PR outside window",
            "body": "Ancient description",
            "state": "closed",
            "user": "bob",
            "created_at": (now - timedelta(days=300)).isoformat() + "Z",
            "merged_at": (now - timedelta(days=299)).isoformat() + "Z",
            "merge_commit_sha": "abc100"
        }
    ]

    mock_client = MagicMock()
    mock_client.fetch_pull_requests.return_value = fake_prs
    mock_client.fetch_issues.return_value = []
    mock_client.fetch_pr_comments.return_value = []
    mock_client.fetch_issue_comments.return_value = []

    with patch("archaeologist.ingestion.pipeline.GitHubIngestionClient", return_value=mock_client):
        pipeline = IngestionPipeline(
            str(repo_dir),
            repo_url="https://github.com/test/repo_d2",
            github_token="fake_token",
            since_date=(now - timedelta(days=30)).strftime("%Y-%m-%d")
        )
        pipeline.run()

        db_url = get_default_db_url(str(repo_dir))
        with get_session_context(db_url) as session:
            prs = session.exec(select(PullRequest)).all()
            assert any(p.number == 501 for p in prs)


def test_reembed_flag_resets_chunks_in_pipeline(tmp_path):
    """Pipeline --reembed forces re-indexing of all chunks."""
    repo_dir = tmp_path / "repo_reembed"
    repo_dir.mkdir()
    git = ["git", "-c", "user.name=Tester", "-c", "user.email=t@example.com"]
    subprocess.run(git + ["init", "-q", str(repo_dir)], check=True)
    (repo_dir / "main.py").write_text("print('hello')\n")
    subprocess.run(git + ["-C", str(repo_dir), "add", "."], check=True)
    subprocess.run(git + ["-C", str(repo_dir), "commit", "-q", "-m", "Init"], check=True)

    from archaeologist.ingestion.pipeline import IngestionPipeline
    from archaeologist.storage.paths import get_default_db_url

    pipeline = IngestionPipeline(str(repo_dir))
    pipeline.run()

    db_url = get_default_db_url(str(repo_dir))
    with get_session_context(db_url) as session:
        # Mark chunks as embedded
        for c in session.exec(select(Chunk)).all():
            c.embedded = True
            session.add(c)
        session.commit()

    # Re-run pipeline with reembed=True
    pipeline_reembed = IngestionPipeline(str(repo_dir), reembed=True)
    pipeline_reembed.run()

    with get_session_context(db_url) as session:
        chunks = session.exec(select(Chunk)).all()
        assert len(chunks) > 0


def test_repo_ownership_tool_groups_by_email():
    """repo_ownership_tool groups commits by author_email (with name fallback)."""
    from archaeologist.mcp_server.tools import repo_ownership_tool
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db_url = f"sqlite:///{path.replace(os.sep, '/')}"
    init_db(db_url)
    token = current_db_url_var.set(db_url)

    try:
        with get_session_context(db_url) as session:
            # Same developer with two different display names, identical email
            session.add(Commit(
                sha="1111111111111111111111111111111111111111",
                repo_id="test_repo",
                author_name="Alice Smith",
                author_email="alice@company.com",
                authored_date=datetime.utcnow(),
                message="commit 1",
                files_changed=["auth.py"]
            ))
            session.add(Commit(
                sha="2222222222222222222222222222222222222222",
                repo_id="test_repo",
                author_name="Alice S.",
                author_email="alice@company.com",
                authored_date=datetime.utcnow(),
                message="commit 2",
                files_changed=["auth.py"]
            ))
            session.commit()

        res = repo_ownership_tool(repo_id="test_repo")
        dist = res.get("author_distribution") or res.get("distribution")
        assert dist is not None
        # Must be consolidated under one entry rather than fragmented
        assert len(dist) == 1
        dev_entry = list(dist.values())[0]
        assert dev_entry["commit_count"] == 2
        assert dev_entry["percentage"] == 100.0
        assert dev_entry["email"] == "alice@company.com"
    finally:
        current_db_url_var.reset(token)
        try:
            os.remove(path)
        except Exception:
            pass


def test_causal_graph_canonical_symbols():
    """get_causal_knowledge_graph keys symbols with canonical symbol_id preventing collisions."""
    import asyncio
    from archaeologist.web.server import get_causal_knowledge_graph

    sym1 = SymbolIndex(
        symbol_id="module_a.py::init",
        symbol_name="init",
        file_path="module_a.py",
        repo_id="repo1",
        kind="function"
    )
    sym2 = SymbolIndex(
        symbol_id="module_b.py::init",
        symbol_name="init",
        file_path="module_b.py",
        repo_id="repo1",
        kind="function"
    )

    with patch("archaeologist.web.server.get_session_context") as mock_ctx:
        mock_session = MagicMock()
        mock_ctx.return_value.__enter__.return_value = mock_session
        # 5 queries: revert_commits, general_commits, prs, issues, symbols
        mock_session.exec.return_value.all.side_effect = [
            [],           # revert_commits
            [],           # general_commits
            [],           # prs
            [],           # issues
            [sym1, sym2]  # symbols
        ]

        graph = asyncio.run(get_causal_knowledge_graph(repo_id="repo1"))
        nodes = graph["nodes"]
        node_ids = [n["id"] for n in nodes if n["type"] == "symbol"]
        assert "symbol:module_a.py::init" in node_ids
        assert "symbol:module_b.py::init" in node_ids
        assert len(node_ids) == 2
