import subprocess
from unittest.mock import MagicMock, patch

from sqlmodel import select

from archaeologist.ingestion.pipeline import IngestionPipeline
from archaeologist.mcp_server.tools import blame_explain_tool, search_history_tool
from archaeologist.storage.db import get_session_context
from archaeologist.storage.models import Commit


def test_code_review_blame_explain_attaches_diff(tmp_path):
    """Verifies that blame_explain_tool extracts and attaches actual file diffs to commit data."""
    repo_dir = tmp_path / "repo_blame_diff"
    repo_dir.mkdir()
    git = ["git", "-c", "user.name=Reviewer", "-c", "user.email=rev@example.com"]
    subprocess.run(git + ["init", "-q", str(repo_dir)], check=True)

    target_file = repo_dir / "service.py"
    target_file.write_text("def run_service():\n    # Line 2\n    return 'v1'\n")
    subprocess.run(git + ["-C", str(repo_dir), "add", "service.py"], check=True)
    subprocess.run(git + ["-C", str(repo_dir), "commit", "-q", "-m", "Initial service implementation"], check=True)

    # Ingest repository so commit is in SQLite
    pipeline = IngestionPipeline(str(repo_dir), enable_llm_summaries=False)
    pipeline.run()

    # Call blame_explain_tool on line 2
    res = blame_explain_tool(
        repo_path=str(repo_dir),
        file_path="service.py",
        line_start=2,
        line_end=2
    )

    assert "commits" in res
    assert len(res["commits"]) >= 1
    commit_entry = res["commits"][0]
    assert "diff" in commit_entry, "Commit entry must include extracted file diff preview"
    assert "def run_service" in commit_entry["diff"]


def test_code_review_search_history_storage_path_forwarding(tmp_path):
    """Verifies that search_history_tool accepts and initializes VectorStore with custom storage_path."""
    custom_qdrant_dir = str(tmp_path / "custom_qdrant_dir")

    with patch("archaeologist.mcp_server.tools.VectorStore") as mock_vs_cls:
        mock_instance = MagicMock()
        mock_instance.search.return_value = []
        mock_vs_cls.return_value = mock_instance

        results = search_history_tool(
            query="test query",
            repo_id="test_repo",
            storage_path=custom_qdrant_dir
        )

        assert isinstance(results, list)
        mock_vs_cls.assert_called_once()
        _, kwargs = mock_vs_cls.call_args
        assert kwargs.get("storage_path") == custom_qdrant_dir


def test_code_review_pipeline_empty_files_and_symbols_graceful(tmp_path, monkeypatch):
    """Verifies that commits with no modified files or empty messages do not crash structural footprint."""
    monkeypatch.setenv("SKIP_GITHUB_API", "1")
    repo_dir = tmp_path / "repo_empty_commit"
    repo_dir.mkdir()
    git = ["git", "-c", "user.name=Tester", "-c", "user.email=t@example.com"]
    subprocess.run(git + ["init", "-q", str(repo_dir)], check=True)

    # Allow empty commit
    subprocess.run(git + ["-C", str(repo_dir), "commit", "--allow-empty", "-q", "-m", "Empty commit"], check=True)

    pipeline = IngestionPipeline(str(repo_dir), enable_llm_summaries=False)
    pipeline.run()

    with get_session_context() as session:
        commits = session.exec(select(Commit)).all()
        assert len(commits) == 1
        assert commits[0].diff_summary is not None
        assert "Commit" in commits[0].diff_summary


def test_code_review_preflight_github_token_warning(tmp_path, monkeypatch):
    """Verifies that IngestionPipeline reports a tip when repo_url is set but GITHUB_TOKEN is missing."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("SKIP_GITHUB_API", "1")
    repo_dir = tmp_path / "repo_token_test"
    repo_dir.mkdir()
    git = ["git", "-c", "user.name=Tester", "-c", "user.email=t@example.com"]
    subprocess.run(git + ["init", "-q", str(repo_dir)], check=True)
    (repo_dir / "hello.py").write_text("print('hello')\n")
    subprocess.run(git + ["-C", str(repo_dir), "add", "."], check=True)
    subprocess.run(git + ["-C", str(repo_dir), "commit", "-q", "-m", "init"], check=True)

    reported_messages = []
    def record_progress(phase, cur, total, msg, pct):
        reported_messages.append(msg)

    pipeline = IngestionPipeline(
        str(repo_dir),
        repo_url="https://github.com/example/repo",
        progress_callback=record_progress,
        github_token=None
    )
    pipeline.run()

    found_warning = any("GITHUB_TOKEN not set" in m for m in reported_messages)
    assert found_warning, "Pipeline must emit a pre-flight GITHUB_TOKEN reminder"


def test_local_llm_inference_routing(monkeypatch):
    """Verifies that GeminiClientWrapper routes to local model endpoint when LOCAL_LLM_URL is set."""
    from archaeologist.utils.gemini_client import GeminiClientWrapper, get_gemini_api_key

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY_SECONDARY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_LLM_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("LOCAL_LLM_MODEL", "qwen2.5-coder:7b")

    assert get_gemini_api_key() == "local-model"

    client = GeminiClientWrapper()
    assert client.local_llm_url == "http://localhost:11434/v1"
    assert client.local_llm_model == "qwen2.5-coder:7b"

    with patch.object(client, "_call_local_llm", return_value="Local reasoning result") as mock_local:
        res = client.generate_text("Explain why code changed")
        assert res == "Local reasoning result"
        mock_local.assert_called_once()
