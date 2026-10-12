import subprocess
from unittest.mock import MagicMock, patch

from sqlmodel import select

from archaeologist.agent.nodes.synthesize import synthesize_node
from archaeologist.agent.state import AgentState
from archaeologist.ingestion.pipeline import IngestionPipeline
from archaeologist.storage.db import get_session_context
from archaeologist.storage.models import Chunk, Commit, PullRequest


def test_zero_cost_ingestion_no_llm_calls(golden_repo, monkeypatch):
    """Verifies that default ingestion uses zero LLM calls and populates AST structural footprint."""
    monkeypatch.setenv("SKIP_GITHUB_API", "1")
    monkeypatch.delenv("ENABLE_LLM_SUMMARIES", raising=False)

    with patch("archaeologist.ingestion.diff_summarizer.LLMSummarizer.summarize_diff_batch") as mock_summarize:
        pipeline = IngestionPipeline(str(golden_repo), enable_llm_summaries=False)
        pipeline.run()

        # Ingestion must NOT call LLM summarizer by default
        mock_summarize.assert_not_called()

    # Verify that commits received high-signal structural patch footprints
    with get_session_context() as session:
        commits = session.exec(select(Commit)).all()
        assert len(commits) > 0
        for c in commits:
            assert c.diff_summary is not None
            # Structural footprint should contain file churn or AST symbols
            assert "lines" in c.diff_summary or "AST Symbols" in c.diff_summary or "Files modified" in c.diff_summary

        # Check commit chunks
        commit_chunks = session.exec(select(Chunk).where(Chunk.source_type == "commit")).all()
        assert len(commit_chunks) > 0
        for chk in commit_chunks:
            assert "Diff Summary:" in chk.text


def test_zero_cost_ingestion_with_pr_rationale(tmp_path, monkeypatch):
    """Verifies that linked PR title and description are injected into structural commit footprint."""
    monkeypatch.setenv("SKIP_GITHUB_API", "1")
    repo_dir = tmp_path / "repo_pr_test"
    repo_dir.mkdir()
    git = ["git", "-c", "user.name=Tester", "-c", "user.email=t@example.com"]
    subprocess.run(git + ["init", "-q", str(repo_dir)], check=True)

    test_file = repo_dir / "auth.py"
    test_file.write_text("def authenticate_user():\n    return True\n")
    subprocess.run(git + ["-C", str(repo_dir), "add", "auth.py"], check=True)
    subprocess.run(git + ["-C", str(repo_dir), "commit", "-q", "-m", "Refactor authentication logic (#42)"], check=True)

    # Ingest repository
    pipeline = IngestionPipeline(str(repo_dir), enable_llm_summaries=False)
    pipeline.run()

    # Add a mock PullRequest #42 to SQLite and re-run pipeline step 5
    from datetime import datetime
    with get_session_context() as session:
        pr_obj = PullRequest(
            number=42,
            repo_id=pipeline.repo_id,
            title="Fix auth timeout bug",
            body="Resolves #40 by adding proper credential caching in authenticate_user",
            state="closed",
            author="security-lead",
            created_at=datetime.utcnow()
        )
        session.add(pr_obj)
        session.commit()

    # Run ingestion again to link PR
    pipeline.run()

    with get_session_context() as session:
        chunks = session.exec(select(Chunk).where(Chunk.source_type == "commit")).all()
        assert len(chunks) >= 1
        found_pr_context = any("Fix auth timeout bug" in c.text for c in chunks)
        assert found_pr_context, "Commit chunk should be enriched with linked PR title & rationale"


def test_jit_diff_expansion_in_synthesize_node(tmp_path, monkeypatch):
    """Verifies that synthesize_node expands top candidate commits with live unified diffs."""
    repo_dir = tmp_path / "repo_jit_test"
    repo_dir.mkdir()
    git = ["git", "-c", "user.name=Tester", "-c", "user.email=t@example.com"]
    subprocess.run(git + ["init", "-q", str(repo_dir)], check=True)

    code_file = repo_dir / "calc.py"
    code_file.write_text("def calculate(x):\n    return x * 2\n")
    subprocess.run(git + ["-C", str(repo_dir), "add", "calc.py"], check=True)
    subprocess.run(git + ["-C", str(repo_dir), "commit", "-q", "-m", "Initial multiply logic"], check=True)

    res = subprocess.run(git + ["-C", str(repo_dir), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    commit_sha = res.stdout.strip()

    # Mock gemini client so synthesize_node generates with prompt inspection
    mock_client = MagicMock()
    captured_prompts = []
    def mock_generate_text(prompt, **kwargs):
        captured_prompts.append(prompt)
        return "Because the multiplication logic was implemented in commit " + commit_sha[:7]

    mock_client.generate_text.side_effect = mock_generate_text

    monkeypatch.setattr("archaeologist.agent.nodes.synthesize.GeminiClientWrapper", lambda api_key: mock_client)
    monkeypatch.setattr("archaeologist.agent.nodes.synthesize.get_gemini_api_key", lambda: "fake_test_key")

    state: AgentState = {
        "question": "Why does calculate function exist?",
        "repo_id": "test_repo",
        "sub_questions": [],
        "current_sub_question_index": 0,
        "search_queries": [],
        "retrieved_chunks": [{
            "source_type": "commit",
            "source_id": commit_sha,
            "text": f"Commit {commit_sha} Initial multiply logic",
            "related_ids": []
        }],
        "evidence_by_chunk_id": {},
        "draft_answer": "Initial draft",
        "verification_passed": True,
        "unverified_claims": [],
        "retry_count": 0,
        "response": None,
        "repo_path": str(repo_dir)
    }

    result = synthesize_node(state)
    assert result["response"] is not None
    assert len(captured_prompts) == 1
    evidence_prompt = captured_prompts[0]

    # Verify that JIT unified diff preview was retrieved and passed to the reasoning model
    assert "--- JIT Unified Diff Preview" in evidence_prompt
    assert "def calculate(x):" in evidence_prompt
