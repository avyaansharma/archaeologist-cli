import os
import pytest
from datetime import datetime, timezone
from sqlmodel import select

from archaeologist.storage.db import init_db, get_session_context
from archaeologist.storage.models import Commit, SymbolIndex
from archaeologist.ingestion.revert_detector import find_reverted_commit, detect_revert_from_message
from archaeologist.ingestion.git_parser import _normalize_numstat_path
from archaeologist.storage.paths import get_default_db_path
from archaeologist.mcp_server.tools import repo_hotspots_tool, repo_ownership_tool
from archaeologist.retrieval.bm25_index import tokenize_text, BM25Index


@pytest.fixture
def clean_db(tmp_path):
    db_file = tmp_path / "test_swarm_deep_p2.db"
    db_url = f"sqlite:///{db_file.as_posix()}"
    init_db(db_url)
    yield db_url


def test_revert_detector_prevents_cross_repo_causal_corruption(clean_db):
    """Bug 1: find_reverted_commit did not filter by repo_id.
    When repo A and repo B both have a commit with the subject 'Fix bug',
    a revert in repo B must NOT link to the commit in repo A.
    """
    now = datetime.now(timezone.utc)
    t1 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    t2 = datetime(2025, 2, 1, tzinfo=timezone.utc)
    t3 = datetime(2025, 3, 1, tzinfo=timezone.utc)

    with get_session_context() as session:
        # Commit in Repo A (older)
        session.add(Commit(
            sha="commit_repo_a",
            repo_id="pallets/flask",
            author_name="Alice",
            author_email="alice@example.com",
            authored_date=t1,
            message="Fix bug in auth",
            files_changed=["auth.py"],
            symbols_modified=[]
        ))
        # Commit in Repo B (newer, but still before revert)
        session.add(Commit(
            sha="commit_repo_b",
            repo_id="psf/requests",
            author_name="Bob",
            author_email="bob@example.com",
            authored_date=t2,
            message="Fix bug in auth",
            files_changed=["requests/auth.py"],
            symbols_modified=[]
        ))

    with get_session_context() as session:
        # Repo A has a revert commit at t3
        # Looking up original commit for Repo A must return commit_repo_a, NOT commit_repo_b
        res_a = find_reverted_commit(
            session,
            reverted_subject="Fix bug in auth",
            before_date=t3,
            repo_id="pallets/flask"
        )
        assert res_a is not None
        assert res_a.sha == "commit_repo_a", f"Cross-repo corruption! Expected commit_repo_a but got {res_a.sha}"


def test_normalize_numstat_path_strips_quotes_on_rename():
    """Bug 2: _normalize_numstat_path left trailing or leading quotes when
    git numstat quotes renamed paths containing spaces.
    """
    # Example 1: Git rename with spaces in target
    raw_path_1 = '"old folder/file.py => new folder/file.py"'
    clean_1 = _normalize_numstat_path(raw_path_1)
    assert clean_1 == "new folder/file.py", f"Expected 'new folder/file.py', got '{clean_1}'"

    # Example 2: Brace syntax with spaces
    raw_path_2 = '"src/{old dir => new dir}/main.py"'
    clean_2 = _normalize_numstat_path(raw_path_2)
    assert clean_2 == "src/new dir/main.py", f"Expected 'src/new dir/main.py', got '{clean_2}'"


def test_init_db_creates_missing_parent_directories(tmp_path):
    """Bug 3: init_db failed with OperationalError if parent directory of SQLite DB didn't exist."""
    nested_dir = tmp_path / "deep" / "nested" / "folder"
    nested_db = nested_dir / "archaeologist.db"
    db_url = f"sqlite:///{nested_db.as_posix()}"

    assert not nested_dir.exists()
    init_db(db_url)
    assert nested_dir.exists(), "init_db did not create missing parent directory"
    assert nested_db.exists(), "init_db did not create database file"


def test_get_default_db_path_strips_url_query_parameters(monkeypatch):
    """Bug 4: get_default_db_path preserved '?check_same_thread=False' query params,
    corrupting the filesystem path.
    """
    monkeypatch.setenv("DATABASE_URL", "sqlite:///C:/data/repo.db?check_same_thread=False&timeout=30")
    db_path = get_default_db_path()
    assert "?" not in db_path, f"Database path contained URL query parameters: {db_path}"
    assert db_path.endswith("repo.db")


def test_analytical_tools_support_repo_basename_matching(clean_db):
    """Bug 5: Ingested commits have repo_id='pallets/flask', but CLI/user passes 'flask'.
    Analytical tools should match the repo basename instead of returning empty results.
    """
    now = datetime.now(timezone.utc)
    with get_session_context() as session:
        session.add(Commit(
            sha="commit_flask_1",
            repo_id="pallets/flask",
            author_name="Armin",
            author_email="armin@example.com",
            authored_date=now,
            message="Improve routing engine",
            files_changed=["flask/app.py", "flask/routing.py"],
            symbols_modified=[]
        ))

    # Querying using repo_id="flask" (basename)
    hotspots = repo_hotspots_tool(top_n=5, repo_id="flask")
    assert len(hotspots) > 0, "repo_hotspots_tool returned 0 hotspots when querying by basename 'flask'!"
    assert any(h["file_path"] == "flask/app.py" for h in hotspots)

    ownership = repo_ownership_tool(repo_id="flask")
    assert ownership["total_commits"] == 1, "repo_ownership_tool returned 0 commits when querying by basename 'flask'!"


def test_bm25_tokenize_text_handles_none_gracefully():
    """Bug 6: tokenize_text raised AttributeError if passed None or non-string."""
    assert tokenize_text(None) == []
    assert tokenize_text("") == []
    
    idx = BM25Index()
    # Should not raise AttributeError when searching with None or empty
    assert idx.search(None) == []
