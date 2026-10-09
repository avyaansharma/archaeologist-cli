"""Concrete reproducible test cases establishing architectural flaws, edge cases,
and vulnerabilities in Codebase Archaeologist.

These tests are expected to FAIL against the current implementation, demonstrating
the bugs flagged during the senior code review.
"""

import os
import sys
import pytest
from datetime import datetime
from sqlmodel import SQLModel, create_engine, Session, select
from sqlite3 import IntegrityError

from archaeologist.storage.models import PullRequest, Issue, Chunk
from archaeologist.ingestion.link_resolver import update_cross_links
from archaeologist.ingestion.symbol_parser import extract_modified_line_numbers_from_diff
from archaeologist.ingestion.git_parser import _parse_commit_record, FIELD_DELIM
from archaeologist.ingestion.chunker import chunk_commit, token_count
from archaeologist.retrieval.bm25_index import BM25Index
from archaeologist.agent.nodes.follow_links import follow_links_node
from archaeologist.agent.state import AgentState
from archaeologist.web.server import _activate_repo_environment
from archaeologist.web.registry import REPOSITORIES
from archaeologist.storage.context import current_client_api_key_var


# ==============================================================================
# BUG 1: Multi-Repository Primary Key Collision in PullRequest and Issue Models
# Target: archaeologist/storage/models.py (lines 22-52)
# ==============================================================================
def test_failing_multi_repo_pk_collision_pull_request_and_issue():
    """Demonstrates that PullRequest and Issue use 'number' as the sole primary key,
    making it impossible to ingest multiple repositories without primary key collisions
    and data loss/corruption.
    """
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        pr_repo_a = PullRequest(
            number=42,
            repo_id="owner/repo-a",
            title="Repo A PR: Fix database pool leak",
            state="merged",
            author="alice",
            created_at=datetime.utcnow()
        )
        session.add(pr_repo_a)
        session.commit()

        # Ingesting second repository with PR #42
        pr_repo_b = PullRequest(
            number=42,
            repo_id="owner/repo-b",
            title="Repo B PR: Update frontend styles",
            state="open",
            author="bob",
            created_at=datetime.utcnow()
        )
        session.add(pr_repo_b)

        # In a multi-repo database, both PRs should coexist.
        # Current behavior: Raises sqlite3.IntegrityError (UNIQUE constraint failed: pullrequest.number)
        session.commit()

        prs = session.exec(select(PullRequest)).all()
        assert len(prs) == 2, f"Expected 2 PRs across repos, but found {len(prs)}"
        repo_ids = {p.repo_id for p in prs}
        assert "owner/repo-a" in repo_ids and "owner/repo-b" in repo_ids


# ==============================================================================
# BUG 2: Cross-Repository Contamination in Link Resolver
# Target: archaeologist/ingestion/link_resolver.py (lines 40-44)
# ==============================================================================
def test_failing_cross_repo_link_contamination_in_link_resolver():
    """Demonstrates that update_cross_links ignores repo_id, falsely linking PRs
    from repo A to issues with the same number in repo B.
    """
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        # Repo A has PR #10 mentioning '#5'
        pr_a = PullRequest(
            number=10,
            repo_id="org/repo-alpha",
            title="Refactor auth",
            body="Fixes #5 by updating tokens",
            state="merged",
            author="alice",
            created_at=datetime.utcnow()
        )
        session.add(pr_a)

        # Repo B has Issue #5 (completely separate repository!)
        issue_b = Issue(
            number=5,
            repo_id="org/repo-beta",
            title="Confidential security vulnerability in payment gateway",
            body="Zero-day auth bypass",
            state="open",
            author="eve",
            created_at=datetime.utcnow(),
            linked_pr_numbers=[]
        )
        session.add(issue_b)
        session.commit()

        # Run cross-linker
        update_cross_links(session)

        # Refresh issue_b
        session.refresh(issue_b)

        # CRITICAL FLAW: Issue #5 in repo-beta now has linked_pr_numbers=[10]
        # even though PR #10 is from repo-alpha!
        assert issue_b.linked_pr_numbers == [], (
            f"Cross-repo contamination! Issue #5 in {issue_b.repo_id} was falsely "
            f"linked to PR #10 from another repo: {issue_b.linked_pr_numbers}"
        )


# ==============================================================================
# BUG 3: Silent Discarding of Deleted Files in Unified Diff Parsing
# Target: archaeologist/ingestion/symbol_parser.py (lines 151-155)
# ==============================================================================
def test_failing_diff_parser_silent_drop_of_deleted_files():
    """Demonstrates that extract_modified_line_numbers_from_diff silently ignores
    deleted files because line.startswith('+++ ') points to '+++ /dev/null',
    preventing any tracking of deleted symbols.
    """
    git_diff_deleted_file = (
        "diff --git a/src/legacy_crypto.py b/src/legacy_crypto.py\n"
        "deleted file mode 100644\n"
        "index e69de29..0000000\n"
        "--- a/src/legacy_crypto.py\n"
        "+++ /dev/null\n"
        "@@ -1,5 +0,0 @@\n"
        "-def insecure_des_encrypt(data, key):\n"
        "-    # Deprecated cipher\n"
        "-    return des(key).encrypt(data)\n"
        "-\n"
        "-CIPHER_NAME = 'DES'\n"
    )

    result = extract_modified_line_numbers_from_diff(git_diff_deleted_file)

    # The parser should capture that 'src/legacy_crypto.py' was modified with deleted lines 1-5
    assert "src/legacy_crypto.py" in result, (
        f"Diff parser completely dropped deleted file 'src/legacy_crypto.py'! Result was: {result}"
    )
    assert 1 in result["src/legacy_crypto.py"]["deleted"], (
        "Deleted line 1 was not recorded for deleted file!"
    )


# ==============================================================================
# BUG 4: Git Rename Path String Corruption in _parse_commit_record
# Target: archaeologist/ingestion/git_parser.py (lines 73-80)
# ==============================================================================
def test_failing_git_numstat_rename_path_corruption():
    """Demonstrates that _parse_commit_record corrupts file paths when git numstat
    outputs rename syntax 'dir/{old.py => new.py}'.
    """
    # Simulated git log record with numstat showing a rename
    record = (
        f"abc1234567{FIELD_DELIM}Alice{FIELD_DELIM}alice@example.com{FIELD_DELIM}"
        f"2026-01-01T12:00:00{FIELD_DELIM}Rename auth service{FIELD_DELIM}NUMSTAT\n"
        "10\t2\tsrc/{auth_v1.py => auth_v2.py}\n"
    )

    commit = _parse_commit_record(record)
    files = commit["files_changed"]

    # The parser stores 'src/{auth_v1.py => auth_v2.py}' as a literal file path,
    # which fails all downstream git show / ast parsing operations.
    assert "src/{auth_v1.py => auth_v2.py}" not in files, (
        f"Raw rename notation leaked into files_changed: {files}"
    )
    assert "src/auth_v2.py" in files, (
        f"Expected target file 'src/auth_v2.py' in files_changed, got: {files}"
    )


# ==============================================================================
# BUG 5: Unbounded Commit Message Overflow in chunk_commit
# Target: archaeologist/ingestion/chunker.py (lines 40-47)
# ==============================================================================
def test_failing_chunk_commit_unbounded_message_overflow():
    """Demonstrates that chunk_commit only truncates diff summary, allowing
    large commit messages (e.g. squash merges, changelogs) to blow past 500 tokens.
    """
    # Create a 1,200-word commit message
    long_message = "Feature explanation: " + " ".join([f"detail_{i}" for i in range(1200)])
    commit = {
        "sha": "1234567890abcdef",
        "author_name": "Developer",
        "authored_date": datetime.utcnow(),
        "message": long_message,
        "files_changed": ["src/main.py"],
        "symbols_modified": ["main"],
        "is_revert": False
    }

    chunks = chunk_commit(commit, diff_summary="Diff summary here")

    assert len(chunks) >= 1
    main_chunk = chunks[0]
    tokens = token_count(main_chunk["text"])

    # Current behavior: tokens will be ~1200+ because raw_msg is never truncated!
    assert tokens <= 550, (
        f"Chunk exceeded token budget! Expected <= 550 tokens, got {tokens} tokens."
    )


# ==============================================================================
# BUG 6: Silent Cross-Repo Chunk Leakage in BM25 Metadata Filtering
# Target: archaeologist/retrieval/bm25_index.py (lines 54-57)
# ==============================================================================
def test_failing_bm25_repo_leakage_when_chunk_repo_none_or_substring():
    """Demonstrates that BM25Index.search leaks chunks whose repo_id is None
    or matches as a substring into queries for other repositories.
    """
    bm25 = BM25Index()
    test_chunks = [
        {
            "id": "c1",
            "repo_id": "confidential-finance",
            "source_type": "commit",
            "source_id": "sha1",
            "text": "Executive payroll bonus authorization and bank routing credentials",
            "file_paths": ["finance/payroll.py"],
            "symbols_modified": [],
            "timestamp": datetime.utcnow(),
            "is_reverted": False
        },
        {
            "id": "c2",
            "repo_id": None,  # Legacy or unattributed chunk in the database
            "text": "Executive payroll bonus summary note for auditors",
            "source_type": "commit",
            "source_id": "sha2",
            "file_paths": ["audit.txt"],
            "symbols_modified": [],
            "timestamp": datetime.utcnow(),
            "is_reverted": False
        },
        {
            "id": "c3",
            "repo_id": "public-docs",
            "text": "Public executive documentation on open salaries",
            "source_type": "commit",
            "source_id": "sha3",
            "file_paths": ["docs/salaries.md"],
            "symbols_modified": [],
            "timestamp": datetime.utcnow(),
            "is_reverted": False
        }
    ]

    bm25.build(test_chunks)

    # Search specifically for repo_id="public-docs"
    results = bm25.search(query="executive payroll bonus", repo_id="public-docs", limit=10)

    returned_ids = [r["chunk"]["id"] for r in results]

    # Chunk c2 has repo_id=None. In bm25_index.py:
    #   if c_repo and c_repo.lower() != repo_id.lower() ...: continue
    # When c_repo is None, the 'if' fails, and the chunk LEAKS into public-docs results!
    assert "c2" not in returned_ids, (
        f"Security flaw: Chunk c2 (repo_id=None) leaked into repo_id='public-docs' query! "
        f"Returned IDs: {returned_ids}"
    )


# ==============================================================================
# BUG 7: High-Numbered Issues/PRs Misclassified as Git SHAs in follow_links_node
# Target: archaeologist/agent/nodes/follow_links.py (lines 30-46)
# ==============================================================================
def test_failing_follow_links_numeric_id_misclassified_as_sha():
    """Demonstrates that follow_links_node treats numeric issue/PR IDs with >= 7 digits
    (e.g. issue#1000001) as commit SHAs, querying source_id.like('1000001%') instead
    of source_id == 'issue#1000001'.
    """
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)

    from archaeologist.storage.db import init_db
    init_db("sqlite:///:memory:")

    with Session(engine) as session:
        # Seed an issue chunk with high numeric ID (>= 7 digits)
        target_chunk = Chunk(
            id="issue_chunk_1",
            source_type="issue",
            source_id="issue#1000001",
            text="High priority issue regarding authentication deadlock",
            timestamp=datetime.utcnow(),
            file_paths=["auth.py"],
            symbols_modified=[],
            related_ids=[]
        )
        session.add(target_chunk)
        session.commit()

        # Mock agent state referencing issue#1000001
        state: AgentState = {
            "question": "What happened in issue 1000001?",
            "repo_id": None,
            "retrieved_chunks": [
                {
                    "id": "commit_chunk_1",
                    "source_type": "commit",
                    "source_id": "sha_123",
                    "text": "Initial commit fixing #1000001",
                    "related_ids": ["#1000001"]
                }
            ]
        }

        # Monkeypatch get_session_context to use our test database
        from unittest.mock import patch
        from contextlib import contextmanager
        @contextmanager
        def mock_ctx():
            with Session(engine) as s:
                yield s

        with patch("archaeologist.agent.nodes.follow_links.get_session_context", mock_ctx):
            updates = follow_links_node(state)

        retrieved = updates.get("retrieved_chunks", [])
        retrieved_ids = [c["id"] for c in retrieved]

        # Current behavior: is_sha evaluates to True for "1000001", so it searches
        # for source_id == "1000001" or like "1000001%", missing "issue#1000001"!
        assert "issue_chunk_1" in retrieved_ids, (
            f"follow_links_node misclassified 'issue#1000001' as a hex commit SHA! "
            f"Failed to retrieve target chunk. Retrieved: {retrieved_ids}"
        )


# ==============================================================================
# BUG 8: Process-Global Environment Mutation Race Condition in FastAPI Server
# Target: archaeologist/web/server.py (lines 48-60)
# ==============================================================================
def test_failing_concurrent_env_contamination_in_server():
    """Demonstrates that _activate_repo_environment directly mutates process-global
    os.environ['GEMINI_API_KEY'], causing client credentials to leak across concurrent
    requests from different users.
    """
    # Setup mock registry entries
    REPOSITORIES["repo_alpha"] = {
        "repo_id": "repo_alpha",
        "name": "Alpha",
        "db_path": "alpha.db",
        "bm25_path": "alpha.bin"
    }
    REPOSITORIES["repo_beta"] = {
        "repo_id": "repo_beta",
        "name": "Beta",
        "db_path": "beta.db",
        "bm25_path": "beta.bin"
    }

    try:
        # Clear initial state
        if "GEMINI_API_KEY" in os.environ:
            del os.environ["GEMINI_API_KEY"]

        # Request 1: User Alpha provides their private API key
        _activate_repo_environment("repo_alpha", client_api_key="SECRET_ALPHA_KEY_999")
        # In a secure context-isolated server, client key is set in ContextVar, NEVER in os.environ
        assert current_client_api_key_var.get() == "SECRET_ALPHA_KEY_999"
        assert os.environ.get("GEMINI_API_KEY") is None

        # Request 2: Concurrent User Beta on repo_beta sends NO API key (expected to use default or fail)
        _activate_repo_environment("repo_beta", client_api_key=None)

        # In a thread-safe / context-isolated server, User Beta must NOT inherit User Alpha's key.
        assert current_client_api_key_var.get() is None
        assert os.environ.get("GEMINI_API_KEY") is None, (
            "Process-global environment mutation leak! "
            "User Alpha's private API key leaked to User Beta's request: "
            f"{os.environ.get('GEMINI_API_KEY')}"
        )
    finally:
        REPOSITORIES.pop("repo_alpha", None)
        REPOSITORIES.pop("repo_beta", None)
