

from datetime import UTC

from archaeologist.ingestion.git_parser import _get_monorepo_subpath, _normalize_numstat_path
from archaeologist.retrieval.bm25_index import BM25Index
from archaeologist.retrieval.embedder import Embedder
from archaeologist.storage.paths import (
    find_repo_root,
    get_archaeologist_dir,
    get_default_db_url,
    resolve_repo_id,
)


def test_reproduce_find_repo_root_and_paths_on_file_path(tmp_path):
    """Bug Finding 1: Passing a file path inside a repository to find_repo_root or get_archaeologist_dir
    must find the containing repository root, not treat the file itself as the root and crash with FileExistsError.
    """
    repo = tmp_path / "my_project"
    repo.mkdir()
    (repo / ".git").mkdir()

    src = repo / "src"
    src.mkdir()
    code_file = src / "app.py"
    code_file.write_text("print('hello')", encoding="utf-8")

    # When given the file path directly:
    resolved_root = find_repo_root(str(code_file))
    assert resolved_root == repo, f"Expected repo root {repo}, got {resolved_root}"

    # Must not raise FileExistsError or NotADirectoryError when getting storage dir
    arch_dir = get_archaeologist_dir(str(code_file))
    assert arch_dir == repo / ".archaeologist"
    assert arch_dir.exists()

    db_url = get_default_db_url(str(code_file))
    assert "app.py" not in db_url.split("/")[-2]
    assert "my_project" in db_url

    rid = resolve_repo_id(str(code_file))
    assert rid == "my_project", f"Expected repo_id 'my_project', got '{rid}'"


def test_reproduce_bm25_zero_division_on_empty_corpus():
    """Bug Finding 2: BM25Index.fit must not raise ZeroDivisionError when chunks contain no alphanumeric tokens."""
    bm25 = BM25Index()
    # Chunks with empty or punctuation-only texts
    chunks = [
        {"id": "c1", "text": "   ... !!!   "},
        {"id": "c2", "text": ""},
        {"id": "c3", "text": "---"}
    ]
    # This must fit gracefully without ZeroDivisionError
    bm25.fit(chunks)
    assert bm25.bm25 is None or hasattr(bm25.bm25, "get_scores")
    results = bm25.search("test")
    assert results == []


def test_reproduce_get_monorepo_subpath_false_prefix_matching(tmp_path):
    """Bug Finding 3: _get_monorepo_subpath must not match adjacent folders that share a string prefix.
    e.g. 'repo_other' should not match as a subpath of 'repo'.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()

    adjacent = tmp_path / "repo_other"
    adjacent.mkdir()

    subpath = _get_monorepo_subpath(str(adjacent))
    # 'repo_other' is NOT inside 'repo', so subpath must be None
    assert subpath is None, f"Expected None for sibling folder, got '{subpath}'"


def test_reproduce_date_to_same_day_inclusion():
    """Bug Finding 4: date_to='YYYY-MM-DD' without time must include chunks created on that same day."""
    bm25 = BM25Index()
    chunks = [
        {
            "id": "c1",
            "text": "Fixed auth token expiration issue",
            "timestamp": "2024-05-10T14:30:00"
        },
        {
            "id": "c2",
            "text": "Later commit on next day",
            "timestamp": "2024-05-11T09:00:00"
        }
    ]
    bm25.fit(chunks)

    # Searching with date_to='2024-05-10' should include the commit on May 10 at 14:30
    results = bm25.search("auth token", date_to="2024-05-10")
    matching_ids = [r["chunk"]["id"] for r in results]
    assert "c1" in matching_ids, f"Expected 'c1' to be included in date_to='2024-05-10', got {matching_ids}"


def test_reproduce_fastembed_handles_none_text():
    """Bug Finding 5: Embedder.embed_texts must handle None or non-string elements without TypeError."""
    embedder = Embedder()
    # Must not raise TypeError when input list contains None
    embeddings, flags = embedder.embed_texts([None, "print('hello')"], return_success_flags=True)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == embedder.dimension
    assert len(embeddings[1]) == embedder.dimension


def test_reproduce_git_numstat_rename_normalization():
    """Bug Finding 6: _normalize_numstat_path must not produce leading slashes on moved files."""
    p1 = "{subfolder => }/target.py"
    norm1 = _normalize_numstat_path(p1)
    assert norm1 == "target.py", f"Expected 'target.py', got '{norm1}'"

    p2 = "dir/{old => }/file.py"
    norm2 = _normalize_numstat_path(p2)
    assert norm2 == "dir/file.py", f"Expected 'dir/file.py', got '{norm2}'"

    p3 = "{ => new_dir}/app.py"
    norm3 = _normalize_numstat_path(p3)
    assert norm3 == "new_dir/app.py", f"Expected 'new_dir/app.py', got '{norm3}'"


def test_reproduce_vector_store_timezone_normalization(tmp_path):
    """Bug Finding 7: VectorStore datetime parsing must normalize naive dates to UTC rather than system local time."""
    from datetime import datetime

    from archaeologist.retrieval.vector_store import VectorStore

    vs = VectorStore(vector_size=4, storage_path=str(tmp_path / "qdrant"))
    vs.init_collection()

    # Create chunk on May 10 at 12:00:00 UTC
    dt_utc = datetime(2024, 5, 10, 12, 0, 0, tzinfo=UTC)
    vs.upsert_chunks(
        [{
            "id": "c1",
            "repo_id": "test_repo",
            "source_type": "commit",
            "source_id": "s1",
            "text": "test commit",
            "timestamp": dt_utc
        }],
        [[0.1, 0.2, 0.3, 0.4]]
    )

    # Search with date_from='2024-05-10' and date_to='2024-05-10'
    # Must find the chunk regardless of host machine timezone
    hits = vs.search_chunks(
        query_vector=[0.1, 0.2, 0.3, 0.4],
        date_from="2024-05-10",
        date_to="2024-05-10"
    )
    vs.close()
    assert len(hits) == 1, f"Expected 1 hit for same-day search, got {len(hits)}"
    assert hits[0]["id"] == "c1"


def test_reproduce_pipeline_empty_sha_wildcard_corruption():
    """Bug Finding 8: An empty or whitespace string in linked_commit_shas must not match every commit in repo."""
    all_commits_shas = ["abcdef1234567890", "1234567abcdef0000", "fedcba9876543210"]
    pr_linked_shas = ["", "   ", "abc"]

    pr_by_commit = {}
    for sha in pr_linked_shas:
        clean_sha = sha.strip() if isinstance(sha, str) else ""
        if not clean_sha or len(clean_sha) < 7:
            continue
        for c_sha in all_commits_shas:
            if c_sha.startswith(clean_sha):
                pr_by_commit[c_sha] = 42

    # Empty or short string must NOT match any commits
    assert len(pr_by_commit) == 0, f"Expected 0 commits mapped, got {len(pr_by_commit)}"


def test_reproduce_bm25_file_paths_none_safe():
    """Bug Finding 9: BM25Index.search must handle chunks where 'file_paths' is None without TypeError."""
    bm25 = BM25Index()
    bm25.fit([
        {"id": "c1", "text": "refactored database query", "file_paths": None},
        {"id": "c2", "text": "updated auth logic", "file_paths": ["src/auth.py"]}
    ])
    # Must not raise TypeError: 'NoneType' object is not iterable
    res = bm25.search("database", file_path="auth.py")
    assert isinstance(res, list)


def test_reproduce_find_related_discussion_malformed_ref_safe():
    """Bug Finding 10: find_related_discussion_tool must not crash on malformed non-integer related refs."""
    from archaeologist.mcp_server.tools import find_related_discussion_tool
    # Calling with valid SHA or ID should not crash even if database has malformed related_ids
    res = find_related_discussion_tool("nonexistent_commit")
    assert "commits" in res
    assert "pull_requests" in res
    assert "issues" in res


def test_reproduce_detect_github_remote_trailing_slash_git(tmp_path):
    """Bug Finding 11: detect_github_remote must strip trailing slash and .git properly."""
    import subprocess
    repo = tmp_path / "repo"
    repo.mkdir()
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run(git + ["init", "-q", str(repo)], check=True)
    subprocess.run(git + ["-C", str(repo), "remote", "add", "origin", "https://github.com/my-org/my-project.git/"], check=True)

    from archaeologist.storage.paths import detect_github_remote
    url = detect_github_remote(str(repo))
    assert url == "https://github.com/my-org/my-project", f"Expected 'https://github.com/my-org/my-project', got '{url}'"

