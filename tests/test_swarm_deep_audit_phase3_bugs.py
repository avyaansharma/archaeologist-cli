import pytest
from archaeologist.utils.gemini_client import GeminiClientWrapper
from archaeologist.ingestion.link_resolver import extract_refs
from archaeologist.utils.security import sanitize_file_path
from archaeologist.storage.db import init_db, get_session_context
from archaeologist.storage.models import PullRequest, Issue, is_valid_pr_or_issue
from archaeologist.retrieval.vector_store import VectorStore
from datetime import datetime, timezone


@pytest.fixture
def clean_db(tmp_path):
    db_file = tmp_path / "test_swarm_deep_p3.db"
    db_url = f"sqlite:///{db_file.as_posix()}"
    init_db(db_url)
    yield db_url


def test_gemini_client_extracts_json_inside_fences_with_conversational_text():
    """Bug 1: generate_json crashed if Gemini returned conversational preamble
    and postamble containing braces outside the ```json code fence.
    """
    raw_llm_response = (
        "Sure, I analyzed the repository history! Here is the JSON:\n"
        "```json\n"
        "{\n"
        '  "sub_questions": ["What caused the bug in auth?"]\n'
        "}\n"
        "```\n"
        "Hope this helps! Context: {repo: flask}"
    )

    client = GeminiClientWrapper.__new__(GeminiClientWrapper)
    # Monkeypatch generate_text to return the raw_llm_response
    client.generate_text = lambda *args, **kwargs: raw_llm_response

    # Should successfully parse the JSON inside the code fence
    parsed = client.generate_json("fake prompt")
    assert isinstance(parsed, dict)
    assert parsed.get("sub_questions") == ["What caused the bug in auth?"]


def test_link_resolver_captures_colons_and_multi_issue_closing_clauses():
    """Bug 2: extract_refs missed 'Fixes: #123' and only extracted the first issue
    in comma-separated closing clauses like 'Fixes #101, #102, and #103'.
    """
    # Test case A: colon after keyword
    text_colon = "Fixes: #42 and closes: #43"
    refs_colon = extract_refs(text_colon)
    assert 42 in refs_colon["closes"], f"Expected 42 in closes, got {refs_colon['closes']}"
    assert 43 in refs_colon["closes"], f"Expected 43 in closes, got {refs_colon['closes']}"

    # Test case B: multiple comma-separated issues
    text_multi = "Resolves #101, #102, and #103 in routing layer"
    refs_multi = extract_refs(text_multi)
    assert 101 in refs_multi["closes"]
    assert 102 in refs_multi["closes"], f"Expected 102 in closes, got {refs_multi['closes']}"
    assert 103 in refs_multi["closes"], f"Expected 103 in closes, got {refs_multi['closes']}"


def test_sanitize_file_path_returns_forward_slashes_on_windows(tmp_path):
    """Bug 3: sanitize_file_path returned native Windows backslashes, causing
    path lookups against forward-slash database records to fail.
    """
    base_dir = tmp_path / "my_repo"
    base_dir.mkdir()
    sub_file = "src/auth/login.py"

    normalized = sanitize_file_path(str(base_dir), sub_file)
    assert "\\" not in normalized, f"Path contained Windows backslashes: '{normalized}'"
    assert normalized == "src/auth/login.py"


def test_is_valid_pr_or_issue_supports_string_nums_and_repo_basename(clean_db):
    """Bug 4: is_valid_pr_or_issue failed if ref_num was a string or repo_id was a basename."""
    now = datetime.now(timezone.utc)
    with get_session_context() as session:
        session.add(PullRequest(
            number=777,
            repo_id="pallets/flask",
            title="Add feature",
            state="merged",
            created_at=now
        ))

    with get_session_context() as session:
        # String number lookup
        assert is_valid_pr_or_issue(session, "777", repo_id="pallets/flask") is True
        # Basename repo_id lookup
        assert is_valid_pr_or_issue(session, 777, repo_id="flask") is True


def test_vector_store_search_supports_repo_basename_matching(tmp_path):
    """Bug 5: Qdrant search_chunks with repo_id='flask' must match vectors
    indexed with repo_id='pallets/flask'.
    """
    store = VectorStore(
        storage_path=(tmp_path / "qdrant_test").as_posix(),
        collection_name="test_repo_basename"
    )
    now = datetime.now(timezone.utc)
    dummy_vec = [0.1] * 768

    store.upsert_chunks(
        chunks=[{
            "id": "chunk_flask_1",
            "repo_id": "pallets/flask",
            "source_type": "commit",
            "source_id": "sha123",
            "text": "Routing engine refactor",
            "timestamp": now
        }],
        embeddings=[dummy_vec]
    )

    # Search with basename 'flask'
    hits = store.search_chunks(
        query_vector=dummy_vec,
        limit=5,
        repo_id="flask"
    )
    assert len(hits) > 0, "search_chunks returned 0 hits when querying by repo basename 'flask'!"
    assert hits[0]["payload"]["repo_id"] == "pallets/flask"
