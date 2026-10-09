import sys
import pytest
from datetime import datetime
from archaeologist.ingestion.symbol_parser import FUNCTION_REGEX, extract_modified_line_numbers_from_diff
from archaeologist.ingestion.link_resolver import SHA_REF
from archaeologist.ingestion.revert_detector import _escape_like
from archaeologist.utils.security import SHA_REGEX
from archaeologist.retrieval.embedder import Embedder
from archaeologist.retrieval.vector_store import VectorStore

def test_sha_ref_ignores_pure_decimal_numbers():
    text = "Fixed in build 20240115, ticket 9876543, commit abc12345"
    matches = SHA_REF.findall(text)
    assert "abc12345" in matches
    assert "20240115" not in matches
    assert "9876543" not in matches

def test_function_regex_requires_explicit_keywords():
    call_line = "    doSomethingRandom(x, y);"
    def_line = "def my_python_func(a, b):"
    js_func = "function myJsFunc(x) {"
    
    assert FUNCTION_REGEX.match(call_line) is None
    assert FUNCTION_REGEX.match(def_line) is not None
    assert FUNCTION_REGEX.match(def_line).group(1) == "my_python_func"
    assert FUNCTION_REGEX.match(js_func) is not None
    assert FUNCTION_REGEX.match(js_func).group(1) == "myJsFunc"

def test_sql_like_wildcard_escaping():
    raw_subject = "Fix 100% CPU_usage & memory"
    escaped = _escape_like(raw_subject)
    assert escaped == "Fix 100\\% CPU\\_usage & memory"

def test_diff_parser_handles_deleted_file():
    diff_text = (
        "--- a/old_file.py\n"
        "+++ b/old_file.py\n"
        "@@ -1,3 +1,3 @@\n"
        "+line 1\n"
        "--- a/deleted.py\n"
        "+++ /dev/null\n"
        "@@ -1,2 +0,0 @@\n"
        "-deleted line 1\n"
        "--- a/new_file.py\n"
        "+++ b/new_file.py\n"
        "@@ -1,2 +1,2 @@\n"
        "+new line\n"
    )
    res = extract_modified_line_numbers_from_diff(diff_text)
    assert "old_file.py" in res
    assert "new_file.py" in res
    assert "/dev/null" not in res
    assert None not in res

def test_sha_regex_misrouting_guard():
    assert not SHA_REGEX.match("issue1234")
    assert SHA_REGEX.match("a1b2c3d")
    assert SHA_REGEX.match("4970a09")

def test_pure_deletion_hunk_line_attribution():
    diff_text = (
        "--- a/src/app.py\n"
        "+++ b/src/app.py\n"
        "@@ -15,5 +15,0 @@\n"
        "-def old_unused_function():\n"
        "-    pass\n"
        "-    return None\n"
    )
    res = extract_modified_line_numbers_from_diff(diff_text)
    assert "src/app.py" in res
    assert 15 in res["src/app.py"]
    assert 16 in res["src/app.py"]

def test_mock_embedder_high_entropy():
    embedder = Embedder()
    vecs = embedder._embed_mock(["hello world", "test query"])
    assert len(vecs) == 2
    assert len(vecs[0]) == embedder.dimension
    # Check that elements across dimension are distinct (not repeating 32-byte pattern)
    unique_vals = len(set(vecs[0][:100]))
    assert unique_vals > 50

def test_vector_store_fallback_flag():
    store = VectorStore()
    assert hasattr(store, "is_in_memory_fallback")
    assert isinstance(store.is_in_memory_fallback, bool)
    store.close()

def test_update_cross_links_accepts_orm_session():
    from sqlmodel import create_engine, Session, SQLModel
    from archaeologist.ingestion.link_resolver import update_cross_links
    from archaeologist.storage.models import PullRequest, Issue
    from datetime import datetime

    test_engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(test_engine)
    with Session(test_engine) as session:
        p_num = 1010
        i_num = 1020

        pr = PullRequest(number=p_num, title=f"Fix issue #{i_num}", body=f"Closes #{i_num}", author="alice", state="closed", created_at=datetime.utcnow())
        issue = Issue(number=i_num, title="Bug in auth", body="Found bug abc12345", state="open", author="bob", created_at=datetime.utcnow())
        session.add(pr)
        session.add(issue)
        session.commit()

        update_cross_links(session)

        updated_pr = session.get(PullRequest, ("", p_num))
        updated_issue = session.get(Issue, ("", i_num))

        assert i_num in updated_pr.linked_issue_numbers
        assert p_num in updated_issue.linked_pr_numbers



def test_increment_retry_preserves_sub_question_index():
    from archaeologist.agent.graph import increment_retry_node
    
    state = {
        "retry_count": 0,
        "current_sub_question_index": 3,
        "draft_answer": "unverified answer",
        "verification_passed": False
    }
    new_state = increment_retry_node(state)
    assert new_state["retry_count"] == 1
    assert new_state["current_sub_question_index"] == 3
    assert new_state["draft_answer"] is None

def test_escape_like_central_security_helper():
    from archaeologist.utils.security import escape_like
    assert escape_like("get_user") == "get\\_user"
    assert escape_like("100%") == "100\\%"

def test_sha_regex_allows_parent_ref_suffixes():
    from archaeologist.utils.security import sanitize_sha
    assert sanitize_sha("edabb99b12be5e1^") == "edabb99b12be5e1^"
    assert sanitize_sha("edabb99b12be5e1^1") == "edabb99b12be5e1^1"
    assert sanitize_sha("edabb99b12be5e1~2") == "edabb99b12be5e1~2"

def test_repo_ownership_per_file_breakdown_ranking():
    from archaeologist.mcp_server.tools import repo_ownership_tool
    res = repo_ownership_tool()
    assert "per_file_breakdown" in res
    breakdown = res["per_file_breakdown"]
    totals = [data["total_commits"] for data in breakdown.values()]
    assert totals == sorted(totals, reverse=True)

def test_chunk_commit_squash_merge_bullets():
    from archaeologist.ingestion.chunker import chunk_commit
    commit = {
        "sha": "abc123456789",
        "author_name": "Alice",
        "message": "Squash merge PR #123:\n* Refactor auth.py transport\n* Fix memory leak in session\n* Add unit tests",
        "authored_date": datetime.utcnow(),
        "files_changed": ["src/auth.py"]
    }
    chunks = chunk_commit(commit, diff_summary="Refactored auth")
    assert isinstance(chunks, list)
    assert len(chunks) == 4
    assert chunks[0]["source_id"] == "abc123456789"
    assert "sub-topic: * Refactor auth.py" in chunks[1]["text"]

def test_plan_node_unverified_claims_prompt(monkeypatch):
    from archaeologist.agent.nodes.plan import plan_node
    
    state = {
        "question": "How does auth work?",
        "sub_questions": ["How does auth work?"],
        "current_sub_question_index": 0,
        "retry_count": 1,
        "unverified_claims": ["Auth uses JWT tokens."]
    }
    
    # Verify plan_node handles retry state with unverified claims without error
    res = plan_node(state)
    assert "search_queries" in res
    assert isinstance(res["search_queries"], list)



