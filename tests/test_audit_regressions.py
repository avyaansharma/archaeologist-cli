"""Regression tests from the 2026-10-08 adversarial audit.

Each test is expected to FAIL initially on unmodified code and PASS once its fix lands.
"""
import contextvars
import os
import pickle
import subprocess
from datetime import datetime
import pytest


def _chunk(cid, text, ts):
    return {
        "id": cid,
        "text": text,
        "timestamp": ts,
        "repo_id": "r",
        "source_type": "commit",
        "source_id": cid,
    }


# D1: a locked local store must refuse writes, not accept them into RAM
def test_d1_locked_store_refuses_writes(tmp_path, monkeypatch):
    import archaeologist.retrieval.vector_store as vs

    monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:9")  # nothing listens here
    real_client = vs.QdrantClient

    def fake_client(*args, **kwargs):
        if kwargs.get("path"):
            raise RuntimeError("Storage folder is already accessed by another instance")
        return real_client(*args, **kwargs)

    monkeypatch.setattr(vs, "QdrantClient", fake_client)

    with pytest.raises(Exception) as exc_info:
        store = vs.VectorStore(vector_size=4, storage_path=str(tmp_path / "qdrant"), writable=True)
        store.init_collection()
        store.upsert_chunks(
            [_chunk("a", "hello", datetime(2024, 5, 1))],
            [[0.1, 0.2, 0.3, 0.4]]
        )
    # Must raise a locked error or runtime error, NOT succeed into memory
    assert "locked" in str(exc_info.value).lower() or "accessed by another" in str(exc_info.value).lower()


# D3: ingest and query must compute the same repo_id
def test_d3_cli_and_pipeline_agree_on_repo_id(tmp_path, monkeypatch):
    monkeypatch.setenv("ARCHAEOLOGIST_DB_URL", f"sqlite:///{tmp_path / 'scratch.db'}")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'scratch.db'}")
    monkeypatch.setenv("BM25_INDEX_PATH", str(tmp_path / "bm25.bin"))
    from archaeologist.cli import _configure_repo_env
    from archaeologist.ingestion.pipeline import IngestionPipeline

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    pipeline = IngestionPipeline(
        repo_path=str(checkout),
        repo_url="https://github.com/acme/widget.git"
    )
    assert _configure_repo_env(repo_path=str(checkout)) == pipeline.repo_id


# D4: loading an index must never execute arbitrary pickled payloads
def test_d4_index_load_never_executes_file_contents(tmp_path):
    marker = tmp_path / "executed"

    class Payload:
        def __reduce__(self):
            return (open, (str(marker), "w"))

    index_file = tmp_path / "bm25_index.bin"
    index_file.write_bytes(pickle.dumps(Payload()))

    from archaeologist.retrieval.bm25_index import BM25Index
    try:
        BM25Index().load(str(index_file))
    except Exception:
        pass
    assert not marker.exists(), "Arbitrary code executed via index load!"


# D5: the request's context must win over the process environment
def test_d5_request_context_beats_process_env(monkeypatch):
    from archaeologist.storage.context import current_db_url_var
    from archaeologist.storage.paths import get_default_db_url

    monkeypatch.setenv("DATABASE_URL", "sqlite:///tenant_b.db")

    def in_request():
        current_db_url_var.set("sqlite:///tenant_a.db")
        return get_default_db_url()

    assert contextvars.copy_context().run(in_request) == "sqlite:///tenant_a.db"


# D5: a keyed request followed by an unkeyed one must not destroy the server key
def test_d5_server_key_survives_keyed_then_unkeyed_request(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'x.db'}")
    monkeypatch.setenv("BM25_INDEX_PATH", str(tmp_path / "bm25.bin"))
    monkeypatch.setenv("GEMINI_API_KEY", "server-key")
    from archaeologist.web import server
    monkeypatch.setattr(server, "_CLIENT_OVERRIDE_KEY", None, raising=False)

    def two_requests():
        server._activate_repo_environment("local-repo", client_api_key="client-key")
        server._activate_repo_environment("local-repo", client_api_key=None)

    contextvars.copy_context().run(two_requests)
    assert os.environ.get("GEMINI_API_KEY") == "server-key"


# D6: a generic DATABASE_URL in the shell must not be adopted
def test_d6_generic_database_url_is_not_adopted(tmp_path, monkeypatch):
    from archaeologist.storage.paths import get_default_db_url

    monkeypatch.setenv("DATABASE_URL", "postgresql://app:secret@prod/app")
    monkeypatch.chdir(tmp_path)
    assert get_default_db_url().startswith("sqlite:///")


# D7: one merge-SHA column, not two
def test_d7_pull_request_has_one_merge_sha_column():
    from archaeologist.storage.models import PullRequest

    both = {"merge_commit_sha", "merged_commit_sha"}
    assert not both <= set(PullRequest.model_fields), "PullRequest should only have merge_commit_sha"


# D8: ISO dates with a Z suffix must filter, not crash
def test_d8_bm25_accepts_utc_suffixed_dates():
    from archaeologist.retrieval.bm25_index import BM25Index

    index = BM25Index()
    index.fit([
        _chunk("a", "retry logic for upstream timeouts", datetime(2024, 5, 1)),
        _chunk("b", "unrelated changelog entry", datetime(2024, 5, 2)),
        _chunk("c", "documentation typo", datetime(2024, 5, 3)),
    ])
    hits = index.search("retry", date_from="2024-01-01T00:00:00Z")
    assert [h["chunk"]["id"] for h in hits] == ["a"]


# D9: importing the models must not replace SQLAlchemy's Session.get
def test_d9_session_get_is_not_patched():
    import archaeologist.storage.models  # noqa: F401
    from sqlalchemy.orm import Session

    assert Session.get.__module__.startswith("sqlalchemy")


# D14: an agent failure must surface as an exception
def test_d14_ask_tool_raises_on_agent_failure(monkeypatch):
    import archaeologist.mcp_server.tools as tools

    class BrokenGraph:
        def invoke(self, *args, **kwargs):
            raise RuntimeError("agent exploded")

    monkeypatch.setattr(tools, "agent_graph", BrokenGraph())
    with pytest.raises(RuntimeError):
        tools.ask_tool("why did this change?")


# D20: a git worktree is a repo root (.git is a file there, not a directory)
def test_d20_repo_root_found_inside_a_worktree(tmp_path):
    from archaeologist.storage.paths import find_repo_root

    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com"]
    main = tmp_path / "main"
    main.mkdir()
    subprocess.run(git + ["init", "-q", str(main)], check=True)
    subprocess.run(git + ["-C", str(main), "commit", "-q", "--allow-empty", "-m", "init"], check=True)
    worktree = tmp_path / "wt"
    subprocess.run(git + ["-C", str(main), "worktree", "add", "-q", str(worktree)], check=True)
    sub = worktree / "src"
    sub.mkdir()
    assert find_repo_root(str(sub)) == worktree.resolve()


# D11: opening the store at query time must never delete vectors
def test_d11_query_time_open_never_deletes_vectors(tmp_path, monkeypatch):
    import archaeologist.retrieval.vector_store as vs

    monkeypatch.setenv("QDRANT_URL", "http://127.0.0.1:9")
    path = str(tmp_path / "qdrant")
    writer = vs.VectorStore(vector_size=8, storage_path=path)
    writer.init_collection()
    writer.upsert_chunks([_chunk("a", "hello", datetime(2024, 5, 1))], [[0.1] * 8])

    reader = vs.VectorStore(vector_size=4, storage_path=path)  # other embedder at query time
    reader.init_collection()
    assert reader.client.count(reader.collection_name).count == 1


# D26: with no Gemini key, an unrelated provider key must not be picked up
def test_d26_offline_mode_ignores_other_provider_keys(monkeypatch):
    for name in ("GEMINI_API_KEY", "GEMINI_API_KEY_SECONDARY", "GOOGLE_API_KEY", "VOYAGE_API_KEY", "USE_FASTEMBED"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-unrelated")
    from archaeologist.retrieval.embedder import Embedder

    assert Embedder().model != "text-embedding-3-small"


# D27: a judge that errors must not count as a pass
def test_d27_verification_fails_closed(monkeypatch):
    import archaeologist.agent.nodes.verify as verify

    class BrokenJudge:
        def __init__(self, *args, **kwargs):
            pass

        def generate_text(self, *args, **kwargs):
            return "draft answer"

        def generate_json(self, *args, **kwargs):
            raise RuntimeError("429 RESOURCE_EXHAUSTED")

    monkeypatch.setattr(verify, "GeminiClientWrapper", BrokenJudge)
    monkeypatch.setattr(verify, "get_gemini_api_key", lambda: "key")
    out = verify.verify_node({
        "question": "q",
        "retrieved_chunks": [],
        "draft_answer": None,
        "verification_passed": False
    })
    assert out["verification_passed"] is False


# D28: revert subjects must be captured whole
@pytest.mark.parametrize("message, expected", [
    ('Revert "Don\'t crash on empty input"', "Don't crash on empty input"),
    ('Revert "Revert "Add retry logic""', 'Revert "Add retry logic"'),
])
def test_d28_revert_subject_is_parsed_whole(message, expected):
    from archaeologist.ingestion.revert_detector import detect_revert_from_message

    assert detect_revert_from_message(message) == expected


# D28: is_reverted marks the commit that was reverted, not the revert itself
def test_d28_is_reverted_marks_the_superseded_commit():
    from archaeologist.ingestion.chunker import chunk_commit

    base = {
        "author_name": "dev",
        "message": "change",
        "files_changed": [],
        "symbols_modified": [],
        "authored_date": datetime(2024, 5, 1)
    }
    reverted = chunk_commit({**base, "sha": "a" * 40, "is_revert": False, "superseded_by_sha": "b" * 40}, None, repo_id="r")[0]
    reverting = chunk_commit({**base, "sha": "b" * 40, "is_revert": True, "reverts_sha": "a" * 40}, None, repo_id="r")[0]
    assert reverted["is_reverted"] is True
    assert reverting["is_reverted"] is False


# D29: large documented classes are chunked on every supported Python
def test_d29_large_documented_class_is_chunked(tmp_path):
    from archaeologist.ingestion.chunker import chunk_codebase

    methods = "".join(
        f"    def method_{i}(self, value):\n"
        f"        return [value + {i} for _ in range({i})] if value else None\n"
        for i in range(80)
    )
    (tmp_path / "big.py").write_text('class Big:\n    """Docstring."""\n' + methods)
    chunks = chunk_codebase(str(tmp_path), repo_id="r")
    assert any("method_5" in c["symbols_modified"] for c in chunks)


# D31: an exact-match hit outranks ordinary hits wherever it sits in the list
def test_d31_fusion_uses_scores_not_list_position():
    from archaeologist.retrieval.fusion import reciprocal_rank_fusion

    def hit(cid, score):
        return {
            "id": cid,
            "score": score,
            "payload": {"id": cid, "source_type": "pr", "text": ""}
        }

    ordinary = [hit(f"o{i}", 0.5) for i in range(30)]
    fused = reciprocal_rank_fusion(ordinary + [hit("exact", 3.0)], [], limit=5)
    assert fused[0]["id"] == "exact"
