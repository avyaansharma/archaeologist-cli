import os
import sys
import time
import tempfile
import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from archaeologist.retrieval.vector_store import VectorStore
from archaeologist.retrieval.bm25_index import BM25Index, _BM25_CACHE
from archaeologist.agent.graph import increment_retry_node, agent_graph
from archaeologist.agent.state import AgentState
from archaeologist.storage.db import init_db, get_session_context
from archaeologist.storage.models import Commit, Chunk


def test_reproduce_qdrant_local_concurrent_access_lock_collision(tmp_path):
    """Critical Bug 1: Local Qdrant fails with RuntimeError when multiple VectorStore
    instances are created for the same directory or when MCP tools / threads run concurrently.
    A thread-safe client cache / singleton pool must be used to allow concurrent operations.
    """
    storage_dir = tmp_path / "qdrant_test_dir"
    os.environ["QDRANT_STORAGE_PATH"] = str(storage_dir)
    os.environ["QDRANT_URL"] = "http://127.0.0.1:9"

    vs1 = VectorStore(vector_size=8)
    vs1.init_collection()

    # In buggy code, vs2 will crash with RuntimeError because vs1 holds the portalocker file lock
    vs2 = VectorStore(vector_size=8)
    vs2.init_collection()

    # Both instances should be operational and able to search/upsert without crashing
    res = vs2.search_chunks([0.1] * 8, limit=2)
    assert isinstance(res, list)

    vs1.close()
    vs2.close()


def test_reproduce_bm25_cache_stale_data_on_disk_update(tmp_path):
    """Critical Bug 2: BM25Index.load() caches loaded objects indefinitely in _BM25_CACHE
    without checking file modification time (mtime). If ingestion updates the index on disk,
    warm processes will serve stale search results.
    """
    idx_path = str(tmp_path / "bm25_test.bin")
    
    # 1. Fit index with initial multi-doc corpus so BM25 IDF scores are strictly positive
    idx1 = BM25Index()
    idx1.fit([
        {"id": "c1", "text": "alpha special keyword feature in database", "repo_id": "repo1", "file_paths": []},
        {"id": "c_other", "text": "general other text without keyword", "repo_id": "repo1", "file_paths": []},
        {"id": "c_extra", "text": "more filler sentences for corpus", "repo_id": "repo1", "file_paths": []},
    ])
    idx1.save(idx_path)

    # 2. Warm up cache with loader
    loader1 = BM25Index()
    loader1.load(idx_path)
    res1 = loader1.search("alpha")
    assert len(res1) == 1
    assert res1[0]["chunk"]["id"] == "c1"

    # 3. Simulate pipeline updating the index on disk with new data
    time.sleep(0.05)  # Ensure mtime increments
    idx2 = BM25Index()
    idx2.fit([
        {"id": "c2", "text": "beta updated feature replacing everything", "repo_id": "repo1", "file_paths": []},
        {"id": "c_other", "text": "general other text without keyword", "repo_id": "repo1", "file_paths": []},
        {"id": "c_extra", "text": "more filler sentences for corpus", "repo_id": "repo1", "file_paths": []},
    ])
    # Save directly to disk (simulating external or pipeline write)
    import pickle
    with open(idx_path, "wb") as f:
        pickle.dump({"bm25": idx2.bm25, "chunks": idx2.chunks}, f)

    # Loader2 should automatically detect that mtime changed!
    loader2 = BM25Index()
    loader2.load(idx_path)
    res2 = loader2.search("beta")
    
    assert len(res2) == 1, "Stale cache bug: BM25Index.load() returned stale data because it did not check file mtime!"
    assert res2[0]["chunk"]["id"] == "c2"


def test_reproduce_agent_retry_resets_completed_sub_questions():
    """Critical Bug 3: increment_retry_node hard-resets current_sub_question_index = 0,
    discarding all previous sub-question retrieval and plans, causing repetitive LLM calls,
    429 quota exhaustion, and latency explosion.
    """
    state = {
        "question": "What is the architecture?",
        "sub_questions": [
            "What is component A?",
            "What is component B?",
            "What is component C?"
        ],
        "current_sub_question_index": 2,  # Sub-question 0 and 1 were already retrieved
        "retrieved_chunks": [{"id": "chunk_a"}, {"id": "chunk_b"}],
        "verification_passed": False,
        "unverified_claims": ["Component C uses Redis."],
        "retry_count": 0
    }

    new_state = increment_retry_node(state)
    
    # In buggy code, current_sub_question_index is reset to 0
    # In hardened code, previously completed subquestions must NOT be re-executed from scratch!
    assert new_state["retry_count"] == 1
    assert new_state["current_sub_question_index"] == 2, (
        "Sub-question reset bug: increment_retry_node reset current_sub_question_index to 0! "
        "This discards all completed sub-questions and restarts reasoning from scratch."
    )


def test_reproduce_pipeline_unbatched_sqlite_lock_contention(tmp_path, monkeypatch):
    """Critical Bug 4: IngestionPipeline wraps all commit parsing in a single uncommitted
    SQLite transaction, locking SQLite for the entire duration of git diff and AST parsing.
    It must commit in periodic batches to release locks and allow concurrent reads.
    """
    from archaeologist.ingestion.pipeline import IngestionPipeline
    
    db_file = tmp_path / "test_pipeline_batch.db"
    db_url = f"sqlite:///{db_file.as_posix()}"
    init_db(db_url)
    monkeypatch.setenv("DATABASE_URL", db_url)

    # Mock git commits (150 commits)
    mock_commits = []
    now = datetime.now(timezone.utc)
    for i in range(150):
        mock_commits.append({
            "sha": f"{i:040x}",
            "author_name": "Dev",
            "author_email": "dev@test.com",
            "authored_date": now,
            "message": f"Commit {i}",
            "files_changed": ["main.py"],
            "insertions": 1,
            "deletions": 0
        })

    pipeline = IngestionPipeline(repo_path=str(tmp_path), repo_id="test_repo")
    
    commit_calls = []
    real_get_session_context = get_session_context
    
    from contextlib import contextmanager
    @contextmanager
    def spy_session():
        with real_get_session_context() as sess:
            real_commit = sess.commit
            def spy_commit(*args, **kwargs):
                commit_calls.append(len(commit_calls) + 1)
                return real_commit(*args, **kwargs)
            sess.commit = spy_commit
            yield sess

    monkeypatch.setattr("archaeologist.ingestion.pipeline.get_session_context", spy_session)
    monkeypatch.setattr("archaeologist.ingestion.pipeline.iter_commits", lambda *args, **kwargs: mock_commits)
    monkeypatch.setattr("archaeologist.ingestion.pipeline.count_commits", lambda *args, **kwargs: 150)
    monkeypatch.setattr("archaeologist.ingestion.pipeline.get_commit_diff", lambda *args, **kwargs: "")
    # Stop pipeline after commit phase to test batching
    monkeypatch.setattr("archaeologist.ingestion.pipeline.IngestionPipeline._report_progress", lambda *args, **kwargs: None)
    
    # Just run the commit ingestion logic
    with get_session_context() as session:
        pass  # Just ensure DB is warm

    # Run the ingestion pipeline (mocking other steps)
    mock_emb = MagicMock()
    mock_emb.embed_texts.return_value = ([], [])
    monkeypatch.setattr("archaeologist.ingestion.pipeline.VectorStore", MagicMock())
    monkeypatch.setattr("archaeologist.ingestion.pipeline.Embedder", lambda *args, **kwargs: mock_emb)
    monkeypatch.setattr("archaeologist.ingestion.pipeline.BM25Index", MagicMock())

    pipeline.run()

    assert len(commit_calls) >= 2, (
        f"Unbatched transaction bug: Expected periodic commits (at least 2 batches for 150 commits), "
        f"but got {len(commit_calls)} commit calls! SQLite database was locked for the entire ingestion."
    )


def test_reproduce_qdrant_cross_process_lock_fallback(tmp_path, monkeypatch):
    """Critical Bug: When another OS process holds the directory lock on local Qdrant,
    QdrantClient raises RuntimeError. VectorStore must catch this and gracefully fall back
    to an in-memory client rather than crashing the command.
    """
    storage_dir = tmp_path / "locked_qdrant"
    os.environ["QDRANT_STORAGE_PATH"] = str(storage_dir)
    os.environ["QDRANT_URL"] = "http://127.0.0.1:9"

    from qdrant_client import QdrantClient
    orig_qdrant_init = QdrantClient.__init__

    def mock_qdrant_init(self, *args, **kwargs):
        if kwargs.get("path") or (args and args[0] != ":memory:"):
            raise RuntimeError(f"Storage folder {storage_dir} is already accessed by another instance of Qdrant client.")
        return orig_qdrant_init(self, *args, **kwargs)

    monkeypatch.setattr(QdrantClient, "__init__", mock_qdrant_init)

    # Must not raise RuntimeError
    vs = VectorStore(vector_size=8, storage_path=str(storage_dir))
    vs.init_collection()

    res = vs.search_chunks([0.1] * 8, limit=2)
    assert isinstance(res, list)
    vs.close()

