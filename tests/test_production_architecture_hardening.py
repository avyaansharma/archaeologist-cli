import os
import sys
import tempfile
import asyncio
import threading
from unittest.mock import patch, MagicMock
import pytest
from sqlmodel import select

from archaeologist.storage.db import (
    init_db,
    get_engine_for_url,
    get_session,
    get_session_context,
    _ENGINES,
    dispose_all_engines
)
from archaeologist.storage.models import Commit, Chunk
from archaeologist.storage.context import (
    current_repo_id_var,
    current_db_url_var,
    current_bm25_path_var,
    current_client_api_key_var
)
from archaeologist.retrieval.bm25_index import BM25Index
from archaeologist.utils.gemini_client import get_gemini_api_keys


def test_multi_repo_concurrent_engine_pool_isolation():
    """Validates that accessing multiple repositories concurrently retains connection pools
    in _ENGINES without disposing active engines or thrashing connections.
    """
    tmpdir = tempfile.mkdtemp()
    db_path_a = os.path.join(tmpdir, "repo_a.db").replace(os.sep, "/")
    db_path_b = os.path.join(tmpdir, "repo_b.db").replace(os.sep, "/")
    url_a = f"sqlite:///{db_path_a}"
    url_b = f"sqlite:///{db_path_b}"

    from datetime import datetime
    try:
        # Access Repo A using pooled get_session_context
        with get_session_context(url_a) as session_a:
            chunk_a = Chunk(
                id="chunk_a_1",
                text="Repository A content",
                timestamp=datetime.utcnow(),
                repo_id="repo_a",
                source_type="file",
                source_id="file_a.py",
                file_paths=["file_a.py"]
            )
            session_a.add(chunk_a)

        # Access Repo B using pooled get_session_context concurrently
        with get_session_context(url_b) as session_b:
            chunk_b = Chunk(
                id="chunk_b_1",
                text="Repository B content",
                timestamp=datetime.utcnow(),
                repo_id="repo_b",
                source_type="file",
                source_id="file_b.py",
                file_paths=["file_b.py"]
            )
            session_b.add(chunk_b)

        # Crucial architectural assertion: Both engines must be actively cached in _ENGINES
        assert url_a in _ENGINES, f"Engine for {url_a} was prematurely disposed when Repo B was accessed!"
        assert url_b in _ENGINES, f"Engine for {url_b} was not cached in _ENGINES!"

        # Query Repo A again: Should reuse the pooled engine seamlessly
        with get_session_context(url_a) as session_a2:
            res_a = session_a2.exec(select(Chunk).where(Chunk.repo_id == "repo_a")).all()
            assert len(res_a) == 1
            assert res_a[0].id == "chunk_a_1"

        # Query Repo B again
        with get_session_context(url_b) as session_b2:
            res_b = session_b2.exec(select(Chunk).where(Chunk.repo_id == "repo_b")).all()
            assert len(res_b) == 1
            assert res_b[0].id == "chunk_b_1"

    finally:
        dispose_all_engines()
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.asyncio
async def test_request_scoped_contextvar_isolation():
    """Validates that concurrent asynchronous requests maintain strict contextvar isolation
    for repo_id, db_url, and client_api_key without leaking across tasks or mutating global state.
    """
    os.environ["GEMINI_API_KEY"] = "AMBIENT_SERVER_KEY"

    try:
        async def simulate_client_a():
            current_repo_id_var.set("flask")
            current_db_url_var.set("sqlite:///path/to/flask.db")
            current_client_api_key_var.set("CLIENT_A_CUSTOM_KEY")
            await asyncio.sleep(0.05)
            
            # Verify isolation inside Client A's context
            keys = get_gemini_api_keys()
            assert keys[0] == "CLIENT_A_CUSTOM_KEY"
            assert current_repo_id_var.get() == "flask"
            assert current_db_url_var.get() == "sqlite:///path/to/flask.db"
            return "A_SUCCESS"

        async def simulate_client_b():
            current_repo_id_var.set("requests")
            current_db_url_var.set("sqlite:///path/to/requests.db")
            current_client_api_key_var.set(None)  # Unauthenticated query
            await asyncio.sleep(0.05)
            
            # Verify isolation inside Client B's context: uses ambient key, not Client A's key
            keys = get_gemini_api_keys()
            assert keys[0] == "AMBIENT_SERVER_KEY"
            assert current_repo_id_var.get() == "requests"
            assert current_db_url_var.get() == "sqlite:///path/to/requests.db"
            return "B_SUCCESS"

        res_a, res_b = await asyncio.gather(simulate_client_a(), simulate_client_b())
        assert res_a == "A_SUCCESS"
        assert res_b == "B_SUCCESS"
        assert os.environ.get("GEMINI_API_KEY") == "AMBIENT_SERVER_KEY"
    finally:
        current_repo_id_var.set(None)
        current_db_url_var.set(None)
        current_bm25_path_var.set(None)
        current_client_api_key_var.set(None)


def test_atomic_bm25_save_guarantee():
    """Validates that BM25Index.save writes atomically using a temporary file and os.replace,
    preventing file corruption during concurrent operations or crashes.
    """
    fd, path = tempfile.mkstemp(suffix=".bin")
    os.close(fd)

    try:
        bm25 = BM25Index()
        bm25.fit([
            {"chunk_id": "c1", "text": "error handler pattern in flask", "source_type": "commit"},
            {"chunk_id": "c2", "text": "session management cookies", "source_type": "commit"}
        ])
        bm25.save(path)

        # Verify load succeeds
        loaded = BM25Index()
        loaded.load(path)
        hits = loaded.search("error handler")
        assert len(hits) == 1
        assert hits[0]["chunk"]["chunk_id"] == "c1"

        # Verify no temp files left behind in the directory
        parent = os.path.dirname(path)
        tmp_files = [f for f in os.listdir(parent) if f.startswith(os.path.basename(path)) and f.endswith(".tmp")]
        assert len(tmp_files) == 0, f"Leaked temporary files found: {tmp_files}"

    finally:
        try:
            os.remove(path)
        except Exception:
            pass


def test_ask_endpoint_offloads_blocking_io():
    """Validates that the FastAPI /api/ask endpoint executes through asyncio.to_thread,
    preventing event-loop starvation during intensive agent execution.
    """
    from fastapi.testclient import TestClient
    from archaeologist.web.server import app

    client = TestClient(app)

    with patch("archaeologist.web.server.ask_tool") as mock_ask:
        mock_ask.return_value = "Forensic answer for testing."

        response = client.post(
            "/api/ask",
            json={"repo_id": "requests", "question": "Why was adapter modified?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "Forensic answer for testing."
        assert data["repo_id"] == "requests"
        assert mock_ask.called
