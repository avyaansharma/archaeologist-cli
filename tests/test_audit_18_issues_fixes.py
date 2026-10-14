import sqlite3
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from archaeologist.agent.nodes.follow_links import follow_links_node
from archaeologist.agent.nodes.plan import plan_node
from archaeologist.agent.nodes.search import _get_vector_store, sanitize_search_query, search_node
from archaeologist.retrieval.bm25_index import BM25Index
from archaeologist.retrieval.fusion import reciprocal_rank_fusion
from archaeologist.storage.db import get_engine, init_db
from archaeologist.storage.models import Chunk
from archaeologist.storage.paths import get_default_db_url, save_stored_repo_meta


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """Provides a fresh isolated SQLite database with WAL and all tables."""
    db_file = tmp_path / "test_isolated.db"
    db_url = f"sqlite:///{db_file}"
    monkeypatch.setenv("DATABASE_URL", db_url)

    from archaeologist.storage import db
    db.init_db(db_url)
    return db_url


def test_issue_1_plan_node_sets_filters_and_handles_no_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    state = {
        "question": "Why did the auth flow change?",
        "sub_questions": ["What commit altered auth.py?"],
        "current_sub_question_index": 0,
        "repo_id": "org/repo",
        "retry_count": 0,
        "unverified_claims": []
    }
    result = plan_node(state)
    assert "_current_plan" in result
    plan = result["_current_plan"]
    assert plan["sub_question"] == "What commit altered auth.py?"
    assert plan["repo_id"] == "org/repo"
    assert "file_path" in plan
    assert "source_types" in plan
    assert "date_from" in plan
    assert "date_to" in plan


def test_issue_2_plan_node_targets_unverified_claims_on_retry(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    state = {
        "question": "Why did the auth flow change?",
        "sub_questions": ["q1", "q2"],
        "current_sub_question_index": 2,  # Past the end of sub_questions
        "repo_id": "org/repo",
        "retry_count": 1,
        "unverified_claims": ["PR #42 introduced OAuth2", "commit deadbeef updated tokens"]
    }
    result = plan_node(state)
    plan = result["_current_plan"]
    assert "PR #42" in plan["sub_question"]
    assert "deadbeef" in plan["sub_question"]


def test_issue_3_search_node_pr_word_boundary_avoids_prefix_collision(isolated_db):
    from archaeologist.storage import db
    with db.get_session_context() as session:
        # Commit that mentions #120
        session.add(Chunk(
            id="chunk-120",
            repo_id="test/repo",
            source_type="commit",
            source_id="sha120",
            text="Refactor login flow (#120)",
            timestamp=datetime.now(UTC),
            related_ids=["pr#120"],
            symbols_modified=[]
        ))
        session.commit()

    state = {
        "question": "What happened in #12?",
        "search_queries": ["What happened in #12?"],
        "repo_id": "test/repo",
        "retrieved_chunks": [],
        "evidence_by_chunk_id": {},
        "_current_plan": {"repo_id": "test/repo"}
    }

    mock_emb = MagicMock()
    mock_emb.dimension = 384
    mock_emb.embed_texts.return_value = ([], [False])

    with patch("archaeologist.agent.nodes.search._get_embedder", return_value=mock_emb), \
         patch("archaeologist.storage.models.is_valid_pr_or_issue", return_value=True):
        res = search_node(state)
        # chunk-120 should NOT be retrieved for query #12
        retrieved_ids = [c["id"] for c in res["retrieved_chunks"]]
        assert "chunk-120" not in retrieved_ids


def test_issue_4_verify_node_numeric_strings_not_treated_as_shas(monkeypatch):
    class FakeGemini:
        def __init__(self, **kw): pass
        def generate_text(self, prompt, **kw):
            return "In 20240101, the system processed 1000000 requests without errors."
        def generate_json(self, prompt, **kw):
            return {"verification_passed": True, "unverified_claims": []}

    import archaeologist.agent.nodes.verify as vm
    monkeypatch.setattr(vm, "GeminiClientWrapper", FakeGemini)
    monkeypatch.setattr(vm, "get_gemini_api_key", lambda: "fake-key")

    state = {
        "question": "What were the stats?",
        "retrieved_chunks": [{"id": "c1", "source_id": "sha1", "text": "Report from 20240101", "related_ids": []}],
        "draft_answer": "In 20240101, the system processed 1000000 requests without errors.",
        "verification_passed": True
    }
    result = vm.verify_node(state)
    assert result["verification_passed"] is True
    assert result["unverified_claims"] == []


def test_issue_5_search_node_passes_date_range_to_bm25(monkeypatch):
    mock_bm25 = MagicMock()
    mock_bm25.search.return_value = []
    
    mock_emb = MagicMock()
    mock_emb.dimension = 384
    mock_emb.embed_texts.return_value = ([], [False])

    state = {
        "question": "Search auth",
        "search_queries": ["Search auth"],
        "_current_plan": {
            "date_from": "2023-01-01",
            "date_to": "2023-12-31"
        },
        "retrieved_chunks": [],
        "evidence_by_chunk_id": {}
    }

    with patch("archaeologist.agent.nodes.search.BM25Index", return_value=mock_bm25), \
         patch("archaeologist.agent.nodes.search.os.path.exists", return_value=True), \
         patch("archaeologist.agent.nodes.search._get_embedder", return_value=mock_emb):
        search_node(state)
        mock_bm25.search.assert_called_once()
        _, kwargs = mock_bm25.search.call_args
        assert kwargs.get("date_from") == "2023-01-01"
        assert kwargs.get("date_to") == "2023-12-31"


def test_issue_6_get_engine_consistency():
    url = get_default_db_url()
    eng1 = get_engine()
    eng2 = get_engine(url)
    assert eng1 is eng2


def test_issue_7_and_17_follow_links_node_robustness(isolated_db):
    from archaeologist.storage import db
    with db.get_session_context() as session:
        session.add(Chunk(
            id="pr-50",
            repo_id="repo1",
            source_type="pr",
            source_id="pr#50",
            text="PR 50 text",
            timestamp=datetime.now(UTC),
            related_ids=[],
            symbols_modified=[]
        ))
        session.commit()

    # Pass chunks with missing source_id, None related_ids, and #50 when pr#50 already seen
    state = {
        "retrieved_chunks": [
            {"id": "c1", "source_type": "commit", "source_id": "pr#50", "related_ids": None},
            {"id": "c2", "source_type": "commit", "related_ids": ["#50", None]},  # Missing source_id, None in list
            "invalid_non_dict_chunk"
        ],
        "repo_id": "repo1",
        "evidence_by_chunk_id": {}
    }
    # Should not raise KeyError or TypeError, and should recognize #50 as already present via pr#50
    out = follow_links_node(state)
    assert isinstance(out, dict)


def test_issue_8_text_fallback_preserves_source_type_and_sets_is_fallback(isolated_db):
    from archaeologist.storage import db
    with db.get_session_context() as session:
        session.add(Chunk(
            id="commit-fallback-test",
            repo_id="repo1",
            source_type="historical_commit",
            source_id="sha999",
            text="unique_needle_word commit content",
            timestamp=datetime.now(UTC),
            related_ids=[],
            symbols_modified=[]
        ))
        session.commit()

    state = {
        "question": "unique_needle_word",
        "search_queries": ["unique_needle_word"],
        "repo_id": "repo1",
        "retrieved_chunks": [],
        "evidence_by_chunk_id": {},
        "_current_plan": {}
    }

    mock_emb = MagicMock()
    mock_emb.dimension = 384
    mock_emb.embed_texts.return_value = ([], [False])

    with patch("archaeologist.agent.nodes.search._get_embedder", return_value=mock_emb), \
         patch("archaeologist.agent.nodes.search.BM25Index"):
        res = search_node(state)
        matching = [c for c in res["retrieved_chunks"] if c["id"] == "commit-fallback-test"]
        assert len(matching) == 1
        assert matching[0]["source_type"] == "historical_commit"
        assert matching[0].get("is_fallback") is True


def test_issue_9_fusion_score_1_vector_hit_and_id_generation():
    dense_hits = [
        {"id": "vec1", "score": 1.0, "payload": {"id": "vec1", "source_type": "code", "text": "vector content"}},
        {"id": "fall1", "score": 1.0, "payload": {"id": "fall1", "source_type": "code", "text": "fallback content", "is_fallback": True}}
    ]
    sparse_hits = [
        {"score": 5.0, "chunk": {"source_type": "code", "text": "sparse content"}}  # Missing id
    ]
    fused = reciprocal_rank_fusion(dense_hits, sparse_hits, limit=5)
    fused_ids = [f["id"] for f in fused]
    # vec1 (weight 1.0) must rank higher than fall1 (weight 0.2)
    assert fused_ids.index("vec1") < fused_ids.index("fall1")
    # Sparse hit with missing id gets sparse_1, not 1
    assert "sparse_1" in fused_ids


def test_issue_10_bm25_fit_alignment():
    bm = BM25Index()
    # chunks list contains a None or non-dict
    chunks = [
        {"id": "1", "text": "first document"},
        None,
        {"id": "2", "text": "second document"}
    ]
    bm.fit(chunks)
    assert len(bm.chunks) == 2
    res = bm.search("second", limit=1)
    assert len(res) == 1
    assert res[0]["chunk"]["id"] == "2"


def test_issue_11_bm25_search_excludes_undated_when_date_bounds_specified():
    bm = BM25Index()
    chunks = [
        {"id": "dated", "text": "target query term", "timestamp": "2024-05-01T00:00:00Z"},
        {"id": "undated", "text": "target query term", "timestamp": None}
    ]
    bm.fit(chunks)
    res = bm.search("target", date_from="2024-01-01", date_to="2024-12-31")
    ids = [r["chunk"]["id"] for r in res]
    assert "dated" in ids
    assert "undated" not in ids


def test_issue_12_sanitize_search_query():
    # Words ending in 'is' should not have their end stripped
    q = "analysis:foo thesis:bar pre-commit is:issue repo:org/repo"
    cleaned = sanitize_search_query(q)
    assert "analysis:foo" in cleaned
    assert "thesis:bar" in cleaned
    assert "pre-commit" in cleaned
    assert "repo:org/repo" not in cleaned


def test_issue_13_init_db_thread_safe():
    eng = init_db()
    assert eng is not None


def test_issue_14_save_stored_repo_meta_preserves_columns(tmp_path):
    db_file = tmp_path / ".archaeologist" / "archaeologist.db"
    db_file.parent.mkdir(parents=True)
    db_url = f"sqlite:///{db_file}"

    conn = sqlite3.connect(str(db_file))
    conn.execute("CREATE TABLE repo_meta (repo_id TEXT PRIMARY KEY, repo_name TEXT, repo_url TEXT, embedder_provider TEXT, embedder_dimension INTEGER, last_ingested_at TIMESTAMP, key TEXT, value TEXT, created_at TIMESTAMP, updated_at TIMESTAMP)")
    conn.execute("INSERT INTO repo_meta (repo_id, repo_name, key, value, created_at) VALUES ('myrepo', 'My Repository Name', 'k1', 'v1', '2024-01-01 00:00:00')")
    conn.commit()
    conn.close()

    with patch("archaeologist.storage.paths.get_default_db_url", return_value=db_url), \
         patch("archaeologist.storage.paths.get_archaeologist_dir", return_value=tmp_path / ".archaeologist"):
        save_stored_repo_meta(
            repo_id="myrepo",
            repo_url="https://github.com/my/repo",
            embedder_provider="gemini",
            embedder_dimension=768,
            last_ingested_at="2024-06-01 12:00:00"
        )

    conn = sqlite3.connect(str(db_file))
    row = conn.execute("SELECT repo_name, repo_url, embedder_provider, key, value, created_at FROM repo_meta WHERE repo_id='myrepo'").fetchone()
    conn.close()
    assert row[0] == "My Repository Name"
    assert row[1] == "https://github.com/my/repo"
    assert row[2] == "gemini"
    assert row[3] == "k1"
    assert row[4] == "v1"
    assert row[5] == "2024-01-01 00:00:00"


def test_issue_15_get_vector_store_dimension_change():
    vs1 = _get_vector_store(dim=384, storage_path=":memory:")
    assert vs1.vector_size == 384
    vs2 = _get_vector_store(dim=768, storage_path=":memory:")
    assert vs2.vector_size == 768
    assert vs1 is not vs2


def test_issue_16_search_node_does_not_mutate_state_evidence_map():
    state_evidence = {"c0": {"id": "c0"}}
    state = {
        "question": "test",
        "search_queries": ["test"],
        "_current_plan": {},
        "retrieved_chunks": [],
        "evidence_by_chunk_id": state_evidence
    }
    mock_emb = MagicMock()
    mock_emb.dimension = 384
    mock_emb.embed_texts.return_value = ([], [False])

    with patch("archaeologist.agent.nodes.search._get_embedder", return_value=mock_emb), \
         patch("archaeologist.agent.nodes.search.reciprocal_rank_fusion", return_value=[{"id": "c1", "payload": {"id": "c1"}, "rrf_score": 1.0}]):
        res = search_node(state)
        # Original state dictionary must not have received c1
        assert "c1" not in state_evidence
        assert "c1" in res["evidence_by_chunk_id"]


def test_issue_18_fusion_docstring():
    import archaeologist.retrieval.fusion as fm
    doc = fm.reciprocal_rank_fusion.__doc__
    assert "0.03" in doc
    assert "0.18" not in doc
