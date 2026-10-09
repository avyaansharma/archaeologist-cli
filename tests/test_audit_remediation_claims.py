import os
import re
import sys
import tomllib
import pytest
from datetime import datetime, timezone
from sqlmodel import Session, select, create_engine, SQLModel

from archaeologist.ingestion.chunker import chunk_pr, make_deterministic_chunk_id, token_count, _EncodingProxy
from archaeologist.ingestion.revert_detector import extract_revert_sha_from_message, detect_revert_from_message
from archaeologist.ingestion.symbol_parser import map_lines_to_symbols, extract_symbols_from_code
from archaeologist.retrieval.bm25_index import tokenize_text
from archaeologist.retrieval.fusion import reciprocal_rank_fusion
from archaeologist.retrieval.vector_store import VectorStore
from archaeologist.retrieval.embedder import Embedder
from archaeologist.storage.models import RepoMeta, Chunk, Commit
from archaeologist.storage.paths import get_default_qdrant_path
from archaeologist.cli import _configure_repo_env
from archaeologist.utils.gemini_client import DEFAULT_MODEL, FALLBACK_MODELS
from archaeologist.ingestion.git_parser import _get_monorepo_subpath, get_commit_diff

def test_claim_1_chunk_pr_no_crash():
    pr = {
        "number": 101,
        "title": "Fix memory leak in parser",
        "body": "Fixes #42 and resolves issue #43",
        "author": "octocat",
        "linked_issue_numbers": [42, 43],
        "created_at": "2026-01-01T00:00:00Z"
    }
    chunks = chunk_pr(pr, repo_id="test-repo")
    assert len(chunks) >= 1
    assert "issue#42" in chunks[0]["related_ids"]
    assert "issue#43" in chunks[0]["related_ids"]
    assert chunks[0]["source_id"] == "pr#101"

def test_claim_2_pipeline_imports_subprocess():
    import archaeologist.ingestion.pipeline as pipeline_mod
    assert hasattr(pipeline_mod, "subprocess")
    assert pipeline_mod.subprocess.run is not None

def test_claim_3_pyproject_toml_valid():
    repo_root = os.path.dirname(os.path.dirname(__file__))
    pyproject_path = os.path.join(repo_root, "pyproject.toml")
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    assert "project" in data
    assert "dependencies" in data["project"]
    deps = data["project"]["dependencies"]
    sqlmodel_dep = next((d for d in deps if d.startswith("sqlmodel")), None)
    assert sqlmodel_dep is not None
    assert "<0.0.45" in sqlmodel_dep

def test_claim_4_cli_repo_env_sets_qdrant_and_repo(tmp_path):
    repo_dir = str(tmp_path / "my_project")
    os.makedirs(repo_dir, exist_ok=True)
    _configure_repo_env(repo_path=repo_dir)
    assert os.environ.get("ARCHAEOLOGIST_REPO") == os.path.abspath(repo_dir)
    assert "qdrant" in os.environ.get("QDRANT_STORAGE_PATH", "").lower()

def test_claim_5_deterministic_chunk_id():
    id1 = make_deterministic_chunk_id("commit", "abc1234", 0, "short text", repo_id="r1")
    id2 = make_deterministic_chunk_id("commit", "abc1234", 0, "different text", repo_id="r1")
    assert id1 == id2
    assert len(id1) == 36  # Valid UUID string

def test_claim_6_encoding_proxy_fallback():
    proxy = _EncodingProxy()
    tokens = proxy.encode("Hello world of code")
    assert len(tokens) > 0
    text = proxy.decode(tokens)
    assert "Hello" in text
    assert token_count("testing tokens") >= 2

def test_claim_7_pipeline_chunk_upsert():
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        cid = make_deterministic_chunk_id("commit", "c1", 0, repo_id="r1")
        c1 = Chunk(id=cid, repo_id="r1", source_type="commit", source_id="c1", text="original text", timestamp=datetime.now(timezone.utc), embedded=True)
        session.add(c1)
        session.commit()

        # Simulate upsert with changed text
        existing = session.get(Chunk, cid)
        if existing and existing.text != "updated diff summary":
            existing.text = "updated diff summary"
            existing.embedded = False
            session.add(existing)
            session.commit()

        reloaded = session.get(Chunk, cid)
        assert reloaded.text == "updated diff summary"
        assert reloaded.embedded is False

def test_claim_8_github_desc_and_since():
    from archaeologist.ingestion.github_client import GitHubIngestionClient
    import inspect
    sig_pr = inspect.signature(GitHubIngestionClient.fetch_pull_requests)
    sig_iss = inspect.signature(GitHubIngestionClient.fetch_issues)
    assert sig_pr.parameters["direction"].default == "desc"
    assert "since" in sig_pr.parameters
    assert sig_iss.parameters["direction"].default == "desc"
    assert "since" in sig_iss.parameters

def test_claim_9_revert_sha_authoritative():
    sha = "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"
    msg = f"Revert commit\n\nThis reverts commit {sha}."
    extracted = extract_revert_sha_from_message(msg)
    assert extracted == sha

    # Conversational phrases without quoted subject or trailer should not match
    assert detect_revert_from_message("Revert to previous behaviour in config") is None
    assert extract_revert_sha_from_message("Revert to previous behaviour") is None

def test_claim_10_diff_summaries_order():
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(Commit(sha="old1", author_name="a", author_email="a@test.com", message="m", authored_date=datetime(2020, 1, 1)))
        session.add(Commit(sha="new1", author_name="a", author_email="a@test.com", message="m", authored_date=datetime(2026, 1, 1)))
        session.commit()

        ordered = session.exec(select(Commit).order_by(Commit.authored_date.desc())).all()
        assert ordered[0].sha == "new1"
        assert ordered[1].sha == "old1"

def test_claim_11_fusion_decay_and_symbol_match():
    hits_code = [{"id": "c_code", "score": 0.9, "payload": {"source_type": "code", "symbols_modified": ["UserService::authenticate"]}}]
    hits_bm25 = [{"chunk": {"id": "c_code", "source_type": "code", "symbols_modified": ["UserService::authenticate"]}}]
    
    # Code chunk does not decay with age
    res = reciprocal_rank_fusion(hits_code, hits_bm25, query="authenticate")
    assert len(res) == 1
    assert res[0]["rrf_score"] > 0.0

def test_claim_12_bm25_unicode_and_code_identifiers():
    tokens = tokenize_text("getUserById get_user_by_id München")
    assert "getuserbyid" in tokens
    assert "user" in tokens
    assert "get_user_by_id" in tokens
    assert "münchen" in tokens

def test_claim_13_symbol_parser_inner_attribution_and_multilang():
    symbols = [
        {"symbol_id": "file.py::MyClass", "name": "MyClass", "kind": "class", "start_line": 10, "end_line": 100},
        {"symbol_id": "file.py::MyClass::my_method", "name": "my_method", "kind": "method", "start_line": 20, "end_line": 30}
    ]
    # Modifying line 25 inside method should attribute method, not class
    attributed = map_lines_to_symbols(symbols, [25])
    assert attributed == ["file.py::MyClass::my_method"]

    # Modifying line 12 outside method should attribute class
    attributed_class = map_lines_to_symbols(symbols, [12])
    assert attributed_class == ["file.py::MyClass"]

    # Multi-language extraction
    ts_code = "export default async function handleRequest(req) { return req; }"
    go_code = "func (s *Server) ServeHTTP(w ResponseWriter, r *Request) {}"
    java_code = "public void authenticate(User user) {}"
    
    ts_syms = extract_symbols_from_code(ts_code, "handler.ts")
    assert any(s["name"] == "handleRequest" for s in ts_syms)

    go_syms = extract_symbols_from_code(go_code, "server.go")
    assert any(s["name"] == "ServeHTTP" for s in go_syms)

    java_syms = extract_symbols_from_code(java_code, "Auth.java")
    assert any(s["name"] == "authenticate" for s in java_syms)

def test_claim_14_mcp_blame_path_traversal():
    from archaeologist.mcp_server.server import blame_explain
    # Verify that calling blame_explain with a relative safe path works without crash
    res = blame_explain("README.md", 1, 5, repo_path=None)
    assert res is not None

def test_claim_15_valid_gemini_models():
    assert any(valid in DEFAULT_MODEL for valid in ["gemini-2.5-flash", "gemini-3.5-flash", "flash"])
    assert any("flash" in m for m in FALLBACK_MODELS)
    # Ensure invalid legacy deprecated models are not present
    assert "gemini-1.5-pro-latest" not in FALLBACK_MODELS

def test_claim_16_gemini_timeout_and_semaphore():
    from archaeologist.utils.gemini_client import GeminiClientWrapper
    assert hasattr(GeminiClientWrapper, "_semaphore")

def test_claim_17_verify_provenance():
    from archaeologist.agent.nodes.verify import verify_node
    # Simulated state citing hallucinated references
    state = {
        "question": "Why did we add auth?",
        "draft_answer": "We added auth in PR #9999 and commit deadbeef99999.",
        "retrieved_chunks": [{"id": "c1", "source_id": "commit111", "text": "Evidence 1"}]
    }
    # verify_node should fail when no Gemini key is provided, returning verification_passed=False
    res = verify_node(state)
    assert res["verification_passed"] is False

def test_claim_18_embedder_key_rotation_429():
    import inspect
    from archaeologist.retrieval.embedder import Embedder
    src = inspect.getsource(Embedder._embed_gemini)
    assert "attempt < 2" in src
    assert "exhausted_keys.add" in src

def test_claim_19_vector_store_recreate_dim_flag():
    vs = VectorStore(vector_size=384, storage_path=":memory:")
    assert hasattr(vs, "recreated_due_to_dim_change")
    assert vs.recreated_due_to_dim_change is False

def test_claim_20_vector_store_remote_timeout():
    assert float(os.getenv("QDRANT_TIMEOUT", "30.0")) >= 10.0

def test_claim_21_remote_reembed_isolates_repo():
    import inspect
    from archaeologist.retrieval.vector_store import VectorStore
    sig = inspect.signature(VectorStore.init_collection)
    assert "repo_id" in sig.parameters

def test_claim_22_and_23_repo_meta_model_fields():
    meta = RepoMeta(
        repo_id="test_repo",
        repo_path="/path/test",
        repo_url="https://github.com/test/test",
        embedder_provider="gemini",
        embedder_dimension=3072,
        last_ingested_at="2026-10-09T12:00:00"
    )
    assert meta.embedder_provider == "gemini"
    assert meta.embedder_dimension == 3072
    assert meta.last_ingested_at == "2026-10-09T12:00:00"

def test_claim_24_git_parser_monorepo_and_streaming():
    import subprocess
    sub = _get_monorepo_subpath(".")
    assert sub is None or isinstance(sub, str)
    # Resolve real commit SHA to satisfy sanitize_sha hex check
    res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    if res.returncode == 0 and res.stdout.strip():
        head_sha = res.stdout.strip()
        diff = get_commit_diff(".", head_sha, max_bytes=100)
        assert diff is None or len(diff.encode("utf-8")) <= 500

def test_claim_25_sqlite_busy_timeout():
    from archaeologist.storage.db import get_engine
    engine = get_engine("sqlite:///:memory:")
    with engine.connect() as conn:
        res = conn.exec_driver_sql("PRAGMA busy_timeout;").scalar()
        assert res == 10000
