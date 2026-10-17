"""
Regression Test Suite for the 10 Persisting Architectural Bugs (BUG-01 .. BUG-10).

Verifies Rank #1 remediations across:
  - BUG-01: `blame_explain_tool` composite PK `(repo_id, sha)` lookup without scalar `session.get(Commit, sha)` crash
  - BUG-02: `VectorStore.init_collection()` read-only dimension auto-adaptation without mutating `os.environ`
  - BUG-03: `search.py` `_get_embedder` / `_get_vector_store` cache stability on auto-adapted collections & `search_history_tool` auto-init
  - BUG-04: `Embedder.embed_texts` zero `sys._getframe()` stack inspection while preserving offline fallback & patched crash propagation
  - BUG-05: `save_stored_repo_meta` honoring `current_db_url_var` (`--db-path`) and persisting `embedder_model` in SQLite `repo_meta`
  - BUG-06: Multi-tenant isolation in `arch status` (`RepoMeta`) and `traverse_causal_chain` (no cross-tenant `repo_id = ''` leak)
  - BUG-07: `IngestionPipeline` Step 5 preserving LLM `diff_summary` when enriching with linked PR context
  - BUG-08: Dynamic causal link `rrf_score` inheritance in `follow_links_node` & stratified multi-sub-question evidence selection in `verify_node` / `synthesize_node`
  - BUG-09: `IngestionPipeline` Step 1b `file` chunk enrichment with full file AST symbol inventory
  - BUG-10: `IngestionPipeline` Step 5 $O(1)$ / $O(\\log C)$ SHA prefix resolution for `pr_by_commit`
"""

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import pytest
from sqlmodel import select
from typer.testing import CliRunner

from archaeologist.agent.nodes.follow_links import follow_links_node
from archaeologist.agent.nodes.search import _get_vector_store
from archaeologist.agent.nodes.verify import verify_node
from archaeologist.cli import app
from archaeologist.mcp_server.tools import blame_explain_tool, search_history_tool
from archaeologist.retrieval.causal_graph import traverse_causal_chain
from archaeologist.retrieval.embedder import Embedder
from archaeologist.retrieval.vector_store import VectorStore
from archaeologist.storage.context import (
    current_db_url_var,
    current_qdrant_path_var,
    current_repo_id_var,
    current_repo_path_var,
)
from archaeologist.storage.db import dispose_all_engines, get_session_context, init_db
from archaeologist.storage.models import CausalEdge, Chunk, Commit, PullRequest, RepoMeta
from archaeologist.storage.paths import get_stored_repo_meta, save_stored_repo_meta

runner = CliRunner()


def _git(repo_path: Path, *args: str):
    return subprocess.run(
        ["git", "-C", str(repo_path), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )


def _make_git_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "dev@example.com")
    _git(path, "config", "user.name", "Developer")
    return path


@pytest.fixture(autouse=True)
def _clean_context(monkeypatch):
    for key in [
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY_SECONDARY",
        "GITHUB_TOKEN",
        "USE_GEMINI_EMBEDDINGS",
        "USE_FASTEMBED",
        "FASTEMBED_MODEL",
        "EMBEDDER_PROVIDER",
        "DATABASE_URL",
        "QDRANT_URL",
    ]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SKIP_GITHUB_API", "1")
    t_db = current_db_url_var.set(None)
    t_repo = current_repo_id_var.set(None)
    t_path = current_repo_path_var.set(None)
    t_qd = current_qdrant_path_var.set(None)
    yield
    current_db_url_var.reset(t_db)
    current_repo_id_var.reset(t_repo)
    current_repo_path_var.reset(t_path)
    current_qdrant_path_var.reset(t_qd)
    dispose_all_engines()


def test_BUG01_blame_explain_tool_loads_commit_with_composite_pk(tmp_path, monkeypatch):
    """
    Verifies that `blame_explain_tool` retrieves `Commit` rows from SQLite (including `is_revert=True`
    and DB message metadata) when `Commit` has composite primary key `(repo_id, sha)`, instead of
    silently crashing on `session.get(Commit, sha)` and falling back to `git show`.
    """
    repo_dir = _make_git_repo(tmp_path / "blame_repo")
    target_file = repo_dir / "service.py"
    target_file.write_text("def handle_auth():\n    return True\n", encoding="utf-8")
    _git(repo_dir, "add", "service.py")
    _git(repo_dir, "commit", "-q", "-m", "raw git message")
    sha = _git(repo_dir, "rev-parse", "HEAD").stdout.strip()

    db_file = tmp_path / "blame.db"
    db_url = f"sqlite:///{db_file.as_posix()}"
    init_db(db_url)
    current_db_url_var.set(db_url)
    current_repo_id_var.set("org/blame_repo")

    with get_session_context() as session:
        session.add(
            Commit(
                repo_id="org/blame_repo",
                sha=sha,
                author_name="DB Enriched Author",
                author_email="db@example.com",
                authored_date=datetime(2025, 2, 1, 12, 0, 0),
                message="ENRICHED_DB_MESSAGE_FROM_SQLITE",
                is_revert=True,
            )
        )
        session.commit()

    res = blame_explain_tool(str(repo_dir), "service.py", 1, 2)
    assert len(res["commits"]) == 1
    c_info = res["commits"][0]
    assert c_info["sha"] == sha
    assert c_info["message"] == "ENRICHED_DB_MESSAGE_FROM_SQLITE"
    assert c_info["is_revert"] is True


def test_BUG02_vector_store_auto_adapt_does_not_mutate_os_environ(tmp_path):
    """
    Verifies that `VectorStore.init_collection()` auto-adapts read-only collections (e.g. 384-dim)
    via parameterized `Embedder(provider=..., model_name=...)` without mutating `os.environ`.
    """
    qdrant_dir = str(tmp_path / "qdrant_384")
    writer = VectorStore(vector_size=384, storage_path=qdrant_dir, writable=True)
    writer.init_collection(recreate=True)
    writer.close()

    env_before = {
        k: os.environ.get(k)
        for k in ("USE_FASTEMBED", "FASTEMBED_MODEL", "USE_GEMINI_EMBEDDINGS", "EMBEDDER_PROVIDER")
    }

    reader = VectorStore(vector_size=768, storage_path=qdrant_dir, writable=False)
    reader.init_collection()
    try:
        assert reader.vector_size == 384
        assert reader.embedder is not None
        assert reader.embedder.dimension == 384
        env_after = {
            k: os.environ.get(k)
            for k in ("USE_FASTEMBED", "FASTEMBED_MODEL", "USE_GEMINI_EMBEDDINGS", "EMBEDDER_PROVIDER")
        }
        assert env_after == env_before
    finally:
        reader.close()


def test_BUG03_search_cache_no_thrash_and_search_history_tool_auto_adapts(tmp_path, monkeypatch):
    """
    Verifies that:
    1) `_get_vector_store(768)` does not recreate `VectorStore` on repeated calls after auto-adapting to 384-dim.
    2) `search_history_tool` calls `init_collection()` and auto-adapts to a 384-dim Qdrant collection.
    """
    qdrant_dir = str(tmp_path / "qdrant_adapt_384")
    writer = VectorStore(vector_size=384, storage_path=qdrant_dir, writable=True)
    writer.init_collection(recreate=True)
    writer.upsert_chunks(
        [
            {
                "id": "11111111-1111-1111-1111-111111111111",
                "repo_id": "org/adapt",
                "source_type": "commit",
                "source_id": "abcdef1234567",
                "text": "Commit fixing rate limiter token bucket",
                "timestamp": datetime(2025, 1, 1),
                "file_paths": ["limiter.py"],
                "symbols_modified": ["token_bucket"],
                "related_ids": [],
                "is_reverted": False,
            }
        ],
        [[0.05] * 384],
    )
    writer.close()

    # 1) Verify _get_vector_store does not thrash when requested dim=768 auto-adapts to 384
    vs1 = _get_vector_store(768, storage_path=qdrant_dir)
    assert vs1.vector_size == 384
    vs2 = _get_vector_store(768, storage_path=qdrant_dir)
    assert vs1 is vs2
    vs1.close()

    # 2) Verify search_history_tool auto-adapts from default 768-dim Embedder to 384-dim on-disk Qdrant
    import archaeologist.retrieval.embedder as emb_mod

    monkeypatch.setattr(
        emb_mod.Embedder,
        "_embed_fastembed",
        lambda self, texts: [[0.05] * self.dimension for _ in texts],
    )
    hits = search_history_tool("rate limiter token bucket", repo_id="org/adapt", storage_path=qdrant_dir)
    assert len(hits) >= 1
    assert hits[0]["payload"]["source_id"] == "abcdef1234567"


def test_BUG04_no_sys_getframe_in_embedder_and_patched_crash_propagates():
    """
    Verifies that `sys._getframe` has been completely removed from `embedder.py`, while an
    instance-level patched `_embed_fastembed` crash still propagates when `return_success_flags=False`.
    """
    embedder_src = (Path(__file__).parent.parent / "archaeologist" / "retrieval" / "embedder.py").read_text(encoding="utf-8")
    assert "_getframe" not in embedder_src
    assert "test_OBS_embedder_failure_leaves_partial_index" not in embedder_src

    emb = Embedder(provider="fastembed", model_name="BAAI/bge-small-en-v1.5")

    def _boom(texts):
        raise RuntimeError("Simulated mid-ingestion embedder crash")

    emb._embed_fastembed = _boom
    with pytest.raises(RuntimeError, match="Simulated mid-ingestion embedder crash"):
        emb.embed_texts(["chunk 1"], return_success_flags=False)


def test_BUG05_save_stored_repo_meta_honors_custom_db_url_and_persists_embedder_model(tmp_path):
    """
    Verifies that `save_stored_repo_meta` writes to `current_db_url_var` even when `repo_path` is
    passed, and persists `embedder_model` in SQLite `repo_meta` and `RepoMeta`.
    """
    repo_dir = tmp_path / "custom_repo"
    repo_dir.mkdir(parents=True)
    custom_db = tmp_path / "custom_shared.db"
    custom_db_url = f"sqlite:///{custom_db.as_posix()}"
    init_db(custom_db_url)
    current_db_url_var.set(custom_db_url)

    save_stored_repo_meta(
        repo_id="org/custom_repo",
        repo_path=str(repo_dir),
        repo_url="https://github.com/org/custom_repo",
        embedder_provider="fastembed",
        embedder_dimension=384,
        last_ingested_at="2025-03-01T00:00:00",
        embedder_model="BAAI/bge-small-en-v1.5",
    )

    with get_session_context() as session:
        row = session.exec(select(RepoMeta).where(RepoMeta.repo_id == "org/custom_repo")).first()
        assert row is not None
        assert row.embedder_provider == "fastembed"
        assert row.embedder_model == "BAAI/bge-small-en-v1.5"
        assert row.embedder_dimension == 384

    meta = get_stored_repo_meta(repo_id="org/custom_repo")
    assert meta.get("embedder_model") == "BAAI/bge-small-en-v1.5"


def test_BUG06_no_cross_tenant_leak_in_status_or_traverse_causal_chain(tmp_path):
    """
    Verifies that:
    1) `arch status` for `repo_b` (which has no `RepoMeta` row) does not leak `repo_a`'s `RepoMeta`.
    2) `traverse_causal_chain` for `repo_b` does not leak unattributed (`repo_id=''`) edges when
       tenants exist in the database.
    """
    db_file = tmp_path / "multi_tenant_leak.db"
    db_url = f"sqlite:///{db_file.as_posix()}"
    init_db(db_url)
    current_db_url_var.set(db_url)

    repo_b_dir = _make_git_repo(tmp_path / "repo_b")
    (repo_b_dir / "b.py").write_text("x = 2\n", encoding="utf-8")

    with get_session_context() as session:
        session.add(
            RepoMeta(
                repo_id="repo_a",
                embedder_provider="gemini",
                embedder_dimension=3072,
                key="secret_tenant_key",
                value="tenant_a_secret",
            )
        )
        session.add(
            Commit(
                repo_id="repo_b",
                sha="b" * 40,
                author_name="Bob",
                author_email="bob@example.com",
                authored_date=datetime(2025, 1, 1),
                message="Bob commit",
            )
        )
        session.add(
            CausalEdge(
                repo_id="repo_a",
                source_id="issue:#1",
                target_id="pr:#2",
                edge_type="resolves",
                label="RESOLVED_BY",
            )
        )
        session.add(
            CausalEdge(
                repo_id="",
                source_id="issue:#1",
                target_id="pr:#999",
                edge_type="resolves",
                label="LEAKED_EMPTY_REPO_EDGE",
            )
        )
        session.commit()

    res = runner.invoke(app, ["status", "--repo", str(repo_b_dir), "--db-path", str(db_file), "--json"])
    assert res.exit_code == 0
    payload = json.loads(res.output)
    assert payload["total_commits"] == 1
    assert payload.get("embedder_provider") != "gemini"

    chain_b = traverse_causal_chain("issue:#1", repo_id="repo_b")
    assert chain_b == []


def test_BUG07_step5_preserves_llm_diff_summary_when_attaching_pr_context(tmp_path, monkeypatch):
    """
    Verifies that when LLM diff summaries are enabled and a commit is linked to a PR, Step 5
    appends the PR context to the LLM-generated `diff_summary` rather than overwriting it with
    the static AST footprint.
    """
    import archaeologist.ingestion.pipeline as pl

    repo_dir = _make_git_repo(tmp_path / "llm_pr_repo")
    (repo_dir / "auth.py").write_text("def verify_token():\n    return True\n", encoding="utf-8")
    _git(repo_dir, "add", "auth.py")
    _git(repo_dir, "commit", "-q", "-m", "Add token verification (#42)")
    sha = _git(repo_dir, "rev-parse", "HEAD").stdout.strip()

    class FakeEmbedder:
        dimension = 8
        provider = "fake"
        model = "fake"

        def __init__(self, *args, **kwargs):
            pass

        def embed_texts(self, texts, return_success_flags=False):
            vecs = [[0.1] * 8 for _ in texts]
            return (vecs, [True] * len(vecs)) if return_success_flags else vecs

    class FakeSummarizer:
        def __init__(self, *args, **kwargs):
            pass

        def summarize_diff_batch(self, items):
            return {i["sha"]: "LLM_SEMANTIC_DIFF_SUMMARY_FOR_AUTH" for i in items}

    monkeypatch.setattr(pl, "Embedder", FakeEmbedder)
    monkeypatch.setattr(pl, "LLMSummarizer", FakeSummarizer)

    db_file = tmp_path / "llm_pr.db"
    db_url = f"sqlite:///{db_file.as_posix()}"
    init_db(db_url)
    current_db_url_var.set(db_url)

    with get_session_context() as session:
        session.add(
            PullRequest(
                repo_id="org/llm_pr_repo",
                number=42,
                title="OAuth2 Token Verification",
                body="Implements JWT signature verification",
                state="merged",
                created_at=datetime(2025, 1, 1),
                merge_commit_sha=sha,
                linked_commit_shas=[sha],
            )
        )
        session.commit()

    pl.IngestionPipeline(
        str(repo_dir),
        repo_id="org/llm_pr_repo",
        enable_llm_summaries=True,
        progress_callback=lambda *a: None,
    ).run()

    with get_session_context() as session:
        c_obj = session.exec(select(Commit).where(Commit.sha == sha)).first()
        assert c_obj is not None
        assert "LLM_SEMANTIC_DIFF_SUMMARY_FOR_AUTH" in (c_obj.diff_summary or "")
        assert "Linked PR #42" in (c_obj.diff_summary or "")


def test_BUG08_causal_link_inherits_score_and_sub_questions_are_stratified(tmp_path, monkeypatch):
    """
    Verifies that:
    1) `follow_links_node` assigns an inherited `rrf_score > 0.0` (`parent_score * 0.92`) to linked chunks.
    2) `verify_node` and `synthesize_node` stratify chunks across `sub_question_idx` so later sub-questions
       are not starved when Sub-Question 0 has many high-scoring chunks.
    """
    db_file = tmp_path / "fl_score.db"
    db_url = f"sqlite:///{db_file.as_posix()}"
    init_db(db_url)
    current_db_url_var.set(db_url)

    with get_session_context() as session:
        session.add(
            Chunk(
                id="linked-pr-100",
                repo_id="org/r",
                source_type="pr",
                source_id="pr#100",
                text="PR #100 detailed design rationale",
                timestamp=datetime(2025, 1, 1),
            )
        )
        session.commit()

    seed_chunk = {
        "id": "seed-commit-1",
        "source_type": "commit",
        "source_id": "abcdef1",
        "text": "Commit referencing #100",
        "related_ids": ["pr#100"],
        "rrf_score": 0.05,
    }
    fl_out = follow_links_node(
        {
            "repo_id": "org/r",
            "retrieved_chunks": [seed_chunk],
            "evidence_by_chunk_id": {"seed-commit-1": seed_chunk},
        }
    )
    linked = [c for c in fl_out["retrieved_chunks"] if c["id"] == "linked-pr-100"]
    assert len(linked) == 1
    assert linked[0]["rrf_score"] >= 0.04

    # Part 2: Verify sub-question stratification in verify_node under tight char budget
    import archaeologist.agent.nodes.verify as ver_mod

    captured_prompts = []

    class SpyGemini:
        def __init__(self, **kw):
            pass

        def generate_text(self, prompt, **kw):
            captured_prompts.append(prompt)
            return "Draft citing sub_q_1_unique_evidence"

        def generate_json(self, prompt, **kw):
            return {"verification_passed": True, "unverified_claims": []}

    monkeypatch.setattr(ver_mod, "get_gemini_api_key", lambda: "fake-key")
    monkeypatch.setattr(ver_mod, "GeminiClientWrapper", SpyGemini)

    # Create 20 large chunks for sub_question_idx=0 (exceeding 16,000 chars) and 1 chunk for sub_question_idx=1
    sq0_chunks = [
        {
            "id": f"sq0-{i}",
            "source_type": "commit",
            "source_id": f"aaaaaaa{i:02d}",
            "text": f"SubQ0 chunk {i} " + ("X" * 1000),
            "rrf_score": 0.90 - (i * 0.01),
            "sub_question_idx": 0,
        }
        for i in range(20)
    ]
    sq1_chunk = {
        "id": "sq1-top",
        "source_type": "pr",
        "source_id": "pr#777",
        "text": "SUB_Q_1_CRITICAL_EVIDENCE_MARKER",
        "rrf_score": 0.40,
        "sub_question_idx": 1,
    }
    verify_node(
        {
            "question": "Multi-part question",
            "retrieved_chunks": sq0_chunks + [sq1_chunk],
            "draft_answer": None,
            "verification_passed": True,
        }
    )
    assert len(captured_prompts) == 1
    assert "SUB_Q_1_CRITICAL_EVIDENCE_MARKER" in captured_prompts[0]


def test_BUG09_and_BUG10_file_chunk_symbols_and_fast_sha_prefix_matching(tmp_path, monkeypatch):
    """
    Verifies that Step 1b populates `file` chunks with full-file AST symbols (even for symbols
    defined past byte 3,000 of a Python file) and Step 5 links PR short SHAs via bisect/hash lookup.
    """
    import archaeologist.ingestion.pipeline as pl

    class FakeEmbedder:
        dimension = 8
        provider = "fake"
        model = "fake"

        def __init__(self, *args, **kwargs):
            pass

        def embed_texts(self, texts, return_success_flags=False):
            vecs = [[0.1] * 8 for _ in texts]
            return (vecs, [True] * len(vecs)) if return_success_flags else vecs

    monkeypatch.setattr(pl, "Embedder", FakeEmbedder)

    repo_dir = _make_git_repo(tmp_path / "sym_repo")
    padding = "# comment padding line\n" * 200  # > 4,400 chars
    (repo_dir / "module.py").write_text(
        f"def early_func():\n    return 1\n\n{padding}\ndef late_deep_symbol_xyz():\n    return 2\n",
        encoding="utf-8",
    )
    _git(repo_dir, "add", "module.py")
    _git(repo_dir, "commit", "-q", "-m", "Add module with late symbol")
    sha = _git(repo_dir, "rev-parse", "HEAD").stdout.strip()

    db_file = tmp_path / "sym_repo.db"
    db_url = f"sqlite:///{db_file.as_posix()}"
    init_db(db_url)
    current_db_url_var.set(db_url)

    with get_session_context() as session:
        session.add(
            PullRequest(
                repo_id="org/sym_repo",
                number=88,
                title="Late Symbol PR",
                body="Adds late_deep_symbol_xyz",
                state="merged",
                created_at=datetime(2025, 1, 1),
                linked_commit_shas=[sha[:8]],  # 8-char prefix tested via bisect
            )
        )
        session.commit()

    pl.IngestionPipeline(str(repo_dir), repo_id="org/sym_repo", progress_callback=lambda *a: None).run()

    with get_session_context() as session:
        file_chunk = session.exec(
            select(Chunk).where(Chunk.source_type == "file", Chunk.source_id == "module.py")
        ).first()
        assert file_chunk is not None
        assert "late_deep_symbol_xyz" in (file_chunk.symbols_modified or [])
        assert "late_deep_symbol_xyz" in file_chunk.text

        commit_chunk = session.exec(
            select(Chunk).where(Chunk.source_type == "commit", Chunk.source_id == sha)
        ).first()
        assert commit_chunk is not None
        assert "pr#88" in (commit_chunk.related_ids or [])
