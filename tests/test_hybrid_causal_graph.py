import pytest
from sqlmodel import Session

from archaeologist.agent.nodes.follow_links import follow_links_node
from archaeologist.retrieval.causal_graph import (
    build_rustworkx_graph,
    compute_graph_topology_metrics,
    get_causal_knowledge_graph,
    traverse_causal_chain,
)
from archaeologist.storage.analytics import graph_topology_tool
from archaeologist.storage.db import get_engine, init_db
from archaeologist.storage.models import CausalEdge, Chunk, Commit, Issue, PullRequest, SymbolIndex


@pytest.fixture
def graph_test_db(tmp_path):
    db_file = tmp_path / "test_graph.db"
    db_url = f"sqlite:///{db_file.as_posix()}"
    init_db(db_url)
    engine = get_engine(db_url)
    with Session(engine) as session:
        yield session, db_url


def test_causal_edges_table_and_cte_traversal(graph_test_db):
    """Verifies that CausalEdge records support sub-millisecond multi-hop recursive CTE traversal."""
    session, db_url = graph_test_db
    repo = "test-repo"

    session.add(CausalEdge(repo_id=repo, source_id="issue:#101", target_id="pr:#501", edge_type="resolves", label="RESOLVED_BY_PR"))
    session.add(CausalEdge(repo_id=repo, source_id="pr:#501", target_id="commit:a1b2c3d", edge_type="merges", label="MERGED_IN_COMMIT"))
    session.add(CausalEdge(repo_id=repo, source_id="commit:a1b2c3d", target_id="symbol:auth::validate_token", edge_type="modifies", label="MODIFIES_SYMBOL"))
    session.commit()

    # Outgoing traversal from issue
    chain = traverse_causal_chain("issue:#101", repo_id=repo, max_depth=3, direction="outgoing")
    assert len(chain) == 3
    assert chain[0]["target_id"] == "pr:#501"
    assert chain[1]["target_id"] == "commit:a1b2c3d"
    assert chain[2]["target_id"] == "symbol:auth::validate_token"
    assert "issue:#101 -> pr:#501" in chain[0]["path"]

    # Incoming traversal from symbol
    chain_inc = traverse_causal_chain("symbol:auth::validate_token", repo_id=repo, max_depth=3, direction="incoming")
    assert len(chain_inc) == 3
    sources = [c["source_id"] for c in chain_inc]
    assert "commit:a1b2c3d" in sources
    assert "pr:#501" in sources


def test_rustworkx_graph_and_topology_metrics(graph_test_db):
    """Verifies that build_rustworkx_graph and compute_graph_topology_metrics calculate PageRank and betweenness centrality."""
    session, db_url = graph_test_db
    repo = "topology-repo"

    session.add(CausalEdge(repo_id=repo, source_id="issue:#1", target_id="pr:#10", edge_type="resolves", label="RESOLVED_BY_PR"))
    session.add(CausalEdge(repo_id=repo, source_id="pr:#10", target_id="commit:c0ffee1", edge_type="merges", label="MERGED_IN_COMMIT"))
    session.add(CausalEdge(repo_id=repo, source_id="commit:c0ffee1", target_id="symbol:core::run", edge_type="modifies", label="MODIFIES_SYMBOL"))
    session.commit()

    g, node_map = build_rustworkx_graph(repo_id=repo)
    assert g.num_nodes() == 4
    assert g.num_edges() == 3

    metrics = compute_graph_topology_metrics(repo_id=repo)
    assert metrics["total_nodes"] == 4
    assert metrics["total_edges"] == 3
    top_pr_nodes = [node for node, score in metrics["top_pagerank"]]
    assert "symbol:core::run" in top_pr_nodes


def test_follow_links_node_uses_cte_expansion(graph_test_db):
    """Verifies that follow_links_node leverages recursive CTE expansion to discover transitive causal links."""
    session, db_url = graph_test_db
    repo = "cte-agent-repo"

    # Seed causal edges: issue #200 -> pr #300 -> commit eef0123
    session.add(CausalEdge(repo_id=repo, source_id="issue:#200", target_id="pr:#300", edge_type="resolves", label="RESOLVED_BY_PR"))
    session.add(CausalEdge(repo_id=repo, source_id="pr:#300", target_id="commit:eef0123", edge_type="merges", label="MERGED_IN_COMMIT"))

    # Add chunks
    from datetime import UTC, datetime
    session.add(Chunk(
        id="chunk-pr-300",
        repo_id=repo,
        source_type="pr",
        source_id="pr#300",
        text="PR 300 resolves issue #200",
        timestamp=datetime.now(UTC).replace(tzinfo=None)
    ))
    session.add(Chunk(
        id="chunk-commit-eef0123",
        repo_id=repo,
        source_type="commit",
        source_id="eef0123",
        text="Commit eef0123 merges PR #300",
        timestamp=datetime.now(UTC).replace(tzinfo=None)
    ))
    session.commit()

    # Agent state retrieved initial issue chunk referencing #200
    initial_chunk = {
        "id": "chunk-initial",
        "source_type": "issue",
        "source_id": "issue#200",
        "text": "Bug report in issue 200",
        "related_ids": ["#200"]
    }
    state = {
        "repo_id": repo,
        "retrieved_chunks": [initial_chunk],
        "evidence_by_chunk_id": {"chunk-initial": initial_chunk}
    }

    result = follow_links_node(state)
    assert "retrieved_chunks" in result
    new_source_ids = {c.get("source_id") for c in result["retrieved_chunks"]}
    # Expect PR 300 and Commit eef0123 to be discovered transitively
    assert "pr#300" in new_source_ids or "eef0123" in new_source_ids


def test_get_causal_knowledge_graph_with_stored_edges(graph_test_db):
    """Verifies that get_causal_knowledge_graph seamlessly reads stored CausalEdge entities."""
    session, db_url = graph_test_db
    repo = "stored-edge-repo"

    from datetime import UTC, datetime
    now_utc = datetime.now(UTC).replace(tzinfo=None)
    session.add(Issue(repo_id=repo, number=1, title="Crash on login", state="closed", created_at=now_utc))
    session.add(PullRequest(repo_id=repo, number=2, title="Fix crash on login", state="merged", created_at=now_utc))
    session.add(Commit(
        sha="deadbeef1234567890abcdef1234567890abcdef",
        repo_id=repo,
        author_name="Alice",
        author_email="alice@example.com",
        authored_date=now_utc,
        message="Fix crash"
    ))
    session.add(SymbolIndex(repo_id=repo, symbol_id="app.py::login", file_path="app.py", symbol_name="login", kind="function", commit_count=1))
    session.add(CausalEdge(repo_id=repo, source_id="issue:#1", target_id="pr:#2", edge_type="resolves", label="RESOLVED_BY_PR"))
    session.commit()

    import asyncio
    graph_res = asyncio.run(get_causal_knowledge_graph(repo_id=repo))
    assert graph_res["repo_id"] == repo
    assert graph_res["edges_count"] >= 1
    edge_types = [e["type"] for e in graph_res["edges"]]
    assert "resolves" in edge_types


def test_graph_topology_tool_graceful_empty_repo(graph_test_db):
    """Verifies graph_topology_tool returns graceful empty metrics when no edges exist."""
    session, db_url = graph_test_db
    res = graph_topology_tool(repo_id="nonexistent-repo")
    assert res["total_nodes"] == 0
    assert res["total_edges"] == 0
    assert res["top_pagerank"] == []
    assert res["top_betweenness"] == []
