import os
from collections.abc import Coroutine
from typing import Any

from sqlmodel import select

from archaeologist.mcp_server.tools import _apply_repo_id_filter
from archaeologist.storage.context import (
    current_bm25_path_var,
    current_db_url_var,
    current_repo_id_var,
)
from archaeologist.storage.db import get_session_context
from archaeologist.storage.models import CausalEdge, Commit, Issue, PullRequest, SymbolIndex
from archaeologist.storage.paths import (
    get_default_bm25_path,
    get_default_db_url,
)


class AsyncDict(dict, Coroutine):
    """A dictionary that can also be directly awaited or passed to asyncio.run()."""

    def send(self, val: Any) -> Any:
        raise StopIteration(self)

    def throw(self, *args: Any) -> Any:
        raise StopIteration(self)

    def close(self) -> None:
        pass

    def __await__(self) -> Any:
        async def _coro():
            return self

        return _coro().__await__()


def activate_repo_context(repo_id: str) -> dict[str, Any]:
    """Sets request-scoped ContextVars for the specified repository."""
    current_repo_id_var.set(repo_id)
    current_db = current_db_url_var.get() or os.environ.get("DATABASE_URL") or get_default_db_url()
    current_bm25 = current_bm25_path_var.get() or os.environ.get("BM25_INDEX_PATH") or get_default_bm25_path()
    return {
        "repo_id": repo_id,
        "name": repo_id,
        "db_path": current_db,
        "bm25_path": current_bm25,
    }


def traverse_causal_chain(
    seed_id: str,
    repo_id: str | None = None,
    max_depth: int = 3,
    direction: str = "both"
) -> list[dict[str, Any]]:
    """Executes a native SQLite Recursive CTE traversal over causal_edges in < 2ms.
    
    Supports:
      direction='outgoing' (seed -> targets)
      direction='incoming' (sources -> seed)
      direction='both'     (bidirectional causal neighborhood)
    """
    clean_seed = str(seed_id).strip()
    if not clean_seed:
        return []

    # Format seed variants (e.g. 101 -> issue:#101, pr:#101, commit:101)
    seed_variants = {clean_seed}
    if clean_seed.isdigit():
        seed_variants.update({f"issue:#{clean_seed}", f"pr:#{clean_seed}", f"#{clean_seed}"})
    elif clean_seed.startswith("#") and clean_seed[1:].isdigit():
        num = clean_seed[1:]
        seed_variants.update({f"issue:#{num}", f"pr:#{num}", num})
    elif clean_seed.lower().startswith("pr#") and clean_seed[3:].isdigit():
        num = clean_seed[3:]
        seed_variants.update({f"pr:#{num}", f"#{num}", num})
    elif clean_seed.lower().startswith("issue#") and clean_seed[6:].isdigit():
        num = clean_seed[6:]
        seed_variants.update({f"issue:#{num}", f"#{num}", num})
    elif len(clean_seed) >= 7 and all(c in "0123456789abcdefABCDEF" for c in clean_seed):
        seed_variants.add(f"commit:{clean_seed[:7]}")

    results = []
    with get_session_context() as session:
        allow_empty_repo_fallback = not bool(repo_id)
        if repo_id:
            try:
                distinct_repos = session.connection().exec_driver_sql(
                    "SELECT COUNT(DISTINCT repo_id) FROM causal_edges WHERE repo_id IS NOT NULL AND repo_id != ''"
                ).scalar() or 0
                if distinct_repos == 0:
                    allow_empty_repo_fallback = True
            except Exception:
                pass

        for seed_var in seed_variants:
            query = """
            WITH RECURSIVE traverse(node_id, depth, edge_type, label, path) AS (
                SELECT
                    target_id AS node_id,
                    1 AS depth,
                    edge_type,
                    label,
                    source_id || ' -> ' || target_id AS path
                FROM causal_edges
                WHERE source_id = :seed
                  AND (
                      :repo_id IS NULL
                      OR :repo_id = ''
                      OR repo_id = :repo_id
                      OR LOWER(repo_id) = LOWER(:repo_id)
                      OR LOWER(repo_id) LIKE '%/' || LOWER(:repo_id)
                      OR (:allow_empty = 1 AND repo_id = '')
                  )

                UNION

                SELECT
                    e.target_id AS node_id,
                    t.depth + 1 AS depth,
                    e.edge_type,
                    e.label,
                    t.path || ' -> ' || e.target_id AS path
                FROM causal_edges e
                JOIN traverse t ON e.source_id = t.node_id
                WHERE t.depth < :max_depth
                  AND (
                      :repo_id IS NULL
                      OR :repo_id = ''
                      OR e.repo_id = :repo_id
                      OR LOWER(e.repo_id) = LOWER(:repo_id)
                      OR LOWER(e.repo_id) LIKE '%/' || LOWER(:repo_id)
                      OR (:allow_empty = 1 AND e.repo_id = '')
                  )
            )
            SELECT DISTINCT node_id, depth, edge_type, label, path
            FROM traverse
            ORDER BY depth ASC;
            """
            rows = session.connection().exec_driver_sql(
                query,
                {
                    "seed": seed_var,
                    "repo_id": repo_id,
                    "max_depth": max_depth,
                    "allow_empty": 1 if allow_empty_repo_fallback else 0,
                }
            ).fetchall()

            for row in rows:
                results.append({
                    "target_id": row[0],
                    "depth": row[1],
                    "edge_type": row[2],
                    "label": row[3],
                    "path": row[4]
                })

            if direction in ("incoming", "both"):
                query_inc = """
                WITH RECURSIVE traverse_inc(node_id, depth, edge_type, label, path) AS (
                    SELECT
                        source_id AS node_id,
                        1 AS depth,
                        edge_type,
                        label,
                        source_id || ' -> ' || target_id AS path
                    FROM causal_edges
                    WHERE target_id = :seed
                      AND (
                          :repo_id IS NULL
                          OR :repo_id = ''
                          OR repo_id = :repo_id
                          OR LOWER(repo_id) = LOWER(:repo_id)
                          OR LOWER(repo_id) LIKE '%/' || LOWER(:repo_id)
                          OR (:allow_empty = 1 AND repo_id = '')
                      )

                    UNION

                    SELECT
                        e.source_id AS node_id,
                        t.depth + 1 AS depth,
                        e.edge_type,
                        e.label,
                        e.source_id || ' -> ' || t.path AS path
                    FROM causal_edges e
                    JOIN traverse_inc t ON e.target_id = t.node_id
                    WHERE t.depth < :max_depth
                      AND (
                          :repo_id IS NULL
                          OR :repo_id = ''
                          OR e.repo_id = :repo_id
                          OR LOWER(e.repo_id) = LOWER(:repo_id)
                          OR LOWER(e.repo_id) LIKE '%/' || LOWER(:repo_id)
                          OR (:allow_empty = 1 AND e.repo_id = '')
                      )
                )
                SELECT DISTINCT node_id, depth, edge_type, label, path
                FROM traverse_inc
                ORDER BY depth ASC;
                """
                rows_inc = session.connection().exec_driver_sql(
                    query_inc,
                    {
                        "seed": seed_var,
                        "repo_id": repo_id,
                        "max_depth": max_depth,
                        "allow_empty": 1 if allow_empty_repo_fallback else 0,
                    }
                ).fetchall()
                for row in rows_inc:
                    results.append({
                        "source_id": row[0],
                        "depth": row[1],
                        "edge_type": row[2],
                        "label": row[3],
                        "path": row[4]
                    })

    # Deduplicate results keeping shortest depth
    seen_nodes = set()
    deduped = []
    for r in sorted(results, key=lambda x: x["depth"]):
        nid = r.get("target_id") or r.get("source_id")
        if nid not in seen_nodes:
            seen_nodes.add(nid)
            deduped.append(r)
    return deduped


def build_rustworkx_graph(repo_id: str | None = None):
    """Constructs a high-performance rustworkx.PyDiGraph from SQLite causal_edges."""
    import rustworkx as rx

    g = rx.PyDiGraph()
    node_map = {}

    with get_session_context() as session:
        query = select(CausalEdge)
        if repo_id:
            query = _apply_repo_id_filter(query, CausalEdge.repo_id, repo_id)
        edges = session.exec(query).all()

        for e in edges:
            if e.source_id not in node_map:
                node_map[e.source_id] = g.add_node(e.source_id)
            if e.target_id not in node_map:
                node_map[e.target_id] = g.add_node(e.target_id)
            g.add_edge(node_map[e.source_id], node_map[e.target_id], {
                "edge_type": e.edge_type,
                "label": e.label,
                "highlight": e.highlight
            })

    return g, node_map


def compute_graph_topology_metrics(repo_id: str | None = None) -> dict[str, Any]:
    """Computes PageRank and Betweenness Centrality on the repository graph using rustworkx."""
    import rustworkx as rx

    g, _ = build_rustworkx_graph(repo_id=repo_id)
    if g.num_nodes() == 0:
        return {
            "total_nodes": 0,
            "total_edges": 0,
            "top_pagerank": [],
            "top_betweenness": [],
            "pagerank": {},
            "betweenness": {}
        }

    pr = rx.pagerank(g)
    bc = rx.betweenness_centrality(g)

    pr_ranked = sorted([(g[idx], round(score, 6)) for idx, score in pr.items()], key=lambda x: x[1], reverse=True)
    bc_ranked = sorted([(g[idx], round(score, 6)) for idx, score in bc.items()], key=lambda x: x[1], reverse=True)

    return {
        "total_nodes": g.num_nodes(),
        "total_edges": g.num_edges(),
        "top_pagerank": pr_ranked[:20],
        "top_betweenness": bc_ranked[:20],
    }


def get_causal_knowledge_graph(
    repo_id: str,
    limit: int = 250,
    include_all: bool = True
) -> dict[str, Any]:
    """Returns comprehensive structured nodes and causal edges across Issues, PRs, Commits, Reverts, and AST Symbols."""
    config = activate_repo_context(repo_id)

    nodes = []
    edges = []
    node_ids = set()
    edge_set = set()

    def add_edge(src, tgt, edge_type, label, highlight=False):
        edge_key = (src, tgt, edge_type)
        if edge_key not in edge_set and src in node_ids and tgt in node_ids:
            edge_set.add(edge_key)
            edges.append({
                "source": src,
                "target": tgt,
                "type": edge_type,
                "label": label,
                "highlight": highlight,
            })

    actual_limit = int(limit.default) if hasattr(limit, "default") else int(limit)

    with get_session_context() as session:
        # 1. Fetch Issues
        iss_stmt = select(Issue).order_by(Issue.created_at.desc())
        iss_stmt = _apply_repo_id_filter(iss_stmt, Issue.repo_id, repo_id)
        issues = session.exec(iss_stmt).all()
        for iss in issues:
            iss_id = f"issue:#{iss.number}"
            if iss_id not in node_ids:
                node_ids.add(iss_id)
                nodes.append({
                    "id": iss_id,
                    "type": "issue",
                    "label": f"Issue #{iss.number}",
                    "title": iss.title,
                    "author": iss.author or "contributor",
                    "state": iss.state or "closed",
                    "labels": iss.labels or [],
                    "date": iss.created_at.strftime("%Y-%m-%d") if iss.created_at else "",
                    "linked_prs": iss.linked_pr_numbers or [],
                    "linked_commits": iss.linked_commit_shas or [],
                })

        # 2. Fetch Pull Requests
        pr_stmt = select(PullRequest).order_by(PullRequest.created_at.desc())
        pr_stmt = _apply_repo_id_filter(pr_stmt, PullRequest.repo_id, repo_id)
        prs = session.exec(pr_stmt).all()
        for pr in prs:
            pr_id = f"pr:#{pr.number}"
            if pr_id not in node_ids:
                node_ids.add(pr_id)
                nodes.append({
                    "id": pr_id,
                    "type": "pr",
                    "label": f"PR #{pr.number}",
                    "title": pr.title,
                    "author": pr.author or "contributor",
                    "state": pr.state or "merged",
                    "date": pr.created_at.strftime("%Y-%m-%d") if pr.created_at else "",
                    "linked_issues": pr.linked_issue_numbers or [],
                    "linked_commits": pr.linked_commit_shas or [],
                })

        # 3. Fetch Reverts & Significant Commits
        revert_stmt = select(Commit).where(
            (Commit.is_revert == True) | (Commit.reverts_sha != None) | (Commit.superseded_by_sha != None)  # noqa: E711
        )
        revert_stmt = _apply_repo_id_filter(revert_stmt, Commit.repo_id, repo_id)
        revert_commits = session.exec(revert_stmt).all()

        # General commits
        general_stmt = select(Commit).order_by(Commit.authored_date.desc())
        general_stmt = _apply_repo_id_filter(general_stmt, Commit.repo_id, repo_id).limit(actual_limit)
        general_commits = session.exec(general_stmt).all()

        # Merge uniquely
        all_commits_dict = {c.sha: c for c in (list(revert_commits) + list(general_commits))}

        # Ensure all commits linked from PRs and Issues are loaded without arbitrary slicing
        needed_shas = set()
        for pr in prs:
            for s in (pr.linked_commit_shas or []):
                if len(s) == 40 and s not in all_commits_dict:
                    needed_shas.add(s)
        for iss in issues:
            for s in (iss.linked_commit_shas or []):
                if len(s) == 40 and s not in all_commits_dict:
                    needed_shas.add(s)

        if needed_shas:
            # Batch fetch missing commits
            chunk_size = 200
            needed_list = list(needed_shas)
            for i in range(0, len(needed_list), chunk_size):
                sub_needed = needed_list[i:i + chunk_size]
                extra_stmt = select(Commit).where(Commit.sha.in_(sub_needed))
                extra_stmt = _apply_repo_id_filter(extra_stmt, Commit.repo_id, repo_id)
                extra_commits = session.exec(extra_stmt).all()
                for c in extra_commits:
                    all_commits_dict[c.sha] = c

        commits = list(all_commits_dict.values())

        # Populate Commit Nodes
        for c in commits:
            cid = f"commit:{c.sha[:7]}"
            if cid not in node_ids:
                node_ids.add(cid)
                is_rev = bool(c.is_revert or c.reverts_sha or c.superseded_by_sha)
                nodes.append({
                    "id": cid,
                    "type": "revert" if is_rev else "commit",
                    "label": f"commit {c.sha[:7]}",
                    "title": c.message.split("\n")[0] if c.message else "Commit",
                    "sha": c.sha,
                    "author": c.author_name or "unknown",
                    "date": c.authored_date.strftime("%Y-%m-%d") if c.authored_date else "",
                    "symbols": c.symbols_modified or [],
                    "files": c.files_changed or [],
                    "is_revert": c.is_revert,
                    "reverts_sha": c.reverts_sha,
                    "superseded_by_sha": c.superseded_by_sha,
                    "insertions": c.insertions,
                    "deletions": c.deletions,
                })

        # 4. Fetch AST Symbols
        sym_stmt = select(SymbolIndex).order_by(SymbolIndex.commit_count.desc())
        sym_stmt = _apply_repo_id_filter(sym_stmt, SymbolIndex.repo_id, repo_id)
        if not include_all:
            sym_stmt = sym_stmt.limit(60)
        symbols = session.exec(sym_stmt).all()
        symbol_alias_map: dict[str, str] = {}
        from collections import Counter
        name_counts = Counter(s.symbol_name for s in symbols)
        for sym in symbols:
            if name_counts.get(sym.symbol_name, 0) > 1 or "::" in (sym.symbol_id or ""):
                canonical_key = sym.symbol_id or f"{sym.file_path}::{sym.symbol_name}"
            else:
                canonical_key = sym.symbol_name
            sym_id = f"symbol:{canonical_key}"
            if sym_id not in node_ids:
                node_ids.add(sym_id)
                nodes.append({
                    "id": sym_id,
                    "type": "symbol",
                    "label": sym.symbol_name,
                    "kind": sym.kind or "symbol",
                    "file_path": sym.file_path,
                    "commits_count": sym.commit_count,
                })
            symbol_alias_map[sym.symbol_name] = sym_id
            if sym.symbol_id:
                symbol_alias_map[sym.symbol_id] = sym_id
                if ":" in sym.symbol_id:
                    symbol_alias_map[sym.symbol_id.split(":")[-1]] = sym_id

        # 5. Build Causal Edges (Reading persistent causal_edges when available or computing dynamically)
        stored_edges = []
        try:
            edge_query = select(CausalEdge)
            edge_query = _apply_repo_id_filter(edge_query, CausalEdge.repo_id, repo_id)
            stored_edges = session.exec(edge_query).all()
        except Exception:
            stored_edges = []

        if stored_edges:
            for se in stored_edges:
                src = se.source_id
                tgt = se.target_id
                if tgt not in node_ids and tgt.startswith("symbol:"):
                    raw_sym = tgt.removeprefix("symbol:")
                    tgt = (
                        symbol_alias_map.get(raw_sym)
                        or (symbol_alias_map.get(raw_sym.split(":")[-1]) if ":" in raw_sym else None)
                        or tgt
                    )
                add_edge(src, tgt, se.edge_type, se.label, se.highlight)

        # Supplement with dynamic edge computation (deduplicated via edge_set)
        # Issue -> PR ("RESOLVED_BY")
        for iss in issues:
            iss_id = f"issue:#{iss.number}"
            for pr_num in (iss.linked_pr_numbers or []):
                pr_id = f"pr:#{pr_num}"
                add_edge(iss_id, pr_id, "resolves", "RESOLVED_BY_PR")

        for pr in prs:
            pr_id = f"pr:#{pr.number}"
            for is_num in (pr.linked_issue_numbers or []):
                iss_id = f"issue:#{is_num}"
                add_edge(iss_id, pr_id, "resolves", "RESOLVES_ISSUE")

            for sha in (pr.linked_commit_shas or []):
                c_id = f"commit:{sha[:7]}"
                add_edge(pr_id, c_id, "merges", "MERGED_IN_COMMIT")

        for c in commits:
            cid = f"commit:{c.sha[:7]}"
            if c.reverts_sha:
                target_id = f"commit:{c.reverts_sha[:7]}"
                add_edge(cid, target_id, "reverts", "REVERTS_COMMIT", highlight=True)
            if c.superseded_by_sha:
                target_id = f"commit:{c.superseded_by_sha[:7]}"
                add_edge(cid, target_id, "superseded_by", "SUPERSEDED_BY", highlight=True)

            for sym_name in (c.symbols_modified or []):
                target_sym_id = symbol_alias_map.get(sym_name)
                if not target_sym_id and ":" in sym_name:
                    target_sym_id = symbol_alias_map.get(sym_name.split(":")[-1])
                if not target_sym_id:
                    direct_id = f"symbol:{sym_name}"
                    if direct_id in node_ids:
                        target_sym_id = direct_id
                    elif ":" in sym_name and f"symbol:{sym_name.split(':')[-1]}" in node_ids:
                        target_sym_id = f"symbol:{sym_name.split(':')[-1]}"
                if target_sym_id:
                    add_edge(cid, target_sym_id, "modifies", "MODIFIES_SYMBOL")

    return AsyncDict({
        "repo_id": repo_id,
        "repo_name": config["name"],
        "total_repo_commits": len(commits),
        "nodes_count": len(nodes),
        "edges_count": len(edges),
        "nodes": nodes,
        "edges": edges,
    })
